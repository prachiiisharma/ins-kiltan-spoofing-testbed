"""
Streaming NMEA-0183 feature extractor.

The SAME code is used for:
  * offline training on MARSIM pcaps   (build_features.py)
  * live detection in the testbed hub  (detector.py -> NEO-6M / MQTT stream)
so the model sees identical features in both places.

Feed it one sentence at a time:
    fx = NmeaFeatureExtractor()
    rows = fx.feed(receiver_id, arrival_time, "$GPGGA,...")   # list of finished epochs
    rows += fx.flush()                                        # at end of stream

Each row is one GPS fix ("epoch") of one receiver: raw values + consistency features.
Model features deliberately avoid absolute lat/lon/speed so a model trained on MARSIM
(20 kn, North Sea route) still applies to your own ship platform.
"""
import math
from datetime import datetime, date, timezone

EARTH_R = 6371008.8
KN = 0.514444  # m/s per knot

# ---- features given to the anomaly model ------------------------------------------------
# single receiver: works with one NEO-6M
SINGLE_RX_FEATURES = [
    "speed_resid_kn", "abs_cog_resid_deg", "abs_rot_deg_s", "dr_resid_m", "abs_alt_diff_m",
    "clock_step_s", "clock_dev_s", "hdop", "num_sats", "n_in_view",
    "cn0_mean", "cn0_std", "cn0_range", "sat_jaccard",
    "dr_resid_m_r10", "speed_resid_kn_r10", "clock_step_s_r10std", "cn0_std_r10",
]
# extra features when two receivers are mounted a known distance apart (MARSIM: 4 m)
DUAL_RX_FEATURES = SINGLE_RX_FEATURES + ["rx_dist_dev_m", "rx_sog_diff_kn", "rx_clock_diff_s"]


# ---- small helpers -------------------------------------------------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _latlon(v, hemi, deg_digits):
    if not v:
        return None
    try:
        d = float(v[:deg_digits]) + float(v[deg_digits:]) / 60.0
    except ValueError:
        return None
    return -d if hemi in ("S", "W") else d


def _utc(hhmmss, ddmmyy=None, fallback_date=None):
    if not hhmmss or len(hhmmss) < 6:
        return None, fallback_date
    d = fallback_date
    if ddmmyy and len(ddmmyy) == 6:
        try:
            yy = int(ddmmyy[4:6])
            d = date(2000 + yy if yy < 80 else 1900 + yy, int(ddmmyy[2:4]), int(ddmmyy[0:2]))
        except ValueError:
            pass
    if d is None:
        d = date(2000, 1, 1)
    try:
        sec = float(hhmmss[4:])
        t = datetime(d.year, d.month, d.day, int(hhmmss[0:2]), int(hhmmss[2:4]), int(sec),
                     int(round((sec % 1) * 1e6)) % 1000000, tzinfo=timezone.utc)
    except ValueError:
        return None, d
    return t.timestamp(), d


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def destination(lat, lon, brg_deg, dist_m):
    """Dead-reckoning step: position after moving dist_m on bearing brg_deg."""
    p1, l1, b, d = math.radians(lat), math.radians(lon), math.radians(brg_deg), dist_m / EARTH_R
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def wrap180(a):
    return (a + 180) % 360 - 180


def nmea_checksum_ok(s):
    if "*" not in s:
        return True  # no checksum given -> accept
    body, cs = s[1:].split("*", 1)
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(cs[:2], 16)
    except ValueError:
        return False


# ---- extractor -----------------------------------------------------------------------------
class _RxState:
    def __init__(self):
        self.epoch = None          # dict being filled for the current fix
        self.prev = None           # last finished epoch (raw values)
        self.date = None           # last date seen in RMC
        self.gsv = {}              # prn -> (elev, az, snr) for the current GSV cycle
        self.gsv_done = {}         # last complete GSV cycle
        self.clock_base = []       # first offsets, for clock_dev baseline


class NmeaFeatureExtractor:
    TIME_SENTENCES = ("GGA", "RMC", "GLL")

    def __init__(self, baseline_epochs=10, check_checksum=True):
        self.rx = {}
        self.baseline_epochs = baseline_epochs
        self.check_checksum = check_checksum
        self.bad_sentences = 0

    # public API --------------------------------------------------------------
    def feed(self, rx_id, arrival_t, sentence):
        """Feed one NMEA sentence. Returns a list of finished epoch rows (0 or 1)."""
        sentence = sentence.strip()
        if len(sentence) < 7 or sentence[0] not in "$!":
            return []
        if self.check_checksum and not nmea_checksum_ok(sentence):
            self.bad_sentences += 1
            return []
        st = self.rx.setdefault(rx_id, _RxState())
        fields = sentence[1:].split("*")[0].split(",")
        kind = fields[0][-3:]
        out = []
        if kind in self.TIME_SENTENCES:
            utc, st.date = _utc(fields[1] if kind != "GLL" else (fields[5] if len(fields) > 5 else None),
                                fields[9] if kind == "RMC" and len(fields) > 9 else None, st.date)
            if st.epoch is not None and utc is not None and st.epoch.get("utc") is not None \
                    and abs(utc - st.epoch["utc"]) > 1e-3:
                out.append(self._finish(rx_id, st))
            if st.epoch is None:
                st.epoch = {"rx": rx_id, "t": arrival_t, "utc": utc}
            elif st.epoch.get("utc") is None:
                st.epoch["utc"] = utc
        if st.epoch is None:
            st.epoch = {"rx": rx_id, "t": arrival_t, "utc": None}
        getattr(self, "_p_" + kind, lambda *a: None)(st, fields)
        return [r for r in out if r is not None]

    def flush(self):
        out = []
        for rx_id, st in self.rx.items():
            if st.epoch is not None:
                r = self._finish(rx_id, st)
                if r is not None:
                    out.append(r)
        return out

    # sentence parsers ---------------------------------------------------------
    def _p_GGA(self, st, f):
        e = st.epoch
        if len(f) < 10:
            return
        e["lat"] = _latlon(f[2], f[3], 2) if f[2] else e.get("lat")
        e["lon"] = _latlon(f[4], f[5], 3) if f[4] else e.get("lon")
        e["fix_q"], e["num_sats"], e["hdop"], e["alt_m"] = _f(f[6]), _f(f[7]), _f(f[8]), _f(f[9])

    def _p_RMC(self, st, f):
        e = st.epoch
        if len(f) < 9:
            return
        e["status_ok"] = 1.0 if f[2] == "A" else 0.0
        if f[3] and e.get("lat") is None:
            e["lat"], e["lon"] = _latlon(f[3], f[4], 2), _latlon(f[5], f[6], 3)
        e["sog_kn"], e["cog_deg"] = _f(f[7]), _f(f[8])

    def _p_GLL(self, st, f):
        e = st.epoch
        if len(f) > 4 and f[1] and e.get("lat") is None:
            e["lat"], e["lon"] = _latlon(f[1], f[2], 2), _latlon(f[3], f[4], 3)

    def _p_VTG(self, st, f):
        e = st.epoch
        if len(f) > 5:
            if e.get("cog_deg") is None:
                e["cog_deg"] = _f(f[1])
            if e.get("sog_kn") is None:
                e["sog_kn"] = _f(f[5])

    def _p_GSA(self, st, f):
        e = st.epoch
        if len(f) >= 18:
            e["pdop"], e["hdop_gsa"], e["vdop"] = _f(f[15]), _f(f[16]), _f(f[17])
            e["used_prns"] = {p for p in f[3:15] if p}

    def _p_GSV(self, st, f):
        if len(f) < 4:
            return
        total, num = _f(f[1]), _f(f[2])
        if total is None or num is None:
            return
        # st.gsv: message number -> {prn: (elev, az, snr)}; robust to out-of-order parts
        if num in st.gsv:          # same part seen again -> a new GSV cycle has started
            st.gsv = {}
        st.epoch["n_in_view"] = _f(f[3])
        part = {}
        for i in range(4, len(f) - 3, 4):
            if f[i]:
                part[f[i]] = (_f(f[i + 1]), _f(f[i + 2]), _f(f[i + 3]))
        st.gsv[num] = part
        if all(float(k) in st.gsv for k in range(1, int(total) + 1)):
            st.gsv_done = {p: v for d in st.gsv.values() for p, v in d.items()}
            st.gsv = {}

    # epoch -> feature row ----------------------------------------------------------
    def _finish(self, rx_id, st):
        e, st.epoch = st.epoch, None
        if e.get("lat") is None or e.get("lon") is None:
            return None
        sats = st.gsv_done
        snr = [v[2] for v in sats.values() if v[2] is not None]
        row = {
            "rx": rx_id, "t": e["t"], "utc": e.get("utc"),
            "lat": e["lat"], "lon": e["lon"], "alt_m": e.get("alt_m"),
            "sog_kn": e.get("sog_kn"), "cog_deg": e.get("cog_deg"),
            "fix_q": e.get("fix_q"), "num_sats": e.get("num_sats"),
            "hdop": e.get("hdop") if e.get("hdop") is not None else e.get("hdop_gsa"),
            "pdop": e.get("pdop"), "vdop": e.get("vdop"),
            "n_in_view": e.get("n_in_view", len(sats) or None),
            "cn0_mean": sum(snr) / len(snr) if snr else None,
            "cn0_std": (sum((s - sum(snr) / len(snr)) ** 2 for s in snr) / len(snr)) ** 0.5 if snr else None,
            "cn0_range": (max(snr) - min(snr)) if snr else None,
            "_prns": set(sats.keys()),
        }
        # receiver clock vs arrival clock (replay / meaconing shift this)
        off = (row["utc"] - row["t"]) if row["utc"] is not None else None
        row["clock_offset_s"] = off
        if off is not None and len(st.clock_base) < self.baseline_epochs:
            st.clock_base.append(off)
        base = sorted(st.clock_base)[len(st.clock_base) // 2] if st.clock_base else None
        row["clock_dev_s"] = (off - base) if (off is not None and base is not None) else None

        p = st.prev
        if p is not None:
            dt = row["t"] - p["t"]
            if row["utc"] is not None and p["utc"] is not None and row["utc"] - p["utc"] > 0:
                dt_fix = row["utc"] - p["utc"]
            else:
                dt_fix = dt if dt > 0 else None
            dist = haversine_m(p["lat"], p["lon"], row["lat"], row["lon"])
            row["dist_m"] = dist
            if dt_fix:
                row["speed_implied_kn"] = dist / dt_fix / KN
                if row["sog_kn"] is not None:
                    row["speed_resid_kn"] = row["speed_implied_kn"] - row["sog_kn"]
                if p.get("sog_kn") is not None and p.get("cog_deg") is not None:
                    dlat, dlon = destination(p["lat"], p["lon"], p["cog_deg"], p["sog_kn"] * KN * dt_fix)
                    row["dr_resid_m"] = haversine_m(dlat, dlon, row["lat"], row["lon"])
                if row["cog_deg"] is not None and p.get("cog_deg") is not None:
                    row["abs_rot_deg_s"] = abs(wrap180(row["cog_deg"] - p["cog_deg"])) / dt_fix
            if dist > 0.5 and row["cog_deg"] is not None:
                row["abs_cog_resid_deg"] = abs(wrap180(bearing_deg(p["lat"], p["lon"], row["lat"], row["lon"])
                                                       - row["cog_deg"]))
            if row["alt_m"] is not None and p.get("alt_m") is not None:
                row["abs_alt_diff_m"] = abs(row["alt_m"] - p["alt_m"])
            if off is not None and p.get("clock_offset_s") is not None:
                row["clock_step_s"] = off - p["clock_offset_s"]
            a, b = row["_prns"], p["_prns"]
            row["sat_jaccard"] = len(a & b) / len(a | b) if (a or b) else None
        st.prev = row
        out = {k: v for k, v in row.items() if k != "_prns"}
        return out


# ---- table-level helpers (pandas) ---------------------------------------------------------
def add_rolling(df, window=10):
    """Rolling features per receiver. df must be sorted by time within each receiver."""
    g = df.groupby("rx", sort=False)
    df["dr_resid_m_r10"] = g["dr_resid_m"].transform(lambda s: s.rolling(window, min_periods=3).mean())
    df["speed_resid_kn_r10"] = g["speed_resid_kn"].transform(lambda s: s.rolling(window, min_periods=3).mean())
    df["clock_step_s_r10std"] = g["clock_step_s"].transform(lambda s: s.rolling(window, min_periods=3).std())
    df["cn0_std_r10"] = g["cn0_std"].transform(lambda s: s.rolling(window, min_periods=3).mean())
    return df


def combine_receivers(df, primary_rx, secondary_rx, baseline_dist_m=None, baseline_epochs=10):
    """
    One row per second for the primary receiver, with cross-receiver features added.
    baseline_dist_m: known antenna separation (MARSIM: 4 m). If None it is estimated
    from the first `baseline_epochs` seconds.
    """
    import pandas as pd
    a = df[df.rx == primary_rx].copy()
    b = df[df.rx == secondary_rx][["t", "lat", "lon", "sog_kn", "clock_offset_s"]].copy()
    if b.empty:
        for c in ("rx_dist_m", "rx_dist_dev_m", "rx_sog_diff_kn", "rx_clock_diff_s"):
            a[c] = None
        return a
    a = a.sort_values("t")
    b = b.sort_values("t")
    m = pd.merge_asof(a, b, on="t", direction="nearest", tolerance=0.6, suffixes=("", "_b"))
    ok = m["lat_b"].notna()
    m["rx_dist_m"] = None
    m.loc[ok, "rx_dist_m"] = [haversine_m(r.lat, r.lon, r.lat_b, r.lon_b)
                              for r in m.loc[ok, ["lat", "lon", "lat_b", "lon_b"]].itertuples()]
    m["rx_dist_m"] = m["rx_dist_m"].astype(float)
    base = baseline_dist_m if baseline_dist_m is not None else m["rx_dist_m"].head(baseline_epochs).median()
    m["rx_dist_dev_m"] = (m["rx_dist_m"] - base).abs()
    m["rx_sog_diff_kn"] = (m["sog_kn"] - m["sog_kn_b"]).abs()
    m["rx_clock_diff_s"] = (m["clock_offset_s"] - m["clock_offset_s_b"]).abs()
    return m.drop(columns=["lat_b", "lon_b", "sog_kn_b", "clock_offset_s_b"])
