"""Loaders that turn *real* recordings into the same table schema as the
simulator, so the same attack injectors can label them.

  load_nmea_log(path)        NEO-6M / any NMEA 0183 log  -> gnss table
  load_ais_csv(path)         Danish Maritime Authority or NOAA MarineCadastre CSV
                             (auto-detected)              -> ais table
  load_ais_nmea_log(path)    AIS-catcher / rtl_ais raw AIVDM log -> ais table

Real recordings are assumed clean (no spoofing) and become the "authentic"
class; attacks are then injected with attacks_gnss / attacks_ais.
"""
import re
import numpy as np
import pandas as pd
import pynmea2

_TS = re.compile(r"^(\S+)\s+([!$].*)$")


def load_nmea_log(path, date=None):
    """Parse GGA/RMC/GSA/GSV into one row per epoch (keyed on GGA/RMC time).
    Lines may optionally be prefixed by a timestamp (as written by many loggers)."""
    with open(path, errors="ignore") as fh:
        return parse_nmea_lines(fh, date)


def parse_nmea_lines(lines, date=None):
    """Same as load_nmea_log but from any iterable of sentences. An item may also be
    a (capture_time_s, sentence) tuple, e.g. from a pcap; the first capture time of
    each epoch is kept in column `cap_time`."""
    # An epoch starts whenever the GGA/RMC time differs from the *current* epoch's
    # time (not any earlier one), so replayed / repeated timestamps stay separate.
    epochs, cur, cur_ts = [], None, None
    last_date = date
    for line in lines:
        cap = None
        if isinstance(line, tuple):
            cap, line = line
        line = line.strip()
        m = _TS.match(line)
        if m:
            line = m.group(2)
        if not line.startswith("$"):
            continue
        try:
            msg = pynmea2.parse(line, check=True)
        except Exception:
            continue
        typ = msg.sentence_type
        if typ in ("GGA", "RMC") and msg.timestamp is not None:
            if cur is None or msg.timestamp != cur_ts:
                cur = dict(ts=msg.timestamp, cap_time=cap, sat_prn=[], sat_el=[], sat_az=[], sat_snr=[])
                cur_ts = msg.timestamp
                epochs.append(cur)
            if typ == "GGA":
                q = int(msg.gps_qual or 0)
                cur.update(fix_quality=q, lat=msg.latitude if q else np.nan, lon=msg.longitude if q else np.nan,
                           alt_m=float(msg.altitude) if msg.altitude not in (None, "") else np.nan,
                           nsat_used=int(msg.num_sats or 0),
                           hdop=float(msg.horizontal_dil) if (q and msg.horizontal_dil) else np.nan)
            else:
                if msg.datestamp:
                    last_date = msg.datestamp
                cur.update(sog_kn=float(msg.spd_over_grnd) if msg.spd_over_grnd not in (None, "") else np.nan,
                           cog_deg=float(msg.true_course) if msg.true_course not in (None, "") else np.nan,
                           date=last_date)
        elif typ == "GSA" and cur is not None:
            f = lambda v: float(v) if v not in (None, "") else np.nan
            cur.update(pdop=f(msg.pdop), vdop=f(msg.vdop))
        elif typ == "GSV" and cur is not None:
            if int(msg.msg_num) == 1:
                cur.update(sat_prn=[], sat_el=[], sat_az=[], sat_snr=[])
            for k in range(1, 5):
                prn = getattr(msg, f"sv_prn_num_{k}", "")
                if prn:
                    cur["sat_prn"].append(int(prn))
                    cur["sat_el"].append(int(getattr(msg, f"elevation_deg_{k}") or 0))
                    cur["sat_az"].append(int(getattr(msg, f"azimuth_{k}") or 0))
                    sv = getattr(msg, f"snr_{k}")
                    cur["sat_snr"].append(int(sv) if sv not in (None, "") else 0)
    rows = []
    for e in epochs:
        ts = e.pop("ts")
        d = e.get("date") or last_date or pd.Timestamp("1970-01-01").date()
        e["utc"] = pd.Timestamp.combine(d, ts.replace(tzinfo=None)).tz_localize("UTC")
        rows.append(e)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).reset_index(drop=True)   # keep log order (do not sort: replays)
    for c in ["fix_quality", "lat", "lon", "alt_m", "sog_kn", "cog_deg", "hdop", "vdop", "pdop", "nsat_used"]:
        if c not in df:
            df[c] = np.nan
    df["fix_quality"] = df.fix_quality.fillna(0).astype(int)
    tt = (df.utc - df.utc.iloc[0]).dt.total_seconds()
    # clean real logs: elapsed UTC; logs whose UTC jumps backwards (e.g. our replay
    # attacks) fall back to the epoch index, which equals seconds for 1 Hz output
    df["t"] = tt if tt.is_monotonic_increasing else np.arange(len(df), dtype=float)
    snr = df.sat_snr.map(lambda s: [x for x in s if x > 0])
    df["nsat_visible"] = df.sat_prn.map(len)
    df["snr_mean"] = snr.map(lambda s: np.mean(s) if s else np.nan)
    df["snr_std"] = snr.map(lambda s: np.std(s) if s else np.nan)
    df["snr_max"] = snr.map(lambda s: max(s) if s else np.nan)
    df["snr_min"] = snr.map(lambda s: min(s) if s else np.nan)
    df["benign_event"] = "none"
    return df.drop(columns=["date"], errors="ignore")


_COLMAPS = {
    "dma": {"# Timestamp": "utc", "MMSI": "mmsi", "Latitude": "lat", "Longitude": "lon", "SOG": "sog_kn",
            "COG": "cog_deg", "Heading": "heading", "ROT": "rot_dpm", "Navigational status": "nav_status",
            "Name": "ship_name", "Ship type": "ship_type", "Type of mobile": "mobile_type"},
    "noaa": {"BaseDateTime": "utc", "MMSI": "mmsi", "LAT": "lat", "LON": "lon", "SOG": "sog_kn",
             "COG": "cog_deg", "Heading": "heading", "Status": "nav_status", "VesselName": "ship_name",
             "VesselType": "ship_type"},
}


def load_ais_csv(path, bbox=None, max_rows=None):
    """bbox = (lat_min, lat_max, lon_min, lon_max) to crop large national files."""
    head = pd.read_csv(path, nrows=1).columns
    kind = "dma" if "# Timestamp" in head else "noaa" if "BaseDateTime" in head else None
    if kind is None:
        raise ValueError(f"Unrecognised AIS CSV columns: {list(head)}")
    cm = _COLMAPS[kind]
    df = pd.read_csv(path, usecols=[c for c in cm if c in head], nrows=max_rows, low_memory=False).rename(columns=cm)
    df["utc"] = pd.to_datetime(df.utc, dayfirst=(kind == "dma"), utc=True)
    if kind == "dma" and "mobile_type" in df:
        df = df[df.mobile_type.isin(["Class A", "Class B"])]
        df["msg_type"] = np.where(df.pop("mobile_type") == "Class A", 1, 18)
    else:
        df["msg_type"] = 1
    if bbox:
        df = df[df.lat.between(bbox[0], bbox[1]) & df.lon.between(bbox[2], bbox[3])]
    df = df.dropna(subset=["lat", "lon"]).sort_values("utc").reset_index(drop=True)
    df["t"] = (df.utc - df.utc.iloc[0]).dt.total_seconds()
    df["heading"] = pd.to_numeric(df.get("heading"), errors="coerce")
    for c, v in [("channel", "A"), ("rot_dpm", 0.0), ("nav_status", 0), ("is_own", False), ("benign_event", "none")]:
        if c not in df:
            df[c] = v
    df["nav_status"] = pd.to_numeric(df.nav_status, errors="coerce").fillna(0)
    df["vessel_uid"] = "mmsi_" + df.mmsi.astype(str)
    df["truth_lat"], df["truth_lon"] = df.lat, df.lon
    return df


def load_ais_nmea_log(path):
    """Raw AIVDM log, one sentence per line, optionally '<ISO time> <sentence>'."""
    from pyais import decode
    rows, buf = [], []
    for line in lines:
        cap = None
        if isinstance(line, tuple):
            cap, line = line
        line = line.strip()
        m = _TS.match(line)
        ts = pd.Timestamp(m.group(1)) if m else pd.NaT
        s = m.group(2) if m else line
        if not s.startswith("!AIVD"):
            continue
        parts = s.split(",")
        buf.append(s)
        if parts[1] != parts[2]:
            continue
        try:
            d = decode(*buf).asdict()
        except Exception:
            buf = []
            continue
        buf = []
        rows.append(dict(utc=ts, mmsi=d.get("mmsi"), msg_type=d.get("msg_type"), channel=parts[4],
                         lat=d.get("lat"), lon=d.get("lon"), sog_kn=d.get("speed"), cog_deg=d.get("course"),
                         heading=d.get("heading"), rot_dpm=d.get("turn"), nav_status=d.get("status"),
                         ship_name=d.get("shipname"), ship_type=d.get("ship_type"),
                         is_own=s.startswith("!AIVDO")))
    df = pd.DataFrame(rows)
    if df.utc.notna().any():
        df["t"] = (df.utc - df.utc.dropna().iloc[0]).dt.total_seconds()
    df["vessel_uid"] = "mmsi_" + df.mmsi.astype(str)
    df["benign_event"] = "none"
    return df
