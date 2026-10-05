"""AIS transmitter + receiver model.

Reporting intervals follow ITU-R M.1371 (Class A: 10 s / 6 s / 2 s by speed,
faster while turning; Class B "CS": 30 s). Positions of other vessels carry
their own GNSS error. Own ship's AIS (VDO) is fed from *its own measured GNSS*,
so a GNSS spoof on own ship propagates into own-ship AIS exactly as on a real
ship. Reception probability falls with range (VHF horizon).
"""
import numpy as np
import pandas as pd

from .geo import haversine_m, enu_to_ll
from .gnss import gauss_markov

# Maritime Identification Digits commonly seen off the west coast of India
MIDS = [419, 419, 419, 636, 538, 563, 477, 370, 351, 248, 229, 215, 566, 403, 470]

NAMES = ["SAGAR", "VAISHNAVI", "JAL", "TARANG", "KALPANA", "MALABAR STAR", "OCEAN PEARL",
         "ARABIAN DAWN", "SEA HAWK", "KONKAN", "SAMUDRA", "DEEP BLUE", "KAVERI", "GODAVARI",
         "MERCURY", "PACIFIC GRACE", "NORDIC SUN", "ATLANTIC ACE", "GULF PRIDE", "EVER BRIGHT"]


def random_mmsi(rng, used=None):
    used = used or set()
    while True:
        m = int(rng.choice(MIDS)) * 1_000_000 + int(rng.integers(0, 999_999))
        if m not in used:
            return m


def report_interval(sog_kn, rot_dps, ais_class):
    if ais_class == "B":
        return 30.0 if sog_kn > 2 else 180.0
    turning = abs(rot_dps) > 0.2
    if sog_kn < 0.5:
        return 180.0
    if sog_kn <= 14:
        return 3.33 if turning else 10.0
    if sog_kn <= 23:
        return 2.0 if turning else 6.0
    return 2.0


def rx_probability(range_m, ais_class):
    r = range_m / (1.0 if ais_class == "A" else 0.6)
    if r < 20_000:
        return 0.97
    if r < 60_000:
        return 0.97 - 0.85 * (r - 20_000) / 40_000
    return 0.05


def generate_ais(rng, vessel: dict, truth: pd.DataFrame, own_truth: pd.DataFrame, ref_lat, ref_lon,
                 t0_utc, own_gnss: pd.DataFrame = None, own_aux: pd.DataFrame = None, gaps=None):
    """vessel: {uid, mmsi, name, profile, is_own}. Returns AIS message rows."""
    prof = vessel["profile"]
    is_own = vessel["is_own"]
    n = len(truth)
    ex = gauss_markov(rng, n, 3.0, 80)
    ey = gauss_markov(rng, n, 3.0, 80)
    gaps = gaps or []
    rows = []
    t_next = rng.uniform(0, 10)
    t_static = rng.uniform(0, 360)
    channel = "A"
    after_gap = False
    T = truth.t.values
    while t_next < T[-1]:
        i = int(np.searchsorted(T, t_next))
        i = min(i, n - 1)
        tr = truth.iloc[i]
        sog_true_kn = tr.sog_ms / 0.514444
        interval = report_interval(sog_true_kn, tr.rot_dps, prof.ais_class)
        t_tx = t_next
        t_next += interval * rng.uniform(0.95, 1.05)

        in_gap = any(g["start"] <= t_tx < g["end"] for g in gaps)
        if in_gap:
            after_gap = True
            continue
        if not is_own:
            rng_m = haversine_m(own_truth.lat.iat[i], own_truth.lon.iat[i], tr.lat, tr.lon)
            if rng.random() > rx_probability(rng_m, prof.ais_class):
                continue

        if is_own:
            g = own_gnss.iloc[i]
            if g.fix_quality == 0 or not np.isfinite(g.lat):
                lat, lon, sog, cog = 91.0, 181.0, np.nan, np.nan   # AIS "not available"
            else:
                lat, lon, sog, cog = g.lat, g.lon, g.sog_kn, g.cog_deg
            heading = own_aux.gyro_heading_deg.iat[i]
        else:
            lat, lon = enu_to_ll(tr.x_m + ex[i], tr.y_m + ey[i], ref_lat, ref_lon)
            sog = max(sog_true_kn + rng.normal(0, 0.1), 0)
            cog = (tr.cog_deg + rng.normal(0, 0.5 if sog_true_kn > 2 else 5)) % 360
            heading = (tr.heading_deg + rng.normal(0, 0.5)) % 360
        channel = "B" if channel == "A" else "A"
        msg_type = 1 if prof.ais_class == "A" else 18
        nav = 7 if prof.name == "fishing" else 0
        rows.append(dict(t=round(t_tx, 2), mmsi=vessel["mmsi"], msg_type=msg_type, channel=channel,
                         lat=float(lat), lon=float(lon), sog_kn=float(np.round(sog, 1)) if np.isfinite(sog) else np.nan,
                         cog_deg=float(np.round(cog, 1)) if np.isfinite(cog) else np.nan,
                         heading=int(round(heading)) % 360, rot_dpm=float(np.round(tr.rot_dps * 60, 1)),
                         nav_status=nav, ship_name=None, ship_type=None,
                         vessel_uid=vessel["uid"], is_own=is_own,
                         truth_lat=tr.lat, truth_lon=tr.lon,
                         benign_event="after_reception_gap" if after_gap else "none"))
        after_gap = False

        if t_tx >= t_static:   # periodic static & voyage data
            t_static += 360
            rows.append(dict(t=round(t_tx + 0.5, 2), mmsi=vessel["mmsi"],
                             msg_type=5 if prof.ais_class == "A" else 24, channel=channel,
                             lat=np.nan, lon=np.nan, sog_kn=np.nan, cog_deg=np.nan, heading=np.nan,
                             rot_dpm=np.nan, nav_status=np.nan, ship_name=vessel["name"],
                             ship_type=prof.ship_type, vessel_uid=vessel["uid"], is_own=is_own,
                             truth_lat=np.nan, truth_lon=np.nan, benign_event="none"))
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(1, "utc", t0_utc + pd.to_timedelta(df.t, unit="s"))
    return df
