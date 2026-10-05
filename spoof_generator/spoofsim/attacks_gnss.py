"""Data-layer GNSS spoofing attacks (NMEA level, no RF).

Every attack edits the *measured* receiver output inside [start, start+duration)
and labels those epochs. Works identically on simulated data and on real
NEO-6M logs loaded via `real_data.load_nmea_log`.

Attack catalogue
----------------
jump         sudden position offset (300 m - 5 km), velocity left consistent
drag_off     slow, velocity-consistent pull-off (the stealthy capture attack)
replay       position/velocity/time replayed from D seconds earlier
freeze       position and velocity frozen at onset value
time_jump    UTC offset only; position untouched (timing attack)
Each attack also carries a `signal_signature` flag: when True the spoofer
"overpowers" the sky, giving uniformly high C/N0 and a tidy constellation
(the classic signal-level tell); when False it matches the authentic levels.
"""
import numpy as np
import pandas as pd

from .geo import ll_to_enu, enu_to_ll, vel_to_sog_cog, KN

GNSS_ATTACKS = ["jump", "drag_off", "replay", "freeze", "time_jump"]


def init_labels(df):
    df["is_spoofed"] = 0
    df["attack_type"] = "none"
    df["attack_id"] = ""
    df["spoof_signature"] = 0
    return df


def random_gnss_attack(rng, kind, start, duration):
    p = dict(type=kind, start=float(start), duration=float(duration),
             signal_signature=bool(rng.random() < 0.6))
    if kind == "jump":
        p.update(offset_m=float(rng.uniform(300, 5000)), bearing_deg=float(rng.uniform(0, 360)))
    elif kind == "drag_off":
        p.update(accel_ms2=float(rng.uniform(0.002, 0.02)), max_drift_ms=float(rng.uniform(0.3, 3.0)),
                 bearing_deg=float(rng.uniform(0, 360)))
    elif kind == "replay":
        p.update(delay_s=float(rng.uniform(30, 600)))
    elif kind == "time_jump":
        p.update(offset_s=float(rng.choice([-1, 1]) * rng.uniform(2, 3600)))
    return p


def apply_gnss_attack(df: pd.DataFrame, spec: dict, attack_id: str, ref_lat, ref_lon, rng):
    t = df.t.values
    m = (t >= spec["start"]) & (t < spec["start"] + spec["duration"]) & (df.fix_quality.values > 0)
    if not m.any():
        return df
    idx = np.where(m)[0]
    k = spec["type"]
    x, y = ll_to_enu(df.lat.values, df.lon.values, ref_lat, ref_lon)
    x, y = np.array(x, float), np.array(y, float)
    tau = t[idx] - spec["start"]

    if k == "jump":
        b = np.radians(spec["bearing_deg"])
        x[idx] += spec["offset_m"] * np.sin(b)
        y[idx] += spec["offset_m"] * np.cos(b)
    elif k == "drag_off":
        b = np.radians(spec["bearing_deg"])
        v = np.minimum(spec["accel_ms2"] * tau, spec["max_drift_ms"])
        t_sat = spec["max_drift_ms"] / spec["accel_ms2"]
        d = np.where(tau < t_sat, 0.5 * spec["accel_ms2"] * tau ** 2,
                     0.5 * spec["accel_ms2"] * t_sat ** 2 + spec["max_drift_ms"] * (tau - t_sat))
        x[idx] += d * np.sin(b)
        y[idx] += d * np.cos(b)
        # spoofer keeps Doppler consistent with the fake track
        ve = df.sog_kn.values[idx] * KN * np.sin(np.radians(df.cog_deg.values[idx])) + v * np.sin(b)
        vn = df.sog_kn.values[idx] * KN * np.cos(np.radians(df.cog_deg.values[idx])) + v * np.cos(b)
        sog, cog = vel_to_sog_cog(ve, vn)
        df.loc[df.index[idx], "sog_kn"] = sog / KN
        df.loc[df.index[idx], "cog_deg"] = cog
    elif k == "replay":
        src = np.searchsorted(t, t[idx] - spec["delay_s"])
        src = np.clip(src, 0, len(t) - 1)
        x[idx], y[idx] = x[src], y[src]
        for c in ["sog_kn", "cog_deg"]:
            df.loc[df.index[idx], c] = df[c].values[src]
        df.loc[df.index[idx], "utc"] = df.utc.iloc[src].array
    elif k == "freeze":
        x[idx], y[idx] = x[idx[0]], y[idx[0]]
        for c in ["sog_kn", "cog_deg", "alt_m"]:
            df.loc[df.index[idx], c] = df[c].values[idx[0]]
    elif k == "time_jump":
        df.loc[df.index[idx], "utc"] = (df.utc.iloc[idx] + pd.Timedelta(milliseconds=round(spec["offset_s"] * 1000))).astype(df.utc.dtype).array

    lat, lon = enu_to_ll(x[idx], y[idx], ref_lat, ref_lon)
    df.loc[df.index[idx], "lat"] = lat
    df.loc[df.index[idx], "lon"] = lon

    if spec.get("signal_signature"):
        # Spoofer transmits all PRNs from one antenna: near-identical, high C/N0,
        # no sector masking, clean geometry.
        level = rng.uniform(44, 49)
        spread = rng.uniform(0.6, 1.5)
        for j in idx:
            nsat = len(df.at[df.index[j], "sat_snr"])
            snr = np.clip(np.round(level + rng.normal(0, spread, nsat)), 15, 55).astype(int).tolist()
            df.at[df.index[j], "sat_snr"] = snr
        s = df.loc[df.index[idx], "sat_snr"]
        df.loc[df.index[idx], "snr_mean"] = s.map(np.mean).values
        df.loc[df.index[idx], "snr_std"] = s.map(np.std).values
        df.loc[df.index[idx], "snr_max"] = s.map(max).values
        df.loc[df.index[idx], "snr_min"] = s.map(min).values
        # spoofer's simulated sky: clean, slowly varying geometry
        h0 = rng.uniform(0.7, 1.0)
        df.loc[df.index[idx], "hdop"] = np.round(h0 + 0.08 * np.sin(2 * np.pi * tau / rng.uniform(600, 1800))
                                                 + rng.normal(0, 0.01, len(idx)), 2)
        df.loc[df.index[idx], "spoof_signature"] = 1

    df.loc[df.index[idx], "is_spoofed"] = 1
    df.loc[df.index[idx], "attack_type"] = k
    df.loc[df.index[idx], "attack_id"] = attack_id
    return df
