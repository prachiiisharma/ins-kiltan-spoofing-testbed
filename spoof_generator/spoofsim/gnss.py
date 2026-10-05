"""Own-ship GNSS receiver model (NEO-6M-like, GPS L1 only).

Produces per-epoch (1 Hz) measurements with:
  * first-order Gauss-Markov position error (~2-3 m 1-sigma) + white noise
  * Doppler-grade velocity noise
  * a slowly moving satellite constellation with elevation-dependent C/N0
  * HDOP/VDOP/PDOP computed from the actual visible geometry
Benign degradations (multipath/partial masking, short fix outages) are
scheduled here and labelled in `benign_event`.
"""
import numpy as np
import pandas as pd

from .geo import enu_to_ll, vel_to_sog_cog


class Constellation:
    def __init__(self, rng, n_sats=31):
        self.prn = rng.choice(np.arange(1, 33), n_sats, replace=False)
        # each satellite traces a slow great-circle-ish pass: el(t), az(t)
        self.el_peak = rng.uniform(15, 88, n_sats)
        self.phase = rng.uniform(0, 2 * np.pi, n_sats)
        self.period = rng.uniform(4, 6, n_sats) * 3600          # visible arc length
        self.az0 = rng.uniform(0, 360, n_sats)
        self.az_rate = rng.uniform(-0.01, 0.01, n_sats)            # deg/s
        self.snr_bias = rng.normal(0, 1.5, n_sats)

    def at(self, t):
        el = self.el_peak * np.sin(self.phase + 2 * np.pi * t / self.period)
        az = (self.az0 + self.az_rate * t) % 360
        return el, az


def dops(el_deg, az_deg):
    if len(el_deg) < 4:
        return np.nan, np.nan, np.nan
    el, az = np.radians(el_deg), np.radians(az_deg)
    H = np.column_stack([-np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), -np.sin(el), np.ones_like(el)])
    try:
        Q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    hdop = np.sqrt(Q[0, 0] + Q[1, 1])
    vdop = np.sqrt(Q[2, 2])
    return hdop, vdop, np.sqrt(hdop ** 2 + vdop ** 2)


def gauss_markov(rng, n, sigma, tau, dt=1.0):
    a = np.exp(-dt / tau)
    w = rng.normal(0, sigma * np.sqrt(1 - a * a), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for i in range(1, n):
        out[i] = a * out[i - 1] + w[i]
    return out


def simulate_gnss(rng, truth: pd.DataFrame, ref_lat, ref_lon, t0_utc: pd.Timestamp,
                  degradations=None, elev_mask=10.0, max_channels=12):
    """degradations: list of {type: 'multipath'|'outage', start, end}"""
    n = len(truth)
    const = Constellation(rng)
    sig = rng.uniform(1.8, 3.0)
    ex = gauss_markov(rng, n, sig, rng.uniform(40, 120)) + rng.normal(0, 0.6, n)
    ey = gauss_markov(rng, n, sig, rng.uniform(40, 120)) + rng.normal(0, 0.6, n)
    alt = 2.0 + gauss_markov(rng, n, sig * 1.8, 90)
    degradations = degradations or []

    rows, sat_prn, sat_el, sat_az, sat_snr = [], [], [], [], []
    for i in range(n):
        t = truth.t.iat[i]
        ev = "none"
        for d in degradations:
            if d["start"] <= t < d["end"]:
                ev = "gnss_" + d["type"]
                deg = d
        el, az = const.at(t)
        vis = el > elev_mask
        snr = 28 + 17 * np.sin(np.radians(np.clip(el, 0, 90))) + const.snr_bias + rng.normal(0, 1.2, len(el))
        noise_mult = 1.0
        fix = 1
        if ev == "gnss_multipath":
            # superstructure/sector masking + reflected-signal noise
            sector = (az - deg.setdefault("sector", rng.uniform(0, 360))) % 360 < 180
            vis &= ~(sector & (el < 60))
            snr = snr - rng.uniform(3, 8)
            noise_mult = deg.setdefault("mult", rng.uniform(1.5, 3.5))
        elif ev == "gnss_outage":
            vis[:] = False
            fix = 0
        idx = np.where(vis)[0]
        idx = idx[np.argsort(-el[idx])][:max_channels]
        snr = np.clip(snr, 15, 50)
        h, v, p = dops(el[idx], az[idx])
        if len(idx) < 4:
            fix = 0
        hf = min(h, 2.5) if np.isfinite(h) else 1.0      # DOP amplification, capped
        e_err = ex[i] * noise_mult * hf
        n_err = ey[i] * noise_mult * hf
        xm = truth.x_m.iat[i] + e_err
        ym = truth.y_m.iat[i] + n_err
        lat, lon = enu_to_ll(xm, ym, ref_lat, ref_lon)
        vstd = 0.05 * noise_mult
        ve = truth.ve_ms.iat[i] + rng.normal(0, vstd)
        vn = truth.vn_ms.iat[i] + rng.normal(0, vstd)
        sog, cog = vel_to_sog_cog(ve, vn)
        s = snr[idx]
        rows.append((t, fix, float(lat) if fix else np.nan, float(lon) if fix else np.nan,
                     alt[i] if fix else np.nan, sog / 0.514444 if fix else np.nan,
                     cog if fix else np.nan, h if fix else np.nan, v if fix else np.nan,
                     p if fix else np.nan, len(idx) if fix else 0,
                     len(np.where(el > elev_mask)[0]),
                     s.mean() if len(s) else np.nan, s.std() if len(s) else np.nan,
                     s.max() if len(s) else np.nan, s.min() if len(s) else np.nan, ev))
        sat_prn.append(const.prn[idx].tolist())
        sat_el.append(np.round(el[idx]).astype(int).tolist())
        sat_az.append(np.round(az[idx]).astype(int).tolist())
        sat_snr.append(np.round(snr[idx]).astype(int).tolist())

    df = pd.DataFrame(rows, columns=["t", "fix_quality", "lat", "lon", "alt_m", "sog_kn", "cog_deg",
                                     "hdop", "vdop", "pdop", "nsat_used", "nsat_visible",
                                     "snr_mean", "snr_std", "snr_max", "snr_min", "benign_event"])
    df.insert(1, "utc", t0_utc + pd.to_timedelta(df.t, unit="s"))
    df["sat_prn"], df["sat_el"], df["sat_az"], df["sat_snr"] = sat_prn, sat_el, sat_az, sat_snr
    return df


def simulate_ownship_aux(rng, truth: pd.DataFrame):
    """Ship's gyrocompass + speed log (1 Hz). Optional sensors for DR on a real
    ship; on the ESP32 testbed use the IMU stream instead."""
    n = len(truth)
    hdg = (truth.heading_deg.values + rng.normal(0, 0.4, n) + rng.normal(0, 0.3)) % 360
    stw = truth.stw_ms.values / 0.514444 * (1 + rng.normal(0, 0.02)) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"t": truth.t.values, "gyro_heading_deg": hdg, "log_stw_kn": np.clip(stw, 0, None)})
