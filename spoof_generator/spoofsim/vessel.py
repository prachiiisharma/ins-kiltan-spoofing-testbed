"""Ground-truth vessel motion.

Each vessel steers toward random waypoints with a rate-limited heading
controller and a first-order speed response. Ground velocity is

    v_ground = v_through_water + ocean_current + wind_leeway

so the truth already contains the legitimate environmental drift that the
dead-reckoning layer must model (and that a naive detector would flag).
Benign manoeuvre events (hard turns, crash stops, sprints) are scheduled on the
own ship and recorded in the `benign_event` column as hard negatives.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .geo import enu_to_ll, KN, vel_to_sog_cog, wrap180


@dataclass
class Profile:
    name: str
    speed_kn: tuple          # (min, max) cruise speed
    max_rate_dps: float      # max rate of turn, deg/s
    speed_tau_s: float       # speed response time constant
    wp_dist_m: tuple         # distance to next waypoint
    ship_type: int           # AIS ship type code
    length_m: tuple
    ais_class: str = "A"


PROFILES = {
    "patrol":   Profile("patrol", (10, 24), 2.5, 25, (4000, 15000), 35, (90, 110)),
    "merchant": Profile("merchant", (10, 16), 0.4, 120, (15000, 40000), 70, (120, 300)),
    "tanker":   Profile("tanker", (9, 14), 0.3, 180, (15000, 40000), 80, (180, 330)),
    "fishing":  Profile("fishing", (2, 8), 2.0, 20, (500, 3000), 30, (12, 30), "B"),
    "tug":      Profile("tug", (5, 11), 1.5, 30, (2000, 8000), 52, (25, 40)),
}

# Box kept offshore of Ezhimala (coast runs ~75.1-75.3E at these latitudes)
SEA_BOX = dict(lat=(11.5, 12.8), lon=(74.0, 74.95))


@dataclass
class Environment:
    current_e: float
    current_n: float
    current_var: float       # amplitude of slow (tidal-like) variation, m/s
    current_period_s: float
    wind_e: float
    wind_n: float

    @staticmethod
    def random(rng):
        cs, cd = rng.uniform(0.1, 0.7), rng.uniform(0, 2 * np.pi)   # SW-monsoon-ish range
        ws, wd = rng.uniform(3, 13), rng.uniform(0, 2 * np.pi)
        return Environment(cs * np.sin(cd), cs * np.cos(cd), rng.uniform(0.0, 0.2),
                           rng.uniform(6, 12.4) * 3600, ws * np.sin(wd), ws * np.cos(wd))

    def current(self, t):
        f = 1 + (self.current_var / max(np.hypot(self.current_e, self.current_n), 1e-3)) \
            * np.sin(2 * np.pi * t / self.current_period_s)
        return self.current_e * f, self.current_n * f


def _in_box(lat, lon):
    return SEA_BOX["lat"][0] < lat < SEA_BOX["lat"][1] and SEA_BOX["lon"][0] < lon < SEA_BOX["lon"][1]


def simulate_vessel(rng, profile: Profile, env: Environment, duration_s: int, lat0: float,
                    lon0: float, ref_lat: float, ref_lon: float, x0=None, y0=None,
                    events=None, dt=1.0, leeway_coef=None):
    """Return a 1 Hz truth DataFrame for one vessel.

    events: list of dicts {type, start, end} for benign manoeuvres:
        hard_turn  - heading change of 90-180 deg at elevated rate
        crash_stop - speed to ~0 with short time constant
        sprint     - speed to profile max
        zigzag     - alternating +/-30 deg turns
    """
    n = int(duration_s / dt)
    from .geo import ll_to_enu
    if x0 is None:
        x0, y0 = ll_to_enu(lat0, lon0, ref_lat, ref_lon)
    leeway_coef = rng.uniform(0.015, 0.04) if leeway_coef is None else leeway_coef

    x, y = float(x0), float(y0)
    hdg = rng.uniform(0, 360)
    spd = rng.uniform(*profile.speed_kn) * KN
    tgt_spd = spd
    wp = None
    events = events or []
    out = np.zeros((n, 11))
    ev_label = np.array(["none"] * n, dtype=object)
    zig_sign, zig_next = 1, 0
    prev_vfwd = spd

    for i in range(n):
        t = i * dt
        # --- waypoint management -------------------------------------------------
        if wp is None or np.hypot(wp[0] - x, wp[1] - y) < 300:
            for _ in range(50):
                d = rng.uniform(*profile.wp_dist_m)
                b = rng.uniform(0, 2 * np.pi)
                cand = (x + d * np.sin(b), y + d * np.cos(b))
                la, lo = enu_to_ll(cand[0], cand[1], ref_lat, ref_lon)
                if _in_box(la, lo):
                    wp = cand
                    break
            else:  # head back toward box centre
                from .geo import ll_to_enu as _l
                cx, cy = _l(np.mean(SEA_BOX["lat"]), np.mean(SEA_BOX["lon"]), ref_lat, ref_lon)
                wp = (float(cx), float(cy))
            tgt_spd = rng.uniform(*profile.speed_kn) * KN

        desired = np.degrees(np.arctan2(wp[0] - x, wp[1] - y)) % 360
        max_rate = profile.max_rate_dps
        tau = profile.speed_tau_s
        spd_cmd = tgt_spd

        active = [e for e in events if e["start"] <= t < e["end"]]
        for e in active:
            ev_label[i] = e["type"]
            if e["type"] == "hard_turn":
                desired = e.setdefault("_hdg", (hdg + e.get("delta", 150)) % 360)
                max_rate = max(profile.max_rate_dps * 1.6, 3.0)
            elif e["type"] == "crash_stop":
                spd_cmd, tau = 0.3, 15
            elif e["type"] == "sprint":
                spd_cmd, tau = profile.speed_kn[1] * KN * 1.1, 20
            elif e["type"] == "zigzag":
                if t >= zig_next:
                    zig_sign *= -1
                    zig_next = t + 60
                    e["_hdg"] = (hdg + 30 * zig_sign) % 360
                desired = e.get("_hdg", desired)
                max_rate = max(profile.max_rate_dps, 2.0)

        err = wrap180(desired - hdg)
        rate = float(np.clip(0.05 * err, -max_rate, max_rate))
        hdg = (hdg + rate * dt) % 360
        spd += (spd_cmd - spd) * dt / tau
        spd = max(spd, 0.0)

        ce, cn = env.current(t)
        lw = leeway_coef * np.hypot(env.wind_e, env.wind_n)
        lwe = lw * env.wind_e / max(np.hypot(env.wind_e, env.wind_n), 1e-6)
        lwn = lw * env.wind_n / max(np.hypot(env.wind_e, env.wind_n), 1e-6)
        ve = spd * np.sin(np.radians(hdg)) + ce + lwe
        vn = spd * np.cos(np.radians(hdg)) + cn + lwn
        x += ve * dt
        y += vn * dt
        afwd = (spd - prev_vfwd) / dt
        prev_vfwd = spd
        out[i] = (t, x, y, hdg, spd, ve, vn, rate, afwd, ce + lwe, cn + lwn)

    df = pd.DataFrame(out, columns=["t", "x_m", "y_m", "heading_deg", "stw_ms", "ve_ms", "vn_ms",
                                    "rot_dps", "accel_fwd_ms2", "drift_e_ms", "drift_n_ms"])
    df["lat"], df["lon"] = enu_to_ll(df.x_m, df.y_m, ref_lat, ref_lon)
    df["sog_ms"], df["cog_deg"] = vel_to_sog_cog(df.ve_ms, df.vn_ms)
    df["benign_event"] = ev_label
    return df
