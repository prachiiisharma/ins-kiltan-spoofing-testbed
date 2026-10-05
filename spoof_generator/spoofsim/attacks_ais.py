"""Data-layer AIS spoofing attacks (AIVDM level, no RF).

Attack catalogue
----------------
position_jump       a real vessel's reports are offset 1-20 km for a window
position_drift      a real vessel's reports are dragged off gradually
ghost_vessel        a non-existent vessel with a new MMSI (plausible or
                    kinematically impossible track)
mmsi_duplication    a second transmitter re-uses a real MMSI elsewhere
                    (may target own ship = impersonation of own ship)
kinematic_mismatch  reported SOG/COG contradict the positional track
                    (e.g. claims to be stationary while moving)
invalid_identity    MMSI replaced by an invalid / unallocated identity
"""
import numpy as np
import pandas as pd

from .geo import ll_to_enu, enu_to_ll

AIS_ATTACKS = ["position_jump", "position_drift", "ghost_vessel", "mmsi_duplication",
               "kinematic_mismatch", "invalid_identity"]

INVALID_MMSIS = [0, 123456789, 111111111, 999999999, 100000001, 12345678, 199123456]


def init_labels(df):
    df["is_spoofed"] = 0
    df["attack_type"] = "none"
    df["attack_id"] = ""
    return df


def random_ais_attack(rng, kind, start, duration, target_mmsi):
    p = dict(type=kind, start=float(start), duration=float(duration), target_mmsi=int(target_mmsi))
    if kind == "position_jump":
        p.update(offset_m=float(rng.uniform(1000, 20000)), bearing_deg=float(rng.uniform(0, 360)))
    elif kind == "position_drift":
        p.update(drift_ms=float(rng.uniform(0.5, 4.0)), bearing_deg=float(rng.uniform(0, 360)))
    elif kind == "ghost_vessel":
        p.update(plausible=bool(rng.random() < 0.5), speed_factor=float(rng.uniform(3, 8)))
    elif kind == "mmsi_duplication":
        p.update(offset_m=float(rng.uniform(10000, 60000)), bearing_deg=float(rng.uniform(0, 360)))
    elif kind == "kinematic_mismatch":
        p.update(variant=str(rng.choice(["claims_stationary", "sog_scaled", "cog_offset"])),
                 sog_factor=float(rng.uniform(0.2, 0.5)), cog_offset_deg=float(rng.uniform(60, 150)))
    elif kind == "invalid_identity":
        p.update(fake_mmsi=int(rng.choice(INVALID_MMSIS)))
    return p


def messages_from_track(track: pd.DataFrame, mmsi, uid, name, t0_utc, interval=10.0, rng=None,
                        report_scale=None, origin=None, ref=None):
    """Fake-transmitter messages along `track` (always received: spoofer is close)."""
    rows = []
    T = track.t.values
    tt = T[0] + (rng.uniform(0, interval) if rng is not None else 0)
    ch = "A"
    while tt < T[-1]:
        i = int(np.searchsorted(T, tt))
        r = track.iloc[min(i, len(T) - 1)]
        lat, lon = r.lat, r.lon
        if report_scale is not None:   # stretch displacement -> impossible implied speed
            x, y = ll_to_enu(lat, lon, *ref)
            x = origin[0] + (x - origin[0]) * report_scale
            y = origin[1] + (y - origin[1]) * report_scale
            lat, lon = enu_to_ll(x, y, *ref)
        ch = "B" if ch == "A" else "A"
        rows.append(dict(t=round(tt, 2), mmsi=int(mmsi), msg_type=1, channel=ch, lat=float(lat), lon=float(lon),
                         sog_kn=round(r.sog_ms / 0.514444, 1), cog_deg=round(r.cog_deg, 1),
                         heading=int(round(r.heading_deg)) % 360, rot_dpm=round(r.rot_dps * 60, 1),
                         nav_status=0, ship_name=None, ship_type=None, vessel_uid=uid, is_own=False,
                         truth_lat=np.nan, truth_lon=np.nan, benign_event="none"))
        tt += interval * (rng.uniform(0.95, 1.05) if rng is not None else 1)
    rows.append(dict(t=round(T[0] + 1, 2), mmsi=int(mmsi), msg_type=5, channel="A", lat=np.nan, lon=np.nan,
                     sog_kn=np.nan, cog_deg=np.nan, heading=np.nan, rot_dpm=np.nan, nav_status=np.nan,
                     ship_name=name, ship_type=70, vessel_uid=uid, is_own=False,
                     truth_lat=np.nan, truth_lon=np.nan, benign_event="none"))
    df = pd.DataFrame(rows)
    df.insert(1, "utc", t0_utc + pd.to_timedelta(df.t, unit="s"))
    return df


def apply_ais_attack(ais: pd.DataFrame, spec: dict, attack_id: str, ref_lat, ref_lon, rng,
                     make_track=None, t0_utc=None, used_mmsi=None):
    """make_track(lat0, lon0, start_s, duration_s) -> truth DataFrame (for ghost / dup)."""
    k = spec["type"]
    s, e = spec["start"], spec["start"] + spec["duration"]
    pos = ais.msg_type.isin([1, 2, 3, 18])

    def label(mask):
        ais.loc[mask, "is_spoofed"] = 1
        ais.loc[mask, "attack_type"] = k
        ais.loc[mask, "attack_id"] = attack_id

    if k in ("position_jump", "position_drift", "kinematic_mismatch", "invalid_identity"):
        m = (ais.mmsi == spec["target_mmsi"]) & (ais.t >= s) & (ais.t < e)
        mp = m & pos
        if k != "invalid_identity" and not mp.any():
            return ais, False
        if k in ("position_jump", "position_drift"):
            x, y = ll_to_enu(ais.loc[mp, "lat"].values, ais.loc[mp, "lon"].values, ref_lat, ref_lon)
            b = np.radians(spec["bearing_deg"])
            d = spec["offset_m"] if k == "position_jump" else spec["drift_ms"] * (ais.loc[mp, "t"].values - s)
            la, lo = enu_to_ll(x + d * np.sin(b), y + d * np.cos(b), ref_lat, ref_lon)
            ais.loc[mp, "lat"], ais.loc[mp, "lon"] = la, lo
        elif k == "kinematic_mismatch":
            v = spec["variant"]
            if v == "claims_stationary":
                ais.loc[mp, "sog_kn"] = np.round(rng.uniform(0, 0.3, mp.sum()), 1)
                ais.loc[mp, "nav_status"] = 1   # "at anchor"
            elif v == "sog_scaled":
                ais.loc[mp, "sog_kn"] = np.round(ais.loc[mp, "sog_kn"] * spec["sog_factor"], 1)
            else:
                ais.loc[mp, "cog_deg"] = np.round((ais.loc[mp, "cog_deg"] + spec["cog_offset_deg"]) % 360, 1)
        else:
            if not m.any():
                return ais, False
            ais.loc[m, "mmsi"] = spec["fake_mmsi"]
            mp = m
        label(mp)
        return ais, True

    # ---- attacks that add a fake transmitter -------------------------------------
    own = ais[ais.is_own & pos]
    near = own[(own.t >= s)]
    if near.empty:
        return ais, False
    olat, olon = near.lat.iat[0], near.lon.iat[0]
    if not (np.isfinite(olat) and abs(olat) <= 90):
        return ais, False
    if k == "ghost_vessel":
        b, dist = rng.uniform(0, 2 * np.pi), rng.uniform(3000, 15000)
        mmsi = int(spec.get("fake_mmsi") or _new_mmsi(rng, used_mmsi))
        spec["fake_mmsi"] = mmsi
        name = "GHOST"
    else:
        b, dist = np.radians(spec["bearing_deg"]), spec["offset_m"]
        mmsi = spec["target_mmsi"]
        name = None
    ox, oy = ll_to_enu(olat, olon, ref_lat, ref_lon)
    la0, lo0 = enu_to_ll(ox + dist * np.sin(b), oy + dist * np.cos(b), ref_lat, ref_lon)
    track = make_track(float(la0), float(lo0), s, spec["duration"])
    kw = {}
    if k == "ghost_vessel" and not spec["plausible"]:
        x0, y0 = ll_to_enu(track.lat.iat[0], track.lon.iat[0], ref_lat, ref_lon)
        kw = dict(report_scale=spec["speed_factor"], origin=(float(x0), float(y0)), ref=(ref_lat, ref_lon))
    fake = messages_from_track(track, mmsi, f"spoofer_{attack_id}", name or "", t0_utc, rng=rng, **kw)
    if name is None:   # duplicated identity: no fresh static msg with a different name
        fake = fake[fake.msg_type != 5]
    fake = init_labels(fake)
    fake["is_spoofed"], fake["attack_type"], fake["attack_id"] = 1, k, attack_id
    out = pd.concat([ais, fake], ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)
    return out, True


def _new_mmsi(rng, used):
    from .ais import random_mmsi
    m = random_mmsi(rng, used)
    if used is not None:
        used.add(m)
    return m
