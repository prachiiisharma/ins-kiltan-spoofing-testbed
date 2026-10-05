"""One voyage = own ship + surrounding AIS traffic + scheduled attacks and
benign events. Everything is driven by a single seed, so any voyage can be
regenerated exactly."""
import numpy as np
import pandas as pd

from .vessel import PROFILES, Environment, simulate_vessel, SEA_BOX
from .gnss import simulate_gnss, simulate_ownship_aux
from .imu import simulate_imu
from .ais import generate_ais, random_mmsi, NAMES
from .geo import ll_to_enu, enu_to_ll
from . import attacks_gnss as AG
from . import attacks_ais as AA


def _windows(rng, n, total, dur_range, warmup, gap=120, max_tries=200):
    """n non-overlapping (start, duration) windows after `warmup`."""
    out = []
    for _ in range(max_tries):
        if len(out) == n:
            break
        d = rng.uniform(*dur_range)
        s = rng.uniform(warmup, total - d - 30)
        if all(s + d + gap < a or s > a + b + gap for a, b in out):
            out.append((s, d))
    return sorted(out)


def build_voyage(seed: int, cfg: dict, voyage_id: str, gnss_attack_cycle=None, ais_attack_cycle=None):
    rng = np.random.default_rng(seed)
    dur = int(rng.uniform(*cfg["duration_min"]) * 60)
    t0 = pd.Timestamp("2026-06-01T00:00:00Z") + pd.Timedelta(days=int(rng.integers(0, 120)),
                                                                seconds=int(rng.integers(0, 86400)))
    ref_lat, ref_lon = 12.0, 74.6
    lat0 = rng.uniform(11.75, 12.55)
    lon0 = rng.uniform(74.25, 74.8)
    env = Environment.random(rng)

    # ---------------- own ship + benign manoeuvres ------------------------------------
    n_man = int(rng.integers(*cfg["benign_manoeuvres"]))
    man = [dict(type=str(rng.choice(["hard_turn", "crash_stop", "sprint", "zigzag"])), start=s, end=s + d,
                delta=float(rng.choice([-1, 1]) * rng.uniform(90, 180)))
           for s, d in _windows(rng, n_man, dur, (90, 420), 300)]
    own_truth = simulate_vessel(rng, PROFILES["patrol"], env, dur, lat0, lon0, ref_lat, ref_lon, events=man)

    n_deg = int(rng.integers(*cfg["benign_gnss_degradations"]))
    degr = [dict(type=str(rng.choice(["multipath", "multipath", "outage"])), start=s, end=s + d)
            for s, d in _windows(rng, n_deg, dur, (20, 240), 300)]
    for d in degr:
        if d["type"] == "outage":
            d["end"] = d["start"] + min(d["end"] - d["start"], 60)
    gnss = simulate_gnss(rng, own_truth, ref_lat, ref_lon, t0, degradations=degr)
    # manoeuvre labels from truth -> gnss epochs (keep GNSS degradation label if both)
    gnss.loc[gnss.benign_event == "none", "benign_event"] = own_truth.benign_event[gnss.benign_event == "none"].values
    aux = simulate_ownship_aux(rng, own_truth)
    imu = simulate_imu(rng, own_truth, rate_hz=cfg["imu_rate_hz"])

    # ---------------- GNSS attacks ----------------------------------------------------
    gnss = AG.init_labels(gnss)
    n_ga = int(rng.choice(len(cfg["gnss_attacks_per_voyage_p"]), p=cfg["gnss_attacks_per_voyage_p"]))
    attacks = []
    for j, (s, d) in enumerate(_windows(rng, n_ga, dur, tuple(cfg["attack_duration_s"]), cfg["warmup_s"])):
        kind = gnss_attack_cycle() if gnss_attack_cycle else str(rng.choice(AG.GNSS_ATTACKS))
        spec = AG.random_gnss_attack(rng, kind, s, d)
        aid = f"{voyage_id}_G{j}"
        gnss = AG.apply_gnss_attack(gnss, spec, aid, ref_lat, ref_lon, rng)
        spec.update(attack_id=aid, stream="gnss")
        attacks.append(spec)

    # ---------------- traffic ---------------------------------------------------------
    used = set()
    own_mmsi = 419000000 + int(rng.integers(100000, 999999))   # Indian MID
    used.add(own_mmsi)
    vessels = [dict(uid="own", mmsi=own_mmsi, name="OWN SHIP", profile=PROFILES["patrol"], is_own=True,
                    truth=own_truth)]
    n_tr = int(rng.integers(*cfg["traffic_vessels"]))
    kinds = ["merchant", "merchant", "tanker", "fishing", "fishing", "fishing", "tug"]
    ox, oy = ll_to_enu(lat0, lon0, ref_lat, ref_lon)
    for k in range(n_tr):
        p = PROFILES[str(rng.choice(kinds))]
        b, r = rng.uniform(0, 2 * np.pi), rng.uniform(2000, 35000)
        la, lo = enu_to_ll(ox + r * np.sin(b), oy + r * np.cos(b), ref_lat, ref_lon)
        la = float(np.clip(la, SEA_BOX["lat"][0] + .02, SEA_BOX["lat"][1] - .02))
        lo = float(np.clip(lo, SEA_BOX["lon"][0] + .02, SEA_BOX["lon"][1] - .02))
        tr = simulate_vessel(rng, p, env, dur, la, lo, ref_lat, ref_lon)
        m = random_mmsi(rng, used)
        used.add(m)
        vessels.append(dict(uid=f"v{k:02d}", mmsi=m, name=str(rng.choice(NAMES)), profile=p, is_own=False, truth=tr))

    gaps_by_uid = {}
    for v in vessels[1:]:
        if rng.random() < cfg["ais_gap_prob"]:
            s = rng.uniform(300, dur - 700)
            gaps_by_uid[v["uid"]] = [dict(start=s, end=s + rng.uniform(180, 600))]

    ais = pd.concat([generate_ais(rng, v, v["truth"], own_truth, ref_lat, ref_lon, t0,
                                  own_gnss=gnss, own_aux=aux, gaps=gaps_by_uid.get(v["uid"]))
                     for v in vessels], ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)
    ais = AA.init_labels(ais)
    # own-ship AIS inherits GNSS spoof (it is fed by the spoofed receiver)
    own_pos = ais.is_own & ais.msg_type.isin([1, 2, 3, 18])
    gi = np.clip(np.searchsorted(gnss.t.values, ais.loc[own_pos, "t"].values), 0, len(gnss) - 1)
    sp = gnss.is_spoofed.values[gi] == 1
    sub = ais.index[own_pos][sp]
    ais.loc[sub, "is_spoofed"] = 1
    ais.loc[sub, "attack_type"] = "propagated_gnss_" + gnss.attack_type.values[gi][sp]
    ais.loc[sub, "attack_id"] = gnss.attack_id.values[gi][sp]

    # ---------------- AIS attacks -----------------------------------------------------
    def make_track(la, lo, start, d):
        p = PROFILES[str(rng.choice(["merchant", "fishing", "tug", "patrol"]))]
        tr = simulate_vessel(rng, p, env, int(d), la, lo, ref_lat, ref_lon)
        tr["t"] += start
        return tr

    n_aa = int(rng.choice(len(cfg["ais_attacks_per_voyage_p"]), p=cfg["ais_attacks_per_voyage_p"]))
    for j, (s, d) in enumerate(_windows(rng, n_aa, dur, tuple(cfg["attack_duration_s"]), cfg["warmup_s"], gap=0)):
        kind = ais_attack_cycle() if ais_attack_cycle else str(rng.choice(AA.AIS_ATTACKS))
        cands = [v for v in vessels[1:] if (v["profile"].ais_class == "A" or kind != "kinematic_mismatch")]
        tgt = vessels[0] if (kind == "mmsi_duplication" and rng.random() < 0.3) else cands[int(rng.integers(len(cands)))]
        spec = AA.random_ais_attack(rng, kind, s, d, tgt["mmsi"])
        aid = f"{voyage_id}_A{j}"
        ais, ok = AA.apply_ais_attack(ais, spec, aid, ref_lat, ref_lon, rng, make_track=make_track,
                                      t0_utc=t0, used_mmsi=used)
        if ok:
            spec.update(attack_id=aid, stream="ais", target_uid=tgt["uid"])
            attacks.append(spec)

    for df in (gnss, ais):
        df.insert(0, "voyage_id", voyage_id)
    imu.insert(0, "voyage_id", voyage_id)
    aux.insert(0, "voyage_id", voyage_id)
    truth = pd.concat([v["truth"].assign(vessel_uid=v["uid"], mmsi=v["mmsi"]) for v in vessels], ignore_index=True)
    truth.insert(0, "voyage_id", voyage_id)
    meta = dict(voyage_id=voyage_id, seed=seed, start_utc=str(t0), duration_s=dur, own_mmsi=own_mmsi,
                n_traffic=n_tr, current_e_ms=env.current_e, current_n_ms=env.current_n,
                wind_e_ms=env.wind_e, wind_n_ms=env.wind_n,
                benign_manoeuvres=[{k: v for k, v in m.items() if not k.startswith("_")} for m in man],
                benign_gnss=degr, ais_gaps={k: v for k, v in gaps_by_uid.items()},
                attacks=attacks)
    return dict(gnss=gnss, aux=aux, imu=imu, ais=ais, truth=truth, meta=meta)
