"""Generate the labelled GNSS + AIS spoofing dataset.

    python build_dataset.py                       # uses config/default.yaml
    python build_dataset.py --config my.yaml --out data --workers 8

Output (data/<dataset_name>/):
    gnss.parquet / gnss.csv     own-ship GNSS epochs (1 Hz) + gyro/log + labels
    ais.parquet  / ais.csv      every received AIS message (all vessels) + labels
    imu.parquet                 MPU6050-like IMU, 10 Hz
    truth.parquet               ground-truth trajectories of every real vessel (1 Hz)
    attacks.csv                 one row per injected attack (onset, duration, params)
    voyages.json                per-voyage metadata, environment, benign events, split
    raw/<voyage_id>/            raw NMEA (gnss.nmea) + AIVDM (ais.nmea) logs
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

from spoofsim.scenario import build_voyage
from spoofsim.attacks_gnss import GNSS_ATTACKS
from spoofsim.attacks_ais import AIS_ATTACKS
from spoofsim.nmea_io import write_gnss_log, write_ais_log


def _cycle(seq, offset):
    state = {"k": offset}

    def nxt():
        v = seq[state["k"] % len(seq)]
        state["k"] += 1
        return v
    return nxt


def _one(args):
    i, cfg = args
    vid = f"V{i:03d}"
    # balanced attack-type coverage across voyages
    return build_voyage(cfg["base_seed"] + i, cfg, vid,
                        gnss_attack_cycle=_cycle(GNSS_ATTACKS, i),
                        ais_attack_cycle=_cycle(AIS_ATTACKS, i))


def assign_splits(metas, frac, seed):
    """Voyage-level split; reshuffle until every split sees every attack type."""
    ids = [m["voyage_id"] for m in metas]
    types = {m["voyage_id"]: {a["type"] for a in m["attacks"]} for m in metas}
    need = set(GNSS_ATTACKS) | set(AIS_ATTACKS)
    rng = np.random.default_rng(seed)
    for _ in range(5000):
        perm = list(rng.permutation(ids))
        n_tr = int(round(frac["train"] * len(ids)))
        n_va = int(round(frac["val"] * len(ids)))
        sp = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
        if all(need <= set().union(*(types[v] for v in vs)) for vs in sp.values()):
            return {v: k for k, vs in sp.items() for v in vs}
    print("WARNING: no split covers every attack type in every split (too few voyages?) - using last shuffle")
    return {v: k for k, vs in sp.items() for v in vs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--n", type=int, help="override n_voyages")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.n:
        cfg["n_voyages"] = a.n
    out = os.path.join(a.out, cfg["dataset_name"])
    os.makedirs(out, exist_ok=True)

    t = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        voyages = list(ex.map(_one, [(i, cfg) for i in range(cfg["n_voyages"])]))
    print(f"simulated {len(voyages)} voyages in {time.time() - t:.0f}s")

    metas = [v["meta"] for v in voyages]
    split = assign_splits(metas, cfg["split"], cfg["base_seed"])
    for m in metas:
        m["split"] = split[m["voyage_id"]]

    gnss = pd.concat([v["gnss"].merge(v["aux"], on=["voyage_id", "t"], how="left") for v in voyages],
                     ignore_index=True)
    ais = pd.concat([v["ais"] for v in voyages], ignore_index=True)
    imu = pd.concat([v["imu"] for v in voyages], ignore_index=True)
    truth = pd.concat([v["truth"] for v in voyages], ignore_index=True)
    for df in (gnss, ais, imu, truth):
        df.insert(1, "split", df.voyage_id.map(split))
    ais["mmsi"] = ais.mmsi.astype("int64")
    for c in ["ship_name"]:
        ais[c] = ais[c].astype("string")
    ais["ship_type"] = pd.to_numeric(ais.ship_type, errors="coerce")
    truth = truth.drop(columns=["x_m", "y_m", "ve_ms", "vn_ms"])
    # own ship at 1 Hz; traffic truth every 10 s is plenty for evaluation
    truth = truth[(truth.vessel_uid == "own") | (truth.t % 10 == 0)].reset_index(drop=True)
    truth = truth.astype({c: "float32" for c in truth.select_dtypes("float64").columns if c not in ("lat", "lon", "t")})
    for df in (gnss, ais, truth, imu):
        for c in ("voyage_id", "split", "benign_event", "attack_type", "vessel_uid", "attack_id"):
            if c in df:
                df[c] = df[c].astype("category")
    imu = imu.astype({c: "float32" for c in imu.columns if c.endswith(("ms2", "rads"))})

    gnss.to_parquet(compression="zstd", path=f"{out}/gnss.parquet", index=False)
    ais.to_parquet(compression="zstd", path=f"{out}/ais.parquet", index=False)
    imu.to_parquet(compression="zstd", path=f"{out}/imu.parquet", index=False)
    truth.to_parquet(compression="zstd", path=f"{out}/truth.parquet", index=False)
    if cfg.get("write_csv"):
        g = gnss.copy()
        for c in ["sat_prn", "sat_el", "sat_az", "sat_snr"]:
            g[c] = g[c].map(lambda v: ";".join(map(str, v)))
        g.to_csv(f"{out}/gnss.csv", index=False, float_format="%.7g")
        ais.to_csv(f"{out}/ais.csv", index=False, float_format="%.7g")

    rows = []
    for m in metas:
        for at in m["attacks"]:
            r = {k: v for k, v in at.items()}
            r.update(voyage_id=m["voyage_id"], split=m["split"])
            rows.append(r)
    att = pd.DataFrame(rows)
    lead = ["attack_id", "voyage_id", "split", "stream", "type", "start", "duration"]
    att[lead + [c for c in att.columns if c not in lead]].to_csv(f"{out}/attacks.csv", index=False)
    json.dump(metas, open(f"{out}/voyages.json", "w"), indent=1, default=str)

    raw = cfg.get("write_raw_logs", "test")
    if raw:
        for v in voyages:
            m = v["meta"]
            if raw is True or raw == "all" or m["split"] == raw:
                d = f"{out}/raw/{m['voyage_id']}"
                os.makedirs(d, exist_ok=True)
                write_gnss_log(v["gnss"], f"{d}/gnss.nmea")
                write_ais_log(v["ais"], f"{d}/ais.nmea")
    json.dump(cfg, open(f"{out}/config_used.json", "w"), indent=1)
    print(f"wrote {out}  ({time.time() - t:.0f}s total)")


if __name__ == "__main__":
    main()
