"""
Step 3: turn MARSIM pcaps into a labelled feature table.

    python build_features.py --data data/marsim --out features/marsim
    python build_features.py --data data/marsim --out features/marsim_small --limit 2000   # quick test

Output: a folder of Parquet part files, one row per second per pcap, with
  file, scenario, file_label, params, t_rel, label (1 = spoofed at this second), features...

A second is labelled spoofed only if the file is spoofed AND t_rel >= 60 s (MARSIM attacks
start at 60 s), so you can measure detection delay, not just "was the file spoofed".
"""
import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from nmea_features import NmeaFeatureExtractor, add_rolling, combine_receivers
from pcap_reader import iter_nmea

ATTACK_START_S = 60.0
META_COLS = ["file", "scenario", "file_label", "params", "rx", "t", "t_rel", "label"]


def parse_filename(name):
    """'{scenario}-{label}-{index}-{k1}-{v1}-...pcap' -> dict (used if dataset.json is missing)."""
    stem = os.path.basename(name).rsplit(".", 1)[0]
    parts = stem.split("-")
    params = {}
    rest = parts[3:]
    for i in range(0, len(rest) - 1, 2):
        params[rest[i]] = rest[i + 1]
    return {"filename": os.path.basename(name), "scenario": parts[0], "label": parts[1],
            "index": parts[2] if len(parts) > 2 else "0", "parameters": params}


def load_index(data_dir):
    idx_path = os.path.join(data_dir, "dataset.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)
    print("dataset.json not found - parsing file names instead", file=sys.stderr)
    return [parse_filename(f) for f in sorted(os.listdir(data_dir)) if f.endswith((".pcap", ".pcapng"))]


def find_file(data_dir, filename):
    p = os.path.join(data_dir, filename)
    if os.path.exists(p):
        return p
    for root, _, files in os.walk(data_dir):  # tolerate an extra sub-folder after extraction
        if filename in files:
            return os.path.join(root, filename)
    return None


def process_one(args):
    entry, data_dir, primary, secondary, rx_spacing = args
    path = find_file(data_dir, entry["filename"])
    if path is None:
        return None
    fx = NmeaFeatureExtractor()
    rows = []
    for src, t, s in iter_nmea(path):
        rows.extend(fx.feed(src, t, s))
    rows.extend(fx.flush())
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["rx", "t"])
    df = add_rolling(df)
    rxs = sorted(df.rx.unique())
    p = primary if primary in rxs else rxs[0]
    s = secondary if secondary in rxs else (rxs[1] if len(rxs) > 1 else None)
    df = combine_receivers(df, p, s, baseline_dist_m=rx_spacing) if s else df[df.rx == p].copy()
    t0 = df["t"].min()
    df["t_rel"] = df["t"] - t0
    spoofed = entry["label"] == "spoofed"
    df["label"] = ((df["t_rel"] >= ATTACK_START_S) & spoofed).astype(np.int8)
    df["file"] = entry["filename"]
    df["scenario"] = entry["scenario"]
    df["file_label"] = entry["label"]
    df["params"] = json.dumps(entry.get("parameters", {}), sort_keys=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the extracted .pcap files (+ dataset.json)")
    ap.add_argument("--out", required=True, help="output folder for Parquet parts")
    ap.add_argument("--limit", type=int, default=0, help="only use N files (random, stratified) for a quick run")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--primary-rx", default="192.168.0.10")
    ap.add_argument("--secondary-rx", default="192.168.0.11")
    ap.add_argument("--rx-spacing-m", type=float, default=4.0, help="known antenna separation (MARSIM: 4 m)")
    ap.add_argument("--files-per-part", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    index = load_index(a.data)
    if a.limit and a.limit < len(index):
        rng = np.random.default_rng(a.seed)
        groups = {}
        for e in index:
            groups.setdefault((e["scenario"], e["label"]), []).append(e)
        per = max(1, a.limit // len(groups))
        index = [e for g in groups.values() for e in rng.permutation(np.array(g, dtype=object))[:per]]
    os.makedirs(a.out, exist_ok=True)
    print(f"processing {len(index)} files with {a.workers} workers -> {a.out}")

    jobs = [(e, a.data, a.primary_rx, a.secondary_rx, a.rx_spacing_m) for e in index]
    buf, part, done, missing, t_start = [], 0, 0, 0, time.time()

    def write_part():
        nonlocal buf, part
        if not buf:
            return
        df = pd.concat(buf, ignore_index=True)
        for c in df.columns:
            if df[c].dtype == np.float64 and c not in ("t", "utc"):
                df[c] = df[c].astype(np.float32)
        df.to_parquet(os.path.join(a.out, f"part-{part:04d}.parquet"), index=False)
        part += 1
        buf = []

    with Pool(a.workers) as pool:
        for df in pool.imap_unordered(process_one, jobs, chunksize=8):
            done += 1
            if df is None:
                missing += 1
            else:
                buf.append(df)
            if len(buf) >= a.files_per_part:
                write_part()
            if done % 500 == 0 or done == len(jobs):
                el = time.time() - t_start
                print(f"  {done}/{len(jobs)} files  ({el / done * (len(jobs) - done) / 60:.1f} min left)")
    write_part()
    print(f"done. parts: {part}, files skipped (missing/empty): {missing}")


if __name__ == "__main__":
    main()
