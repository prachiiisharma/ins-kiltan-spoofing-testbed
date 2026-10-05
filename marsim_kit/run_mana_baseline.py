"""
Step 5: run the original MANA detectors on the SAME test files, as a baseline for your report.

Needs a clone of https://github.com/fkie-cad/mana (no install needed). Then:
    python run_mana_baseline.py --mana-dir ../mana --data data/marsim --split-from results/iforest_single/per_file_results.csv
        --out results/mana_baseline

--split-from makes MANA run on exactly the test files your model was evaluated on,
so the comparison table is fair. Without it, --limit N random files are used.

Output: per_file_results.csv + metrics.json in the same format as train_iforest.py,
plus which MANA method fired first in each file.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from build_features import find_file, load_index
from datetime import datetime

from pcap_reader import iter_nmea, iter_udp
from train_iforest import summarise, ATTACK_START_S

DETECTION_THRESHOLD = 0.1  # same as MANA's own example


def run_one(args):
    entry, data_dir, methods_json, mana_dir = args
    src = os.path.join(os.path.abspath(mana_dir), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from mana.handler import DetectionHandler
    from mana.method import load_methods_json

    path = find_file(data_dir, entry["filename"])
    if path is None:
        return None
    t0 = next(iter_udp(path))[1]
    hits = []

    def on_attack(device_id, spoofing_indicator, method, state):
        hits.append((state.update_time.timestamp() - t0, type(method).__name__))

    dev, classes, opts = load_methods_json(methods_json)
    handler = DetectionHandler(device_ids=dev, method_classes=classes, method_options=opts,
                               on_spoofing_attack=on_attack, detection_threshold=DETECTION_THRESHOLD)
    # MANA's own PcapFeeder crashes on recent scapy ('EDecimal' timestamps) and then silently
    # reports no detections, so we feed MANA's handler with our own pcap reader instead.
    for src, t, sentence in iter_nmea(path):
        handler.handle(device_id=src, time=datetime.fromtimestamp(t), sentence=sentence)
    ta, method = (min(hits) if hits else (np.nan, None))
    spoofed = entry["label"] == "spoofed"
    return {"file": entry["filename"], "scenario": entry["scenario"], "file_label": entry["label"],
            "params": json.dumps(entry.get("parameters", {}), sort_keys=True),
            "first_alarm_s": ta, "first_method": method, "flagged": bool(hits),
            "early_alarm": bool(hits) and ta < ATTACK_START_S,
            "detected_after_onset": spoofed and bool(hits) and ta >= ATTACK_START_S,
            "delay_s": (ta - ATTACK_START_S) if spoofed and hits and ta >= ATTACK_START_S else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mana-dir", required=True, help="folder where you cloned github.com/fkie-cad/mana")
    ap.add_argument("--methods", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "mana_methods.json"))
    ap.add_argument("--split-from", help="per_file_results.csv from train_iforest.py (use its test files)")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    index = load_index(a.data)
    if a.split_from:
        keep = set(pd.read_csv(a.split_from)["file"])
        index = [e for e in index if e["filename"] in keep]
    elif a.limit and a.limit < len(index):
        index = list(np.random.default_rng(0).permutation(np.array(index, dtype=object))[:a.limit])
    print(f"running MANA on {len(index)} files ...")
    with Pool(a.workers) as pool:
        rows = [r for r in pool.imap_unordered(run_one, [(e, a.data, a.methods, a.mana_dir) for e in index], chunksize=4) if r]
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(a.out, "per_file_results.csv"), index=False)
    m = summarise(r)
    m["first_method_counts"] = r["first_method"].value_counts().to_dict()
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    per = pd.DataFrame({s: summarise(g) for s, g in r.groupby("scenario")}).T
    per.to_csv(os.path.join(a.out, "per_scenario.csv"))
    print(json.dumps(m, indent=2))
    print(per[["files", "strict_f1", "strict_recall", "false_alarm_rate_normal_files", "median_delay_s"]])


if __name__ == "__main__":
    main()
