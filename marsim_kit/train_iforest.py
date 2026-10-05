"""
Step 4: train + evaluate the Isolation Forest spoofing detector on MARSIM features.

    python train_iforest.py --features features/marsim --out results/iforest_dual  --mode dual
    python train_iforest.py --features features/marsim --out results/iforest_single --mode single

--mode single uses only features a single NEO-6M can provide (no second receiver),
          which is what transfers to your hardware testbed.
--mode dual   adds the two-receiver distance check (MARSIM has receivers 4 m apart).

Protocol (write this in your methodology):
  * split by FILE (never by row) into train / val / test = 60 / 20 / 20, stratified by
    scenario and label, so no recording leaks between sets
  * the model is trained ONLY on normal seconds (unsupervised, like it would be at sea)
  * the score threshold and the "K consecutive anomalies" alarm rule are tuned on val
  * all reported numbers come from test

Outputs in --out:
  model.joblib           model + feature list + threshold + K  (load it in detector.py)
  metrics.json           row-level AUC, file-level P/R/F1, false-alarm rate, detection delay
  per_scenario.csv       same metrics per attacker type
  per_file_results.csv   one line per test file (alarm time, delay, params) -> for heat-maps
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from nmea_features import SINGLE_RX_FEATURES, DUAL_RX_FEATURES

ATTACK_START_S = 60.0
WARMUP_S = 10.0  # rolling windows / clock baseline need a few seconds


def split_files(meta, seed):
    rng = np.random.default_rng(seed)
    split = {}
    for _, g in meta.groupby(["scenario", "file_label"]):
        files = rng.permutation(g["file"].values)
        n = len(files)
        n_tr, n_va = int(round(n * 0.6)), int(round(n * 0.2))
        for f in files[:n_tr]:
            split[f] = "train"
        for f in files[n_tr:n_tr + n_va]:
            split[f] = "val"
        for f in files[n_tr + n_va:]:
            split[f] = "test"
    return split


def first_alarm(t_rel, flags, k):
    """time of the first run of k consecutive anomalous seconds after warm-up, else NaN"""
    run = 0
    for t, f in zip(t_rel, flags):
        if t < WARMUP_S:
            continue
        run = run + 1 if f else 0
        if run >= k:
            return t
    return np.nan


def evaluate_files(df, score_col, thr, k):
    res = []
    for f, g in df.groupby("file", sort=False):
        g = g.sort_values("t_rel")
        ta = first_alarm(g["t_rel"].values, (g[score_col].values > thr), k)
        spoofed = g["file_label"].iat[0] == "spoofed"
        res.append({"file": f, "scenario": g["scenario"].iat[0], "file_label": g["file_label"].iat[0],
                    "params": g["params"].iat[0], "first_alarm_s": ta,
                    "flagged": not np.isnan(ta),
                    "early_alarm": (not np.isnan(ta)) and ta < ATTACK_START_S,
                    "detected_after_onset": spoofed and (not np.isnan(ta)) and ta >= ATTACK_START_S,
                    "delay_s": (ta - ATTACK_START_S) if spoofed and not np.isnan(ta) and ta >= ATTACK_START_S
                    else np.nan})
    return pd.DataFrame(res)


def summarise(r):
    sp, no = r[r.file_label == "spoofed"], r[r.file_label == "unspoofed"]
    # (a) MANA-comparable: was the file flagged at all?
    tp, fn, fp = sp.flagged.sum(), (~sp.flagged).sum(), no.flagged.sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    # (b) strict: alarm must come after the attack starts; any earlier alarm is a false alarm
    s_tp = sp.detected_after_onset.sum()
    s_fp = no.flagged.sum() + sp.early_alarm.sum()
    s_p = s_tp / (s_tp + s_fp) if s_tp + s_fp else 0.0
    s_r = s_tp / len(sp) if len(sp) else 0.0
    return {
        "files": int(len(r)),
        "file_precision": round(float(p), 4), "file_recall": round(float(rc), 4),
        "file_f1": round(float(2 * p * rc / (p + rc)) if p + rc else 0.0, 4),
        "strict_precision": round(float(s_p), 4), "strict_recall": round(float(s_r), 4),
        "strict_f1": round(float(2 * s_p * s_r / (s_p + s_r)) if s_p + s_r else 0.0, 4),
        "false_alarm_rate_normal_files": round(float(no.flagged.mean()) if len(no) else 0.0, 4),
        "median_delay_s": None if sp.delay_s.isna().all() else round(float(sp.delay_s.median()), 2),
        "p90_delay_s": None if sp.delay_s.isna().all() else round(float(sp.delay_s.quantile(0.9)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["single", "dual"], default="single")
    ap.add_argument("--max-train-rows", type=int, default=300_000)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    feats = SINGLE_RX_FEATURES if a.mode == "single" else DUAL_RX_FEATURES
    cols = ["file", "scenario", "file_label", "params", "t_rel", "label"] + feats
    df = pd.read_parquet(a.features, columns=cols)
    print(f"loaded {len(df):,} rows from {df.file.nunique():,} files")

    split = split_files(df[["file", "scenario", "file_label"]].drop_duplicates(), a.seed)
    df["split"] = df["file"].map(split)
    df = df[df.t_rel >= WARMUP_S].copy()

    med = df.loc[(df.split == "train") & (df.label == 0), feats].median()
    X = df[feats].astype(np.float32).fillna(med).fillna(0.0)

    train_mask = (df.split == "train") & (df.label == 0)
    Xtr = X[train_mask]
    if len(Xtr) > a.max_train_rows:
        Xtr = Xtr.sample(a.max_train_rows, random_state=a.seed)
    print(f"training Isolation Forest on {len(Xtr):,} normal seconds, {len(feats)} features ({a.mode})")
    model = IsolationForest(n_estimators=a.n_estimators, max_samples=256, contamination="auto",
                            random_state=a.seed, n_jobs=-1).fit(Xtr.to_numpy())
    df["score"] = -model.score_samples(X.to_numpy())  # higher = more anomalous

    # ---- tune threshold + K on validation --------------------------------------------
    val = df[df.split == "val"]
    normal_val = val.loc[val.label == 0, "score"]
    best = None
    for q in (0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999):
        thr = float(normal_val.quantile(q))
        for k in (1, 2, 3, 5, 8):
            s = summarise(evaluate_files(val, "score", thr, k))
            key = (s["strict_f1"], -(s["median_delay_s"] or 1e9))
            if best is None or key > best[0]:
                best = (key, thr, k, q, s)
    _, thr, k, q, val_s = best
    print(f"chosen on val: threshold = {q} quantile of normal scores ({thr:.4f}), K = {k}  "
          f"-> val strict F1 {val_s['strict_f1']}")

    # ---- final test evaluation ----------------------------------------------------------
    test = df[df.split == "test"]
    r = evaluate_files(test, "score", thr, k)
    metrics = summarise(r)
    metrics["row_roc_auc"] = round(float(roc_auc_score(test.label, test.score)), 4) if test.label.nunique() > 1 else None
    metrics["row_avg_precision"] = round(float(average_precision_score(test.label, test.score)), 4) \
        if test.label.nunique() > 1 else None
    metrics.update({"mode": a.mode, "features": feats, "threshold": thr, "threshold_quantile": q, "K": k,
                    "train_rows": int(len(Xtr)),
                    "n_files": {s: int((pd.Series(split) == s).sum()) for s in ("train", "val", "test")}})
    per_scn = pd.DataFrame({s: summarise(g) for s, g in r.groupby("scenario")}).T
    per_scn.index.name = "scenario"

    joblib.dump({"model": model, "features": feats, "fill": med.to_dict(), "threshold": thr, "K": k,
                 "mode": a.mode, "warmup_s": WARMUP_S}, os.path.join(a.out, "model.joblib"))
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    per_scn.to_csv(os.path.join(a.out, "per_scenario.csv"))
    r.to_csv(os.path.join(a.out, "per_file_results.csv"), index=False)

    print("\nTEST RESULTS")
    for key in ("row_roc_auc", "file_precision", "file_recall", "file_f1", "strict_precision", "strict_recall",
                "strict_f1", "false_alarm_rate_normal_files", "median_delay_s", "p90_delay_s"):
        print(f"  {key:32s} {metrics[key]}")
    print("\nper scenario:\n", per_scn[["files", "strict_f1", "strict_recall", "false_alarm_rate_normal_files",
                                        "median_delay_s"]].to_string())
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    main()
