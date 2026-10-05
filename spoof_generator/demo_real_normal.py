"""Proof of concept: learn 'normal' from real AIS, detect injected spoofing.

    python demo_real_normal.py data/real_ais_nyc

1. Isolation Forest trained ONLY on real normal tracks (train split, moving reports).
2. Test tracks (never seen) get attacks injected in a window:
     position_jump   +1..5 km offset
     position_drift  pulled off at 2..8 kn
     kinematic       reported SOG cut to 20-50 % (claims slower than it moves)
3. Score every test report; report detection and false-alarm rates.
Also compares real vs simulated traffic kinematics (realism check).
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from spoofsim.geo import ll_to_enu, enu_to_ll
from prepare_real_ais import kinematic_features

d0 = sys.argv[1]
tr = pd.read_parquet(f"{d0}/tracks.parquet")
fe = pd.read_parquet(f"{d0}/features.parquet")
FEATS = ["speed_resid_kn", "course_resid_deg", "accel_kn_per_min", "turn_rate_deg_per_min", "v_implied_kn"]

def X(f):
    x = f[FEATS].copy()
    x["speed_resid_kn"] = x.speed_resid_kn.abs()
    x["accel_kn_per_min"] = x.accel_kn_per_min.abs()
    x["turn_rate_deg_per_min"] = x.turn_rate_deg_per_min.abs()
    return np.log1p(x.clip(lower=0).fillna(0))

train = fe[(fe.split == "train") & (fe.sog_kn > 1)]
model = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(X(train))

# ---- inject attacks into moving test tracks ---------------------------------------
rng = np.random.default_rng(1)
test = tr[tr.split == "test"].copy()
mov = test.groupby("track_id").sog_kn.median()
cands = mov[mov > 3].index.tolist()
rng.shuffle(cands)
kinds = ["position_jump", "position_drift", "kinematic"]
for i, tid in enumerate(cands):
    k = kinds[i % 3]
    idx = test.index[test.track_id == tid]
    n = len(idx)
    if n < 25: continue
    s = int(rng.integers(5, n - 15)); e = min(n, s + int(rng.integers(10, 30)))
    w = idx[s:e]
    x, y = ll_to_enu(test.loc[w, "lat"].values, test.loc[w, "lon"].values, 40.7, -74.0)
    b = rng.uniform(0, 2 * np.pi)
    if k == "position_jump":
        dd = rng.uniform(1000, 5000)
    elif k == "position_drift":
        dd = rng.uniform(2, 8) * 0.514444 * (test.loc[w, "t"].values - test.loc[w[0], "t"])
    if k != "kinematic":
        la, lo = enu_to_ll(x + dd * np.sin(b), y + dd * np.cos(b), 40.7, -74.0)
        test.loc[w, "lat"], test.loc[w, "lon"] = la, lo
    else:
        test.loc[w, "sog_kn"] = test.loc[w, "sog_kn"] * rng.uniform(.2, .5)
    test.loc[w, "is_spoofed"] = 1
    test.loc[w, "attack_type"] = k
ft = kinematic_features(test.drop(columns=[c for c in test.columns if c in fe.columns and c not in tr.columns]))
ft = ft[ft.dt_s.notna() & (ft.sog_kn > 1)]
ft["score"] = -model.score_samples(X(ft))
thr = np.quantile(-model.score_samples(X(fe[(fe.split == "val") & (fe.sog_kn > 1)])), 0.99)   # 1 % FA on val
ft["alert"] = ft.score > thr

lines = [f"trained on {len(train):,} real normal moving reports ({train.track_id.nunique()} tracks)",
         f"test: {len(ft):,} moving reports, {int(ft.is_spoofed.sum())} spoofed, "
         f"{ft[ft.is_spoofed==1].track_id.nunique()} attacked tracks",
         f"ROC-AUC (report level): {roc_auc_score(ft.is_spoofed, ft.score):.3f}",
         f"threshold set for 1% false alarms on validation tracks",
         f"false-alarm rate on clean test reports: {ft[ft.is_spoofed==0].alert.mean():.2%}", ""]
att = ft[ft.is_spoofed == 1]
per = att.groupby("attack_type").agg(reports=("alert", "size"), report_recall=("alert", "mean"))
trk = att.groupby(["attack_type", "track_id"]).alert.any().groupby("attack_type").mean().rename("attack_detected")
lines.append(per.join(trk).round(3).to_string())
txt = "\n".join(lines); print(txt)
open(f"{d0}/demo_results.txt", "w").write(txt + "\n")

# ---- realism: real NY vs simulated traffic (resampled to ~70 s) -------------------
sim = pd.read_parquet("data/syndicate9_spoof_v1/ais.parquet",
                      columns=["voyage_id", "split", "t", "mmsi", "msg_type", "lat", "lon", "sog_kn", "cog_deg",
                               "heading", "is_spoofed", "is_own"])
sim = sim[sim.msg_type.isin([1, 18]) & (sim.is_spoofed == 0) & ~sim.is_own & (np.abs(sim.lat) <= 90)]
sim["track_id"] = sim.voyage_id.astype(str) + "_" + sim.mmsi.astype(str)
sim = sim.sort_values(["track_id", "t"])
sim["bucket"] = (sim.t // 70).astype(int)
sim = sim.groupby(["track_id", "bucket"]).first().reset_index()     # thin to MarineCadastre-like rate
sim["t"] = sim.groupby("track_id").t.transform(lambda s: s - s.iloc[0])
fs = kinematic_features(sim.reset_index(drop=True))
fs = fs[fs.dt_s.notna() & (fs.sog_kn > 1)]
fr = fe[fe.sog_kn > 1]

fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
spec = [("sog_kn", "reported speed [kn]", (0, 35)), ("speed_resid_kn", "implied − reported speed [kn]", (-5, 5)),
        ("course_resid_deg", "track vs COG [deg]", (0, 90)), ("turn_rate_deg_per_min", "turn rate [deg/min]", (-60, 60)),
        ("accel_kn_per_min", "acceleration [kn/min]", (-8, 8))]
for ax, (c, lab, rg) in zip(axs.flat, spec):
    b = np.linspace(*rg, 70)
    ax.hist(fr[c].clip(*rg), bins=b, density=True, histtype="stepfilled", alpha=.45, color="#2b6cb0", label="real NY harbour")
    ax.hist(fs[c].clip(*rg), bins=b, density=True, histtype="step", lw=1.8, color="#c05621", label="Syndicate-9 simulated")
    ax.set_xlabel(lab); ax.set_yscale("log"); ax.legend(fontsize=8)
ax = axs.flat[5]
for k, col in [("none", "0.6"), ("position_jump", "#c53030"), ("position_drift", "#d69e2e"), ("kinematic", "#2f855a")]:
    s = ft[ft.attack_type == k].score if k != "none" else ft[ft.is_spoofed == 0].score
    ax.hist(s, bins=60, density=True, histtype="step", lw=1.8, color=col, label="clean" if k == "none" else k)
ax.axvline(thr, ls="--", color="k", lw=1); ax.text(thr, ax.get_ylim()[1] * .9, " threshold", fontsize=8)
ax.set_xlabel("anomaly score (Isolation Forest, trained on real normal)"); ax.legend(fontsize=8)
fig.suptitle("Real New York harbour AIS (14 Jan 2024) vs simulated traffic, and anomaly scores on injected attacks")
fig.tight_layout(); fig.savefig(f"{d0}/real_vs_sim_and_scores.png", dpi=110)
print("figure:", f"{d0}/real_vs_sim_and_scores.png")
