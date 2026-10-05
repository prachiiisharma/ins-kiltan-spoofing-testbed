"""Sanity-check a generated dataset and draw overview figures.

    python validate_dataset.py data/syndicate9_spoof_v1

Checks
  1. label balance per split / stream / attack type
  2. raw-log round trip: re-parse raw NMEA + AIVDM with the *real-data loaders*
     and compare with the tables (this also tests the loaders you will use on
     your NEO-6M and AIS-catcher recordings)
  3. attack effect: position error vs ground truth per attack type
Figures go to <dataset>/figures/.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spoofsim.geo import haversine_m
from spoofsim.real_data import load_nmea_log, load_ais_nmea_log


def main(d):
    figdir = os.path.join(d, "figures")
    os.makedirs(figdir, exist_ok=True)
    g = pd.read_parquet(f"{d}/gnss.parquet")
    a = pd.read_parquet(f"{d}/ais.parquet")
    tr = pd.read_parquet(f"{d}/truth.parquet")
    att = pd.read_csv(f"{d}/attacks.csv")
    metas = json.load(open(f"{d}/voyages.json"))
    ok = True
    report = []
    p = lambda s="": (print(s), report.append(s))

    # ---------------- 1. balance ----------------------------------------------------
    p("## 1. Size and label balance")
    p(f"voyages: {len(metas)}  |  GNSS epochs: {len(g):,}  |  AIS messages: {len(a):,}  |  attacks: {len(att)}")
    p(f"voyage hours: {g.groupby('voyage_id', observed=True).t.max().sum() / 3600:.1f}")
    p("\nSpoofed fraction by split:")
    apos = a[a.msg_type.isin([1, 2, 3, 18])]
    tab = pd.DataFrame({"voyages": g.groupby("split", observed=True).voyage_id.nunique(),
                        "gnss_epochs": g.groupby("split", observed=True).size(),
                        "gnss_spoofed_%": (100 * g.groupby("split", observed=True).is_spoofed.mean()).round(1),
                        "ais_pos_msgs": apos.groupby("split", observed=True).size(),
                        "ais_spoofed_%": (100 * apos.groupby("split", observed=True).is_spoofed.mean()).round(1)})
    p(tab.to_string())
    p("\nAttack instances per type x split:")
    ct = pd.crosstab(att.type, att.split)
    p(ct.to_string())
    missing = [(t, s) for t in ct.index for s in ct.columns if ct.loc[t, s] == 0]
    if missing:
        p(f"WARNING: attack types missing from a split: {missing}")
    p("\nBenign hard-negative epochs (GNSS):")
    p(g[g.benign_event != "none"].benign_event.astype(str).value_counts().to_string())
    p(f"AIS messages right after a reception gap: {(a.benign_event == 'after_reception_gap').sum()}")

    # ---------------- 2. round trip -------------------------------------------------
    p("\n## 2. Raw log round trip")
    raws = sorted(os.listdir(f"{d}/raw")) if os.path.isdir(f"{d}/raw") else []
    for vid in raws[:3]:
        gl = load_nmea_log(f"{d}/raw/{vid}/gnss.nmea")
        gt = g[(g.voyage_id == vid)].reset_index(drop=True)
        fx = gt.fix_quality.values > 0
        n_ok = len(gl) == len(gt)
        # replay/time_jump attacks deliberately repeat/shift UTC, so compare row-wise on fixed epochs
        both = min(len(gl), len(gt))
        dpos = haversine_m(gl.lat.values[:both], gl.lon.values[:both], gt.lat.values[:both], gt.lon.values[:both])
        dpos = dpos[fx[:both] & np.isfinite(dpos)]
        al = load_ais_nmea_log(f"{d}/raw/{vid}/ais.nmea")
        at = a[a.voyage_id == vid].reset_index(drop=True)
        apl = al[al.msg_type.isin([1, 18])]
        apt = at[at.msg_type.isin([1, 18])]
        da = haversine_m(apl.lat.values, apl.lon.values, apt.lat.values, apt.lon.values) if len(apl) == len(apt) else np.array([np.inf])
        da = da[np.isfinite(da) & (np.abs(apt.lat.values) <= 90)] if len(apl) == len(apt) else da
        mm = (apl.mmsi.values == apt.mmsi.values).all() if len(apl) == len(apt) else False
        good = len(gl) >= len(gt) - 5 and dpos.max() < 0.05 and len(apl) == len(apt) and da.max() < 0.5 and mm
        ok &= bool(good)
        p(f"{vid}: GNSS epochs {len(gl)}/{len(gt)} max pos diff {dpos.max() * 100:.2f} cm | "
          f"AIS pos msgs {len(apl)}/{len(apt)} max diff {da.max():.2f} m, MMSI match {mm} -> {'OK' if good else 'FAIL'}")
        if not n_ok:
            p(f"   note: {len(gt) - len(gl)} epoch(s) merged by the parser because a replay/time-jump attack re-used a UTC stamp")

    # ---------------- 3. attack effect ----------------------------------------------
    p("\n## 3. Own-ship GNSS position error vs truth (m)")
    own = tr[tr.vessel_uid == "own"][["voyage_id", "t", "lat", "lon"]].rename(columns={"lat": "tlat", "lon": "tlon"})
    gg = g.merge(own, on=["voyage_id", "t"], how="left")
    gg["err_m"] = haversine_m(gg.lat, gg.lon, gg.tlat, gg.tlon)
    gg["cls"] = np.where(gg.is_spoofed == 1, gg.attack_type.astype(str),
                         np.where(gg.benign_event != "none", "benign:" + gg.benign_event.astype(str), "clean"))
    eff = gg.groupby("cls").err_m.describe(percentiles=[.5, .95])[["count", "50%", "95%", "max"]].round(1)
    p(eff.to_string())

    # ---------------- figures -------------------------------------------------------
    gatt = att[att.stream == "gnss"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, kind in zip(axs.flat, ["jump", "drag_off", "replay", "freeze", "time_jump"]):
        row = gatt[gatt.type == kind]
        if row.empty:
            ax.set_visible(False)
            continue
        r = row.iloc[0]
        v = gg[gg.voyage_id == r.voyage_id]
        ax.plot(v.tlon, v.tlat, color="0.6", lw=2.5, label="truth")
        m = v.attack_id == r.attack_id
        ax.plot(v.lon[~m], v.lat[~m], ".", ms=1, color="#2b6cb0", label="GNSS (authentic)")
        ax.plot(v.lon[m], v.lat[m], ".", ms=1.5, color="#c53030", label="GNSS (spoofed)")
        ax.set_title(f"{kind}  ({r.voyage_id}, {r.duration:.0f} s)")
        ax.set_aspect(1 / np.cos(np.radians(12)))
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, markerscale=6, loc="best")
    ax = axs.flat[5]
    for cls in ["clean", "benign:gnss_multipath", "jump", "drag_off", "replay", "freeze"]:
        x = gg[gg.cls == cls].err_m.dropna()
        if len(x):
            ax.hist(np.log10(x.clip(lower=0.1)), bins=np.linspace(-1, 4, 80), histtype="step",
                    weights=np.full(len(x), 1 / len(x)), lw=1.5, label=cls)
    ax.set_yscale("log")
    ax.set_ylabel("fraction of epochs")
    ax.set_xlabel("log10 position error vs truth [m]")
    ax.set_title("Error distribution by class")
    ax.legend(fontsize=7)
    fig.suptitle("Own-ship GNSS spoofing attacks — one example per type", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{figdir}/gnss_attacks.png", dpi=110)

    # AIS figure: map of one voyage with several attacks
    best = att[att.stream == "ais"].groupby("voyage_id").size().idxmax()
    v = a[(a.voyage_id == best) & a.msg_type.isin([1, 18])]
    fig, ax = plt.subplots(figsize=(10, 8))
    for uid, s in v[v.is_spoofed == 0].groupby("vessel_uid", observed=True):
        ax.plot(s.lon, s.lat, ".", ms=2, color="#2b6cb0" if uid != "own" else "black", alpha=.6)
    for k, s in v[v.is_spoofed == 1].groupby("attack_type", observed=True):
        ax.plot(s.lon, s.lat, "o", ms=3, label=k, alpha=.8)
    ax.set_title(f"AIS picture for {best}: blue = authentic traffic, black = own ship, colours = spoofed messages")
    ax.set_aspect(1 / np.cos(np.radians(12)))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{figdir}/ais_attacks_{best}.png", dpi=110)

    # timeline: anomaly-relevant signals for one voyage with a signature attack
    sig = gatt[gatt.signal_signature == True]  # noqa: E712
    r = (sig if len(sig) else gatt).iloc[0]
    v = gg[gg.voyage_id == r.voyage_id]
    fig, axs = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    axs[0].semilogy(v.t / 60, v.err_m.clip(lower=.1)); axs[0].set_ylabel("pos err vs truth [m]")
    axs[1].plot(v.t / 60, v.snr_mean, label="mean C/N0"); axs[1].plot(v.t / 60, v.snr_std, label="std C/N0")
    axs[1].legend(fontsize=8); axs[1].set_ylabel("dB-Hz")
    axs[2].plot(v.t / 60, v.hdop); axs[2].set_ylabel("HDOP")
    axs[3].plot(v.t / 60, v.nsat_used); axs[3].set_ylabel("sats used"); axs[3].set_xlabel("minutes")
    for _, at_ in gatt[gatt.voyage_id == r.voyage_id].iterrows():
        for ax in axs:
            ax.axvspan(at_.start / 60, (at_.start + at_.duration) / 60, color="red", alpha=.15)
    for ev in [m for m in metas if m["voyage_id"] == r.voyage_id][0]["benign_gnss"]:
        for ax in axs:
            ax.axvspan(ev["start"] / 60, ev["end"] / 60, color="green", alpha=.15)
    axs[0].set_title(f"{r.voyage_id}: red = spoofing windows, green = benign GNSS degradation")
    fig.tight_layout()
    fig.savefig(f"{figdir}/timeline_{r.voyage_id}.png", dpi=110)

    open(f"{d}/validation_report.md", "w").write("\n".join(report) + "\n")
    p(f"\nRound trip: {'PASS' if ok else 'FAIL'}  |  figures in {figdir}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
