"""Convert the MarSim pcap captures (NMEA-over-UDP) into one labelled table.

    python marsim_to_table.py --root data/marsim --out data/marsim_gnss.parquet
    python marsim_to_table.py --root data/marsim --out data/test.parquet --limit 200   # quick test

File names look like
    A1-spoofed-0-distance_to_ship-0-time_difference-0.015.pcap
    A3-unspoofed-9-shift_angle-90-shift_speed-68.pcap
i.e.  <scenario>-<spoofed|unspoofed>-<run index>-<param>-<value>-<param>-<value>
Every spoofed file has an unspoofed twin with the same scenario, index and
parameters. The twins are compared epoch by epoch to estimate when the attack
starts inside each spoofed capture (onset_s); epochs from then on get
is_spoofed = 1. If no twin is found, the whole spoofed file is labelled 1.

Output
    <out>.parquet         one row per GNSS epoch, same columns as the
                          syndicate9_spoof_v1 gnss table + marsim columns
    <out>_files.csv       one row per pcap: label, parameters, epochs, onset_s
"""
import argparse
import glob
import os
import re
import struct
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from spoofsim.real_data import parse_nmea_lines
from spoofsim.geo import haversine_m

NAME = re.compile(r"^(?P<scenario>[A-Za-z0-9]+)-(?P<label>spoofed|unspoofed)-(?P<index>\d+)-(?P<rest>.*)$")


def parse_name(path):
    base = os.path.basename(path)[:-5]
    m = NAME.match(base)
    if not m:
        return None
    rest = m.group("rest").split("-")
    params = {rest[i]: rest[i + 1] for i in range(0, len(rest) - 1, 2)}
    return dict(file=base, scenario=m.group("scenario"), file_label=m.group("label"),
                run_index=int(m.group("index")), params=params)


def read_pcap_udp(path):
    """Fast pure-python reader for classic pcap (not pcapng). Yields
    (capture_time_s, utf8 text) for every UDP payload."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 24:
        return
    magic = data[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        e = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        e = ">"
    else:
        yield from _read_with_scapy(path)   # pcapng or unusual format
        return
    nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    linktype = struct.unpack(e + "I", data[20:24])[0]
    off = 24
    n = len(data)
    while off + 16 <= n:
        ts_s, ts_f, incl, _ = struct.unpack(e + "IIII", data[off:off + 16])
        off += 16
        pkt = data[off:off + incl]
        off += incl
        t = ts_s + ts_f / (1e9 if nano else 1e6)
        if linktype == 1:            # Ethernet
            if len(pkt) < 14:
                continue
            eth = struct.unpack("!H", pkt[12:14])[0]
            p = 14
            if eth == 0x8100:        # VLAN tag
                eth = struct.unpack("!H", pkt[16:18])[0]
                p = 18
            if eth != 0x0800:
                continue
        elif linktype in (101, 228):  # raw IPv4
            p = 0
        elif linktype == 113:         # Linux cooked capture
            p = 16
        else:
            continue
        if len(pkt) < p + 20 or pkt[p + 9] != 17:   # not UDP
            continue
        ihl = (pkt[p] & 0x0F) * 4
        payload = pkt[p + ihl + 8:]
        if payload:
            yield t, payload.decode("ascii", errors="ignore")


def _read_with_scapy(path):
    from scapy.all import rdpcap, UDP
    for pk in rdpcap(path):
        if UDP in pk:
            yield float(pk.time), bytes(pk[UDP].payload).decode("ascii", errors="ignore")


def convert_file(path):
    meta = parse_name(path)
    if meta is None:
        return None, None
    lines = []
    for t, txt in read_pcap_udp(path):
        for ln in txt.splitlines():   # one UDP packet may carry several sentences
            if ln.strip():
                lines.append((t, ln))
    df = parse_nmea_lines(lines)
    info = dict(file=meta["file"], scenario=meta["scenario"], file_label=meta["file_label"],
                run_index=meta["run_index"], n_lines=len(lines), n_epochs=len(df), **meta["params"])
    if df.empty:
        return None, info
    for k in ("file", "scenario", "file_label", "run_index"):
        df[k] = meta[k]
    for k, v in meta["params"].items():
        try:
            df[k] = float(v)
        except ValueError:
            df[k] = v
    df["t_cap"] = df.cap_time - df.cap_time.iloc[0]
    df["epoch_idx"] = np.arange(len(df))
    return df, info


def estimate_onsets(df, files, tol_m=1.0):
    """Spoofed file vs. its unspoofed twin: first epoch where position or speed
    differ. Returns dict file -> onset epoch index (None if never differs)."""
    # the twin has the identical name with "unspoofed" in place of "spoofed"
    have = set(files.file)
    twin = {f: f.replace("-spoofed-", "-unspoofed-", 1) for f in files.file[files.file_label == "spoofed"]}
    twin = {k: v for k, v in twin.items() if v in have}
    groups = dict(tuple(df.groupby("file", sort=False)))
    onset = {}
    for _, r in files[files.file_label == "spoofed"].iterrows():
        u = twin.get(r.file)
        if u is None or r.file not in groups or u not in groups:
            onset[r.file] = (0, "no_twin")
            continue
        a, b = groups[r.file], groups[u]
        n = min(len(a), len(b))
        d = haversine_m(a.lat.values[:n], a.lon.values[:n], b.lat.values[:n], b.lon.values[:n])
        dv = np.abs(np.nan_to_num(a.sog_kn.values[:n]) - np.nan_to_num(b.sog_kn.values[:n]))
        diff = np.where((np.nan_to_num(d) > tol_m) | (dv > 0.2))[0]
        onset[r.file] = (int(diff[0]), "twin") if len(diff) else (None, "identical_to_twin")
    return onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/marsim")
    ap.add_argument("--out", default="data/marsim_gnss.parquet")
    ap.add_argument("--limit", type=int, help="only convert the first N files (quick test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.root, "**", "*.pcap"), recursive=True))
    if a.limit:
        # keep spoofed/unspoofed twins together when limiting
        metas = [(p, parse_name(p)) for p in paths]
        metas = [(p, m) for p, m in metas if m]
        metas.sort(key=lambda pm: (pm[1]["scenario"], pm[1]["run_index"], str(pm[1]["params"]), pm[1]["file_label"]))
        paths = [p for p, _ in metas[:a.limit]]
    print(f"{len(paths)} pcap files -> converting with {a.workers} workers ...")
    t0 = time.time()
    tables, infos = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (df, info) in enumerate(ex.map(convert_file, paths, chunksize=32), 1):
            if info:
                infos.append(info)
            if df is not None:
                tables.append(df)
            if i % 2000 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} files  ({time.time() - t0:.0f}s)")
    if not tables:
        raise SystemExit("no GPS epochs found - check --root")
    df = pd.concat(tables, ignore_index=True)
    files = pd.DataFrame(infos)

    on = estimate_onsets(df, files)
    files["onset_epoch"] = files.file.map(lambda f: on.get(f, (None, ""))[0])
    files["onset_method"] = files.file.map(lambda f: on.get(f, (None, "unspoofed"))[1])
    df = df.merge(files[["file", "onset_epoch"]], on="file", how="left")
    spoof_file = df.file_label == "spoofed"
    df["is_spoofed"] = (spoof_file & df.onset_epoch.notna() & (df.epoch_idx >= df.onset_epoch.fillna(1e12))).astype(int)
    df["attack_type"] = np.where(df.is_spoofed == 1, "marsim_" + df.scenario, "none")
    # onset in seconds (capture clock) for each spoofed file
    first = df[df.is_spoofed == 1].groupby("file").t_cap.min()
    files["onset_s"] = files.file.map(first)

    for c in ("file", "scenario", "file_label", "attack_type", "benign_event"):
        df[c] = df[c].astype("category")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out, index=False, compression="zstd")
    fcsv = a.out.rsplit(".", 1)[0] + "_files.csv"
    files.to_csv(fcsv, index=False)

    print(f"\nwrote {a.out}: {len(df):,} epochs from {df.file.nunique():,} files ({time.time() - t0:.0f}s)")
    print(f"wrote {fcsv}")
    print("\nfiles per scenario x label:")
    print(pd.crosstab(files.scenario, files.file_label).to_string())
    print("\nepochs per file: median", int(files.n_epochs.median()), " min", int(files.n_epochs.min()),
          " max", int(files.n_epochs.max()))
    sp = files[files.file_label == "spoofed"]
    print("\nattack onset found by twin comparison:")
    print(sp.groupby("scenario").onset_method.value_counts().unstack(fill_value=0).to_string())
    print("\nonset time in capture (s), spoofed files:")
    print(sp.groupby("scenario").onset_s.describe()[["count", "min", "50%", "max"]].round(1).to_string())
    print(f"\nepoch-level spoofed fraction: {df.is_spoofed.mean():.3f}")


if __name__ == "__main__":
    main()
