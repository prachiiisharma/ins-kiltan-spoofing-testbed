"""Turn a MarineCadastre daily AIS file into a clean "normal behaviour" dataset.

    python prepare_real_ais.py --src ais-2024-01-14.csv.zst --region nyc --out data/real_ais_nyc

Works on the plain .csv, the .csv.zst (needs `pip install zstandard`) and the
older .zip/.csv layout. Only the chosen region is kept, so the 6 M-row US file
never has to fit in memory.

Steps
  1. crop to the region box, drop duplicate reports
  2. split each vessel into tracks at gaps > --gap_min minutes
  3. keep tracks with >= --min_points reports
  4. compute per-report kinematic features (the same ones the detector uses)
  5. voyage-level train / val / test split by track

Output (<out>/)
  tracks.parquet      one row per AIS report, all labelled is_spoofed = 0
  features.parquet    same rows + kinematic features (first report of each track dropped)
  summary.txt         counts and feature statistics
"""
import argparse
import io
import os

import numpy as np
import pandas as pd

from spoofsim.geo import haversine_m, bearing_deg, wrap180

REGIONS = {  # lat_min, lat_max, lon_min, lon_max
    "nyc": (40.40, 40.95, -74.35, -73.65),
    "la_lb": (33.55, 33.85, -118.35, -118.05),
    "houston": (29.30, 29.80, -95.30, -94.60),
}

MC_COLS = {  # new (2024+) and old MarineCadastre column names -> ours
    "mmsi": "mmsi", "MMSI": "mmsi",
    "base_date_time": "utc", "BaseDateTime": "utc",
    "latitude": "lat", "LAT": "lat", "longitude": "lon", "LON": "lon",
    "sog": "sog_kn", "SOG": "sog_kn", "cog": "cog_deg", "COG": "cog_deg",
    "heading": "heading", "Heading": "heading",
    "vessel_name": "ship_name", "VesselName": "ship_name",
    "vessel_type": "ship_type", "VesselType": "ship_type",
    "status": "nav_status", "Status": "nav_status",
    "length": "length_m", "Length": "length_m",
    "transceiver": "ais_class", "TransceiverClass": "ais_class",
}


def _open(path):
    if path.endswith(".zst"):
        import zstandard
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                                encoding="utf-8", errors="ignore")
    return path


def load_region(path, box):
    keep = []
    for ch in pd.read_csv(_open(path), chunksize=1_000_000, low_memory=False):
        ch = ch.rename(columns={k: v for k, v in MC_COLS.items() if k in ch.columns})
        ch = ch[ch.lat.between(box[0], box[1]) & ch.lon.between(box[2], box[3])]
        keep.append(ch[[c for c in set(MC_COLS.values()) if c in ch.columns]])
    d = pd.concat(keep, ignore_index=True)
    d["utc"] = pd.to_datetime(d.utc, utc=True)
    return d.drop_duplicates(["mmsi", "utc"]).sort_values(["mmsi", "utc"]).reset_index(drop=True)


def make_tracks(d, gap_min, min_points):
    gap = d.groupby("mmsi").utc.diff().dt.total_seconds()
    new = gap.isna() | (gap > gap_min * 60)
    d["track_id"] = (d.mmsi.astype(str) + "_" + new.groupby(d.mmsi).cumsum().astype(str))
    n = d.groupby("track_id").utc.transform("size")
    d = d[n >= min_points].reset_index(drop=True)
    d["t"] = d.groupby("track_id").utc.transform(lambda s: (s - s.iloc[0]).dt.total_seconds())
    return d


def kinematic_features(d):
    """Per-report features computed from the previous report of the same track.
    Rate-independent where possible, so they transfer between 1-min AIS and
    1 Hz GNSS (resample the 1 Hz stream to the same interval before comparing)."""
    g = d.groupby("track_id", sort=False)
    plat, plon, pt = g.lat.shift(), g.lon.shift(), g.t.shift()
    psog, pcog = g.sog_kn.shift(), g.cog_deg.shift()
    f = pd.DataFrame(index=d.index)
    f["dt_s"] = d.t - pt
    f["step_m"] = haversine_m(plat.values, plon.values, d.lat.values, d.lon.values)
    f["v_implied_kn"] = f.step_m / f.dt_s / 0.514444
    f["sog_mean_kn"] = (d.sog_kn + psog) / 2
    f["speed_resid_kn"] = f.v_implied_kn - f.sog_mean_kn          # position vs reported speed
    brg = bearing_deg(plat.values, plon.values, d.lat.values, d.lon.values)
    cogm = (pcog + wrap180(d.cog_deg - pcog) / 2) % 360
    f["course_resid_deg"] = np.where(f.step_m > 30, np.abs(wrap180(brg - cogm)), 0.0)  # track vs reported COG
    f["accel_kn_per_min"] = (d.sog_kn - psog) / f.dt_s * 60
    f["turn_rate_deg_per_min"] = wrap180(d.cog_deg - pcog) / f.dt_s * 60
    hd = d.heading.where(d.heading < 360)
    f["heading_cog_diff_deg"] = pd.Series(np.abs(wrap180(hd - d.cog_deg)), index=d.index).where(d.sog_kn > 2)
    return pd.concat([d, f], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--region", default="nyc", choices=REGIONS)
    ap.add_argument("--out", default="data/real_ais_nyc")
    ap.add_argument("--gap_min", type=float, default=15)
    ap.add_argument("--min_points", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = load_region(a.src, REGIONS[a.region])
    n_raw, v_raw = len(d), d.mmsi.nunique()
    d = make_tracks(d, a.gap_min, a.min_points)
    rng = np.random.default_rng(a.seed)
    tids = d.track_id.unique()
    perm = rng.permutation(tids)
    k1, k2 = int(.7 * len(perm)), int(.85 * len(perm))
    split = {**{t: "train" for t in perm[:k1]}, **{t: "val" for t in perm[k1:k2]}, **{t: "test" for t in perm[k2:]}}
    d["split"] = d.track_id.map(split)
    d["is_spoofed"], d["attack_type"], d["source"] = 0, "none", f"marinecadastre_{a.region}"
    d["moving"] = d.sog_kn > 1.0
    d.to_parquet(f"{a.out}/tracks.parquet", index=False)

    f = kinematic_features(d)
    f = f[f.dt_s.notna()]
    f.to_parquet(f"{a.out}/features.parquet", index=False)

    lines = [f"source: {a.src}  region: {a.region} {REGIONS[a.region]}",
             f"reports in region: {n_raw:,}  vessels: {v_raw}",
             f"after track split (gap>{a.gap_min} min, >= {a.min_points} pts): {len(d):,} reports, "
             f"{d.track_id.nunique()} tracks, {d.mmsi.nunique()} vessels",
             f"tracks per split: {pd.Series(split).value_counts().to_dict()}",
             f"moving reports (SOG > 1 kn): {d.moving.mean():.1%}",
             "", "report interval (s): " + str(f.dt_s.describe(percentiles=[.1, .5, .9]).round(0).to_dict()),
             "", "feature distribution on MOVING reports:"]
    cols = ["v_implied_kn", "speed_resid_kn", "course_resid_deg", "accel_kn_per_min", "turn_rate_deg_per_min",
            "heading_cog_diff_deg"]
    lines.append(f.loc[f.sog_kn > 1, cols].describe(percentiles=[.5, .95, .99]).T
                 [["count", "50%", "95%", "99%", "max"]].round(2).to_string())
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
