p = "marsim_to_table.py"
s = open(p, encoding="utf-8").read()
if "persist=5" in s:
    print("already up to date")
else:
    NEW_FUNC = r'''def estimate_onsets(df, files, tol_m=20.0, persist=5):
    """Spoofed file vs. its unspoofed twin, matched on GPS time. Onset = first epoch
    after which the spoofed position stays > tol_m from the twin for `persist` epochs."""
    have = set(files.file)
    twin = {f: f.replace("-spoofed-", "-unspoofed-", 1) for f in files.file[files.file_label == "spoofed"]}
    twin = {k: v for k, v in twin.items() if v in have}
    groups = dict(tuple(df.groupby("file", sort=False, observed=True)))
    onset = {}
    for f in files.file[files.file_label == "spoofed"]:
        u = twin.get(f)
        if u is None or f not in groups or u not in groups:
            onset[f] = (0, "no_twin")
            continue
        a = groups[f].reset_index(drop=True)
        b = groups[u].reset_index(drop=True)
        if a.utc.is_unique and b.utc.is_unique and a.utc.isin(b.utc).mean() > 0.8:
            m = a[["utc", "epoch_idx", "lat", "lon"]].merge(b[["utc", "lat", "lon"]], on="utc", suffixes=("", "_t"))
            how = "twin_time"
        else:
            n = min(len(a), len(b))
            m = a.loc[:n - 1, ["epoch_idx", "lat", "lon"]].copy()
            m["lat_t"], m["lon_t"] = b.lat.values[:n], b.lon.values[:n]
            how = "twin_index"
        d = np.nan_to_num(haversine_m(m.lat.values, m.lon.values, m.lat_t.values, m.lon_t.values))
        far = d > tol_m
        run = np.convolve(far.astype(int), np.ones(persist, int), "valid") == persist
        idx = np.where(run)[0]
        onset[f] = (int(m.epoch_idx.values[idx[0]]), how) if len(idx) else (None, "within_tol_of_twin")
    return onset


'''
    NEW_LIMIT = r'''        pairs = {}
        for p, m in metas:
            pairs.setdefault((m["scenario"], m["run_index"], str(m["params"])), []).append(p)
        keys = sorted(pairs)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(keys), size=min(len(keys), max(1, a.limit // 2)), replace=False)
        paths = [p for k in sorted(pick) for p in pairs[keys[k]]]
'''
    a = s.index("def estimate_onsets"); b = s.index("def main():")
    s = s[:a] + NEW_FUNC + s[b:]
    old = '        paths = [p for p, _ in metas[:a.limit]]\n'
    if old in s:
        s = s.replace(old, NEW_LIMIT)
    s = s.replace('    ap.add_argument("--workers"', '    ap.add_argument("--tol_m", type=float, default=20.0)\n    ap.add_argument("--workers"', 1)
    s = s.replace("    on = estimate_onsets(df, files)", "    on = estimate_onsets(df, files, tol_m=a.tol_m)")
    open(p, "w", encoding="utf-8").write(s)
    print("updated" if "persist=5" in s and "tol_m=a.tol_m" in s and "pairs = {}" in s else "PARTIAL - tell Claude")