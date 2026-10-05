"""
Step 2: download and extract the MARSIM dataset from Zenodo (DOI 10.5281/zenodo.8202802).

    python download_marsim.py                 # downloads to downloads/, extracts to data/marsim
    python download_marsim.py --no-extract    # download only

* resumes an interrupted download (just run it again)
* checks the MD5 checksum published on Zenodo
* only needs the Python standard library
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

RECORD = "8202802"
API = f"https://zenodo.org/api/records/{RECORD}"
FALLBACK = {"key": "marsim_dataset.tar.bz2",
            "url": f"https://zenodo.org/records/{RECORD}/files/marsim_dataset.tar.bz2?download=1",
            "size": None, "checksum": None}


def record_files():
    try:
        with urllib.request.urlopen(API, timeout=30) as r:
            data = json.load(r)
        out = []
        for f in data.get("files", []):
            out.append({"key": f["key"], "url": f["links"]["self"], "size": f.get("size"),
                        "checksum": f.get("checksum")})
        return out or [FALLBACK]
    except Exception as e:
        print(f"(could not read Zenodo API: {e}; using the known file name)")
        return [FALLBACK]


def _fetch_once(url, dest, have):
    """One connection attempt. Appends to dest from byte `have`. Returns bytes now on disk."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored Range -> start over
            have = 0
        with open(dest, "ab" if have else "wb") as f:
            got, t0, last = have, time.time(), have
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                el = max(time.time() - t0, 1e-6)
                yield got, (got - last) / el / 1e6


def download(url, dest, size=None, max_failures=30):
    """Download with automatic resume: keeps reconnecting until the whole file is on disk."""
    failures = 0
    while True:
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if size and have == size:
            print(f"\r  download complete: {have / 1e9:.2f} GB                    ")
            return True
        if size and have > size:  # corrupt leftover -> start again
            os.remove(dest)
            have = 0
        before = have
        try:
            for got, rate in _fetch_once(url, dest, have):
                pct = f"{got / size * 100:5.1f}%" if size else ""
                sys.stdout.write(f"\r  {got / 1e9:6.3f} GB {pct}  {rate:5.2f} MB/s     ")
                sys.stdout.flush()
        except Exception as e:  # connection reset, timeout, ...
            sys.stdout.write(f"\n  connection dropped ({type(e).__name__}); reconnecting ...\n")
        now = os.path.getsize(dest) if os.path.exists(dest) else 0
        if not size:  # size unknown: trust a clean end of stream
            print()
            return True
        if now >= size:
            continue
        if now == before:
            failures += 1  # no progress at all this attempt
        else:
            failures = 0
        if failures >= max_failures:
            print(f"\n  giving up after {max_failures} attempts with no progress. "
                  f"Run the script again later (it resumes from {now / 1e9:.2f} GB).")
            return False
        wait = min(30, 2 + failures * 2)
        sys.stdout.write(f"\n  stopped at {now / 1e9:.3f} GB of {size / 1e9:.2f} GB; resuming in {wait} s ...\n")
        time.sleep(wait)


def md5_ok(path, checksum):
    if not checksum or not checksum.startswith("md5:"):
        return True
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() == checksum.split(":", 1)[1]


def extract(archive, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"extracting {archive} -> {out_dir} (43k files, this takes a while) ...")
    with tarfile.open(archive, "r:*") as tar:
        n = 0
        for m in tar:
            if not (m.isfile() or m.isdir()):
                continue
            if sys.version_info >= (3, 12):
                tar.extract(m, out_dir, filter="data")
            else:
                tar.extract(m, out_dir)
            n += 1
            if n % 2000 == 0:
                print(f"  {n} files")
    # flatten a single top-level folder so pcaps + dataset.json sit directly in out_dir
    for name in ("dataset.json",):
        for root, _, files in os.walk(out_dir):
            if name in files and root != out_dir:
                for f in os.listdir(root):
                    shutil.move(os.path.join(root, f), os.path.join(out_dir, f))
                break
    print("extraction done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", default="downloads")
    ap.add_argument("--out", default=os.path.join("data", "marsim"))
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.download_dir, exist_ok=True)

    files = record_files()
    for f in files:
        size = f"{f['size'] / 1e9:.2f} GB" if f["size"] else "size unknown"
        print(f"- {f['key']} ({size})")
    for f in files:
        dest = os.path.join(a.download_dir, f["key"])
        free = shutil.disk_usage(a.download_dir).free
        if free < 20e9:
            print(f"WARNING: only {free / 1e9:.1f} GB free; plan for about 20 GB "
                  f"(archive + 43k extracted files + features)")
        print(f"downloading {f['key']} ...")
        if not download(f["url"], dest, f["size"]):
            sys.exit(1)
        print("checking MD5 fingerprint ...")
        if not md5_ok(dest, f["checksum"]):
            os.remove(dest)
            print("MD5 MISMATCH: the file is corrupted, so it was deleted. Run the script again.")
            sys.exit(1)
        print("MD5 ok - file is complete and undamaged")
        if not a.no_extract and f["key"].endswith((".tar.bz2", ".tar.gz", ".tgz", ".tar", ".tar.xz")):
            extract(dest, a.out)


if __name__ == "__main__":
    main()
