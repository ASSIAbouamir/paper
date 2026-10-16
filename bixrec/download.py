"""Download raw data (run on Colab or any machine with internet).

    python -m bixrec.download --root data/raw --datasets ml1m beauty sports

ML-1M  : GroupLens official zip.
Amazon : 2014 "5-core" review files + metadata (McAuley et al.), the standard
         Beauty / Sports benchmarks of S3-Rec, UniSRec, etc.
If a URL fails, download the file manually and place it in --root with the same name.
"""

import argparse
import os
import urllib.request
import zipfile

AMAZON_BASES = [
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/",
    "https://jmcauley.ucsd.edu/data/amazon/categoryFiles/",
]
AMAZON = {"beauty": "Beauty", "sports": "Sports_and_Outdoors", "toys": "Toys_and_Games"}


def fetch(url: str, dest: str) -> bool:
    try:
        print("GET", url, flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(dest + ".part", dest)
        return True
    except Exception as e:  # noqa: BLE001
        print("  failed:", e)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--datasets", nargs="+", default=["ml1m", "beauty", "sports"])
    a = ap.parse_args()
    os.makedirs(a.root, exist_ok=True)
    for ds in a.datasets:
        if ds == "ml1m":
            out = os.path.join(a.root, "ml-1m")
            if os.path.exists(os.path.join(out, "ratings.dat")):
                continue
            z = os.path.join(a.root, "ml-1m.zip")
            if fetch("https://files.grouplens.org/datasets/movielens/ml-1m.zip", z):
                zipfile.ZipFile(z).extractall(a.root)
            continue
        cat = AMAZON[ds]
        for fname in (f"reviews_{cat}_5.json.gz", f"meta_{cat}.json.gz"):
            dest = os.path.join(a.root, fname)
            if os.path.exists(dest):
                continue
            if not any(fetch(b + fname, dest) for b in AMAZON_BASES):
                print(f"!! could not download {fname}; place it manually in {a.root}")


if __name__ == "__main__":
    main()
