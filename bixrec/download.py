"""Download raw data (run on any machine with internet).

    python -m bixrec.download --root data/raw --datasets s3rec_beauty s3rec_sports ml100k beauty sports

ml1m          : GroupLens official zip.
ml100k        : RecBole atomic files (ml-100k.inter / ml-100k.item from the RecBole repository).
s3rec_beauty, s3rec_sports : the S3-Rec release (Zhou et al., CIKM 2020) used by DIF-SR / DIFF and most side-
                information work: '<Cat>.txt' sequences + '<Cat>_item2attributes.json' (anonymised attribute ids).
beauty, sports, toys : Amazon 2014 "5-core" reviews + metadata (McAuley et al.); same interactions as the S3-Rec
                release, but with the real product text (title, brand, categories).
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
S3REC_BASE = "https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/master/data/"
S3REC = {"s3rec_beauty": "Beauty", "s3rec_sports": "Sports_and_Outdoors", "s3rec_toys": "Toys_and_Games"}
RECBOLE_ML100K = "https://raw.githubusercontent.com/RUCAIBox/RecBole/master/recbole/dataset_example/ml-100k/"


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


def fetch_all(urls_and_names, root):
    for url, fname in urls_and_names:
        dest = os.path.join(root, fname)
        if not os.path.exists(dest) and not fetch(url, dest):
            print(f"!! could not download {fname}; place it manually in {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--datasets", nargs="+", default=["ml1m", "beauty", "sports"],
                    choices=["ml1m", "ml100k"] + sorted(AMAZON) + sorted(S3REC))
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
        elif ds == "ml100k":
            fetch_all([(RECBOLE_ML100K + f, f) for f in ("ml-100k.inter", "ml-100k.item")], a.root)
        elif ds in S3REC:
            cat = S3REC[ds]
            fetch_all([(S3REC_BASE + f, f) for f in (f"{cat}.txt", f"{cat}_item2attributes.json")], a.root)
        else:
            cat = AMAZON[ds]
            for fname in (f"reviews_{cat}_5.json.gz", f"meta_{cat}.json.gz"):
                dest = os.path.join(a.root, fname)
                if os.path.exists(dest):
                    continue
                if not any(fetch(b + fname, dest) for b in AMAZON_BASES):
                    print(f"!! could not download {fname}; place it manually in {a.root}")


if __name__ == "__main__":
    main()
