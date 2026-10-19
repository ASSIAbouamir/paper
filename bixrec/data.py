"""Dataset preparation for BiXRec v2.

Every dataset is turned into one file, ``dataset.pt``, holding:
    num_items   : int, items are 1..num_items (0 = padding)
    user_seqs   : list[list[int]], chronological, de-duplicated (first occurrence kept)
    item_text   : list[str] of length num_items + 1 (index 0 = "")
    stats       : dict

Protocol (identical for all datasets, no hidden cohorts):
    * iterative k-core filter (users and items with >= k interactions), k = 5
    * chronological order per user, ties broken by file order
    * repeated (user, item) pairs removed, first occurrence kept, so the test
      target is never in the input history and history masking can only remove
      competitors
    * leave-one-out: train = seq[:-2], valid target = seq[-2], test target = seq[-1]

Usage:
    python -m bixrec.data ml1m   --raw data/raw/ml-1m           --out data/ml1m
    python -m bixrec.data amazon --raw data/raw/reviews_Beauty_5.json.gz \
                                 --meta data/raw/meta_Beauty.json.gz --out data/beauty
    python -m bixrec.data csv    --raw interactions.csv --meta item_text.csv --out data/x
    python -m bixrec.data txt    --raw ml-1m.txt --out data/ml1m_txt   (no text, sanity checks only)
    python -m bixrec.data recbole --raw ml-100k.inter --meta ml-100k.item --out data/ml100k
    python -m bixrec.data s3rec  --raw Beauty.txt --meta Beauty_item2attributes.json --out data/beauty_attr
    python -m bixrec.data compare --raw data/beauty --meta data/beauty_text   (same interactions? -> 1.0)
"""

import argparse
import ast
import gzip
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def k_core(rows: List[Tuple[str, str, float]], k: int = 5) -> List[Tuple[str, str, float]]:
    """Iterative k-core on (user, item, ts) rows."""
    while True:
        uc = Counter(r[0] for r in rows)
        ic = Counter(r[1] for r in rows)
        kept = [r for r in rows if uc[r[0]] >= k and ic[r[1]] >= k]
        if len(kept) == len(rows):
            return kept
        rows = kept


def build(rows: List[Tuple[str, str, float]], text_of: Dict[str, str], k: int, name: str) -> dict:
    """rows: (user, item, timestamp) in file order. Returns the dataset dict."""
    # de-duplicate (user, item): keep the earliest interaction (stable on file order)
    order = sorted(range(len(rows)), key=lambda i: (rows[i][2], i))
    seen, dedup = set(), []
    for i in order:
        u, it, ts = rows[i]
        if (u, it) in seen:
            continue
        seen.add((u, it))
        dedup.append((u, it, ts, i))
    rows_d = [(u, it, (ts, i)) for u, it, ts, i in dedup]
    rows_d = k_core(rows_d, k)

    items = sorted({r[1] for r in rows_d})
    item2id = {it: j + 1 for j, it in enumerate(items)}
    per_user = defaultdict(list)
    for u, it, key in rows_d:
        per_user[u].append((key, item2id[it]))
    users = sorted(per_user)
    user_seqs = [[i for _, i in sorted(per_user[u])] for u in users]

    item_text = [""] * (len(items) + 1)
    missing = 0
    for it, j in item2id.items():
        t = text_of.get(it, "")
        if not t:
            missing += 1
        item_text[j] = t

    lens = np.array([len(s) for s in user_seqs])
    stats = {
        "name": name,
        "users": len(user_seqs),
        "items": len(items),
        "interactions": int(lens.sum()),
        "density": float(lens.sum() / (len(user_seqs) * len(items))),
        "seq_len_mean": float(lens.mean()),
        "seq_len_median": float(np.median(lens)),
        "seq_len_min": int(lens.min()),
        "seq_len_max": int(lens.max()),
        "items_without_text": missing,
        "k_core": k,
    }
    return {"num_items": len(items), "user_seqs": user_seqs, "item_text": item_text,
            "raw_item_ids": [""] + items, "stats": stats}


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def read_ml1m(raw_dir: str):
    rows = []
    with open(os.path.join(raw_dir, "ratings.dat"), encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            if len(p) >= 4:
                rows.append((p[0], p[1], float(p[3])))
    text = {}
    with open(os.path.join(raw_dir, "movies.dat"), encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            if len(p) >= 3:
                text[p[0]] =f"Title: {p[1]}. Genres: {', '.join(p[2].split('|'))}."
    return rows, text


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def _parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)  # 2014 metadata uses python literals


def read_amazon(reviews_path: str, meta_path: str):
    rows = []
    with _open(reviews_path) as f:
        for line in f:
            d = _parse_line(line)
            if d is None:
                continue
            u = d.get("reviewerID") or d.get("user_id")
            it = d.get("asin") or d.get("parent_asin")
            ts = d.get("unixReviewTime") or d.get("timestamp")
            if u and it and ts is not None:
                rows.append((u, it, float(ts)))
    text = {}
    if meta_path:
        with _open(meta_path) as f:
            for line in f:
                d = _parse_line(line)
                if d is None:
                    continue
                it = d.get("asin") or d.get("parent_asin")
                # only descriptive fields: title, brand, categories (+ description if no title). Never
                # 'related' / also_bought / also_viewed / salesRank, which encode other users' interactions.
                title = str(d.get("title", "") or "").strip()
                brand = str(d.get("brand", "") or d.get("store", "") or "").strip()
                cats = d.get("categories") or d.get("category") or []
                paths = cats if cats and isinstance(cats[0], list) else [cats]
                names = []  # every category path, minus each path's root (e.g. "Beauty"), deduplicated
                for p in paths:
                    for c in (p[1:] if len(p) > 1 else p):
                        if c and str(c) not in names:
                            names.append(str(c))
                cats = ", ".join(names)
                parts = []
                if title:
                    parts.append(f"Title: {title}.")
                if brand:
                    parts.append(f"Brand: {brand}.")
                if cats:
                    parts.append(f"Category: {cats}.")
                if not title:
                    desc = d.get("description", "")
                    if isinstance(desc, list):
                        desc = " ".join(desc)
                    desc = str(desc).strip()
                    if desc:
                        parts.append(f"Description: {desc[:300]}.")
                text[it] = " ".join(parts)
    return rows, text


def read_csv(inter_path: str, meta_path: str):
    """interactions: user,item,timestamp (header). meta: item,text (header)."""
    import csv
    rows = []
    with open(inter_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["user"], r["item"], float(r["timestamp"])))
    text = {}
    if meta_path:
        with open(meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                text[r["item"]] = r["text"]
    return rows, text


def read_txt(path: str):
    """'user item' per line, already chronological (SASRec format)."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            p = line.split()
            if len(p) >= 2:
                rows.append((p[0], p[1], float(i)))
    return rows, {}


def read_recbole(inter_path: str, item_path: str):
    """RecBole atomic files (e.g. ml-100k.inter / ml-100k.item from the RecBole repository)."""
    rows = []
    with open(inter_path, encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = {h.split(":")[0]: i for i, h in enumerate(header)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows.append((p[cols["user_id"]], p[cols["item_id"]], float(p[cols["timestamp"]])))
    text = {}
    if item_path:
        with open(item_path, encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\n").split("\t")
            cols = {h.split(":")[0]: i for i, h in enumerate(header)}
            for line in f:
                p = line.rstrip("\n").split("\t")
                get = lambda k: p[cols[k]].strip() if k in cols and cols[k] < len(p) else ""
                parts = [f"Title: {get('movie_title')}."]
                if get("release_year"):
                    parts.append(f"Year: {get('release_year')}.")
                if get("class"):
                    parts.append(f"Genres: {', '.join(get('class').split())}.")
                text[get("item_id")] = " ".join(parts)
    return rows, text


def read_s3rec(txt_path: str, attr_path: str):
    """S3-Rec benchmark format: 'user i1 i2 ...' per line (chronological) and item2attributes.json.
    Attribute ids (anonymised brand/category ids) become the item's side-information 'text' as tokens."""
    rows = []
    with open(txt_path) as f:
        for line in f:
            p = line.split()
            for t, it in enumerate(p[1:]):
                rows.append((p[0], it, float(t)))
    text = {}
    if attr_path:
        with open(attr_path) as f:
            for it, attrs in json.load(f).items():
                text[it] = " ".join(f"attr{a}" for a in attrs)
    return rows, text


def same_interactions(a: dict, b: dict) -> float:
    """Fraction of users whose sequence in dataset b matches one in dataset a, up to item relabelling.
    Each sequence is described by the global popularity of its items, in order (ids differ between
    releases); 1.0 means both datasets have the same users, sequences, orders and leave-one-out targets."""
    def signatures(ds):
        pop = Counter(i for s in ds["user_seqs"] for i in s)
        return Counter(tuple(pop[i] for i in s) for s in ds["user_seqs"])
    sa, sb = signatures(a), signatures(b)
    return sum((sa & sb).values()) / max(sum(sb.values()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ml1m", "amazon", "csv", "txt", "recbole", "s3rec", "compare"])
    ap.add_argument("--raw", required=True, help="raw file/dir (compare: first prepared data dir)")
    ap.add_argument("--meta", default="", help="item text file (compare: second prepared data dir)")
    ap.add_argument("--out", default="")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if a.kind == "compare":
        load = lambda d: torch.load(os.path.join(d, "dataset.pt"), weights_only=False)
        da, db = load(a.raw), load(a.meta)
        print(json.dumps({"a": da["stats"], "b": db["stats"], "same_interactions": same_interactions(da, db)},
                         indent=1))
        return
    if not a.out:
        ap.error("--out is required")
    if a.kind == "ml1m":
        rows, text = read_ml1m(a.raw)
    elif a.kind == "amazon":
        rows, text = read_amazon(a.raw, a.meta)
    elif a.kind == "csv":
        rows, text = read_csv(a.raw, a.meta)
    elif a.kind == "recbole":
        rows, text = read_recbole(a.raw, a.meta)
    elif a.kind == "s3rec":
        rows, text = read_s3rec(a.raw, a.meta)
    else:
        rows, text = read_txt(a.raw)
    ds = build(rows, text, a.k, os.path.basename(os.path.normpath(a.out)))
    os.makedirs(a.out, exist_ok=True)
    torch.save(ds, os.path.join(a.out, "dataset.pt"))
    print(json.dumps(ds["stats"], indent=2))


if __name__ == "__main__":
    main()
