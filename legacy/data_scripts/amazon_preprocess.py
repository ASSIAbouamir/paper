"""
Amazon Product Reviews Preprocessing for Sequential Recommendation.

Supports:
  - Amazon Beauty (All_Beauty)       : ~87K ratings, 1,210 items
  - Amazon Sports & Outdoors         : ~296K ratings, 18,357 items

Data source: Amazon Reviews 2023 (McAuley Lab)
  https://amazon-reviews-2023.github.io/

Processing pipeline:
  1. Download raw .jsonl.gz ratings + metadata
  2. Filter users/items by min interaction count
  3. Sort interactions chronologically per user
  4. Strict Leave-One-Out sequential split (no data leakage)
  5. Build item metadata text for LLM embedding
  6. Cold-start cohort simulation (truncate to <= threshold interactions)
"""

import os
import sys
import json
import gzip
import pickle
import urllib.request
import ssl
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Amazon Reviews 2023 download URLs ─────────────────────────────────────────

AMAZON_URLS = {
    "All_Beauty": {
        "ratings": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz",
        "meta":    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz",
    },
    "Sports_and_Outdoors": {
        "ratings": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz",
        "meta":    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz",
    },
}

# Fallback smaller mirrors / alternate format (5-core pre-filtered)
AMAZON_FALLBACK_URLS = {
    "All_Beauty": {
        "ratings": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/All_Beauty.json.gz",
    },
    "Sports_and_Outdoors": {
        "ratings": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/Sports_and_Outdoors_5.json.gz",
    },
}


def _download_gz(url: str, dest_path: str) -> bool:
    """Downloads a .gz file from url to dest_path. Returns True on success."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10_000:
        print(f"  [Download] Already exists: {dest_path}")
        return True
    print(f"  [Download] {url}")
    print(f"          → {dest_path}")
    try:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        print(f"  [Download] OK — {os.path.getsize(dest_path):,} bytes")
        return True
    except Exception as e:
        print(f"  [Download] FAILED: {e}")
        return False


def _iter_jsonl_gz(path: str):
    """Iterates over lines of a gzip-compressed JSONL file."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _save_data(obj: Any, path: str) -> None:
    try:
        import torch
        torch.save(obj, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def preprocess_amazon(cfg, category: str = "All_Beauty") -> Dict[str, Any]:
    """
    Full preprocessing pipeline for Amazon product reviews.

    Args:
        cfg: DatasetConfig object (from dataset_registry)
        category: Amazon category string ("All_Beauty" or "Sports_and_Outdoors")

    Returns:
        Processed data dict compatible with the rest of the pipeline.
    """
    data_dir = cfg.base_dir
    os.makedirs(data_dir, exist_ok=True)

    ratings_gz = os.path.join(data_dir, f"{category}_ratings.jsonl.gz")
    meta_gz    = os.path.join(data_dir, f"{category}_meta.jsonl.gz")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    urls = AMAZON_URLS.get(category, {})
    ratings_ok = _download_gz(urls.get("ratings", ""), ratings_gz)
    meta_ok    = _download_gz(urls.get("meta", ""), meta_gz)

    # Try fallback if primary failed
    if not ratings_ok:
        fallback = AMAZON_FALLBACK_URLS.get(category, {})
        ratings_ok = _download_gz(fallback.get("ratings", ""), ratings_gz)

    if not ratings_ok:
        raise RuntimeError(
            f"Could not download Amazon '{category}' ratings. "
            f"Please manually download from https://amazon-reviews-2023.github.io/ "
            f"and place the file at: {ratings_gz}"
        )

    # ── Step 2: Parse ratings ─────────────────────────────────────────────────
    print(f"\n[Amazon] Parsing ratings for '{category}'...")
    raw_interactions = []  # (user_id_str, item_id_str, rating, timestamp)

    for record in _iter_jsonl_gz(ratings_gz):
        # Amazon Reviews 2023 format: {"rating", "title", "text", "parent_asin", "user_id", "timestamp", ...}
        # Legacy Amazon format:       {"overall", "asin", "reviewerID", "unixReviewTime", ...}
        user_id  = record.get("user_id") or record.get("reviewerID")
        item_id  = record.get("parent_asin") or record.get("asin")
        rating   = float(record.get("rating") or record.get("overall") or 0)
        ts       = int(record.get("timestamp") or record.get("unixReviewTime") or 0)

        if user_id and item_id and ts > 0:
            raw_interactions.append((user_id, item_id, rating, ts))

    print(f"[Amazon] Raw interactions: {len(raw_interactions):,}")

    # ── Step 3: Parse metadata ────────────────────────────────────────────────
    print(f"[Amazon] Parsing metadata...")
    raw_meta: Dict[str, Dict] = {}

    if meta_ok and os.path.exists(meta_gz):
        for record in _iter_jsonl_gz(meta_gz):
            asin = record.get("parent_asin") or record.get("asin")
            if asin:
                title       = record.get("title") or record.get("description", [""])[0] or asin
                categories  = record.get("categories") or []
                # Flatten nested categories list
                if categories and isinstance(categories[0], list):
                    categories = [cat for sublist in categories for cat in sublist]
                features    = record.get("features") or []
                description = record.get("description") or []
                text_parts  = [str(title)]
                if categories:
                    text_parts.append("Categories: " + ", ".join(str(c) for c in categories[:5]))
                if features:
                    text_parts.append("Features: " + ". ".join(str(f) for f in features[:3]))
                if description:
                    text_parts.append(str(description[0])[:200])

                raw_meta[asin] = {
                    "title": str(title),
                    "categories": [str(c) for c in categories[:5]],
                    "text": " ".join(text_parts)[:512],
                }

    print(f"[Amazon] Metadata entries: {len(raw_meta):,}")

    # ── Step 4: Filter by interaction count ──────────────────────────────────
    min_interactions = cfg.min_interactions
    min_item_freq    = cfg.min_item_freq

    # Count per user and item
    user_counter = Counter(r[0] for r in raw_interactions)
    item_counter = Counter(r[1] for r in raw_interactions)

    valid_users = {u for u, c in user_counter.items() if c >= min_interactions}
    valid_items = {i for i, c in item_counter.items() if c >= min_item_freq}

    # K-core filtering (iterative until stable)
    print(f"[Amazon] K-core filtering (min_user={min_interactions}, min_item={min_item_freq})...")
    prev_n = -1
    for iteration in range(20):
        filtered = [
            r for r in raw_interactions
            if r[0] in valid_users and r[1] in valid_items
        ]
        user_counter = Counter(r[0] for r in filtered)
        item_counter = Counter(r[1] for r in filtered)
        valid_users = {u for u, c in user_counter.items() if c >= min_interactions}
        valid_items = {i for i, c in item_counter.items() if c >= min_item_freq}
        n = len(filtered)
        print(f"  Iter {iteration+1}: {len(valid_users):,} users, {len(valid_items):,} items, {n:,} interactions")
        if n == prev_n:
            break
        prev_n = n

    raw_interactions = filtered
    print(f"[Amazon] After K-core: {len(raw_interactions):,} interactions, "
          f"{len(valid_users):,} users, {len(valid_items):,} items")

    # ── Step 5: Re-index users and items ─────────────────────────────────────
    user_ids = sorted(valid_users)
    item_ids = sorted(valid_items)

    user2id = {u: idx + 1 for idx, u in enumerate(user_ids)}
    item2id = {i: idx + 1 for idx, i in enumerate(item_ids)}
    id2item = {idx: i for i, idx in item2id.items()}

    num_users = len(user2id)
    num_items = len(item2id)

    # ── Step 6: Item frequencies ──────────────────────────────────────────────
    item_frequencies = np.zeros(num_items + 1, dtype=np.int32)
    for _, item_id, _, _ in raw_interactions:
        if item_id in item2id:
            item_frequencies[item2id[item_id]] += 1

    # ── Step 7: Item metadata ─────────────────────────────────────────────────
    item_metadata = {}
    for orig_id in item_ids:
        mapped_id = item2id[orig_id]
        meta = raw_meta.get(orig_id, {})
        item_metadata[mapped_id] = {
            "raw_id": orig_id,
            "title": meta.get("title", orig_id),
            "categories": meta.get("categories", []),
            "text": meta.get("text", f"Product ID: {orig_id}"),
            "frequency": int(item_frequencies[mapped_id]),
        }

    # ── Step 8: Build chronological user sequences ────────────────────────────
    user_interactions: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for user_str, item_str, _, ts in raw_interactions:
        if user_str in user2id and item_str in item2id:
            uid = user2id[user_str]
            mid = item2id[item_str]
            user_interactions[uid].append((mid, ts))

    user_sequences: Dict[int, List[int]] = {}
    for uid in sorted(user_interactions.keys()):
        user_interactions[uid].sort(key=lambda x: x[1])
        # Deduplicate consecutive identical items (re-review artifacts)
        seq = []
        prev = -1
        for item, _ in user_interactions[uid]:
            if item != prev:
                seq.append(item)
                prev = item
        user_sequences[uid] = seq

    # ── Step 9: Cold-start cohort ─────────────────────────────────────────────
    cold_start_threshold = cfg.cold_start_threshold
    np.random.seed(42)
    shuffled_uids = np.random.permutation(list(user_sequences.keys()))
    num_cold = int(len(shuffled_uids) * 0.20)
    cold_start_users = set(int(u) for u in shuffled_uids[:num_cold])

    # ── Step 10: Leave-One-Out split ─────────────────────────────────────────
    train_user_seq: Dict[int, List[int]] = {}
    val_user_seq:   Dict[int, Tuple]     = {}
    test_user_seq:  Dict[int, Tuple]     = {}

    skipped = 0
    for uid, full_seq in user_sequences.items():
        if len(full_seq) < 3:
            skipped += 1
            continue

        if uid in cold_start_users:
            cold_seq = full_seq[:cold_start_threshold]
            if len(cold_seq) < 3:
                cold_seq = full_seq[:3]
            train_user_seq[uid] = cold_seq[:-2]
            val_user_seq[uid]   = (cold_seq[:-2], cold_seq[-2])
            test_user_seq[uid]  = (cold_seq[:-1], cold_seq[-1])
        else:
            train_user_seq[uid] = full_seq[:-2]
            val_user_seq[uid]   = (full_seq[:-2], full_seq[-2])
            test_user_seq[uid]  = (full_seq[:-1], full_seq[-1])

    if skipped > 0:
        print(f"[Amazon] Skipped {skipped} users with < 3 interactions after dedup.")

    # Recalculate num_users after filtering
    final_users = set(train_user_seq.keys())
    num_users = len(final_users)

    processed_data = {
        "dataset_name": cfg.name,
        "display_name": cfg.display_name,
        "num_users": num_users,
        "num_items": num_items,
        "user2id": user2id,
        "item2id": item2id,
        "id2item": id2item,
        "item_metadata": item_metadata,
        "item_frequencies": item_frequencies,
        "user_sequences": user_sequences,
        "train_user_seq": train_user_seq,
        "val_user_seq": val_user_seq,
        "test_user_seq": test_user_seq,
        "cold_start_users": cold_start_users,
        "user_genre_profiles": {},  # Amazon uses categories not genres
    }

    _save_data(processed_data, cfg.processed_path)
    print(f"\n[Amazon '{cfg.display_name}'] Preprocessing complete.")
    print(f"  Users: {num_users:,} | Items: {num_items:,} | "
          f"Cold-start cohort: {len(cold_start_users):,}")
    print(f"  Saved to: {cfg.processed_path}")

    return processed_data


if __name__ == "__main__":
    from data.dataset_registry import DATASET_CONFIGS
    # Test with Beauty
    cfg = DATASET_CONFIGS["amazon_beauty"]
    preprocess_amazon(cfg, category="All_Beauty")
