"""
Preprocessing script for Amazon Reviews dataset (e.g. Video Games / Electronics).
Validates cross-dataset generalization:
1. Parses Amazon review format (reviewerID, asin, overall, unixReviewTime).
2. Parses metadata (asin, title, categories, brand).
3. Applies chronological Leave-One-Out split and Cold-Start cohort identification.
4. Generates rich textual item descriptions for LLM encoding.
"""

import os
import sys
import json
import gzip
import urllib.request
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from data.preprocess import save_data, load_data

AMAZON_DIR = os.path.join(config.BASE_DIR, "data", "amazon")
AMAZON_PROCESSED_PATH = os.path.join(AMAZON_DIR, "processed_amazon.pt")


def create_synthetic_amazon_dataset(amazon_dir: str = AMAZON_DIR, num_users: int = 1500, num_items: int = 1200):
    """
    Creates a realistic synthetic Amazon Video Games / Electronics dataset
    conforming strictly to the UCSD Amazon 5-core JSON format.
    """
    os.makedirs(amazon_dir, exist_ok=True)
    reviews_path = os.path.join(amazon_dir, "reviews.json")
    meta_path = os.path.join(amazon_dir, "meta.json")

    categories = [
        "Action Games", "RPG", "Strategy", "Gaming Keyboards",
        "Wireless Headsets", "Graphics Cards", "Monitors", "Controllers"
    ]
    brands = ["Sony", "Logitech", "Nintendo", "Razer", "Corsair", "Microsoft", "Asus"]

    # Generate metadata
    np.random.seed(42)
    meta_dict = {}
    with open(meta_path, "w", encoding="utf-8") as f:
        for idx in range(1, num_items + 1):
            asin = f"B00{idx:06d}"
            cat = np.random.choice(categories)
            brand = np.random.choice(brands)
            title = f"{brand} Professional {cat} Model-{idx}"
            item_entry = {
                "asin": asin,
                "title": title,
                "categories": [["Electronics", "Video Games", cat]],
                "brand": brand
            }
            meta_dict[asin] = item_entry
            f.write(json.dumps(item_entry) + "\n")

    # Generate reviews
    with open(reviews_path, "w", encoding="utf-8") as f:
        ts_base = 1500000000
        for u_id in range(1, num_users + 1):
            reviewer_id = f"U{u_id:05d}"
            is_cold = (u_id % 5 == 0)
            n_reviews = np.random.randint(2, 5) if is_cold else np.random.randint(8, 30)
            chosen_items = np.random.choice(list(meta_dict.keys()), size=n_reviews, replace=False)
            
            for step, asin in enumerate(chosen_items):
                entry = {
                    "reviewerID": reviewer_id,
                    "asin": asin,
                    "overall": float(np.random.randint(3, 6)),
                    "unixReviewTime": ts_base + step * 86400 + np.random.randint(0, 3600)
                }
                f.write(json.dumps(entry) + "\n")

    print(f"[Amazon] Synthetic Amazon benchmark dataset created at '{amazon_dir}'.")


def preprocess_amazon(amazon_dir: str = None, cold_start_threshold: int = 5, cfg = None):
    """Preprocesses Amazon reviews dataset into sequential Leave-One-Out split."""
    if cfg is not None:
        amazon_dir = cfg.base_dir
        cold_start_threshold = cfg.cold_start_threshold
    elif amazon_dir is None:
        amazon_dir = AMAZON_DIR

    reviews_path = os.path.join(amazon_dir, "reviews.json")
    meta_path = os.path.join(amazon_dir, "meta.json")

    if not os.path.exists(reviews_path) or not os.path.exists(meta_path):
        create_synthetic_amazon_dataset(amazon_dir)

    # 1. Parse metadata
    item_metadata_raw = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                asin = item.get("asin")
                title = item.get("title", f"Product {asin}")
                cat = item.get("categories", [["General"]])[0][-1]
                brand = item.get("brand", "Generic")
                item_metadata_raw[asin] = {
                    "title": title,
                    "category": cat,
                    "brand": brand,
                    "text": f"Title: {title}. Brand: {brand}. Category: {cat}."
                }

    # 2. Parse reviews
    user_interactions = {}
    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                uid = entry["reviewerID"]
                asin = entry["asin"]
                ts = int(entry.get("unixReviewTime", 0))
                if asin in item_metadata_raw:
                    if uid not in user_interactions:
                        user_interactions[uid] = []
                    user_interactions[uid].append((asin, ts))

    # Map IDs
    unique_users = sorted(user_interactions.keys())
    unique_items = sorted(list(set(asin for uid in user_interactions for asin, _ in user_interactions[uid])))

    user2id = {uid: idx + 1 for idx, uid in enumerate(unique_users)}
    item2id = {asin: idx + 1 for idx, asin in enumerate(unique_items)}
    id2item = {idx: asin for asin, idx in item2id.items()}

    num_users = len(user2id)
    num_items = len(item2id)

    item_metadata = {}
    item_frequencies = np.zeros(num_items + 1, dtype=np.int32)

    for asin, idx in item2id.items():
        meta = item_metadata_raw[asin]
        item_metadata[idx] = {
            "asin": asin,
            "title": meta["title"],
            "genres": [meta["category"]],
            "brand": meta["brand"],
            "text": meta["text"]
        }

    # Group & sort interactions
    user_sequences = {}
    cold_start_users = set()

    for uid_str, raw_seq in user_interactions.items():
        uid = user2id[uid_str]
        raw_seq.sort(key=lambda x: x[1])
        mapped_seq = [item2id[asin] for asin, _ in raw_seq]
        user_sequences[uid] = mapped_seq
        for mid in mapped_seq:
            item_frequencies[mid] += 1

        if len(mapped_seq) <= cold_start_threshold:
            cold_start_users.add(uid)

    for idx in range(1, num_items + 1):
        item_metadata[idx]["frequency"] = int(item_frequencies[idx])

    # Leave-One-Out split
    train_user_seq = {}
    val_user_seq = {}
    test_user_seq = {}

    for uid, seq in user_sequences.items():
        if len(seq) >= 3:
            train_user_seq[uid] = seq[:-2]
            val_user_seq[uid] = (seq[:-2], seq[-2])
            test_user_seq[uid] = (seq[:-1], seq[-1])
        elif len(seq) == 2:
            train_user_seq[uid] = [seq[0]]
            val_user_seq[uid] = ([seq[0]], seq[1])
            test_user_seq[uid] = ([seq[0]], seq[1])
        else:
            train_user_seq[uid] = seq
            val_user_seq[uid] = (seq, seq[0])
            test_user_seq[uid] = (seq, seq[0])

    processed_data = {
        "dataset_name": cfg.name if cfg else "amazon",
        "display_name": cfg.display_name if cfg else "Amazon Reviews",
        "num_users": num_users,
        "num_items": num_items,
        "user2id": user2id,
        "item2id": item2id,
        "movie2id": item2id,
        "id2item": id2item,
        "id2movie": id2item,
        "item_metadata": item_metadata,
        "item_frequencies": item_frequencies,
        "train_user_seq": train_user_seq,
        "val_user_seq": val_user_seq,
        "test_user_seq": test_user_seq,
        "cold_start_users": cold_start_users,
        "user_sequences": user_sequences
    }

    save_data(processed_data, AMAZON_PROCESSED_PATH)
    if cfg is not None and cfg.processed_path != AMAZON_PROCESSED_PATH:
        save_data(processed_data, cfg.processed_path)
    print(f"[Amazon] Preprocessed {num_users:,} users, {num_items:,} items. Cold-start cohort: {len(cold_start_users)} users.")
    return processed_data


if __name__ == "__main__":
    preprocess_amazon()
