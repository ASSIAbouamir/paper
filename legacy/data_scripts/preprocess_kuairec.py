"""
Preprocessing script for KuaiRec Micro-Video sequential recommendation dataset.
Parses real-world user interactions and micro-video captions & hierarchical categories.
Applies chronological Leave-One-Out (LOO) splitting and formats item metadata for MiniLM embedding.
"""

import os
import sys
import csv
import pickle
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KUAIREC_DIR = os.path.join(BASE_DIR, "data", "kuairec")
KUAIREC_PROCESSED_PATH = os.path.join(KUAIREC_DIR, "processed_kuairec.pt")
INTERACTIONS_PATH = os.path.join(BASE_DIR, "data", "atd_drl_data", "kuairec", "interactions.csv")
CAPTIONS_PATH = os.path.join(BASE_DIR, "data", "kuairec_caption_category.csv")


def preprocess_kuairec(cold_cohort_ratio: float = 0.20, cold_start_threshold: int = 5, cfg = None):
    if cfg is not None:
        cold_start_threshold = cfg.cold_start_threshold
        os.makedirs(cfg.base_dir, exist_ok=True)
    os.makedirs(KUAIREC_DIR, exist_ok=True)
    print(f"[KuaiRec] Starting preprocessing of real KuaiRec dataset...")
    print(f"[KuaiRec] Interactions source: {INTERACTIONS_PATH}")
    print(f"[KuaiRec] Captions source: {CAPTIONS_PATH}")

    # 1. Load item metadata (captions & categories)
    raw_metadata = {}
    if os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = str(row["video_id"]).strip()
                caption = (row.get("caption") or "").strip()
                tag = (row.get("topic_tag") or "").strip()
                cat1 = (row.get("first_level_category_name") or "").strip()
                cat2 = (row.get("second_level_category_name") or "").strip()
                cat3 = (row.get("third_level_category_name") or "").strip()
                
                # Construct clean descriptive text representation
                desc_parts = []
                cats = [c for c in [cat1, cat2, cat3] if c and c != "UNKNOWN"]
                if cats:
                    desc_parts.append(f"Categories: {' > '.join(cats)}")
                if tag and tag != "[]":
                    desc_parts.append(f"Tags: {tag}")
                if caption:
                    desc_parts.append(f"Caption: {caption}")
                
                text_desc = f"Video ID {vid}. " + (". ".join(desc_parts) if desc_parts else "Short video content.")
                raw_metadata[vid] = {
                    "caption": caption,
                    "topic_tag": tag,
                    "categories": cats,
                    "text": text_desc
                }
        print(f"[KuaiRec] Loaded metadata for {len(raw_metadata):,} videos.")
    else:
        print(f"[KuaiRec] Warning: Captions file not found at {CAPTIONS_PATH}")

    # 2. Load interactions
    user_interactions = {}
    all_videos = set()
    total_interactions = 0

    with open(INTERACTIONS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row["user_id"]).strip()
            vid = str(row["video_id"]).strip()
            ts = float(row.get("timestamp", 0.0))
            watch_ratio = float(row.get("watch_ratio", 1.0))
            
            if uid not in user_interactions:
                user_interactions[uid] = []
            user_interactions[uid].append((vid, ts, watch_ratio))
            all_videos.add(vid)
            total_interactions += 1

    print(f"[KuaiRec] Raw interactions: {total_interactions:,} from {len(user_interactions):,} users across {len(all_videos):,} items.")

    # 3. Create ID mappings (1-based for embedding layers, index 0 reserved for padding)
    sorted_uids = sorted(list(user_interactions.keys()), key=lambda x: int(x) if x.isdigit() else x)
    sorted_vids = sorted(list(all_videos), key=lambda x: int(x) if x.isdigit() else x)

    user2id = {uid: idx + 1 for idx, uid in enumerate(sorted_uids)}
    video2id = {vid: idx + 1 for idx, vid in enumerate(sorted_vids)}
    id2video = {idx: vid for vid, idx in video2id.items()}

    num_users = len(user2id)
    num_items = len(video2id)

    # 4. Map item metadata and compute frequencies
    item_frequencies = np.zeros(num_items + 1, dtype=np.int32)
    item_metadata = {}

    for idx in range(1, num_items + 1):
        raw_vid = id2video[idx]
        meta = raw_metadata.get(raw_vid, {
            "caption": "",
            "topic_tag": "",
            "categories": ["Micro-video"],
            "text": f"Micro-video content ID {raw_vid}."
        })
        item_metadata[idx] = {
            "raw_id": raw_vid,
            "caption": meta.get("caption", ""),
            "topic_tag": meta.get("topic_tag", ""),
            "categories": meta.get("categories", []),
            "text": meta.get("text", f"Video {idx}"),
            "frequency": 0
        }

    # 5. Chronological sorting per user sequence
    user_sequences = {}
    for uid_str, inters in user_interactions.items():
        uid = user2id[uid_str]
        inters.sort(key=lambda x: x[1])  # sort by timestamp
        mapped_seq = [video2id[vid] for vid, _, _ in inters]
        user_sequences[uid] = mapped_seq
        for mid in mapped_seq:
            item_frequencies[mid] += 1

    for idx in range(1, num_items + 1):
        item_metadata[idx]["frequency"] = int(item_frequencies[idx])

    # 6. Cold-Start Cohort Designation (20% cohort truncated to 5 interactions for evaluation)
    np.random.seed(42)
    shuffled_uids = np.random.permutation(list(user_sequences.keys()))
    num_cold_users = int(len(shuffled_uids) * cold_cohort_ratio)
    cold_start_users = set(shuffled_uids[:num_cold_users])

    # 7. Leave-One-Out Temporal Split
    train_user_seq = {}
    val_user_seq = {}
    test_user_seq = {}

    for uid, full_seq in user_sequences.items():
        if uid in cold_start_users:
            cold_seq = full_seq[:cold_start_threshold]
            train_user_seq[uid] = cold_seq[:-2]
            val_user_seq[uid] = (cold_seq[:-2], cold_seq[-2])
            test_user_seq[uid] = (cold_seq[:-1], cold_seq[-1])
        else:
            train_user_seq[uid] = full_seq[:-2]
            val_user_seq[uid] = (full_seq[:-2], full_seq[-2])
            test_user_seq[uid] = (full_seq[:-1], full_seq[-1])

    processed_data = {
        "dataset_name": cfg.name if cfg else "kuairec",
        "display_name": cfg.display_name if cfg else "KuaiRec Micro-Video",
        "num_users": num_users,
        "num_items": num_items,
        "user2id": user2id,
        "item2id": video2id,
        "video2id": video2id,
        "movie2id": video2id,
        "id2item": id2video,
        "id2video": id2video,
        "id2movie": id2video,
        "item_metadata": item_metadata,
        "item_frequencies": item_frequencies,
        "train_user_seq": train_user_seq,
        "val_user_seq": val_user_seq,
        "test_user_seq": test_user_seq,
        "cold_start_users": cold_start_users,
        "user_sequences": user_sequences
    }

    with open(KUAIREC_PROCESSED_PATH, "wb") as f:
        pickle.dump(processed_data, f)

    if cfg is not None and cfg.processed_path != KUAIREC_PROCESSED_PATH:
        with open(cfg.processed_path, "wb") as f:
            pickle.dump(processed_data, f)

    print(f"[KuaiRec] Successfully saved processed data to {KUAIREC_PROCESSED_PATH}")
    print(f"[KuaiRec] Summary: {num_users:,} users, {num_items:,} items, {total_interactions:,} interactions.")
    print(f"[KuaiRec] Cold-start cohort: {len(cold_start_users)} users (truncated to <= {cold_start_threshold} interactions).")
    return processed_data


if __name__ == "__main__":
    preprocess_kuairec()
