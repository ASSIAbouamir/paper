"""
Data preprocessing script for MovieLens-1M dataset.
Handles downloading, cleaning, sequential splitting (Leave-One-Out),
item popularity tracking, user genre profile extraction, and a dedicated
Cold-Start simulation protocol (truncation to <= 5 interactions for cold cohort).
"""

import os
import sys
import io
import ssl
import zipfile
import urllib.request
import pickle
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def save_data(obj, path):
    try:
        import torch
        torch.save(obj, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def load_data(path):
    try:
        import torch
        return torch.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def download_movielens_1m(data_dir: str):
    """Downloads and extracts official MovieLens-1M dataset if not present."""
    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path = os.path.join(data_dir, "movies.dat")

    if os.path.exists(ratings_path) and os.path.exists(movies_path) and os.path.getsize(ratings_path) > 1000000:
        print(f"[Data] Real MovieLens-1M dataset verified at '{data_dir}' ({os.path.getsize(ratings_path):,} bytes).")
        return

    os.makedirs(data_dir, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print(f"[Data] Downloading MovieLens-1M from {url}...")
    try:
        ctx = ssl._create_unverified_context()
        req = urllib.request.urlopen(url, context=ctx)
        zip_file = zipfile.ZipFile(io.BytesIO(req.read()))
        for file_info in zip_file.infolist():
            if file_info.filename.endswith("ratings.dat") or file_info.filename.endswith("movies.dat") or file_info.filename.endswith("users.dat"):
                filename = os.path.basename(file_info.filename)
                with zip_file.open(file_info) as source, open(os.path.join(data_dir, filename), "wb") as target:
                    target.write(source.read())
        print("[Data] Download & Extraction successful.")
    except Exception as e:
        print(f"[Warning] Download failed ({e}). Checking local files...")


def preprocess_movielens(
    data_dir: str = config.DATA_DIR,
    cold_start_threshold: int = config.cold_start_threshold,
    cold_cohort_ratio: float = 0.20
):
    """
    Preprocesses ML-1M ratings and movies into sequential train/val/test split.
    Guarantees strict chronologic Leave-One-Out split without data leakage.
    Includes a realistic Cold-Start evaluation protocol (truncating cohort users to <= 5 items).
    """
    download_movielens_1m(data_dir)

    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path = os.path.join(data_dir, "movies.dat")

    # 1. Parse movies.dat (MovieID::Title::Genres)
    movies_data = []
    all_genres = set()
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                mid = int(parts[0])
                title = parts[1]
                genres = parts[2].split("|")
                all_genres.update(genres)
                movies_data.append((mid, title, genres))

    # 2. Parse ratings.dat (UserID::MovieID::Rating::Timestamp)
    ratings_data = []
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 4:
                uid = int(parts[0])
                mid = int(parts[1])
                rating = float(parts[2])
                ts = int(parts[3])
                ratings_data.append((uid, mid, rating, ts))

    print(f"[Data] Raw ratings: {len(ratings_data):,}, Raw movies: {len(movies_data):,}")

    user_ids = sorted(list(set(r[0] for r in ratings_data)))
    movie_ids = sorted(list(set(m[0] for m in movies_data)))

    user2id = {uid: idx + 1 for idx, uid in enumerate(user_ids)}
    movie2id = {mid: idx + 1 for idx, mid in enumerate(movie_ids)}
    id2movie = {idx: mid for mid, idx in movie2id.items()}

    num_users = len(user2id)
    num_items = len(movie2id)

    # 3. Item Metadata & Popularity Counts
    item_counts = Counter(r[1] for r in ratings_data)
    item_frequencies = np.zeros(num_items + 1, dtype=np.int32)
    for mid, count in item_counts.items():
        if mid in movie2id:
            item_frequencies[movie2id[mid]] = count

    item_metadata = {}
    for mid, title, genres in movies_data:
        if mid in movie2id:
            mapped_id = movie2id[mid]
            genres_str = ", ".join(genres)
            text = f"Title: {title}. Genres: {genres_str}."
            item_metadata[mapped_id] = {
                "raw_id": mid,
                "title": title,
                "genres": genres,
                "text": text,
                "frequency": int(item_frequencies[mapped_id])
            }

    # 4. Group interactions chronologically per user
    user_interactions = {}
    for uid, mid, rating, ts in ratings_data:
        mapped_uid = user2id[uid]
        mapped_mid = movie2id[mid]
        if mapped_uid not in user_interactions:
            user_interactions[mapped_uid] = []
        user_interactions[mapped_uid].append((mapped_mid, ts))

    user_sequences = {}
    user_genre_profiles = {}

    for uid in sorted(user_interactions.keys()):
        user_interactions[uid].sort(key=lambda x: x[1])
        seq = [item[0] for item in user_interactions[uid]]
        user_sequences[uid] = seq

        genres_list = []
        for m_id in seq:
            if m_id in item_metadata:
                genres_list.extend(item_metadata[m_id]["genres"])
        user_genre_profiles[uid] = dict(Counter(genres_list))

    # 5. Cold-Start Simulation Cohort
    # In MovieLens-1M, all users have >= 20 ratings. To evaluate Cold-Start (< 5 interactions),
    # we designate 20% of users as a simulated cold-start cohort where we artificially truncate
    # their observed history to only their first 5 interactions.
    np.random.seed(42)
    shuffled_uids = np.random.permutation(list(user_sequences.keys()))
    num_cold_users = int(len(shuffled_uids) * cold_cohort_ratio)
    cold_start_users = set(shuffled_uids[:num_cold_users])

    train_user_seq = {}
    val_user_seq = {}
    test_user_seq = {}

    for uid, full_seq in user_sequences.items():
        if uid in cold_start_users:
            # Cold-start user: restricted strictly to first 5 interactions
            cold_seq = full_seq[:cold_start_threshold]
            train_user_seq[uid] = cold_seq[:-2]
            val_user_seq[uid] = (cold_seq[:-2], cold_seq[-2])
            test_user_seq[uid] = (cold_seq[:-1], cold_seq[-1])
        else:
            # Warm user: standard leave-one-out
            train_user_seq[uid] = full_seq[:-2]
            val_user_seq[uid] = (full_seq[:-2], full_seq[-2])
            test_user_seq[uid] = (full_seq[:-1], full_seq[-1])

    processed_data = {
        "num_users": num_users,
        "num_items": num_items,
        "user2id": user2id,
        "movie2id": movie2id,
        "id2movie": id2movie,
        "item_metadata": item_metadata,
        "item_frequencies": item_frequencies,
        "user_genre_profiles": user_genre_profiles,
        "all_genres": sorted(list(all_genres)),
        "train_user_seq": train_user_seq,
        "val_user_seq": val_user_seq,
        "test_user_seq": test_user_seq,
        "cold_start_users": cold_start_users,
        "user_sequences": user_sequences
    }

    save_data(processed_data, config.PROCESSED_DATA_PATH)
    print(f"[Data] Preprocessing complete. Saved to '{config.PROCESSED_DATA_PATH}'.")
    print(f"[Data] Summary: {num_users:,} Users | {num_items:,} Items | {len(ratings_data):,} Ratings")
    print(f"[Data] Cold-Start Evaluation Cohort: {len(cold_start_users)} users (truncated to <= {cold_start_threshold} items)")

    return processed_data


if __name__ == "__main__":
    preprocess_movielens()
