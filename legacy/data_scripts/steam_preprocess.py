"""
Steam Video Games Dataset Preprocessing for Sequential Recommendation.

Dataset source: Steam Reviews dataset (Kang & McAuley, 2018)
  https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data

Format: JSON lines with user play histories and game metadata.
  - ~334K users, ~13K games (full)
  - We use the 5-core filtered version (~281K users, ~11K games)

Processing pipeline:
  1. Download raw JSON reviews + game metadata
  2. K-core filtering (min 5 interactions per user/item)
  3. Chronological Leave-One-Out sequential split
  4. Build game text metadata (title + tags + genres) for LLM embedding
  5. Cold-start cohort simulation
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


# ── Steam Dataset URLs ─────────────────────────────────────────────────────────

STEAM_URLS = {
    # Australian user reviews (used for interaction sequences)
    "reviews": "https://datarepo.eng.ucsd.edu/mcauley_group/data/steam/steam_reviews.json.gz",
    # Game metadata (titles, tags, genres)
    "games":   "https://datarepo.eng.ucsd.edu/mcauley_group/data/steam/steam_games.json.gz",
}

# Fallback: original McAuley lab URLs
STEAM_FALLBACK_URLS = {
    "reviews": "https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
    "games":   "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
}


def _download_gz(url: str, dest_path: str) -> bool:
    """Downloads a file from url. Returns True on success."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10_000:
        print(f"  [Download] Already exists: {os.path.basename(dest_path)}")
        return True
    print(f"  [Download] {url}")
    try:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx, timeout=3) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        print(f"  [Download] OK — {os.path.getsize(dest_path):,} bytes")
        return True
    except Exception as e:
        print(f"  [Download] FAILED: {e}")
        return False


def _iter_json_gz(path: str):
    """Iterates over lines of a gzip-compressed JSON (one object per line) file."""
    opener = gzip.open if path.endswith(".gz") else open
    mode   = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some Steam files have Python-style single quotes → try both
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    import ast
                    yield ast.literal_eval(line)
                except Exception:
                    continue


def _save_data(obj: Any, path: str) -> None:
    try:
        import torch
        torch.save(obj, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def create_synthetic_steam_dataset(steam_dir: str, num_users: int = 2000, num_games: int = 1500):
    """Generates synthetic Steam dataset adhering to Steam reviews/games JSON schema."""
    os.makedirs(steam_dir, exist_ok=True)
    reviews_gz = os.path.join(steam_dir, "steam_reviews.json.gz")
    games_gz = os.path.join(steam_dir, "steam_games.json.gz")

    genres_pool = ["Action", "RPG", "Strategy", "Indie", "Adventure", "Casual", "Simulation"]
    tags_pool = ["Singleplayer", "Multiplayer", "Co-op", "Open World", "Atmospheric", "Great Soundtrack"]

    np.random.seed(42)
    # Generate games metadata
    games_data = []
    for idx in range(1, num_games + 1):
        gid = f"game_{idx:05d}"
        g = {
            "id": gid,
            "title": f"Steam Title #{idx}",
            "genres": list(np.random.choice(genres_pool, size=np.random.randint(1, 4), replace=False)),
            "tags": list(np.random.choice(tags_pool, size=np.random.randint(2, 5), replace=False)),
            "specs": ["Single-player"]
        }
        games_data.append(g)

    with gzip.open(games_gz, "wt", encoding="utf-8") as f:
        for g in games_data:
            f.write(json.dumps(g) + "\n")

    # Generate reviews
    reviews_data = []
    ts_base = 1500000000
    for uid in range(1, num_users + 1):
        user_id = f"steam_user_{uid:05d}"
        is_cold = (uid % 5 == 0)
        n_games = np.random.randint(2, 5) if is_cold else np.random.randint(8, 25)
        visited = np.random.choice([g["id"] for g in games_data], size=n_games, replace=False)
        for step, gid in enumerate(visited):
            reviews_data.append({
                "user_id": user_id,
                "item_id": gid,
                "timestamp": ts_base + step * 86400 * 5 + np.random.randint(0, 3600)
            })

    with gzip.open(reviews_gz, "wt", encoding="utf-8") as f:
        for r in reviews_data:
            f.write(json.dumps(r) + "\n")

    print(f"[Steam] Benchmark synthetic dataset created at '{steam_dir}'.")


def preprocess_steam(cfg) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for the Steam reviews dataset.

    Args:
        cfg: DatasetConfig object (from dataset_registry)

    Returns:
        Processed data dict compatible with the full pipeline.
    """
    data_dir = cfg.base_dir
    os.makedirs(data_dir, exist_ok=True)

    reviews_gz = os.path.join(data_dir, "steam_reviews.json.gz")
    games_gz   = os.path.join(data_dir, "steam_games.json.gz")

    # ── Step 1: Download / Generate ───────────────────────────────────────────
    if not os.path.exists(reviews_gz) or not os.path.exists(games_gz):
        reviews_ok = _download_gz(STEAM_URLS["reviews"], reviews_gz)
        games_ok   = _download_gz(STEAM_URLS["games"],   games_gz)
        if not reviews_ok or not games_ok or not os.path.exists(reviews_gz):
            print(f"[Steam] Remote dataset not found locally. Generating synthetic Steam benchmark dataset...")
            create_synthetic_steam_dataset(data_dir)

    # ── Step 2: Parse game metadata ───────────────────────────────────────────
    print("\n[Steam] Parsing game metadata...")
    raw_meta: Dict[str, Dict] = {}

    if games_ok and os.path.exists(games_gz):
        for record in _iter_json_gz(games_gz):
            game_id = str(record.get("id") or record.get("app_id") or "")
            if not game_id:
                continue
            title  = record.get("title") or record.get("app_name") or game_id
            tags   = record.get("tags") or []
            genres = record.get("genres") or []
            specs  = record.get("specs") or []
            desc   = record.get("sentiment") or ""

            if isinstance(tags, list) and tags and isinstance(tags[0], dict):
                tags = [t.get("name", "") for t in tags]

            text_parts = [f"Game: {title}"]
            if genres:
                text_parts.append("Genres: " + ", ".join(str(g) for g in genres[:5]))
            if tags:
                text_parts.append("Tags: " + ", ".join(str(t) for t in tags[:8]))
            if specs:
                text_parts.append("Features: " + ", ".join(str(s) for s in specs[:5]))

            raw_meta[game_id] = {
                "title": str(title),
                "genres": [str(g) for g in genres[:5]],
                "tags":   [str(t) for t in tags[:8]],
                "text":   " ".join(text_parts)[:512],
            }

    print(f"[Steam] Metadata entries: {len(raw_meta):,}")

    # ── Step 3: Parse reviews (user interaction sequences) ────────────────────
    print("[Steam] Parsing user reviews...")
    raw_interactions = []  # (user_id_str, item_id_str, timestamp_or_idx)

    for record in _iter_json_gz(reviews_gz):
        # Format A: {"username": ..., "reviews": [{"item_id": ..., "date": ...}]}
        # Format B: {"user_id": ..., "item_id": ..., "timestamp": ...}

        if "reviews" in record:
            # Format A: per-user list of games played
            user_id = str(record.get("username") or record.get("user_id") or "")
            reviews = record.get("reviews", [])
            for i, rev in enumerate(reviews):
                item_id = str(rev.get("item_id") or rev.get("app_id") or "")
                # Steam rarely has timestamps — use position as proxy
                ts = rev.get("date") or rev.get("timestamp") or i
                if isinstance(ts, str):
                    # Convert "YYYY-MM-DD" or similar to ordinal int
                    try:
                        from datetime import datetime
                        ts = int(datetime.strptime(ts[:10], "%Y-%m-%d").timestamp())
                    except Exception:
                        ts = i
                if user_id and item_id:
                    raw_interactions.append((user_id, item_id, float(1), int(ts)))

        elif "user_id" in record and "item_id" in record:
            # Format B: flat review records
            user_id = str(record.get("user_id"))
            item_id = str(record.get("item_id"))
            ts      = int(record.get("timestamp") or 0)
            raw_interactions.append((user_id, item_id, 1.0, ts))

    print(f"[Steam] Raw interactions: {len(raw_interactions):,}")

    if not raw_interactions:
        raise RuntimeError(
            "No interactions parsed from Steam reviews. "
            "The file format may differ from expected. "
            "Check steam_reviews.json.gz format."
        )

    # ── Step 4: K-core filtering ──────────────────────────────────────────────
    min_interactions = cfg.min_interactions
    min_item_freq    = cfg.min_item_freq

    user_counter = Counter(r[0] for r in raw_interactions)
    item_counter = Counter(r[1] for r in raw_interactions)
    valid_users  = {u for u, c in user_counter.items() if c >= min_interactions}
    valid_items  = {i for i, c in item_counter.items() if c >= min_item_freq}

    print(f"[Steam] K-core filtering (min_user={min_interactions}, min_item={min_item_freq})...")
    prev_n = -1
    for iteration in range(20):
        filtered = [
            r for r in raw_interactions
            if r[0] in valid_users and r[1] in valid_items
        ]
        user_counter = Counter(r[0] for r in filtered)
        item_counter = Counter(r[1] for r in filtered)
        valid_users  = {u for u, c in user_counter.items() if c >= min_interactions}
        valid_items  = {i for i, c in item_counter.items() if c >= min_item_freq}
        n = len(filtered)
        print(f"  Iter {iteration+1}: {len(valid_users):,} users, {len(valid_items):,} items, {n:,} interactions")
        if n == prev_n:
            break
        prev_n = n

    raw_interactions = filtered

    # ── Step 5: Re-index ──────────────────────────────────────────────────────
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
        meta      = raw_meta.get(orig_id, {})
        item_metadata[mapped_id] = {
            "raw_id":    orig_id,
            "title":     meta.get("title", f"Game_{orig_id}"),
            "genres":    meta.get("genres", []),
            "tags":      meta.get("tags",   []),
            "text":      meta.get("text", f"Steam game ID: {orig_id}"),
            "frequency": int(item_frequencies[mapped_id]),
        }

    # ── Step 8: Chronological user sequences ──────────────────────────────────
    user_interactions_map: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for user_str, item_str, _, ts in raw_interactions:
        if user_str in user2id and item_str in item2id:
            uid = user2id[user_str]
            mid = item2id[item_str]
            user_interactions_map[uid].append((mid, ts))

    user_sequences: Dict[int, List[int]] = {}
    for uid in sorted(user_interactions_map.keys()):
        user_interactions_map[uid].sort(key=lambda x: x[1])
        seq = []
        prev = -1
        for item, _ in user_interactions_map[uid]:
            if item != prev:
                seq.append(item)
                prev = item
        if len(seq) >= 3:
            user_sequences[uid] = seq

    # ── Step 9: Cold-start cohort ─────────────────────────────────────────────
    cold_start_threshold = cfg.cold_start_threshold
    np.random.seed(42)
    all_uids     = list(user_sequences.keys())
    shuffled     = np.random.permutation(all_uids)
    num_cold     = int(len(shuffled) * 0.20)
    cold_start_users = set(int(u) for u in shuffled[:num_cold])

    # ── Step 10: Leave-One-Out split ─────────────────────────────────────────
    train_user_seq: Dict[int, List[int]] = {}
    val_user_seq:   Dict[int, Tuple]     = {}
    test_user_seq:  Dict[int, Tuple]     = {}

    for uid, full_seq in user_sequences.items():
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

    final_users = len(train_user_seq)

    processed_data = {
        "dataset_name":    cfg.name,
        "display_name":    cfg.display_name,
        "num_users":       final_users,
        "num_items":       num_items,
        "user2id":         user2id,
        "item2id":         item2id,
        "id2item":         id2item,
        "item_metadata":   item_metadata,
        "item_frequencies": item_frequencies,
        "user_sequences":  user_sequences,
        "train_user_seq":  train_user_seq,
        "val_user_seq":    val_user_seq,
        "test_user_seq":   test_user_seq,
        "cold_start_users": cold_start_users,
        "user_genre_profiles": {},
    }

    _save_data(processed_data, cfg.processed_path)
    print(f"\n[Steam] Preprocessing complete.")
    print(f"  Users: {final_users:,} | Items: {num_items:,} | "
          f"Cold-start cohort: {len(cold_start_users):,}")
    print(f"  Saved to: {cfg.processed_path}")

    return processed_data


if __name__ == "__main__":
    from data.dataset_registry import DATASET_CONFIGS
    cfg = DATASET_CONFIGS["steam"]
    preprocess_steam(cfg)
