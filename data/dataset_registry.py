"""
Dataset Registry — Unified interface for all supported datasets.

Supported datasets:
  - "ml1m"           : MovieLens-1M  (6,040 users, 3,706 items)
  - "amazon_beauty"  : Amazon Beauty (22,363 users, 12,101 items)
  - "amazon_sports"  : Amazon Sports & Outdoors (35,598 users, 18,357 items)
  - "steam"          : Steam Video Games (334,730 users, 13,047 items, subset used)

Usage:
    from data.dataset_registry import get_dataset, DATASET_CONFIGS
    processed = get_dataset("amazon_beauty")
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DatasetConfig:
    name: str
    display_name: str
    max_seq_len: int
    min_interactions: int       # Minimum interactions to keep a user
    min_item_freq: int          # Minimum times an item must appear globally
    cold_start_threshold: int   # Max interactions for cold-start simulation
    item_text_field: str        # Key in item_metadata that holds descriptive text
    base_dir: str = ""          # Filled at runtime

    def __post_init__(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.name == "ml1m" and os.path.exists(os.path.join(base, "data", "ml-1m")):
            self.base_dir = os.path.join(base, "data", "ml-1m")
        else:
            self.base_dir = os.path.join(base, "data", self.name)
        os.makedirs(self.base_dir, exist_ok=True)

    @property
    def processed_path(self) -> str:
        return os.path.join(self.base_dir, "processed_data.pt")

    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.base_dir, "llm_embeddings.pt")


# ── Registry of all supported datasets ────────────────────────────────────────

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "ml1m": DatasetConfig(
        name="ml1m",
        display_name="MovieLens-1M",
        max_seq_len=50,
        min_interactions=5,
        min_item_freq=5,
        cold_start_threshold=5,
        item_text_field="text",
    ),
    "amazon_beauty": DatasetConfig(
        name="amazon_beauty",
        display_name="Amazon Beauty",
        max_seq_len=50,
        min_interactions=5,
        min_item_freq=5,
        cold_start_threshold=5,
        item_text_field="text",
    ),
    "amazon_sports": DatasetConfig(
        name="amazon_sports",
        display_name="Amazon Sports & Outdoors",
        max_seq_len=50,
        min_interactions=5,
        min_item_freq=5,
        cold_start_threshold=5,
        item_text_field="text",
    ),
    "amazon": DatasetConfig(
        name="amazon",
        display_name="Amazon Reviews",
        max_seq_len=50,
        min_interactions=5,
        min_item_freq=5,
        cold_start_threshold=5,
        item_text_field="text",
    ),
    "steam": DatasetConfig(
        name="steam",
        display_name="Steam Games",
        max_seq_len=50,
        min_interactions=5,
        min_item_freq=10,
        cold_start_threshold=5,
        item_text_field="text",
    ),
    "yelp": DatasetConfig(
        name="yelp",
        display_name="Yelp Academic",
        max_seq_len=50,
        min_interactions=5,
        min_item_freq=5,
        cold_start_threshold=5,
        item_text_field="text",
    ),
    "kuairec": DatasetConfig(
        name="kuairec",
        display_name="KuaiRec Micro-Video",
        max_seq_len=50,
        min_interactions=5,
        min_item_freq=5,
        cold_start_threshold=5,
        item_text_field="text",
    ),
}


def get_dataset(dataset_name: str, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Returns processed dataset dict for the given dataset name.
    Caches processed data to disk. Re-runs preprocessing only if needed.

    Args:
        dataset_name: One of 'ml1m', 'amazon_beauty', 'amazon_sports', 'amazon', 'steam', 'yelp', 'kuairec'
        force_reprocess: If True, ignores cache and re-runs preprocessing

    Returns:
        processed_data dict with keys:
            num_users, num_items, train_user_seq, val_user_seq,
            test_user_seq, cold_start_users, item_metadata,
            item_frequencies, user_sequences
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset_name]

    # Alternate legacy cache paths
    legacy_caches = {
        "yelp": os.path.join(cfg.base_dir, "processed_yelp.pt"),
        "kuairec": os.path.join(cfg.base_dir, "processed_kuairec.pt"),
        "amazon": os.path.join(cfg.base_dir, "processed_amazon.pt"),
    }
    alt_path = legacy_caches.get(dataset_name)

    if not force_reprocess:
        if os.path.exists(cfg.processed_path):
            print(f"[Registry] Loading cached '{cfg.display_name}' from {cfg.processed_path}")
            return _load_data(cfg.processed_path)
        elif alt_path and os.path.exists(alt_path):
            print(f"[Registry] Loading cached '{cfg.display_name}' from {alt_path}")
            return _load_data(alt_path)

    print(f"[Registry] Preprocessing '{cfg.display_name}'...")

    if dataset_name == "ml1m":
        from data.preprocess import preprocess_movielens
        return preprocess_movielens(data_dir=cfg.base_dir)

    elif dataset_name in ("amazon_beauty", "amazon_sports"):
        from data.amazon_preprocess import preprocess_amazon
        category = "All_Beauty" if dataset_name == "amazon_beauty" else "Sports_and_Outdoors"
        return preprocess_amazon(cfg=cfg, category=category)

    elif dataset_name == "amazon":
        from data.preprocess_amazon import preprocess_amazon as prep_amz
        return prep_amz(amazon_dir=cfg.base_dir, cold_start_threshold=cfg.cold_start_threshold, cfg=cfg)

    elif dataset_name == "steam":
        from data.steam_preprocess import preprocess_steam
        return preprocess_steam(cfg=cfg)

    elif dataset_name == "yelp":
        from data.preprocess_yelp import preprocess_yelp
        return preprocess_yelp(yelp_dir=cfg.base_dir, cold_start_threshold=cfg.cold_start_threshold, cfg=cfg)

    elif dataset_name == "kuairec":
        from data.preprocess_kuairec import preprocess_kuairec
        return preprocess_kuairec(cold_start_threshold=cfg.cold_start_threshold, cfg=cfg)

    else:
        raise NotImplementedError(f"Preprocessor for '{dataset_name}' not implemented yet.")


def list_datasets() -> None:
    """Prints all registered datasets with their configs."""
    print("\n" + "=" * 60)
    print(f"{'Dataset Registry':^60}")
    print("=" * 60)
    legacy_caches = {
        "yelp": "processed_yelp.pt",
        "kuairec": "processed_kuairec.pt",
        "amazon": "processed_amazon.pt",
    }
    for key, cfg in DATASET_CONFIGS.items():
        alt = os.path.join(cfg.base_dir, legacy_caches.get(key, ""))
        is_cached = os.path.exists(cfg.processed_path) or (bool(legacy_caches.get(key)) and os.path.exists(alt))
        cached = "[CACHED]" if is_cached else "[NOT CACHED]"
        print(f"  {key:<20} | {cfg.display_name:<30} | {cached}")
    print()


def _load_data(path: str) -> Dict[str, Any]:
    """Load processed data from disk, supporting both torch and pickle formats."""
    try:
        import torch
        data = torch.load(path, weights_only=False)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

    if isinstance(data, dict):
        num_items = data.get("num_items", 0)
        user_seqs = data.get("user_sequences", data.get("train_user_seq", {}))

        if "item_frequencies" not in data or data["item_frequencies"] is None:
            import numpy as np
            freqs = np.zeros(num_items + 1, dtype=np.int32)
            for uid, seq in user_seqs.items():
                items = seq[0] if isinstance(seq, tuple) else seq
                for mid in items:
                    if 0 < mid <= num_items:
                        freqs[mid] += 1
            data["item_frequencies"] = freqs

        if "cold_start_users" not in data or data["cold_start_users"] is None:
            threshold = data.get("cold_start_threshold", 5)
            cold_users = set()
            for uid, seq in user_seqs.items():
                items = seq[0] if isinstance(seq, tuple) else seq
                if len(items) <= threshold:
                    cold_users.add(uid)
            if not cold_users and user_seqs:
                all_uids = list(user_seqs.keys())
                cold_users = set(all_uids[:max(1, int(len(all_uids) * 0.20))])
            data["cold_start_users"] = cold_users

        if "item2id" not in data and "movie2id" in data:
            data["item2id"] = data["movie2id"]
        if "movie2id" not in data and "item2id" in data:
            data["movie2id"] = data["item2id"]
        if "id2item" not in data and "id2movie" in data:
            data["id2item"] = data["id2movie"]
        if "id2movie" not in data and "id2item" in data:
            data["id2movie"] = data["id2item"]

    return data


if __name__ == "__main__":
    list_datasets()
