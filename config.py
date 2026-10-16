"""
Configuration parameters for RecSys LLM Hybrid model training and evaluation.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- Paths ---
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data", "ml-1m")
    RAW_RATINGS_PATH: str = os.path.join(DATA_DIR, "ratings.dat")
    RAW_MOVIES_PATH: str = os.path.join(DATA_DIR, "movies.dat")
    PROCESSED_DATA_PATH: str = os.path.join(DATA_DIR, "processed_data.pt")
    LLM_EMBEDDINGS_PATH: str = os.path.join(DATA_DIR, "llm_embeddings.pt")
    CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")

    # --- Dataset & Sequence Parameters ---
    max_seq_len: int = 50
    cold_start_threshold: int = 5  # Users with <= 5 total interactions
    test_neg_samples: int = 100    # Number of negative items per positive for eval ranking

    # --- Baseline 1 (SASRec) Hyperparameters ---
    item_embed_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.2

    # --- Baseline 2 (Pure LLM) Parameters ---
    llm_baseline_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- Proposed Approach (Hybrid Model) Parameters ---
    llm_embed_dim: int = 4096        # Dim from 7B LLM (e.g., Qwen/Mistral) or 384/768 for sentence models
    projected_llm_dim: int = 64
    fusion_type: str = "concat"       # Options: "concat", "add", "gate"
    
    # --- Diversity Parameters ---
    lambda_mmr: float = 0.7           # Weight for accuracy vs diversity in MMR (1.0 = pure accuracy, 0.0 = pure diversity)
    top_k: int = 10

    # --- Training Parameters ---
    lr_sasrec: float = 1e-3
    lr_fusion: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 50
    patience: int = 5
    seed: int = 42
    device: str = "cuda"              # Will fallback to cpu automatically if cuda unavailable

    def __post_init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)


config = Config()
