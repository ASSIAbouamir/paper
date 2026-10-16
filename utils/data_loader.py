"""
PyTorch Datasets and DataLoaders for SASRec and Hybrid model training/evaluation.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple


class SASRecTrainDataset(Dataset):
    """Dataset for training SASRec and Hybrid models with next-item prediction."""

    def __init__(self, train_user_seq: Dict[int, List[int]], num_items: int, max_seq_len: int = 50):
        self.user_ids = list(train_user_seq.keys())
        self.user_seq = train_user_seq
        self.num_items = num_items
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        seq = self.user_seq[uid]

        # Input and target creation
        # Input sequence: seq[:-1], Target sequence: seq[1:]
        if len(seq) > 1:
            input_seq = seq[:-1]
            pos_seq = seq[1:]
        else:
            input_seq = seq
            pos_seq = seq

        # Truncate if longer than max_seq_len
        input_seq = input_seq[-self.max_seq_len:]
        pos_seq = pos_seq[-self.max_seq_len:]

        # Pad with 0s on the left
        pad_len = self.max_seq_len - len(input_seq)
        padded_input = [0] * pad_len + input_seq
        padded_pos = [0] * pad_len + pos_seq

        # Negative sampling per item in the sequence
        user_seen = set(seq)
        neg_seq = []
        for _ in pos_seq:
            neg_item = random.randint(1, self.num_items)
            while neg_item in user_seen:
                neg_item = random.randint(1, self.num_items)
            neg_seq.append(neg_item)
        padded_neg = [0] * pad_len + neg_seq

        return (
            torch.tensor(padded_input, dtype=torch.long),
            torch.tensor(padded_pos, dtype=torch.long),
            torch.tensor(padded_neg, dtype=torch.long),
            torch.tensor(uid, dtype=torch.long)
        )


class SASRecEvalDataset(Dataset):
    """Dataset for evaluating SASRec and Hybrid models (Leave-One-Out with negative ranking)."""

    def __init__(
        self,
        eval_user_seq: Dict[int, Tuple[List[int], int]],
        all_user_seq: Dict[int, List[int]],
        num_items: int,
        max_seq_len: int = 50,
        num_neg_samples: int = 100
    ):
        self.user_ids = list(eval_user_seq.keys())
        self.eval_user_seq = eval_user_seq
        self.all_user_seq = all_user_seq
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.num_neg_samples = num_neg_samples

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        history_seq, target_item = self.eval_user_seq[uid]

        # Truncate and pad history sequence
        history_seq = history_seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(history_seq)
        padded_history = [0] * pad_len + history_seq

        # Sample 100 negative candidates
        seen_items = set(self.all_user_seq.get(uid, []))
        neg_candidates = []
        while len(neg_candidates) < self.num_neg_samples:
            neg = random.randint(1, self.num_items)
            if neg not in seen_items and neg not in neg_candidates:
                neg_candidates.append(neg)

        # Candidates tensor: [target_item, neg1, neg2, ...] (target is index 0)
        candidates = [target_item] + neg_candidates

        return (
            torch.tensor(padded_history, dtype=torch.long),
            torch.tensor(candidates, dtype=torch.long),
            torch.tensor(uid, dtype=torch.long)
        )


def get_dataloaders(processed_data: dict, config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Helper function to create Train, Val, and Test DataLoaders."""
    train_dataset = SASRecTrainDataset(
        train_user_seq=processed_data["train_user_seq"],
        num_items=processed_data["num_items"],
        max_seq_len=config.max_seq_len
    )

    val_dataset = SASRecEvalDataset(
        eval_user_seq=processed_data["val_user_seq"],
        all_user_seq=processed_data["user_sequences"],
        num_items=processed_data["num_items"],
        max_seq_len=config.max_seq_len,
        num_neg_samples=config.test_neg_samples
    )

    test_dataset = SASRecEvalDataset(
        eval_user_seq=processed_data["test_user_seq"],
        all_user_seq=processed_data["user_sequences"],
        num_items=processed_data["num_items"],
        max_seq_len=config.max_seq_len,
        num_neg_samples=config.test_neg_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
