"""
Train and Evaluate BiX-GDR on Sequential Recommendation Benchmarks.
Usage:
    python train_bix_gdr.py --dataset ml1m --seeds 42 123 2024
    python train_bix_gdr.py --dataset amazon_beauty --seeds 42 123 2024
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data.dataset_registry import get_dataset, DATASET_CONFIGS
from models.bix_gdr import BiXGDR


class FastCausalDataset(Dataset):
    def __init__(self, user_dict, num_items, max_len=50):
        self.num_items = num_items
        self.max_len = max_len
        self.samples = []
        for uid, seq in user_dict.items():
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            pos_seq = seq[1:]
            if len(input_seq) > max_len:
                input_seq = input_seq[-max_len:]
                pos_seq = pos_seq[-max_len:]
            pad_len = max_len - len(input_seq)
            if pad_len > 0:
                in_padded = np.zeros(max_len, dtype=np.int64)
                pos_padded = np.zeros(max_len, dtype=np.int64)
                in_padded[pad_len:] = input_seq
                pos_padded[pad_len:] = pos_seq
            else:
                in_padded = np.array(input_seq, dtype=np.int64)
                pos_padded = np.array(pos_seq, dtype=np.int64)
            self.samples.append((in_padded, pos_padded))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        in_padded, pos_padded = self.samples[idx]
        neg_padded = np.random.randint(1, self.num_items + 1, size=self.max_len, dtype=np.int64)
        neg_padded[pos_padded == 0] = 0
        return (
            torch.from_numpy(in_padded),
            torch.from_numpy(pos_padded),
            torch.from_numpy(neg_padded),
        )


def make_seq_tensor(seq_dict, num_items, max_len=50):
    users = sorted(seq_dict.keys())
    seqs_list, targets = [], []
    for u in users:
        seq, tgt = seq_dict[u]
        seq = [min(x, num_items) for x in seq if x > 0]
        pad = max_len - len(seq)
        seqs_list.append([0] * pad + seq if pad > 0 else seq[-max_len:])
        targets.append(min(tgt, num_items))
    return torch.tensor(seqs_list, dtype=torch.long), torch.tensor(targets).long()


def eval_batched(model, seqs_tensor, targets_idx, batch_size=512):
    model.eval()
    all_ranks = []
    with torch.no_grad():
        for i in range(0, len(seqs_tensor), batch_size):
            b_seqs = seqs_tensor[i:i + batch_size]
            b_tgt = targets_idx[i:i + batch_size]
            u_c, u_s = model.forward_seq(b_seqs)
            scores = model.compute_all_scores(u_c, u_s)
            ts = scores[torch.arange(len(b_tgt)), b_tgt].unsqueeze(1)
            ranks = (scores > ts).sum(dim=1) + 1
            all_ranks.append(ranks)
    ranks = torch.cat(all_ranks)

    hits5 = (ranks <= 5).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    hits20 = (ranks <= 20).float().mean().item()
    ndcg10 = (1.0 / torch.log2(ranks.float() + 1.0) * (ranks <= 10).float()).mean().item()
    num_hits10 = (ranks <= 10).sum().item()

    return {
        "num_hits10": int(num_hits10),
        "hr5": float(hits5),
        "hr10": float(hits10),
        "hr20": float(hits20),
        "ndcg10": float(ndcg10),
    }


def train_bix_gdr(dataset_name="ml1m", seeds=[42, 123, 2024], epochs=8, lr=2e-3, batch_size=128):
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"\n" + "=" * 80)
    print(f" TRAINING BIX-GDR ON {cfg.display_name} ({dataset_name})")
    print("=" * 80)

    data = get_dataset(dataset_name)
    sem_path = os.path.join(cfg.base_dir, "semantic_table_minilm.pt")
    if not os.path.exists(sem_path):
        raise FileNotFoundError(f"Semantic table not found at {sem_path}")

    sem_table = torch.load(sem_path, weights_only=False).float()
    num_items = min(data["num_items"], sem_table.shape[0] - 1)
    if sem_table.shape[0] <= num_items:
        pad = torch.zeros(num_items + 1 - sem_table.shape[0], sem_table.shape[1])
        sem_table = torch.cat([sem_table, pad], dim=0)

    train_raw = {u: [min(x, num_items) for x in s if x > 0] for u, s in data["train_user_seq"].items()}
    val_dict = {u: ([min(x, num_items) for x in s if x > 0], min(t, num_items)) for u, (s, t) in data["val_user_seq"].items()}
    test_dict = {u: ([min(x, num_items) for x in s if x > 0], min(t, num_items)) for u, (s, t) in data["test_user_seq"].items()}

    val_seqs, val_tgt = make_seq_tensor(val_dict, num_items)
    test_seqs, test_tgt = make_seq_tensor(test_dict, num_items)
    val_tgt_idx = val_tgt - 1
    test_tgt_idx = test_tgt - 1

    print(f"Dataset stats: {len(train_raw)} users, {num_items} items")
    print(f"Test users: {len(test_seqs)}, Validation users: {len(val_seqs)}")

    results = {}
    for seed in seeds:
        print(f"\n--- Training {cfg.display_name} [Seed {seed}] ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            FastCausalDataset(train_raw, num_items, 50),
            batch_size=batch_size, shuffle=True, generator=g
        )

        model = BiXGDR(num_items=num_items, semantic_table=sem_table, hidden_dim=64, max_seq_len=50, num_layers=2, num_heads=2, dropout=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_ndcg = 0.0
        best_weights = None

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            total_loss, nb = 0.0, 0
            for seq_in, seq_pos, _ in train_loader:
                optimizer.zero_grad()
                u_c, u_s = model.forward_seq(seq_in)
                tgt = seq_pos[:, -1]
                valid = (tgt > 0)
                if valid.sum() == 0:
                    continue
                scores = model.compute_all_scores(u_c[valid], u_s[valid])
                loss = F.cross_entropy(scores, tgt[valid] - 1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                nb += 1

            scheduler.step()
            v_res = eval_batched(model, val_seqs, val_tgt_idx)
            if v_res["ndcg10"] > best_val_ndcg:
                best_val_ndcg = v_res["ndcg10"]
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {total_loss/max(nb, 1):.4f} - Val Hits@10: {v_res['num_hits10']} - Val NDCG@10: {v_res['ndcg10']:.4f} ({time.time()-t0:.1f}s)")

        model.load_state_dict(best_weights)
        t_res = eval_batched(model, test_seqs, test_tgt_idx)
        results[str(seed)] = t_res
        print(f"  [*TEST SEED {seed}] Hits@10: {t_res['num_hits10']} | HR@10: {t_res['hr10']:.4f} | NDCG@10: {t_res['ndcg10']:.4f}")

    mean_hits = np.mean([r["num_hits10"] for r in results.values()])
    mean_hr10 = np.mean([r["hr10"] for r in results.values()])
    mean_ndcg10 = np.mean([r["ndcg10"] for r in results.values()])
    print("\n" + "=" * 80)
    print(f"FINAL MULTI-SEED SUMMARY FOR {cfg.display_name}:")
    print(f"  Mean Hits@10: {mean_hits:.1f} / {len(test_seqs)}")
    print(f"  Mean HR@10:   {mean_hr10:.4f}")
    print(f"  Mean NDCG@10: {mean_ndcg10:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", choices=["ml1m", "amazon_beauty", "yelp", "kuairec"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train_bix_gdr(dataset_name=args.dataset, seeds=args.seeds, epochs=args.epochs)
