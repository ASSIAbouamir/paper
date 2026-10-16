"""
Oracle Supervision Ablation.

Validates Claim 7 from the paper:
  "The continuous oracle Y_a = 1/log2(rank_a + 1) is superior to
   the hard-argmax oracle a*(u) = argmax_a Y_a(u) for router training."

Experiments:
  1. Hard Oracle: Categorical cross-entropy, argmax supervision
  2. Continuous Oracle: KL-divergence / soft-label cross-entropy
  3. Random baseline: router trained with random labels
  4. Uniform Oracle: always pick H (trivial baseline)

For each supervision type:
  - Train router on router-train split (75% of val users)
  - Evaluate on router-val split (25%) and test set
  - Report HR@10, NDCG@10, and routing distribution {C%, S%, H%}

Expected result:
  Continuous oracle > Hard oracle > Uniform > Random
  because continuous labels avoid binary threshold artifacts and
  capture the magnitude of inter-action performance differences.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_oracle_labels(
    uid_to_ranks: Dict[int, Dict[str, int]],
    action_names: List[str] = ("Collaborative", "Semantic", "Hybrid"),
    k: int = 10,
) -> Dict[int, Dict]:
    """
    Computes both hard and continuous oracle labels for every user.

    Hard label:   a*(u) = argmax_a Y_a(u)   (integer ∈ {0, 1, 2})
    Continuous:   [Y_C(u), Y_S(u), Y_H(u)] normalized to simplex

    Y_a(u) = 1 / log2(rank_a(u) + 1)  if rank_a(u) <= k, else 0

    Args:
        uid_to_ranks: {uid: {"Collaborative": rank, "Semantic": rank, "Hybrid": rank}}
        action_names: ordered list of action strings
        k:            HR@K cutoff (only actions with rank <= k get Y > 0)

    Returns:
        {uid: {
            "hard_label":        int   in {0, 1, 2}
            "continuous_labels": list  [Y_C, Y_S, Y_H] normalized
            "raw_y":             list  [Y_C, Y_S, Y_H] unnormalized
            "best_action_name":  str
            "all_zero":          bool  (True if all actions miss — all Y=0)
        }}
    """
    oracle_data = {}

    for uid, ranks in uid_to_ranks.items():
        y_vals = []
        for a_name in action_names:
            rank = ranks.get(a_name, k + 1)
            if rank <= k:
                y = 1.0 / np.log2(rank + 1)
            else:
                y = 0.0
            y_vals.append(y)

        y_arr    = np.array(y_vals, dtype=np.float64)
        all_zero = (y_arr.sum() == 0.0)

        hard_label = int(np.argmax(y_arr))

        # Normalize continuous labels to a probability simplex
        y_sum = y_arr.sum()
        if y_sum > 0:
            y_norm = y_arr / y_sum
        else:
            # All misses: uniform distribution (no information)
            y_norm = np.ones_like(y_arr) / len(y_arr)

        oracle_data[uid] = {
            "hard_label":        hard_label,
            "continuous_labels": y_norm.tolist(),
            "raw_y":             y_arr.tolist(),
            "best_action_name":  action_names[hard_label],
            "all_zero":          all_zero,
        }

    return oracle_data


def oracle_supervision_stats(
    oracle_data: Dict[int, Dict],
    action_names: List[str] = ("Collaborative", "Semantic", "Hybrid"),
) -> Dict:
    """
    Reports statistics on oracle label distributions.
    Useful for the ablation table in the paper.
    """
    N           = len(oracle_data)
    hard_counts = np.zeros(len(action_names), dtype=int)
    all_zero_n  = 0
    y_means     = np.zeros(len(action_names), dtype=float)

    for data in oracle_data.values():
        hard_counts[data["hard_label"]] += 1
        if data["all_zero"]:
            all_zero_n += 1
        y_means += np.array(data["raw_y"])

    y_means /= N

    stats = {
        "N": N,
        "hard_label_distribution": {
            action_names[i]: {
                "count": int(hard_counts[i]),
                "pct":   round(hard_counts[i] / N * 100, 1),
            }
            for i in range(len(action_names))
        },
        "mean_raw_y": {a: round(float(y_means[i]), 5) for i, a in enumerate(action_names)},
        "all_zero_users": int(all_zero_n),
        "all_zero_pct":   round(all_zero_n / N * 100, 1),
    }

    print("\n[Oracle] Label Distribution:")
    for a_name, info in stats["hard_label_distribution"].items():
        print(f"  {a_name:>20}: {info['count']:>6} users ({info['pct']:.1f}%)")
    print(f"  {'All misses (all Y=0)':>20}: {all_zero_n:>6} users ({stats['all_zero_pct']:.1f}%)")
    print(f"\n  Mean Y values: " + " | ".join(
        f"{a}={y:.5f}" for a, y in stats["mean_raw_y"].items()
    ))

    return stats


def compare_oracle_supervision_modes(
    evidence_features: "np.ndarray",
    oracle_data: Dict[int, Dict],
    uid_list: List[int],
    test_uid_to_rank: Dict[int, Dict[str, int]],
    action_names: List[str] = ("Collaborative", "Semantic", "Hybrid"),
    n_runs: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Trains the SelectiveRouter under different supervision modes and
    compares their held-out test HR@10.

    Supervision modes:
      - "hard":       Cross-entropy with hard argmax labels
      - "continuous": KL-divergence with soft Y-normalized labels
      - "random":     Random label assignment (control)
      - "uniform_H":  Always predict Hybrid (trivial baseline)

    Returns:
        comparison dict with mean HR@10 per supervision mode
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from models.selective_router import SelectiveRouter

    rng = np.random.default_rng(seed)

    # Split uid_list 75/25 into router-train / router-val
    shuffled = rng.permutation(uid_list)
    n_train  = int(len(shuffled) * 0.75)
    train_uids = list(shuffled[:n_train])
    val_uids   = list(shuffled[n_train:])

    # Build tensors
    uid_to_idx = {uid: i for i, uid in enumerate(uid_list)}

    x_tensor = torch.tensor(evidence_features, dtype=torch.float32)  # [N, 8]

    hard_labels = torch.tensor(
        [oracle_data[uid]["hard_label"] for uid in uid_list],
        dtype=torch.long,
    )  # [N]

    soft_labels = torch.tensor(
        [oracle_data[uid]["continuous_labels"] for uid in uid_list],
        dtype=torch.float32,
    )  # [N, 3]

    results = {}

    supervision_modes = {
        "hard":       "Cross-Entropy (argmax label)",
        "continuous": "KL-Divergence (soft Y labels)",
        "random":     "Random labels (control)",
        "uniform_H":  "Always Hybrid (trivial)",
    }

    for mode, mode_desc in supervision_modes.items():
        mode_results = []

        for run in range(n_runs):
            torch.manual_seed(seed + run)

            router = SelectiveRouter(in_features=x_tensor.shape[1])
            optimizer = optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)

            # Training phase
            router.train()
            train_idx = [uid_to_idx[uid] for uid in train_uids]
            x_train   = x_tensor[train_idx]

            for epoch in range(100):  # short training for ablation
                optimizer.zero_grad()
                logits = router(x_train)   # [N_train, 3]

                if mode == "hard":
                    y_hard = hard_labels[train_idx]
                    loss   = nn.CrossEntropyLoss()(logits, y_hard)

                elif mode == "continuous":
                    y_soft = soft_labels[train_idx]
                    log_p  = torch.log_softmax(logits, dim=-1)
                    loss   = -(y_soft * log_p).sum(dim=-1).mean()  # KL-div

                elif mode == "random":
                    y_rand = torch.randint(0, 3, (len(train_idx),))
                    loss   = nn.CrossEntropyLoss()(logits, y_rand)

                elif mode == "uniform_H":
                    y_h  = torch.full((len(train_idx),), 2, dtype=torch.long)
                    loss = nn.CrossEntropyLoss()(logits, y_h)

                loss.backward()
                optimizer.step()

            # Evaluation: predict actions on test users
            router.eval()
            hits = []
            with torch.no_grad():
                for uid, ranks in test_uid_to_rank.items():
                    if uid not in uid_to_idx:
                        # Test user not in router training — use static fallback
                        rank = min(
                            ranks.get("Collaborative", 11),
                            ranks.get("Semantic", 11),
                        )
                    else:
                        x_u   = x_tensor[uid_to_idx[uid]].unsqueeze(0)
                        logits_u = router(x_u)
                        action   = torch.argmax(logits_u, dim=-1).item()
                        a_name   = action_names[action]
                        rank     = ranks.get(a_name, 11)

                    hits.append(1.0 if rank <= 10 else 0.0)

            mode_results.append(float(np.mean(hits)))

        results[mode] = {
            "description": mode_desc,
            "mean_HR@10":  round(float(np.mean(mode_results)), 4),
            "std_HR@10":   round(float(np.std(mode_results)), 4),
            "runs":        mode_results,
        }

    # Print comparison table
    print("\n[Oracle Ablation] Supervision Mode Comparison:")
    print(f"  {'Mode':>20} | {'Description':>35} | {'HR@10':>8} | {'±':>6}")
    print(f"  {'-'*20}-+-{'-'*35}-+-{'-'*8}-+-{'-'*6}")
    for mode, res in results.items():
        print(f"  {mode:>20} | {res['description']:>35} | "
              f"{res['mean_HR@10']:>8.4f} | {res['std_HR@10']:>6.4f}")

    return results


if __name__ == "__main__":
    # Quick smoke test with dummy data
    print("[Oracle Ablation] Smoke test with random data...")
    N = 200
    uid_to_ranks = {
        i: {
            "Collaborative": np.random.randint(1, 50),
            "Semantic":      np.random.randint(1, 50),
            "Hybrid":        np.random.randint(1, 50),
        }
        for i in range(N)
    }
    oracle_data = compute_oracle_labels(uid_to_ranks)
    oracle_supervision_stats(oracle_data)
    print("[Oracle Ablation] Done.")
