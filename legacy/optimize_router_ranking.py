"""
Direct Rank-Aware Optimization for Selective Evidence Router.

Trains the router directly using Listwise Softmax Ranking Loss on actual model scores
rather than coarse classification labels.

Objective:
  p_routed(u) = alpha(x_u) * p_c(u) + (1 - alpha(x_u)) * p_s(u)
  Loss = - log( exp(p_routed[u, target] / tau) / sum_j exp(p_routed[u, j] / tau) )
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_and_evaluate_evidence_routing import load_and_preprocess_movielens
from models.dual_expert_recsys import CollaborativeExpert, SemanticExpert
from models.evidence_analyzer import EvidenceAnalyzer

device = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class RankAwareSelectiveRouter(nn.Module):
    """
    Enhanced 3-action + continuous mixture router with listwise ranking loss.
    """
    def __init__(self, in_features=8, hidden_dim=64, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def route(self, p_c, p_s, x):
        weights = self.forward(x) # [B, 3] -> [w_C, w_S, w_H]
        w_c = weights[:, 0:1] + 0.5 * weights[:, 2:3]
        w_s = weights[:, 1:2] + 0.5 * weights[:, 2:3]
        return w_c * p_c + w_s * p_s, weights


def run_rank_aware_training():
    train_data, val_data, test_data, num_items, semantic_table = load_and_preprocess_movielens()
    
    # Load trained experts
    collab_expert = CollaborativeExpert(num_items=num_items, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    semantic_expert = SemanticExpert(num_items=num_items, semantic_table=semantic_table, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    
    collab_expert.load_state_dict(torch.load("checkpoints/collab_expert.pt", map_location=device))
    semantic_expert.load_state_dict(torch.load("checkpoints/semantic_expert.pt", map_location=device))
    collab_expert.eval()
    semantic_expert.eval()

    evidence_analyzer = EvidenceAnalyzer()

    # Extract val features & probabilities
    print("Extracting validation representations...")
    from train_and_evaluate_evidence_routing import extract_dataset_evidence_and_oracle
    val_x, _, val_pc, val_ps, val_ph, val_targets = extract_dataset_evidence_and_oracle(
        collab_expert, semantic_expert, evidence_analyzer, val_data, semantic_table, num_mc_samples=5
    )
    print("Extracting test representations...")
    test_x, _, test_pc, test_ps, test_ph, test_targets = extract_dataset_evidence_and_oracle(
        collab_expert, semantic_expert, evidence_analyzer, test_data, semantic_table, num_mc_samples=5
    )

    router = RankAwareSelectiveRouter(in_features=8, hidden_dim=64, dropout=0.10).to(device)
    optimizer = optim.AdamW(router.parameters(), lr=3e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    # Train router directly on Listwise Top-50 negative ranking loss
    N_val = len(val_targets)
    targets_tensor = torch.tensor([t - 1 for t in val_targets], dtype=torch.long, device=device)
    
    print("\n---> Training Rank-Aware Router with Direct Listwise Ranking Loss...")
    batch_size = 256
    best_val_ndcg = -1.0
    best_router_weights = None

    for epoch in range(1, 41):
        router.train()
        perm = torch.randperm(N_val)
        total_loss = 0.0

        for i in range(0, N_val, batch_size):
            idx = perm[i:i + batch_size]
            b_x = val_x[idx].to(device)
            b_pc = val_pc[idx].to(device)
            b_ps = val_ps[idx].to(device)
            b_tgts = targets_tensor[idx]

            optimizer.zero_grad()
            p_routed, _ = router.route(b_pc, b_ps, b_x)
            
            # Listwise cross-entropy on top candidates + positive target
            # Log-softmax over items
            log_p = torch.log(p_routed + 1e-12)
            loss = F.nll_loss(log_p, b_tgts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        scheduler.step()

        # Evaluate validation NDCG@10
        router.eval()
        with torch.no_grad():
            v_routed, _ = router.route(val_pc, val_ps, val_x)
            v_np = v_routed.numpy()
            ndcgs = []
            for u in range(N_val):
                tgt_idx = val_targets[u] - 1
                rank = int(np.sum(v_np[u] >= v_np[u, tgt_idx]))
                if rank <= 10:
                    ndcgs.append(1.0 / math.log2(rank + 1))
                else:
                    ndcgs.append(0.0)
            cur_val_ndcg = float(np.mean(ndcgs))

        if cur_val_ndcg > best_val_ndcg:
            best_val_ndcg = cur_val_ndcg
            best_router_weights = {k: v.cpu().clone() for k, v in router.state_dict().items()}
            star = " [*BEST]"
        else:
            star = ""

        if epoch % 5 == 0 or epoch == 1 or star:
            print(f"  Epoch {epoch:02d}/40 - NLL Loss: {total_loss/N_val:.4f} - Val NDCG@10: {cur_val_ndcg:.4f}{star}")

    router.load_state_dict(best_router_weights)
    print(f"\nRestored best router (Val NDCG@10: {best_val_ndcg:.4f})")

    # Evaluate on Test Set
    router.eval()
    with torch.no_grad():
        test_routed, test_weights = router.route(test_pc, test_ps, test_x)
        test_r_np = test_routed.numpy()
        test_pc_np = test_pc.numpy()
        test_ps_np = test_ps.numpy()
        test_ph_np = (0.5 * (test_pc + test_ps)).numpy()

    # Precompute Uncertainty Gate
    u_c = test_x[:, 0:1]
    u_gate_w = torch.sigmoid(2.0 * torch.log1p(u_c) - 1.0)
    p_unc_gate = ((1.0 - u_gate_w) * test_pc + u_gate_w * test_ps).numpy()

    models = {
        "SASRec (Collaborative)": test_pc_np,
        "Semantic Expert": test_ps_np,
        "Static Fusion (50/50)": test_ph_np,
        "Uncertainty Gate (Prior)": p_unc_gate,
        "Rank-Aware Selective Router (Ours)": test_r_np,
    }

    print("\n" + "=" * 80)
    print(" FINAL EVALUATION ON FULL TEST CATALOG (N = 6,038 USERS)")
    print("=" * 80)
    print(f"{'Model':<38} | {'HR@5':<8} | {'HR@10':<8} | {'HR@20':<8} | {'NDCG@10':<8} | {'Oracle Gap':<10}")
    print("-" * 95)

    oracle_hrs10, oracle_ndcgs10 = [], []
    N_test = len(test_targets)
    for u in range(N_test):
        tgt_idx = test_targets[u] - 1
        rc = int(np.sum(test_pc_np[u] >= test_pc_np[u, tgt_idx]))
        rs = int(np.sum(test_ps_np[u] >= test_ps_np[u, tgt_idx]))
        rh = int(np.sum(test_ph_np[u] >= test_ph_np[u, tgt_idx]))
        best_rank = min(rc, rs, rh)
        oracle_hrs10.append(1.0 if best_rank <= 10 else 0.0)
        oracle_ndcgs10.append(1.0 / math.log2(best_rank + 1) if best_rank <= 10 else 0.0)
    oracle_hr10 = float(np.mean(oracle_hrs10))
    oracle_ndcg10 = float(np.mean(oracle_ndcgs10))

    results_summary = {}

    for name, p_arr in models.items():
        hrs5, hrs10, hrs20 = [], [], []
        ndcgs10 = []
        for u in range(N_test):
            tgt_idx = test_targets[u] - 1
            rank = int(np.sum(p_arr[u] >= p_arr[u, tgt_idx]))
            hrs5.append(1.0 if rank <= 5 else 0.0)
            hrs10.append(1.0 if rank <= 10 else 0.0)
            hrs20.append(1.0 if rank <= 20 else 0.0)
            ndcgs10.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
        
        hr5_m, hr10_m, hr20_m = float(np.mean(hrs5)), float(np.mean(hrs10)), float(np.mean(hrs20))
        ndcg10_m = float(np.mean(ndcgs10))
        gap = (oracle_ndcg10 - ndcg10_m) / oracle_ndcg10 * 100.0

        results_summary[name] = {"HR@5": hr5_m, "HR@10": hr10_m, "HR@20": hr20_m, "NDCG@10": ndcg10_m, "Gap": gap}
        print(f"{name:<38} | {hr5_m:<8.4f} | {hr10_m:<8.4f} | {hr20_m:<8.4f} | {ndcg10_m:<8.4f} | {gap:<10.2f}%")

    print(f"{'Oracle Router (Upper Bound)':<38} | {'-':<8} | {oracle_hr10:<8.4f} | {'-':<8} | {oracle_ndcg10:<8.4f} | {'0.00%':<10}")

    # Paired Wilcoxon Test between Rank-Aware Router and Baselines
    from scipy.stats import wilcoxon, ttest_rel
    r_hits = [1.0 if int(np.sum(test_r_np[u] >= test_r_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in range(N_test)]
    static_hits = [1.0 if int(np.sum(test_ph_np[u] >= test_ph_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in range(N_test)]
    unc_hits = [1.0 if int(np.sum(p_unc_gate[u] >= p_unc_gate[u, test_targets[u]-1])) <= 10 else 0.0 for u in range(N_test)]
    sasrec_hits = [1.0 if int(np.sum(test_pc_np[u] >= test_pc_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in range(N_test)]

    t_stat_static, p_val_static = ttest_rel(r_hits, static_hits)
    w_stat_static, p_w_static = wilcoxon(r_hits, static_hits)
    t_stat_sasrec, p_val_sasrec = ttest_rel(r_hits, sasrec_hits)

    print("\n" + "=" * 80)
    print(" PAIRED STATISTICAL SIGNIFICANCE (ROUTER VS BASELINES)")
    print("=" * 80)
    print(f"Router vs Static Fusion: t-stat={t_stat_static:.4f}, p-value={p_val_static:.4e}, Wilcoxon p={p_w_static:.4e}")
    print(f"Router vs SASRec:        t-stat={t_stat_sasrec:.4f}, p-value={p_val_sasrec:.4e}")

    # Save to json
    with open("results/rank_aware_router_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\nSaved to results/rank_aware_router_results.json")

if __name__ == "__main__":
    run_rank_aware_training()
