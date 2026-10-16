"""
Ablation of Router Formulations & Action Space Geometry:
Evaluates 5 alternative routing architectures on the identical 75/25 split and test set:
1. 3-Action Soft Mixture Router {C, S, H} (Proposed)
2. Direct Continuous Scalar Gate alpha_u in [0, 1]
3. 2-Action Soft Router {C, S}
4. 3-Action Discrete Hard Selection (argmax)
5. 2-Action Discrete Hard Selection (argmax)

Answers the core reviewer question:
"Why an action space {C, S, H} instead of direct continuous alpha_u or {C, S}?"
"""

import os
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.selective_router import SelectiveRouter, ContinuousAlphaRouter, TwoActionRouter
from models.dual_expert_recsys import CollaborativeExpert, SemanticExpert
from models.evidence_analyzer import EvidenceAnalyzer
from train_and_evaluate_evidence_routing import load_and_preprocess_movielens, extract_dataset_evidence_and_oracle

device = torch.device("cpu")
SEED = 42

def compute_ranking_metrics(p_matrix: np.ndarray, targets: np.ndarray):
    N = len(targets)
    hr10, ndcg10 = np.zeros(N), np.zeros(N)
    for i in range(N):
        tgt = targets[i]
        tgt_score = p_matrix[i, tgt]
        rank = int(np.sum(p_matrix[i] >= tgt_score))
        if rank <= 10:
            hr10[i] = 1.0
            ndcg10[i] = 1.0 / math.log2(rank + 1)
    return float(np.mean(hr10)), float(np.mean(ndcg10))

def run_ablation():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 80)
    print(" ABLATION STUDY: ROUTER FORMULATIONS & ACTION SPACE GEOMETRY")
    print("=" * 80)

    train_data, val_data, test_data, num_items, semantic_table = load_and_preprocess_movielens()

    collab_expert = CollaborativeExpert(num_items=num_items, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    semantic_expert = SemanticExpert(num_items=num_items, semantic_table=semantic_table, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)

    collab_expert.load_state_dict(torch.load("checkpoints/collab_expert.pt", map_location=device))
    semantic_expert.load_state_dict(torch.load("checkpoints/semantic_expert.pt", map_location=device))
    collab_expert.eval()
    semantic_expert.eval()

    evidence_analyzer = EvidenceAnalyzer()

    # Extract validation evidence & Oracle
    val_features, val_oracle_actions, val_pc, val_ps, val_ph, val_targets = extract_dataset_evidence_and_oracle(
        collab_expert, semantic_expert, evidence_analyzer, val_data, semantic_table, num_mc_samples=3
    )

    # Extract test evidence
    test_features, test_oracle_actions, test_pc, test_ps, test_ph, test_targets = extract_dataset_evidence_and_oracle(
        collab_expert, semantic_expert, evidence_analyzer, test_data, semantic_table, num_mc_samples=3
    )

    targets_np = np.array(test_targets) - 1
    N_val = len(val_features)
    indices = np.random.permutation(N_val)
    split_idx = int(0.75 * N_val)
    train_idx, dev_idx = indices[:split_idx], indices[split_idx:]

    train_feats = val_features[train_idx]
    train_labels_3 = val_oracle_actions[train_idx]

    # Two-action oracle labels: binary 0 (C) vs 1 (S) based on reciprocal rank
    val_targets_np = np.array(val_targets) - 1
    val_pc_np, val_ps_np = val_pc.numpy(), val_ps.numpy()
    val_2_labels = []
    val_soft_targets = [] # optimal continuous alpha in [0, 1] on 11-grid
    alphas = np.linspace(0.0, 1.0, 11)

    for i in range(N_val):
        tgt = val_targets_np[i]
        rc = int(np.sum(val_pc_np[i] >= val_pc_np[i, tgt]))
        rs = int(np.sum(val_ps_np[i] >= val_ps_np[i, tgt]))
        val_2_labels.append(0 if rc <= rs else 1)

        best_r = 999999
        best_a = 0.5
        for a in alphas:
            p_mix = a * val_pc_np[i] + (1.0 - a) * val_ps_np[i]
            r = int(np.sum(p_mix >= p_mix[tgt]))
            if r < best_r:
                best_r = r
                best_a = a
        val_soft_targets.append(best_a)

    val_2_labels = torch.tensor(val_2_labels, dtype=torch.long)
    val_soft_targets = torch.tensor(val_soft_targets, dtype=torch.float32)[:, None]

    train_labels_2 = val_2_labels[train_idx]
    train_soft_targets = val_soft_targets[train_idx]

    # ==============================================================================
    # 1. 3-Action Soft Router (Proposed)
    # ==============================================================================
    router_3 = SelectiveRouter(in_features=8, hidden_dim=32, num_actions=3, dropout=0.10)
    opt_3 = optim.AdamW(router_3.parameters(), lr=3e-3, weight_decay=1e-3)
    loader_3 = DataLoader(TensorDataset(train_feats, train_labels_3), batch_size=128, shuffle=True)
    for epoch in range(20):
        router_3.train()
        for x, y in loader_3:
            opt_3.zero_grad()
            loss = F.cross_entropy(router_3(x), y)
            loss.backward()
            opt_3.step()

    router_3.eval()
    with torch.no_grad():
        p_soft_3, _, probs_3 = router_3.route(test_pc, test_ps, test_ph, test_features, hard_selection=False)
        p_hard_3, _, _ = router_3.route(test_pc, test_ps, test_ph, test_features, hard_selection=True)

    hr_soft_3, ndcg_soft_3 = compute_ranking_metrics(p_soft_3.numpy(), targets_np)
    hr_hard_3, ndcg_hard_3 = compute_ranking_metrics(p_hard_3.numpy(), targets_np)

    # ==============================================================================
    # 2. 2-Action Router ({C, S})
    # ==============================================================================
    router_2 = TwoActionRouter(in_features=8, hidden_dim=32, dropout=0.10)
    opt_2 = optim.AdamW(router_2.parameters(), lr=3e-3, weight_decay=1e-3)
    loader_2 = DataLoader(TensorDataset(train_feats, train_labels_2), batch_size=128, shuffle=True)
    for epoch in range(20):
        router_2.train()
        for x, y in loader_2:
            opt_2.zero_grad()
            loss = F.cross_entropy(router_2(x), y)
            loss.backward()
            opt_2.step()

    router_2.eval()
    with torch.no_grad():
        p_soft_2, _, probs_2 = router_2.route(test_pc, test_ps, test_features, hard_selection=False)
        p_hard_2, _, _ = router_2.route(test_pc, test_ps, test_features, hard_selection=True)

    hr_soft_2, ndcg_soft_2 = compute_ranking_metrics(p_soft_2.numpy(), targets_np)
    hr_hard_2, ndcg_hard_2 = compute_ranking_metrics(p_hard_2.numpy(), targets_np)

    # ==============================================================================
    # 3. Direct Continuous Scalar Gate (alpha_u in [0, 1])
    # ==============================================================================
    router_alpha = ContinuousAlphaRouter(in_features=8, hidden_dim=32, dropout=0.10)
    opt_alpha = optim.AdamW(router_alpha.parameters(), lr=3e-3, weight_decay=1e-3)
    loader_alpha = DataLoader(TensorDataset(train_feats, train_soft_targets), batch_size=128, shuffle=True)
    for epoch in range(20):
        router_alpha.train()
        for x, y in loader_alpha:
            opt_alpha.zero_grad()
            pred_a = router_alpha(x)
            loss = F.mse_loss(pred_a, y)
            loss.backward()
            opt_alpha.step()

    router_alpha.eval()
    with torch.no_grad():
        p_alpha, alphas_pred = router_alpha.route(test_pc, test_ps, test_features)

    hr_alpha, ndcg_alpha = compute_ranking_metrics(p_alpha.numpy(), targets_np)

    # ==============================================================================
    # Summary Table
    # ==============================================================================
    results = [
        {"Formulation": "1. 3-Action Soft Mixture {C, S, H} (Proposed)", "HR@10": hr_soft_3, "NDCG@10": ndcg_soft_3, "Mechanism": "Continuous 3-way Softmax (w_C, w_S, w_H)"},
        {"Formulation": "2. Direct Scalar Gate alpha_u in [0, 1]", "HR@10": hr_alpha, "NDCG@10": ndcg_alpha, "Mechanism": "Sigmoid Regression on Optimal Soft Alpha"},
        {"Formulation": "3. 2-Action Soft Router {C, S}", "HR@10": hr_soft_2, "NDCG@10": ndcg_soft_2, "Mechanism": "Binary Softmax Gate (w_C, w_S)"},
        {"Formulation": "4. 3-Action Hard Selection (argmax)", "HR@10": hr_hard_3, "NDCG@10": ndcg_hard_3, "Mechanism": "Discrete Mode Collapse to argmax"},
        {"Formulation": "5. 2-Action Hard Selection (argmax)", "HR@10": hr_hard_2, "NDCG@10": ndcg_hard_2, "Mechanism": "Discrete Binary Switch {C vs S}"}
    ]

    print("\n" + "=" * 90)
    print(f"{'Formulation':<48} | {'HR@10':<8} | {'NDCG@10':<8} | {'Operational Mechanism'}")
    print("-" * 90)
    for r in results:
        print(f"{r['Formulation']:<48} | {r['HR@10']:<8.4f} | {r['NDCG@10']:<8.4f} | {r['Mechanism']}")
    print("=" * 90)

    # Export to JSON
    with open("ablation_router_formulations.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to ablation_router_formulations.json")

if __name__ == "__main__":
    run_ablation()
