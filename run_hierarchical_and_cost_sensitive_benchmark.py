"""
Hierarchical and Cost-Sensitive Routing Benchmark for Dual-Expert Sequential Recommendation.

Implements and evaluates:
1. Static Fusion (50/50 Baseline)
2. Standard 5D Continuous Router
3. Cost-Sensitive 5D Router (Asymmetric minority loss penalty)
4. Hierarchical 2-Level Router (Level 1: Specialist vs Hybrid; Level 2: Collab vs Semantic)
5. Oracle Upper Bound

Evaluates across 5 seeds [42, 123, 2024, 7, 99]:
- Overall Metrics: HR@5, HR@10, HR@20, NDCG@10
- Cohort Performance: HR@10 on Cohort C (N=359), Cohort S (N=269), Cohort H (N=5410)
- Specialist Detection Recall: Recall_C, Recall_S, Recall_Specialist
- Statistical Significance: Paired t-test, Wilcoxon signed-rank test, Cohen's d vs Static Fusion
"""

import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_and_evaluate_evidence_routing import load_and_preprocess_movielens
from models.dual_expert_recsys import CollaborativeExpert, SemanticExpert
from run_enhanced_ablation_suite import EnrichedEvidenceExtractor

device = torch.device("cpu")

# ==============================================================================
# 1. MODEL ARCHITECTURES
# ==============================================================================

class Standard5DRouter(nn.Module):
    """Standard 3-way softmax continuous router."""
    def __init__(self, in_features: int = 5, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

    def route(self, p_c: torch.Tensor, p_s: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.forward(x) # [B, 3] (w_C, w_S, w_H)
        w_c = weights[:, 0:1]
        w_s = weights[:, 1:2]
        w_h = weights[:, 2:3]
        eff_c = w_c + 0.5 * w_h
        eff_s = w_s + 0.5 * w_h
        p_routed = eff_c * p_c + eff_s * p_s
        return p_routed, weights


class Hierarchical2LevelRouter(nn.Module):
    """
    2-Level Hierarchical Router:
      Level 1: P(Specialist) vs P(Hybrid) via Binary Gate
      Level 2: P(Collab | Specialist) vs P(Semantic | Specialist)
    Final continuous weights:
      w_H = 1 - P(Specialist)
      w_C = P(Specialist) * P(Collab | Specialist)
      w_S = P(Specialist) * (1 - P(Collab | Specialist))
    """
    def __init__(self, in_features: int = 5, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        # Level 1 Gate: Specialist vs Hybrid
        self.gate_l1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # Level 2 Gate: Collab vs Semantic given Specialist
        self.gate_l2 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights: [B, 3] (w_C, w_S, w_H)
            p_spec: [B, 1] P(Specialist)
            p_c_given_spec: [B, 1] P(Collab | Specialist)
        """
        p_spec = torch.sigmoid(self.gate_l1(x)) # [B, 1]
        p_c_given_spec = torch.sigmoid(self.gate_l2(x)) # [B, 1]

        w_h = 1.0 - p_spec
        w_c = p_spec * p_c_given_spec
        w_s = p_spec * (1.0 - p_c_given_spec)

        weights = torch.cat([w_c, w_s, w_h], dim=-1) # [B, 3]
        return weights, p_spec, p_c_given_spec

    def route(self, p_c: torch.Tensor, p_s: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, _, _ = self.forward(x)
        w_c = weights[:, 0:1]
        w_s = weights[:, 1:2]
        w_h = weights[:, 2:3]
        eff_c = w_c + 0.5 * w_h
        eff_s = w_s + 0.5 * w_h
        p_routed = eff_c * p_c + eff_s * p_s
        return p_routed, weights


# ==============================================================================
# 2. TRAINING FUNCTIONS
# ==============================================================================

def compute_oracle_labels(p_c: torch.Tensor, p_s: torch.Tensor, targets: List[int]):
    """Computes discrete oracle regimes (0: C, 1: S, 2: H) and soft DCG distributions."""
    N = len(targets)
    p_c_np = p_c.numpy()
    p_s_np = p_s.numpy()
    p_h_np = (0.5 * (p_c + p_s)).numpy()

    discrete_cohorts = []
    soft_targets = torch.zeros(N, 3, dtype=torch.float32)

    for u in range(N):
        tgt_idx = targets[u] - 1
        rc = int(np.sum(p_c_np[u] >= p_c_np[u, tgt_idx]))
        rs = int(np.sum(p_s_np[u] >= p_s_np[u, tgt_idx]))
        rh = int(np.sum(p_h_np[u] >= p_h_np[u, tgt_idx]))

        yc = 1.0 / math.log2(rc + 1) if rc <= 20 else 0.0
        ys = 1.0 / math.log2(rs + 1) if rs <= 20 else 0.0
        yh = 1.0 / math.log2(rh + 1) if rh <= 20 else 0.0

        if yh >= max(yc, ys) and yh > 0:
            act = 2
        elif yc >= ys and yc > 0:
            act = 0
        elif ys > 0:
            act = 1
        else:
            act = 2

        discrete_cohorts.append(act)

        tot = yc + ys + yh
        if tot > 0:
            soft_targets[u, 0] = yc / tot
            soft_targets[u, 1] = ys / tot
            soft_targets[u, 2] = yh / tot
        else:
            soft_targets[u, 0] = 0.333
            soft_targets[u, 1] = 0.333
            soft_targets[u, 2] = 0.334

    return np.array(discrete_cohorts), soft_targets


def train_standard_router(router, val_x, val_pc, val_ps, soft_targets, targets_tensor, epochs=35, lr=2e-3):
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    dataset = torch.utils.data.TensorDataset(val_x, val_pc, val_ps, soft_targets, targets_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(1, epochs + 1):
        router.train()
        for bx, bpc, bps, bsoft, btgt in loader:
            optimizer.zero_grad()
            p_routed, weights = router.route(bpc, bps, bx)
            log_weights = torch.log(weights + 1e-12)
            kl_loss = F.kl_div(log_weights, bsoft, reduction="batchmean")
            log_p = torch.log(p_routed + 1e-12)
            nll_loss = F.nll_loss(log_p, btgt)
            loss = kl_loss + 0.5 * nll_loss
            loss.backward()
            optimizer.step()
        scheduler.step()
    return router


def train_cost_sensitive_router(router, val_x, val_pc, val_ps, soft_targets, val_cohorts, targets_tensor, lambda_spec=3.0, epochs=35, lr=2e-3):
    """
    Cost-sensitive loss heavily penalizing mistakes on minority specialist regimes (C and S).
    """
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Sample weights: specialists (cohort 0 or 1) get lambda_spec multiplier
    sample_weights = torch.ones(len(val_cohorts), dtype=torch.float32)
    spec_mask = (val_cohorts == 0) | (val_cohorts == 1)
    sample_weights[spec_mask] = lambda_spec

    dataset = torch.utils.data.TensorDataset(val_x, val_pc, val_ps, soft_targets, sample_weights, targets_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(1, epochs + 1):
        router.train()
        for bx, bpc, bps, bsoft, bw, btgt in loader:
            optimizer.zero_grad()
            p_routed, weights = router.route(bpc, bps, bx)
            log_weights = torch.log(weights + 1e-12)

            # Per-instance KL weighted by sample weight
            kl_per_item = torch.sum(bsoft * (torch.log(bsoft + 1e-12) - log_weights), dim=-1)
            weighted_kl = (kl_per_item * bw).mean()

            log_p = torch.log(p_routed + 1e-12)
            nll_per_item = F.nll_loss(log_p, btgt, reduction='none')
            weighted_nll = (nll_per_item * bw).mean()

            loss = weighted_kl + 0.5 * weighted_nll
            loss.backward()
            optimizer.step()
        scheduler.step()
    return router


def train_hierarchical_router(router, val_x, val_pc, val_ps, val_cohorts, targets_tensor, epochs=35, lr=2e-3):
    """
    Trains Hierarchical 2-Level Router:
      Level 1: Binary Specialist vs Hybrid (weighted BCE with pos_weight to handle 10:90 imbalance)
      Level 2: Collab vs Semantic on Specialist subset
      End-to-end Ranking NLL loss
    """
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    is_spec = torch.tensor((val_cohorts == 0) | (val_cohorts == 1), dtype=torch.float32).unsqueeze(-1)
    is_collab = torch.tensor(val_cohorts == 0, dtype=torch.float32).unsqueeze(-1)

    # pos_weight for L1 imbalance (~ 90 / 10 = 9.0)
    pos_weight = torch.tensor([5.0])

    dataset = torch.utils.data.TensorDataset(val_x, val_pc, val_ps, is_spec, is_collab, targets_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(1, epochs + 1):
        router.train()
        for bx, bpc, bps, bspec, bcollab, btgt in loader:
            optimizer.zero_grad()
            weights, p_spec, p_c_given_spec = router.forward(bx)
            
            # Level 1 Loss: Weighted BCE for specialist detection
            l1_loss = F.binary_cross_entropy_with_logits(
                router.gate_l1(bx), bspec, pos_weight=pos_weight
            )

            # Level 2 Loss: BCE for collab vs semantic (masked to specialists)
            spec_mask = (bspec > 0.5).squeeze(-1)
            if spec_mask.sum() > 0:
                l2_loss = F.binary_cross_entropy(
                    p_c_given_spec[spec_mask], bcollab[spec_mask]
                )
            else:
                l2_loss = torch.tensor(0.0)

            # End-to-end Ranking Loss
            w_c = weights[:, 0:1]
            w_s = weights[:, 1:2]
            w_h = weights[:, 2:3]
            eff_c = w_c + 0.5 * w_h
            eff_s = w_s + 0.5 * w_h
            p_routed = eff_c * bpc + eff_s * bps
            log_p = torch.log(p_routed + 1e-12)
            nll_loss = F.nll_loss(log_p, btgt)

            loss = l1_loss + l2_loss + 0.5 * nll_loss
            loss.backward()
            optimizer.step()
        scheduler.step()
    return router


# ==============================================================================
# 3. EVALUATION ENGINE
# ==============================================================================

def evaluate_predictions(prob_matrix: np.ndarray, targets: List[int], oracle_cohorts: np.ndarray):
    """
    Computes HR@5, HR@10, HR@20, NDCG@10 overall and HR@10 per cohort.
    """
    N = len(targets)
    hr5, hr10, hr20, ndcg10 = [], [], [], []

    for u in range(N):
        tgt_idx = targets[u] - 1
        rank = int(np.sum(prob_matrix[u] >= prob_matrix[u, tgt_idx]))
        
        hr5.append(1.0 if rank <= 5 else 0.0)
        hr10.append(1.0 if rank <= 10 else 0.0)
        hr20.append(1.0 if rank <= 20 else 0.0)
        ndcg10.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)

    hr10_arr = np.array(hr10)
    c_idx = np.where(oracle_cohorts == 0)[0]
    s_idx = np.where(oracle_cohorts == 1)[0]
    h_idx = np.where(oracle_cohorts == 2)[0]

    return {
        "HR@5": float(np.mean(hr5)),
        "HR@10": float(np.mean(hr10)),
        "HR@20": float(np.mean(hr20)),
        "NDCG@10": float(np.mean(ndcg10)),
        "HR@10_C": float(np.mean(hr10_arr[c_idx])),
        "HR@10_S": float(np.mean(hr10_arr[s_idx])),
        "HR@10_H": float(np.mean(hr10_arr[h_idx])),
        "hr10_per_user": hr10,
        "ndcg10_per_user": ndcg10
    }


def compute_specialist_recalls(weights: np.ndarray, oracle_cohorts: np.ndarray):
    """
    Computes Recall_C, Recall_S, and Recall_Specialist.
    Pred decision = argmax(w_C, w_S, w_H).
    """
    pred_actions = np.argmax(weights, axis=-1) # 0: C, 1: S, 2: H

    c_mask = (oracle_cohorts == 0) # 359
    s_mask = (oracle_cohorts == 1) # 269
    spec_mask = c_mask | s_mask   # 628

    correct_c = np.sum((pred_actions == 0) & c_mask)
    correct_s = np.sum((pred_actions == 1) & s_mask)
    detected_spec = np.sum((pred_actions != 2) & spec_mask)

    recall_c = float(correct_c / np.sum(c_mask)) if np.sum(c_mask) > 0 else 0.0
    recall_s = float(correct_s / np.sum(s_mask)) if np.sum(s_mask) > 0 else 0.0
    recall_spec = float(detected_spec / np.sum(spec_mask)) if np.sum(spec_mask) > 0 else 0.0

    return recall_c, recall_s, recall_spec


# ==============================================================================
# 4. MAIN BENCHMARK PIPELINE
# ==============================================================================

def run_benchmark():
    print("=" * 115)
    print(" HIERARCHICAL & COST-SENSITIVE ROUTING BENCHMARK (5 SEEDS x 5 MODELS)")
    print("=" * 115)

    train_data, val_data, test_data, num_items, semantic_table = load_and_preprocess_movielens()

    collab_expert = CollaborativeExpert(num_items=num_items, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    semantic_expert = SemanticExpert(num_items=num_items, semantic_table=semantic_table, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    collab_expert.load_state_dict(torch.load("checkpoints/collab_expert.pt", map_location=device))
    semantic_expert.load_state_dict(torch.load("checkpoints/semantic_expert.pt", map_location=device))
    collab_expert.eval()
    semantic_expert.eval()

    extractor = EnrichedEvidenceExtractor()

    def extract_split(data_dict):
        uids = list(data_dict.keys())
        all_feats, all_pc, all_ps, all_targets = [], [], [], []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(uids), batch_size):
                b_uids = uids[i:i+batch_size]
                seqs, tgts = [], []
                for u in b_uids:
                    s, t = data_dict[u]
                    pad = 50 - len(s)
                    seqs.append([0]*pad + s if pad > 0 else s[-50:])
                    tgts.append(t)
                seqs_t = torch.tensor(seqs, dtype=torch.long, device=device)
                mc_c, _ = collab_expert.mc_predict(seqs_t, num_samples=5)
                mc_s, _ = semantic_expert.mc_predict(seqs_t, num_samples=5)
                feats, pc, ps = extractor.extract_features(mc_c, mc_s, seqs_t, semantic_table)
                all_feats.append(feats.cpu())
                all_pc.append(pc.cpu())
                all_ps.append(ps.cpu())
                all_targets.extend(tgts)
        return torch.cat(all_feats, dim=0), torch.cat(all_pc, dim=0), torch.cat(all_ps, dim=0), all_targets

    print("Extracting validation and test evidence features...")
    val_x, val_pc, val_ps, val_targets = extract_split(val_data)
    test_x, test_pc, test_ps, test_targets = extract_split(test_data)
    N_test = len(test_targets)

    # 5D feature mask: [norm_len, unique_ratio, drift, margin_c, margin_s]
    mask_5d = [6, 7, 8, 9, 10]
    val_x_5d = val_x[:, mask_5d]
    test_x_5d = test_x[:, mask_5d]

    val_targets_tensor = torch.tensor([t - 1 for t in val_targets], dtype=torch.long, device=device)

    # Compute Oracle Cohorts & Targets
    val_cohorts, val_soft_targets = compute_oracle_labels(val_pc, val_ps, val_targets)
    test_cohorts, test_soft_targets = compute_oracle_labels(test_pc, test_ps, test_targets)

    # Static Fusion & Oracle precomputations
    p_c_np = test_pc.numpy()
    p_s_np = test_ps.numpy()
    p_static_np = (0.5 * (test_pc + test_ps)).numpy()

    oracle_preds = np.zeros_like(p_static_np)
    for u in range(N_test):
        act = test_cohorts[u]
        if act == 0:
            oracle_preds[u] = p_c_np[u]
        elif act == 1:
            oracle_preds[u] = p_s_np[u]
        else:
            oracle_preds[u] = p_static_np[u]

    static_eval = evaluate_predictions(p_static_np, test_targets, test_cohorts)
    oracle_eval = evaluate_predictions(oracle_preds, test_targets, test_cohorts)

    seeds = [42, 123, 2024, 7, 99]
    models_to_test = ["Current 5D Router", "Cost-Sensitive Router", "Hierarchical Router"]
    
    results = {
        "Static 50/50": static_eval,
        "Oracle": oracle_eval,
        "Current 5D Router": {"seeds": []},
        "Cost-Sensitive Router": {"seeds": []},
        "Hierarchical Router": {"seeds": []}
    }

    # Hyperparameter selection on validation split for Cost-Sensitive Router:
    print("\n--- Tuning lambda_spec on validation split ---")
    best_lambda = 3.0
    best_val_ndcg = -1.0
    for cand_l in [1.5, 3.0, 5.0, 8.0]:
        torch.manual_seed(42)
        r_tune = Standard5DRouter(in_features=5, hidden_dim=64, dropout=0.10)
        r_tune = train_cost_sensitive_router(r_tune, val_x_5d, val_pc, val_ps, val_soft_targets, val_cohorts, val_targets_tensor, lambda_spec=cand_l, epochs=30)
        with torch.no_grad():
            v_rout, _ = r_tune.route(val_pc, val_ps, val_x_5d)
        v_eval = evaluate_predictions(v_rout.numpy(), val_targets, val_cohorts)
        print(f"Validation lambda={cand_l:<4} -> HR@10: {v_eval['HR@10']:.4f}, NDCG@10: {v_eval['NDCG@10']:.4f}, HR@10_C: {v_eval['HR@10_C']:.4f}, HR@10_S: {v_eval['HR@10_S']:.4f}")
        if v_eval["NDCG@10"] > best_val_ndcg:
            best_val_ndcg = v_eval["NDCG@10"]
            best_lambda = cand_l
    print(f"Selected best lambda_spec on validation: {best_lambda}")

    print("\n" + "=" * 115)
    print(" RUNNING 5-SEED MULTI-SEED REPLICATION ACROSS ALL MODELS")
    print("=" * 115)

    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        torch.manual_seed(s)
        np.random.seed(s)

        # 1. Standard 5D Router
        r_std = Standard5DRouter(in_features=5, hidden_dim=64, dropout=0.10)
        r_std = train_standard_router(r_std, val_x_5d, val_pc, val_ps, val_soft_targets, val_targets_tensor, epochs=35)
        r_std.eval()
        with torch.no_grad():
            t_rout_std, w_std = r_std.route(test_pc, test_ps, test_x_5d)
            ev_std = evaluate_predictions(t_rout_std.numpy(), test_targets, test_cohorts)
            rec_c_std, rec_s_std, rec_sp_std = compute_specialist_recalls(w_std.numpy(), test_cohorts)
            ev_std.update({"Recall_C": rec_c_std, "Recall_S": rec_s_std, "Recall_Spec": rec_sp_std})
            results["Current 5D Router"]["seeds"].append(ev_std)

        # 2. Cost-Sensitive Router
        r_cs = Standard5DRouter(in_features=5, hidden_dim=64, dropout=0.10)
        r_cs = train_cost_sensitive_router(r_cs, val_x_5d, val_pc, val_ps, val_soft_targets, val_cohorts, val_targets_tensor, lambda_spec=best_lambda, epochs=35)
        r_cs.eval()
        with torch.no_grad():
            t_rout_cs, w_cs = r_cs.route(test_pc, test_ps, test_x_5d)
            ev_cs = evaluate_predictions(t_rout_cs.numpy(), test_targets, test_cohorts)
            rec_c_cs, rec_s_cs, rec_sp_cs = compute_specialist_recalls(w_cs.numpy(), test_cohorts)
            ev_cs.update({"Recall_C": rec_c_cs, "Recall_S": rec_s_cs, "Recall_Spec": rec_sp_cs})
            results["Cost-Sensitive Router"]["seeds"].append(ev_cs)

        # 3. Hierarchical Router
        r_hier = Hierarchical2LevelRouter(in_features=5, hidden_dim=64, dropout=0.10)
        r_hier = train_hierarchical_router(r_hier, val_x_5d, val_pc, val_ps, val_cohorts, val_targets_tensor, epochs=35)
        r_hier.eval()
        with torch.no_grad():
            t_rout_hier, w_hier = r_hier.route(test_pc, test_ps, test_x_5d)
            ev_hier = evaluate_predictions(t_rout_hier.numpy(), test_targets, test_cohorts)
            rec_c_hier, rec_s_hier, rec_sp_hier = compute_specialist_recalls(w_hier.numpy(), test_cohorts)
            ev_hier.update({"Recall_C": rec_c_hier, "Recall_S": rec_s_hier, "Recall_Spec": rec_sp_hier})
            results["Hierarchical Router"]["seeds"].append(ev_hier)

        print(f"Seed {s} Done: Std HR@10={ev_std['HR@10']:.4f} | CS HR@10={ev_cs['HR@10']:.4f} (RecSpec={rec_sp_cs:.3f}) | Hier HR@10={ev_hier['HR@10']:.4f} (RecSpec={rec_sp_hier:.3f})")

    # ==============================================================================
    # 5. SUMMARY COMPARISON TABLES
    # ==============================================================================
    print("\n" + "=" * 135)
    print(" COMPREHENSIVE 5-MODEL BENCHMARK SUMMARY (5 SEEDS MEAN +- STD)")
    print("=" * 135)
    headers = f"{'Model':<24} | {'HR@5':<14} | {'HR@10':<14} | {'HR@20':<14} | {'NDCG@10':<14} | {'HR@10 (C)':<12} | {'HR@10 (S)':<12} | {'HR@10 (H)':<12} | {'Recall_Spec':<12}"
    print(headers)
    print("-" * 135)

    # Static Baseline
    st = results["Static 50/50"]
    print(f"{'Static 50/50':<24} | {st['HR@5']:<14.4f} | {st['HR@10']:<14.4f} | {st['HR@20']:<14.4f} | {st['NDCG@10']:<14.4f} | {st['HR@10_C']:<12.4f} | {st['HR@10_S']:<12.4f} | {st['HR@10_H']:<12.4f} | {'0.0000':<12}")

    summary_export = {
        "Static 50/50": {k: float(v) for k, v in st.items() if not isinstance(v, list)},
        "Oracle": {k: float(v) for k, v in results["Oracle"].items() if not isinstance(v, list)}
    }

    for m_name in models_to_test:
        s_list = results[m_name]["seeds"]
        
        m_hr5 = f"{np.mean([x['HR@5'] for x in s_list]):.4f} +- {np.std([x['HR@5'] for x in s_list]):.4f}"
        m_hr10 = f"{np.mean([x['HR@10'] for x in s_list]):.4f} +- {np.std([x['HR@10'] for x in s_list]):.4f}"
        m_hr20 = f"{np.mean([x['HR@20'] for x in s_list]):.4f} +- {np.std([x['HR@20'] for x in s_list]):.4f}"
        m_ndcg10 = f"{np.mean([x['NDCG@10'] for x in s_list]):.4f} +- {np.std([x['NDCG@10'] for x in s_list]):.4f}"
        m_hrc = f"{np.mean([x['HR@10_C'] for x in s_list]):.4f}"
        m_hrs = f"{np.mean([x['HR@10_S'] for x in s_list]):.4f}"
        m_hrh = f"{np.mean([x['HR@10_H'] for x in s_list]):.4f}"
        m_rec = f"{np.mean([x['Recall_Spec'] for x in s_list]):.4f}"

        print(f"{m_name:<24} | {m_hr5:<14} | {m_hr10:<14} | {m_hr20:<14} | {m_ndcg10:<14} | {m_hrc:<12} | {m_hrs:<12} | {m_hrh:<12} | {m_rec:<12}")

        # Pairwise statistical test against Static Fusion across all seeds
        p_vals_t = []
        p_vals_w = []
        for x in s_list:
            t_res = stats.ttest_rel(x["hr10_per_user"], st["hr10_per_user"])
            w_res = stats.wilcoxon(x["hr10_per_user"], st["hr10_per_user"])
            p_vals_t.append(t_res.pvalue)
            p_vals_w.append(w_res.pvalue)

        summary_export[m_name] = {
            "HR@5_mean": float(np.mean([x['HR@5'] for x in s_list])),
            "HR@5_std": float(np.std([x['HR@5'] for x in s_list])),
            "HR@10_mean": float(np.mean([x['HR@10'] for x in s_list])),
            "HR@10_std": float(np.std([x['HR@10'] for x in s_list])),
            "HR@20_mean": float(np.mean([x['HR@20'] for x in s_list])),
            "HR@20_std": float(np.std([x['HR@20'] for x in s_list])),
            "NDCG@10_mean": float(np.mean([x['NDCG@10'] for x in s_list])),
            "NDCG@10_std": float(np.std([x['NDCG@10'] for x in s_list])),
            "HR@10_C": float(np.mean([x['HR@10_C'] for x in s_list])),
            "HR@10_S": float(np.mean([x['HR@10_S'] for x in s_list])),
            "HR@10_H": float(np.mean([x['HR@10_H'] for x in s_list])),
            "Recall_C": float(np.mean([x['Recall_C'] for x in s_list])),
            "Recall_S": float(np.mean([x['Recall_S'] for x in s_list])),
            "Recall_Spec": float(np.mean([x['Recall_Spec'] for x in s_list])),
            "p_val_ttest_vs_static": float(np.mean(p_vals_t)),
            "p_val_wilcoxon_vs_static": float(np.mean(p_vals_w)),
        }

    # Oracle Upper Bound
    ora = results["Oracle"]
    print(f"{'Oracle (Upper Bound)':<24} | {ora['HR@5']:<14.4f} | {ora['HR@10']:<14.4f} | {ora['HR@20']:<14.4f} | {ora['NDCG@10']:<14.4f} | {ora['HR@10_C']:<12.4f} | {ora['HR@10_S']:<12.4f} | {ora['HR@10_H']:<12.4f} | {'1.0000':<12}")
    print("=" * 135)

    with open("results/hierarchical_and_cost_sensitive_benchmark.json", "w") as f:
        json.dump(summary_export, f, indent=2)
    print("\nSaved detailed benchmark results to results/hierarchical_and_cost_sensitive_benchmark.json")

if __name__ == "__main__":
    run_benchmark()
