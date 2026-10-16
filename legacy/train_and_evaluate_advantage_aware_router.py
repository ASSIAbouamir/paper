"""
Advantage-Aware Hierarchical Router (AAHR) & 5-Model Decisive Benchmark.

Implements and evaluates:
1. Static 50/50 Baseline
2. Standard 5D Continuous Router
3. Cost-Sensitive Router (Asymmetric minority loss)
4. Hierarchical 2-Level Router (Gate 1: Specialist vs H; Gate 2: C vs S)
5. Advantage-Aware Hierarchical Router (AAHR) using relative advantages A_C = Y_C - Y_H, A_S = Y_S - Y_H
6. Oracle Upper Bound

Calculates across 5 deterministic seeds [42, 123, 2024, 7, 99]:
- Ranking: HR@5, HR@10, HR@20, NDCG@10
- Cohort Performance: HR@10_C (N=359), HR@10_S (N=269), HR@10_H (N=5410)
- Correctly Defined Routing Recalls:
    RouteRecall_C = N_{C, pred=C} / 359
    RouteRecall_S = N_{S, pred=S} / 269
    RouteRecall_Spec = (N_{C, pred=C} + N_{S, pred=S}) / 628
    RouteRecall_H = N_{H, pred=H} / 5410
- Statistical Tests vs Static Fusion: Paired t-test, Wilcoxon signed-rank test, McNemar test, Cohen's d
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
      Level 1: P(Specialist) vs P(Hybrid)
      Level 2: P(Collab | Specialist) vs P(Semantic | Specialist)
    """
    def __init__(self, in_features: int = 5, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.gate_l1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
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
        p_spec = torch.sigmoid(self.gate_l1(x))
        p_c_given_spec = torch.sigmoid(self.gate_l2(x))

        w_h = 1.0 - p_spec
        w_c = p_spec * p_c_given_spec
        w_s = p_spec * (1.0 - p_c_given_spec)

        weights = torch.cat([w_c, w_s, w_h], dim=-1)
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


class AdvantageAwareHierarchicalRouter(nn.Module):
    """
    Advantage-Aware Hierarchical Router (AAHR):
    Directly estimates relative advantages:
      A_C = Y_C - Y_H (Advantage of Collaborative over Hybrid)
      A_S = Y_S - Y_H (Advantage of Semantic over Hybrid)
    Specialist allocation is triggered when predicted advantage exceeds adaptive threshold tau.
    """
    def __init__(self, in_features: int = 5, hidden_dim: int = 64, dropout: float = 0.10, tau: float = 0.05):
        super().__init__()
        self.tau = tau
        # Dedicated Collab Advantage Branch (conditioned strongly on H_c, M_c)
        self.branch_c = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # Dedicated Semantic Advantage Branch (conditioned strongly on H_s, M_s, drift)
        self.branch_s = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
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
            pred_a_c: [B, 1] predicted advantage A_C
            pred_a_s: [B, 1] predicted advantage A_S
        """
        pred_a_c = self.branch_c(x) # [B, 1]
        pred_a_s = self.branch_s(x) # [B, 1]

        # Convert advantages to routing weights via softplus & temperature
        # Positive advantage -> activates specialist weight
        act_c = F.relu(pred_a_c - self.tau)
        act_s = F.relu(pred_a_s - self.tau)
        
        # Softmax over [act_c, act_s, baseline_h] where baseline_h = tau
        logits = torch.cat([act_c * 5.0, act_s * 5.0, torch.zeros_like(act_c)], dim=-1)
        weights = F.softmax(logits, dim=-1) # [B, 3]

        return weights, pred_a_c, pred_a_s

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
# 2. DATA PROCESSING & TARGET COMPUTATION
# ==============================================================================

def compute_oracle_targets_and_advantages(p_c: torch.Tensor, p_s: torch.Tensor, targets: List[int]):
    """
    Computes per-instance:
      - discrete Oracle actions {0: C, 1: S, 2: H}
      - soft Oracle targets (normalized DCG)
      - relative advantages: A_C = Y_C - Y_H, A_S = Y_S - Y_H
    """
    N = len(targets)
    p_c_np = p_c.numpy()
    p_s_np = p_s.numpy()
    p_h_np = (0.5 * (p_c + p_s)).numpy()

    discrete_cohorts = []
    soft_targets = torch.zeros(N, 3, dtype=torch.float32)
    adv_c = torch.zeros(N, 1, dtype=torch.float32)
    adv_s = torch.zeros(N, 1, dtype=torch.float32)

    for u in range(N):
        tgt_idx = targets[u] - 1
        rc = int(np.sum(p_c_np[u] >= p_c_np[u, tgt_idx]))
        rs = int(np.sum(p_s_np[u] >= p_s_np[u, tgt_idx]))
        rh = int(np.sum(p_h_np[u] >= p_h_np[u, tgt_idx]))

        yc = 1.0 / math.log2(rc + 1) if rc <= 20 else 0.0
        ys = 1.0 / math.log2(rs + 1) if rs <= 20 else 0.0
        yh = 1.0 / math.log2(rh + 1) if rh <= 20 else 0.0

        adv_c[u, 0] = yc - yh
        adv_s[u, 0] = ys - yh

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

    return np.array(discrete_cohorts), soft_targets, adv_c, adv_s


# ==============================================================================
# 3. TRAINING PROCEDURES
# ==============================================================================

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


def train_cost_sensitive_router(router, val_x, val_pc, val_ps, soft_targets, val_cohorts, targets_tensor, lambda_spec=8.0, epochs=35, lr=2e-3):
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

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
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    is_spec = torch.tensor((val_cohorts == 0) | (val_cohorts == 1), dtype=torch.float32).unsqueeze(-1)
    is_collab = torch.tensor(val_cohorts == 0, dtype=torch.float32).unsqueeze(-1)
    pos_weight = torch.tensor([5.0])

    dataset = torch.utils.data.TensorDataset(val_x, val_pc, val_ps, is_spec, is_collab, targets_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(1, epochs + 1):
        router.train()
        for bx, bpc, bps, bspec, bcollab, btgt in loader:
            optimizer.zero_grad()
            weights, p_spec, p_c_given_spec = router.forward(bx)
            l1_loss = F.binary_cross_entropy_with_logits(router.gate_l1(bx), bspec, pos_weight=pos_weight)
            spec_mask = (bspec > 0.5).squeeze(-1)
            if spec_mask.sum() > 0:
                l2_loss = F.binary_cross_entropy(p_c_given_spec[spec_mask], bcollab[spec_mask])
            else:
                l2_loss = torch.tensor(0.0)
            
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


def train_advantage_aware_router(router, val_x, val_pc, val_ps, adv_c, adv_s, targets_tensor, epochs=35, lr=2e-3):
    """
    Trains Advantage-Aware Hierarchical Router:
      L = L_MSE(pred_A_C, A_C) + L_MSE(pred_A_S, A_S) + 0.5 * L_NLL(p_routed, target)
    """
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    dataset = torch.utils.data.TensorDataset(val_x, val_pc, val_ps, adv_c, adv_s, targets_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(1, epochs + 1):
        router.train()
        for bx, bpc, bps, badv_c, badv_s, btgt in loader:
            optimizer.zero_grad()
            weights, pred_ac, pred_as = router.forward(bx)
            
            # Regression on true continuous relative advantages
            loss_adv_c = F.mse_loss(pred_ac, badv_c)
            loss_adv_s = F.mse_loss(pred_as, badv_s)

            # End-to-end Ranking loss
            w_c = weights[:, 0:1]
            w_s = weights[:, 1:2]
            w_h = weights[:, 2:3]
            eff_c = w_c + 0.5 * w_h
            eff_s = w_s + 0.5 * w_h
            p_routed = eff_c * bpc + eff_s * bps
            log_p = torch.log(p_routed + 1e-12)
            nll_loss = F.nll_loss(log_p, btgt)

            loss = loss_adv_c + loss_adv_s + 0.5 * nll_loss
            loss.backward()
            optimizer.step()
        scheduler.step()
    return router


# ==============================================================================
# 4. EVALUATION & RIGOROUS METRICS
# ==============================================================================

def evaluate_predictions(prob_matrix: np.ndarray, targets: List[int], oracle_cohorts: np.ndarray):
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


def compute_rigorous_routing_recalls(weights: np.ndarray, oracle_cohorts: np.ndarray):
    """
    Computes mathematically rigorous Routing Recall:
      RouteRecall_C = N_{C, pred=C} / 359
      RouteRecall_S = N_{S, pred=S} / 269
      RouteRecall_Spec = (N_{C, pred=C} + N_{S, pred=S}) / 628
      RouteRecall_H = N_{H, pred=H} / 5410
    Prediction = argmax(w_C, w_S, w_H).
    """
    pred_actions = np.argmax(weights, axis=-1) # 0: C, 1: S, 2: H

    c_mask = (oracle_cohorts == 0) # N=359
    s_mask = (oracle_cohorts == 1) # N=269
    h_mask = (oracle_cohorts == 2) # N=5410

    n_c = int(np.sum(c_mask))
    n_s = int(np.sum(s_mask))
    n_h = int(np.sum(h_mask))
    n_spec = n_c + n_s # 628

    c_correct = int(np.sum((pred_actions == 0) & c_mask))
    s_correct = int(np.sum((pred_actions == 1) & s_mask))
    h_correct = int(np.sum((pred_actions == 2) & h_mask))

    route_recall_c = float(c_correct / n_c) if n_c > 0 else 0.0
    route_recall_s = float(s_correct / n_s) if n_s > 0 else 0.0
    route_recall_spec = float((c_correct + s_correct) / n_spec) if n_spec > 0 else 0.0
    route_recall_h = float(h_correct / n_h) if n_h > 0 else 0.0

    return {
        "RouteRecall_C": route_recall_c,
        "RouteRecall_S": route_recall_s,
        "RouteRecall_Spec": route_recall_spec,
        "RouteRecall_H": route_recall_h,
        "c_correct": c_correct,
        "s_correct": s_correct,
        "h_correct": h_correct
    }


def compute_mcnemar_p_value(hits_a: List[float], hits_b: List[float]) -> float:
    """Computes exact McNemar test p-value for paired binary classification / hit indicators."""
    a = np.array(hits_a, dtype=bool)
    b = np.array(hits_b, dtype=bool)
    b_only = np.sum(~a & b)
    a_only = np.sum(a & ~b)
    if b_only + a_only == 0:
        return 1.0
    # McNemar with continuity correction
    stat = (abs(b_only - a_only) - 1.0)**2 / (b_only + a_only)
    return float(1.0 - stats.chi2.cdf(stat, df=1))


# ==============================================================================
# 5. BENCHMARK EXECUTION
# ==============================================================================

def run_comprehensive_benchmark():
    print("=" * 135)
    print(" DECISIVE 5-MODEL BENCHMARK: ADVANTAGE-AWARE HIERARCHICAL ROUTER (5 SEEDS)")
    print("=" * 135)

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

    # Compute Oracle Cohorts, Targets, and Advantages
    val_cohorts, val_soft_targets, val_adv_c, val_adv_s = compute_oracle_targets_and_advantages(val_pc, val_ps, val_targets)
    test_cohorts, test_soft_targets, test_adv_c, test_adv_s = compute_oracle_targets_and_advantages(test_pc, test_ps, test_targets)

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
    models = [
        "Current 5D Router",
        "Cost-Sensitive Router",
        "Hierarchical Router",
        "Advantage-Aware Router (AAHR)"
    ]

    benchmark_records = {
        "Static 50/50": static_eval,
        "Oracle": oracle_eval,
        "Current 5D Router": {"seeds": []},
        "Cost-Sensitive Router": {"seeds": []},
        "Hierarchical Router": {"seeds": []},
        "Advantage-Aware Router (AAHR)": {"seeds": []}
    }

    print("\n" + "=" * 135)
    print(" EXECUTING 5-SEED REPLICATION ACROSS ALL LEARNED ARCHITECTURES")
    print("=" * 135)

    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)

        # 1. Standard 5D Router
        r_std = Standard5DRouter(in_features=5, hidden_dim=64, dropout=0.10)
        r_std = train_standard_router(r_std, val_x_5d, val_pc, val_ps, val_soft_targets, val_targets_tensor, epochs=35)
        r_std.eval()
        with torch.no_grad():
            t_rout_std, w_std = r_std.route(test_pc, test_ps, test_x_5d)
            ev_std = evaluate_predictions(t_rout_std.numpy(), test_targets, test_cohorts)
            rec_std = compute_rigorous_routing_recalls(w_std.numpy(), test_cohorts)
            ev_std.update(rec_std)
            benchmark_records["Current 5D Router"]["seeds"].append(ev_std)

        # 2. Cost-Sensitive Router (lambda=8.0)
        r_cs = Standard5DRouter(in_features=5, hidden_dim=64, dropout=0.10)
        r_cs = train_cost_sensitive_router(r_cs, val_x_5d, val_pc, val_ps, val_soft_targets, val_cohorts, val_targets_tensor, lambda_spec=8.0, epochs=35)
        r_cs.eval()
        with torch.no_grad():
            t_rout_cs, w_cs = r_cs.route(test_pc, test_ps, test_x_5d)
            ev_cs = evaluate_predictions(t_rout_cs.numpy(), test_targets, test_cohorts)
            rec_cs = compute_rigorous_routing_recalls(w_cs.numpy(), test_cohorts)
            ev_cs.update(rec_cs)
            benchmark_records["Cost-Sensitive Router"]["seeds"].append(ev_cs)

        # 3. Hierarchical Router
        r_hier = Hierarchical2LevelRouter(in_features=5, hidden_dim=64, dropout=0.10)
        r_hier = train_hierarchical_router(r_hier, val_x_5d, val_pc, val_ps, val_cohorts, val_targets_tensor, epochs=35)
        r_hier.eval()
        with torch.no_grad():
            t_rout_hier, w_hier = r_hier.route(test_pc, test_ps, test_x_5d)
            ev_hier = evaluate_predictions(t_rout_hier.numpy(), test_targets, test_cohorts)
            rec_hier = compute_rigorous_routing_recalls(w_hier.numpy(), test_cohorts)
            ev_hier.update(rec_hier)
            benchmark_records["Hierarchical Router"]["seeds"].append(ev_hier)

        # 4. Advantage-Aware Hierarchical Router (AAHR)
        r_aahr = AdvantageAwareHierarchicalRouter(in_features=5, hidden_dim=64, dropout=0.10, tau=0.03)
        r_aahr = train_advantage_aware_router(r_aahr, val_x_5d, val_pc, val_ps, val_adv_c, val_adv_s, val_targets_tensor, epochs=35)
        r_aahr.eval()
        with torch.no_grad():
            t_rout_aahr, w_aahr = r_aahr.route(test_pc, test_ps, test_x_5d)
            ev_aahr = evaluate_predictions(t_rout_aahr.numpy(), test_targets, test_cohorts)
            rec_aahr = compute_rigorous_routing_recalls(w_aahr.numpy(), test_cohorts)
            ev_aahr.update(rec_aahr)
            benchmark_records["Advantage-Aware Router (AAHR)"]["seeds"].append(ev_aahr)

        print(f"Seed {s:<5} | Std: {ev_std['HR@10']:.4f} | CS: {ev_cs['HR@10']:.4f} (Rec_C={rec_cs['RouteRecall_C']:.3f}, Rec_S={rec_cs['RouteRecall_S']:.3f}, Rec_Spec={rec_cs['RouteRecall_Spec']:.3f}) | Hier: {ev_hier['HR@10']:.4f} | AAHR: {ev_aahr['HR@10']:.4f} (Rec_Spec={rec_aahr['RouteRecall_Spec']:.3f})")

    # ==============================================================================
    # 6. SUMMARY TABLES & STATISTICAL TESTS
    # ==============================================================================
    print("\n" + "=" * 145)
    print(" TABLE 1: OVERALL RANKING PERFORMANCE (5 SEEDS MEAN +- STD)")
    print("=" * 145)
    print(f"{'Model':<30} | {'HR@5':<14} | {'HR@10':<14} | {'HR@20':<14} | {'NDCG@10':<14} | {'p (t-test)':<12} | {'p (Wilcoxon)':<14} | {'p (McNemar)'}")
    print("-" * 145)

    st = benchmark_records["Static 50/50"]
    print(f"{'Static 50/50':<30} | {st['HR@5']:<14.4f} | {st['HR@10']:<14.4f} | {st['HR@20']:<14.4f} | {st['NDCG@10']:<14.4f} | {'-':<12} | {'-':<14} | {'-'}")

    summary_export = {
        "Static 50/50": {k: float(v) for k, v in st.items() if not isinstance(v, list)},
        "Oracle": {k: float(v) for k, v in benchmark_records["Oracle"].items() if not isinstance(v, list)}
    }

    for m_name in models:
        s_list = benchmark_records[m_name]["seeds"]
        
        m_hr5 = f"{np.mean([x['HR@5'] for x in s_list]):.4f} +- {np.std([x['HR@5'] for x in s_list]):.4f}"
        m_hr10 = f"{np.mean([x['HR@10'] for x in s_list]):.4f} +- {np.std([x['HR@10'] for x in s_list]):.4f}"
        m_hr20 = f"{np.mean([x['HR@20'] for x in s_list]):.4f} +- {np.std([x['HR@20'] for x in s_list]):.4f}"
        m_ndcg10 = f"{np.mean([x['NDCG@10'] for x in s_list]):.4f} +- {np.std([x['NDCG@10'] for x in s_list]):.4f}"

        p_t = [stats.ttest_rel(x["hr10_per_user"], st["hr10_per_user"]).pvalue for x in s_list]
        p_w = [stats.wilcoxon(x["hr10_per_user"], st["hr10_per_user"]).pvalue for x in s_list]
        p_m = [compute_mcnemar_p_value(x["hr10_per_user"], st["hr10_per_user"]) for x in s_list]

        print(f"{m_name:<30} | {m_hr5:<14} | {m_hr10:<14} | {m_hr20:<14} | {m_ndcg10:<14} | {np.mean(p_t):<12.4f} | {np.mean(p_w):<14.4f} | {np.mean(p_m):.4f}")

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
            "RouteRecall_C": float(np.mean([x['RouteRecall_C'] for x in s_list])),
            "RouteRecall_S": float(np.mean([x['RouteRecall_S'] for x in s_list])),
            "RouteRecall_Spec": float(np.mean([x['RouteRecall_Spec'] for x in s_list])),
            "RouteRecall_H": float(np.mean([x['RouteRecall_H'] for x in s_list])),
            "p_val_ttest_vs_static": float(np.mean(p_t)),
            "p_val_wilcoxon_vs_static": float(np.mean(p_w)),
            "p_val_mcnemar_vs_static": float(np.mean(p_m))
        }

    ora = benchmark_records["Oracle"]
    print(f"{'Oracle (Upper Bound)':<30} | {ora['HR@5']:<14.4f} | {ora['HR@10']:<14.4f} | {ora['HR@20']:<14.4f} | {ora['NDCG@10']:<14.4f} | {'-':<12} | {'-':<14} | {'-'}")
    print("=" * 145)

    print("\n" + "=" * 145)
    print(" TABLE 2: COHORT PERFORMANCE & RIGOROUS ROUTING RECALL BREAKDOWN")
    print("=" * 145)
    print(f"{'Model':<30} | {'HR@10 (C)':<12} | {'HR@10 (S)':<12} | {'HR@10 (H)':<12} | {'RouteRecall_C':<15} | {'RouteRecall_S':<15} | {'RouteRecall_Spec':<18} | {'RouteRecall_H'}")
    print("-" * 145)

    print(f"{'Static 50/50':<30} | {st['HR@10_C']:<12.4f} | {st['HR@10_S']:<12.4f} | {st['HR@10_H']:<12.4f} | {'0.0000':<15} | {'0.0000':<15} | {'0.0000':<18} | {'1.0000'}")

    for m_name in models:
        s_list = benchmark_records[m_name]["seeds"]
        m_hrc = f"{np.mean([x['HR@10_C'] for x in s_list]):.4f}"
        m_hrs = f"{np.mean([x['HR@10_S'] for x in s_list]):.4f}"
        m_hrh = f"{np.mean([x['HR@10_H'] for x in s_list]):.4f}"
        m_rc = f"{np.mean([x['RouteRecall_C'] for x in s_list]):.4f}"
        m_rs = f"{np.mean([x['RouteRecall_S'] for x in s_list]):.4f}"
        m_rspec = f"{np.mean([x['RouteRecall_Spec'] for x in s_list]):.4f}"
        m_rh = f"{np.mean([x['RouteRecall_H'] for x in s_list]):.4f}"
        print(f"{m_name:<30} | {m_hrc:<12} | {m_hrs:<12} | {m_hrh:<12} | {m_rc:<15} | {m_rs:<15} | {m_rspec:<18} | {m_rh}")

    print(f"{'Oracle (Upper Bound)':<30} | {ora['HR@10_C']:<12.4f} | {ora['HR@10_S']:<12.4f} | {ora['HR@10_H']:<12.4f} | {'1.0000':<15} | {'1.0000':<15} | {'1.0000':<18} | {'1.0000'}")
    print("=" * 145)

    with open("results/advantage_aware_benchmark.json", "w") as f:
        json.dump(summary_export, f, indent=2)
    print("\nSaved complete benchmark results to results/advantage_aware_benchmark.json")

if __name__ == "__main__":
    run_comprehensive_benchmark()
