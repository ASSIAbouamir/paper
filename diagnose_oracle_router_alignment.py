"""
Decision Alignment & Oracle Error Diagnostic Matrix.

Computes exact:
1. 3x3 Confusion Matrix: Oracle Action vs. Router Prediction (Counts & %)
2. Conditional Performance: HR@10 of Router, Static Fusion, SASRec, Semantic inside each cell
3. Misrouting Impact Analysis: Where does the Router lose ground vs. Oracle?
   - False Hybrids: Oracle=C or S, but Router=H
   - False Specialists: Oracle=H, but Router=C or S
   - Cross-Specialist Errors: Oracle=C, Router=S (or vice-versa)
4. Parsimonious Evidence Minimalist Test:
   - A. Static Fusion (50/50)
   - B. Minimal Margin Router: [L, M_c, M_s] (3 features)
   - C. Margin + Dynamics: [L, Unique, Drift, M_c, M_s] (5 features)
   - D. Full 13D Evidence Router
"""

import os
import sys
import math
import json
import torch
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_and_evaluate_evidence_routing import load_and_preprocess_movielens
from models.dual_expert_recsys import CollaborativeExpert, SemanticExpert
from run_enhanced_ablation_suite import EnrichedEvidenceExtractor, ContinuousSoftEvidenceRouter, train_enhanced_router

device = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def run_oracle_router_diagnostic():
    print("=" * 95)
    print(" ORACLE VS. ROUTER DECISION ALIGNMENT & ERROR DIAGNOSTIC MATRIX")
    print("=" * 95)

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

    val_x, val_pc, val_ps, val_targets = extract_split(val_data)
    test_x, test_pc, test_ps, test_targets = extract_split(test_data)
    N_test = len(test_targets)

    # Train Full 13D Router
    router = ContinuousSoftEvidenceRouter(in_features=13, hidden_dim=64, dropout=0.10).to(device)
    router = train_enhanced_router(router, val_x, val_pc, val_ps, val_targets, epochs=35, lr=2e-3)
    router.eval()

    with torch.no_grad():
        test_routed, test_weights = router.route(test_pc, test_ps, test_x)
        p_r_np = test_routed.numpy()
        w_np = test_weights.numpy()

    p_c_np = test_pc.numpy()
    p_s_np = test_ps.numpy()
    p_h_np = (0.5 * (test_pc + test_ps)).numpy()

    # Determine Oracle actions and Router predicted discrete actions
    oracle_actions = []
    router_actions = []

    for u in range(N_test):
        tgt_idx = test_targets[u] - 1
        rc = int(np.sum(p_c_np[u] >= p_c_np[u, tgt_idx]))
        rs = int(np.sum(p_s_np[u] >= p_s_np[u, tgt_idx]))
        rh = int(np.sum(p_h_np[u] >= p_h_np[u, tgt_idx]))

        yc = 1.0 / math.log2(rc + 1) if rc <= 20 else 0.0
        ys = 1.0 / math.log2(rs + 1) if rs <= 20 else 0.0
        yh = 1.0 / math.log2(rh + 1) if rh <= 20 else 0.0

        if yh >= max(yc, ys) and yh > 0:
            act_o = 2 # Hybrid
        elif yc >= ys and yc > 0:
            act_o = 0 # Collaborative
        elif ys > 0:
            act_o = 1 # Semantic
        else:
            act_o = 2 # Default hybrid if all miss

        act_r = int(np.argmax(w_np[u])) # 0: C, 1: S, 2: H

        oracle_actions.append(act_o)
        router_actions.append(act_r)

    oracle_actions = np.array(oracle_actions)
    router_actions = np.array(router_actions)

    action_names = ["Collaborative (C)", "Semantic (S)", "Hybrid (H)"]

    # 1. Compute 3x3 Decision Alignment Matrix
    confusion_matrix = np.zeros((3, 3), dtype=int)
    for o_act, r_act in zip(oracle_actions, router_actions):
        confusion_matrix[o_act, r_act] += 1

    print("\n" + "-" * 80)
    print(" 1. DECISION ALIGNMENT MATRIX: ORACLE (ROWS) VS ROUTER PREDICTION (COLUMNS)")
    print("-" * 80)
    print(f"{'Oracle True Action':<25} | {'Pred C':<12} | {'Pred S':<12} | {'Pred H':<12} | {'Total Oracle':<14}")
    print("-" * 80)
    for i in range(3):
        row_tot = np.sum(confusion_matrix[i])
        pct_str = f"{row_tot} ({row_tot/N_test*100:.1f}%)"
        print(f"{action_names[i]:<25} | {confusion_matrix[i, 0]:<12} | {confusion_matrix[i, 1]:<12} | {confusion_matrix[i, 2]:<12} | {pct_str:<14}")
    print("-" * 80)
    col_tots = [np.sum(confusion_matrix[:, j]) for j in range(3)]
    print(f"{'Total Router Predicted':<25} | {col_tots[0]:<12} | {col_tots[1]:<12} | {col_tots[2]:<12} | {N_test:<14}")
    print("-" * 80)

    # 2. Performance Breakdown Inside Each Oracle Cohort
    print("\n" + "-" * 95)
    print(" 2. CONDITIONAL PERFORMANCE PER ORACLE SUB-POPULATION (WHERE DOES ROUTER SUCCEED / FAIL?)")
    print("-" * 95)
    print(f"{'Oracle Sub-Population':<26} | {'Count (N)':<10} | {'SASRec':<8} | {'Semantic':<8} | {'Static (50/50)':<14} | {'Router':<8} | {'Oracle':<8}")
    print("-" * 95)

    for i in range(3):
        idxs = np.where(oracle_actions == i)[0]
        n_sub = len(idxs)
        if n_sub == 0:
            continue
        
        # Compute HR@10 inside cohort
        h_c = [1.0 if int(np.sum(p_c_np[u] >= p_c_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in idxs]
        h_s = [1.0 if int(np.sum(p_s_np[u] >= p_s_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in idxs]
        h_h = [1.0 if int(np.sum(p_h_np[u] >= p_h_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in idxs]
        h_r = [1.0 if int(np.sum(p_r_np[u] >= p_r_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in idxs]
        
        # Oracle hits
        h_o = []
        for u in idxs:
            rc = int(np.sum(p_c_np[u] >= p_c_np[u, test_targets[u]-1]))
            rs = int(np.sum(p_s_np[u] >= p_s_np[u, test_targets[u]-1]))
            rh = int(np.sum(p_h_np[u] >= p_h_np[u, test_targets[u]-1]))
            h_o.append(1.0 if min(rc, rs, rh) <= 10 else 0.0)

        print(f"{action_names[i]:<26} | {n_sub:<10} | {np.mean(h_c):<8.4f} | {np.mean(h_s):<8.4f} | {np.mean(h_h):<14.4f} | {np.mean(h_r):<8.4f} | {np.mean(h_o):<8.4f}")
    print("-" * 95)

    # 3. Parsimonious Evidence Minimalist Test
    # Feature indices:
    # 0: U_c, 1: U_s, 2: D_JS, 3: H_c, 4: H_s, 5: Delta_H, 6: norm_len, 7: unique, 8: drift, 9: margin_c, 10: margin_s, 11: agreement, 12: raw_len
    parsimonious_configs = [
        ("A. Static Fusion (50/50 Baseline)", None, 0),
        ("B. Minimal Margin Router: [L, M_c, M_s]", [6, 9, 10], 3),
        ("C. Margin + Dynamics: [L, Uniq, Drift, M_c, M_s]", [6, 7, 8, 9, 10], 5),
        ("D. Full 13D Evidence Router", list(range(13)), 13),
    ]

    print("\n" + "=" * 95)
    print(" 3. PARSIMONIOUS EVIDENCE TEST: MINIMAL EVIDENCE FOR ADAPTIVE ROUTING")
    print("=" * 95)
    print(f"{'Configuration':<46} | {'Features':<10} | {'HR@10':<8} | {'NDCG@10':<8} | {'Delta vs Static'}")
    print("-" * 95)

    static_hits = [1.0 if int(np.sum(p_h_np[u] >= p_h_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in range(N_test)]
    static_hr10 = float(np.mean(static_hits))

    for name, mask, feat_dim in parsimonious_configs:
        if mask is None:
            ndcgs = [1.0 / math.log2(int(np.sum(p_h_np[u] >= p_h_np[u, test_targets[u]-1])) + 1) 
                     if int(np.sum(p_h_np[u] >= p_h_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in range(N_test)]
            print(f"{name:<46} | {'0 (None)':<10} | {static_hr10:<8.4f} | {float(np.mean(ndcgs)):<8.4f} | {'Baseline'}")
            continue

        r_mod = ContinuousSoftEvidenceRouter(in_features=feat_dim, hidden_dim=64, dropout=0.10).to(device)
        r_mod = train_enhanced_router(r_mod, val_x, val_pc, val_ps, val_targets, feature_mask=mask, epochs=35, lr=2e-3)
        r_mod.eval()

        with torch.no_grad():
            sub_x = test_x[:, mask]
            t_rout, _ = r_mod.route(test_pc, test_ps, sub_x)
            t_rout_np = t_rout.numpy()

        r_h = [1.0 if int(np.sum(t_rout_np[u] >= t_rout_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in range(N_test)]
        r_n = [1.0 / math.log2(int(np.sum(t_rout_np[u] >= t_rout_np[u, test_targets[u]-1])) + 1) 
               if int(np.sum(t_rout_np[u] >= t_rout_np[u, test_targets[u]-1])) <= 10 else 0.0 for u in range(N_test)]

        hr_m = float(np.mean(r_h))
        ndcg_m = float(np.mean(r_n))
        delta_s = hr_m - static_hr10
        print(f"{name:<46} | {str(feat_dim)+' dims':<10} | {hr_m:<8.4f} | {ndcg_m:<8.4f} | {delta_s:+.4f}")

    print("\n" + "=" * 95)

if __name__ == "__main__":
    run_oracle_router_diagnostic()
