"""
Decisive Statistical Evaluation Script.

Computes exact:
1. Global comparisons: Router vs SASRec, Semantic, Static Fusion, Uncertainty Gate, Oracle
   - HR@10, NDCG@10, Delta, Paired t-test (t, p), Wilcoxon signed-rank (W, p), Cohen's d
2. Stratified analysis across sequence length regimes:
   - L <= 3 (Cold-start / Short)
   - 4 <= L <= 9 (Medium)
   - L >= 10 (Long / Saturated)
   - L = 20 (Deep History)
   Reporting user count N, HR@10, NDCG@10, Delta, and exact p-values for each cohort.
"""

import os
import sys
import math
import json
import torch
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_and_evaluate_evidence_routing import load_and_preprocess_movielens
from models.dual_expert_recsys import CollaborativeExpert, SemanticExpert
from models.evidence_analyzer import EvidenceAnalyzer
from models.selective_router import SelectiveRouter
from optimize_router_ranking import RankAwareSelectiveRouter

device = torch.device("cpu")

def cohens_d(x, y):
    """Calculates Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)

def compute_all_metrics():
    train_data, val_data, test_data, num_items, semantic_table = load_and_preprocess_movielens()
    
    # Load trained models
    collab_expert = CollaborativeExpert(num_items=num_items, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    semantic_expert = SemanticExpert(num_items=num_items, semantic_table=semantic_table, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    collab_expert.load_state_dict(torch.load("checkpoints/collab_expert.pt", map_location=device))
    semantic_expert.load_state_dict(torch.load("checkpoints/semantic_expert.pt", map_location=device))
    collab_expert.eval()
    semantic_expert.eval()

    evidence_analyzer = EvidenceAnalyzer()

    from train_and_evaluate_evidence_routing import extract_dataset_evidence_and_oracle
    test_x, test_oracle, test_pc, test_ps, test_ph, test_targets = extract_dataset_evidence_and_oracle(
        collab_expert, semantic_expert, evidence_analyzer, test_data, semantic_table, num_mc_samples=5
    )

    router = SelectiveRouter(in_features=8, hidden_dim=64, num_actions=3, dropout=0.10).to(device)
    if os.path.exists("checkpoints/best_selective_router.pt"):
        router.load_state_dict(torch.load("checkpoints/best_selective_router.pt", map_location=device))
    router.eval()

    # Re-route test
    with torch.no_grad():
        test_r, _, _ = router.route(test_pc, test_ps, test_ph, test_x, hard_selection=False)
        p_routed_np = test_r.numpy()
        p_c_np = test_pc.numpy()
        p_s_np = test_ps.numpy()
        p_h_np = test_ph.numpy()

    # Uncertainty Gate
    u_c = test_x[:, 0:1]
    u_gate_w = torch.sigmoid(2.0 * torch.log1p(u_c) - 1.0)
    p_unc_np = ((1.0 - u_gate_w) * test_pc + u_gate_w * test_ps).numpy()

    N = len(test_targets)
    seq_lens = [len(test_data[u][0]) for u in test_data.keys()]

    # Per-user hit and ndcg lists
    def get_user_scores(p_dist):
        hits, ndcgs = [], []
        for u in range(N):
            tgt_idx = test_targets[u] - 1
            rank = int(np.sum(p_dist[u] >= p_dist[u, tgt_idx]))
            hits.append(1.0 if rank <= 10 else 0.0)
            ndcgs.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
        return np.array(hits), np.array(ndcgs)

    # Oracle per user
    oracle_hits, oracle_ndcgs = [], []
    for u in range(N):
        tgt_idx = test_targets[u] - 1
        rc = int(np.sum(p_c_np[u] >= p_c_np[u, tgt_idx]))
        rs = int(np.sum(p_s_np[u] >= p_s_np[u, tgt_idx]))
        rh = int(np.sum(p_h_np[u] >= p_h_np[u, tgt_idx]))
        best_r = min(rc, rs, rh)
        oracle_hits.append(1.0 if best_r <= 10 else 0.0)
        oracle_ndcgs.append(1.0 / math.log2(best_r + 1) if best_r <= 10 else 0.0)
    oracle_hits = np.array(oracle_hits)
    oracle_ndcgs = np.array(oracle_ndcgs)

    r_hits, r_ndcgs = get_user_scores(p_routed_np)
    c_hits, c_ndcgs = get_user_scores(p_c_np)
    s_hits, s_ndcgs = get_user_scores(p_s_np)
    h_hits, h_ndcgs = get_user_scores(p_h_np)
    unc_hits, unc_ndcgs = get_user_scores(p_unc_np)

    comparisons = [
        ("Router vs SASRec", r_hits, c_hits, r_ndcgs, c_ndcgs),
        ("Router vs Semantic", r_hits, s_hits, r_ndcgs, s_ndcgs),
        ("Router vs Static Fusion", r_hits, h_hits, r_ndcgs, h_ndcgs),
        ("Router vs Uncertainty Gate", r_hits, unc_hits, r_ndcgs, unc_ndcgs),
        ("Router vs Oracle", r_hits, oracle_hits, r_ndcgs, oracle_ndcgs),
    ]

    print("\n" + "=" * 105)
    print(f" TABLE 1: DECISIVE GLOBAL STATISTICAL EVALUATION (N = {N:,} TEST USERS, FULL CATALOG = {num_items:,} ITEMS)")
    print("=" * 105)
    print(f"{'Comparison':<28} | {'HR@10 (A vs B)':<17} | {'Delta HR':<9} | {'t-stat':<8} | {'p-value (t)':<12} | {'Wilcoxon p':<12} | {'Cohen d (HR)':<12}")
    print("-" * 105)

    for name, a_h, b_h, a_n, b_n in comparisons:
        mean_a = float(np.mean(a_h))
        mean_b = float(np.mean(b_h))
        delta = mean_a - mean_b
        
        if "Oracle" in name:
            t_s, p_t, p_w, d = "-", "-", "-", "-"
        else:
            t_res = stats.ttest_rel(a_h, b_h)
            t_s = f"{t_res.statistic:.3f}"
            p_t = f"{t_res.pvalue:.2e}" if t_res.pvalue >= 1e-4 else f"{t_res.pvalue:.1e}"
            
            try:
                w_res = stats.wilcoxon(a_h, b_h)
                p_w = f"{w_res.pvalue:.2e}" if w_res.pvalue >= 1e-4 else f"{w_res.pvalue:.1e}"
            except Exception:
                p_w = "n/a"
            d = f"{cohens_d(a_h, b_h):.4f}"

        delta_str = f"{delta:+.4f}"
        comp_str = f"{mean_a:.4f} vs {mean_b:.4f}"
        print(f"{name:<28} | {comp_str:<17} | {delta_str:<9} | {t_s:<8} | {p_t:<12} | {p_w:<12} | {d:<12}")

    # Stratified analysis by Sequence Length Regimes
    regimes = [
        ("Short Sequences (L <= 3)", [i for i, l in enumerate(seq_lens) if l <= 3]),
        ("Medium Sequences (4 <= L <= 9)", [i for i, l in enumerate(seq_lens) if 4 <= l <= 9]),
        ("Long Sequences (L >= 10)", [i for i, l in enumerate(seq_lens) if l >= 10]),
        ("Deep Saturated (L >= 20)", [i for i, l in enumerate(seq_lens) if l >= 20]),
    ]

    print("\n" + "=" * 105)
    print(" TABLE 2: STRATIFIED REGIME ANALYSIS BY SEQUENCE LENGTH (ROUTER VS SASREC & STATIC FUSION)")
    print("=" * 105)
    print(f"{'Regime Cohort':<30} | {'Users (N)':<10} | {'SASRec':<8} | {'Semantic':<8} | {'Static':<8} | {'Router':<8} | {'Delta vs SAS':<12} | {'p-val (vs SAS)':<14}")
    print("-" * 105)

    for r_name, idxs in regimes:
        sub_n = len(idxs)
        if sub_n == 0:
            continue
        sub_r_h = r_hits[idxs]
        sub_c_h = c_hits[idxs]
        sub_s_h = s_hits[idxs]
        sub_h_h = h_hits[idxs]

        m_r = float(np.mean(sub_r_h))
        m_c = float(np.mean(sub_c_h))
        m_s = float(np.mean(sub_s_h))
        m_h = float(np.mean(sub_h_h))

        delta_sas = m_r - m_c
        t_res = stats.ttest_rel(sub_r_h, sub_c_h)
        p_val_str = f"{t_res.pvalue:.4f}" if t_res.pvalue >= 1e-4 else f"{t_res.pvalue:.2e}"
        delta_sas_str = f"{delta_sas:+.4f}"

        print(f"{r_name:<30} | {sub_n:<10} | {m_c:<8.4f} | {m_s:<8.4f} | {m_h:<8.4f} | {m_r:<8.4f} | {delta_sas_str:<12} | {p_val_str:<14}")

    print("\n" + "=" * 105)

if __name__ == "__main__":
    compute_all_metrics()
