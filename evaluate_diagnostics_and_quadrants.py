"""
Computes in-depth diagnostic calibrations, quartile risk curves, and 4-quadrant disagreement tables
for the Dual-Expert Selective Evidence Routing Framework.
"""

import math
import json
import torch
import numpy as np

from models.evidence_analyzer import EvidenceAnalyzer
from models.selective_router import SelectiveRouter
from models.dual_expert_recsys import CollaborativeExpert, SemanticExpert
from train_and_evaluate_evidence_routing import load_and_preprocess_movielens, extract_dataset_evidence_and_oracle

device = torch.device("cpu")

def compute_extended_diagnostics():
    print("=" * 80)
    print(" COMPUTING EXTENDED DIAGNOSTICS & 4-QUADRANT EVIDENCE ANALYSIS")
    print("=" * 80)

    # Load dataset & models
    train_data, val_data, test_data, num_items, semantic_table = load_and_preprocess_movielens()

    collab_expert = CollaborativeExpert(num_items=num_items, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    semantic_expert = SemanticExpert(num_items=num_items, semantic_table=semantic_table, hidden_dim=64, max_seq_len=50, dropout=0.2).to(device)
    selective_router = SelectiveRouter(in_features=8, hidden_dim=32, num_actions=3, dropout=0.10).to(device)

    collab_expert.load_state_dict(torch.load("checkpoints/collab_expert.pt", map_location=device))
    semantic_expert.load_state_dict(torch.load("checkpoints/semantic_expert.pt", map_location=device))
    selective_router.load_state_dict(torch.load("checkpoints/best_selective_router.pt", map_location=device))

    evidence_analyzer = EvidenceAnalyzer()

    # Extract test evidence
    test_features, _, test_pc, test_ps, test_ph, test_targets = extract_dataset_evidence_and_oracle(
        collab_expert=collab_expert,
        semantic_expert=semantic_expert,
        evidence_analyzer=evidence_analyzer,
        eval_dict=test_data,
        semantic_table=semantic_table,
        num_mc_samples=5
    )

    N = len(test_targets)
    targets_np = np.array(test_targets) - 1 # 0-indexed

    pc_np = test_pc.numpy()
    ps_np = test_ps.numpy()
    ph_np = test_ph.numpy()

    p_routed, actions, probs = selective_router.route(test_pc, test_ps, test_ph, test_features)
    pr_np = p_routed.detach().numpy()

    uc_np = test_features[:, 0].numpy()
    us_np = test_features[:, 1].numpy()
    djs_np = test_features[:, 2].numpy()

    # Compute ranks
    ranks_c = np.array([int(np.sum(pc_np[i] >= pc_np[i, targets_np[i]])) for i in range(N)])
    ranks_s = np.array([int(np.sum(ps_np[i] >= ps_np[i, targets_np[i]])) for i in range(N)])
    ranks_r = np.array([int(np.sum(pr_np[i] >= pr_np[i, targets_np[i]])) for i in range(N)])

    hits_c = ranks_c <= 10
    hits_s = ranks_s <= 10
    hits_r = ranks_r <= 10

    # 1. Validation: U_c up => Collab error up; U_s up => Semantic error up
    mean_uc_hit_c = float(np.mean(uc_np[hits_c]))
    mean_uc_miss_c = float(np.mean(uc_np[~hits_c]))

    mean_us_hit_s = float(np.mean(us_np[hits_s]))
    mean_us_miss_s = float(np.mean(us_np[~hits_s]))

    # Quartile analysis of U_c vs Collaborative Accuracy
    uc_q25, uc_q50, uc_q75 = np.percentile(uc_np, [25, 50, 75])
    q1_c = hits_c[uc_np <= uc_q25]
    q2_c = hits_c[(uc_np > uc_q25) & (uc_np <= uc_q50)]
    q3_c = hits_c[(uc_np > uc_q50) & (uc_np <= uc_q75)]
    q4_c = hits_c[uc_np > uc_q75]

    uc_quartiles = [
        {"Quartile": "Q1 (Lowest Uc)", "Mean Uc": round(float(np.mean(uc_np[uc_np <= uc_q25])), 5), "Collab HR@10": round(float(np.mean(q1_c)), 4)},
        {"Quartile": "Q2 (Low-Mid Uc)", "Mean Uc": round(float(np.mean(uc_np[(uc_np > uc_q25) & (uc_np <= uc_q50)])), 5), "Collab HR@10": round(float(np.mean(q2_c)), 4)},
        {"Quartile": "Q3 (Mid-High Uc)", "Mean Uc": round(float(np.mean(uc_np[(uc_np > uc_q50) & (uc_np <= uc_q75)])), 5), "Collab HR@10": round(float(np.mean(q3_c)), 4)},
        {"Quartile": "Q4 (Highest Uc)", "Mean Uc": round(float(np.mean(uc_np[uc_np > uc_q75])), 5), "Collab HR@10": round(float(np.mean(q4_c)), 4)},
    ]

    # 2. Four-Quadrant Analysis of (U_c, U_s)
    med_uc = float(np.median(uc_np))
    med_us = float(np.median(us_np))

    quad1_mask = (uc_np <= med_uc) & (us_np <= med_us) # Low Uc, Low Us (Both Reliable)
    quad2_mask = (uc_np > med_uc) & (us_np <= med_us)  # High Uc, Low Us (Trust Semantic)
    quad3_mask = (uc_np <= med_uc) & (us_np > med_us)  # Low Uc, High Us (Trust Collaborative)
    quad4_mask = (uc_np > med_uc) & (us_np > med_us)   # High Uc, High Us (Neither Reliable / Conflict)

    def analyze_quadrant(mask, name, interp):
        sub_n = int(np.sum(mask))
        if sub_n == 0:
            return {}
        return {
            "Quadrant": name,
            "Users": sub_n,
            "Percentage": round((sub_n / N) * 100.0, 1),
            "Mean D_JS": round(float(np.mean(djs_np[mask])), 4),
            "Collab HR@10": round(float(np.mean(hits_c[mask])), 4),
            "Semantic HR@10": round(float(np.mean(hits_s[mask])), 4),
            "Router HR@10": round(float(np.mean(hits_r[mask])), 4),
            "Interpretation": interp
        }

    quadrants = [
        analyze_quadrant(quad1_mask, "Q1: Low Uc, Low Us", "Mutual High Confidence (Fusion Optimal)"),
        analyze_quadrant(quad2_mask, "Q2: High Uc, Low Us", "Collaborative Drift / Trust Semantic Expert"),
        analyze_quadrant(quad3_mask, "Q3: Low Uc, High Us", "Semantic Ambiguity / Trust Collaborative Expert"),
        analyze_quadrant(quad4_mask, "Q4: High Uc, High Us", "Mutual Epistemic Conflict / High Risk"),
    ]

    # 3. Cold Start Router Action Breakdown (% C, % S, % H)
    cold_lengths = [1, 2, 3, 5, 10, 20]
    cold_action_breakdown = []
    uids = list(test_data.keys())

    for L in cold_lengths:
        valid_uids = [u for u in uids if len(test_data[u][0]) >= L + 1][:1500]
        seqs_list = []
        for u in valid_uids:
            prefix_seq = test_data[u][0][:L]
            pad_len = 50 - len(prefix_seq)
            seqs_list.append([0] * pad_len + prefix_seq)
        seqs_t = torch.tensor(seqs_list, dtype=torch.long, device=device)

        with torch.no_grad():
            mc_c, _ = collab_expert.mc_predict(seqs_t, num_samples=3)
            mc_s, _ = semantic_expert.mc_predict(seqs_t, num_samples=3)
            x_u, p_c, p_s = evidence_analyzer.extract_evidence_features(mc_c, mc_s, seqs_t, semantic_table)
            p_h = 0.5 * (p_c + p_s)
            _, act, _ = selective_router.route(p_c, p_s, p_h, x_u)

        counts = torch.bincount(act, minlength=3).tolist()
        tot = len(valid_uids)
        cold_action_breakdown.append({
            "History Length L": L,
            "Collab (%)": round((counts[0] / tot) * 100.0, 1),
            "Semantic (%)": round((counts[1] / tot) * 100.0, 1),
            "Hybrid (%)": round((counts[2] / tot) * 100.0, 1),
        })

    # Load existing results to enrich
    with open("evidence_routing_empirical_results.json", "r") as f:
        existing_data = json.load(f)

    existing_data["extended_diagnostics"] = {
        "uncertainty_validation": {
            "Mean Uc on Collab Hits": round(mean_uc_hit_c, 5),
            "Mean Uc on Collab Misses": round(mean_uc_miss_c, 5),
            "Mean Us on Semantic Hits": round(mean_us_hit_s, 5),
            "Mean Us on Semantic Misses": round(mean_us_miss_s, 5),
            "Uc_Quartile_Curve": uc_quartiles
        },
        "four_quadrant_evidence_analysis": quadrants,
        "cold_start_action_distribution": cold_action_breakdown
    }

    with open("evidence_routing_empirical_results.json", "w") as f:
        json.dump(existing_data, f, indent=2)

    print("\n--- Uncertainty Validation ---")
    print(f"Collab Expert: Mean Uc on Hits = {mean_uc_hit_c:.5f} vs Misses = {mean_uc_miss_c:.5f}")
    print(f"Semantic Expert: Mean Us on Hits = {mean_us_hit_s:.5f} vs Misses = {mean_us_miss_s:.5f}")
    print("\n--- 4-Quadrant Breakdown ---")
    for q in quadrants:
        print(f"{q['Quadrant']:<25} | Users: {q['Users']} ({q['Percentage']}%) | D_JS: {q['Mean D_JS']} | Collab: {q['Collab HR@10']} | Sem: {q['Semantic HR@10']} | Router: {q['Router HR@10']}")
    print("\n--- Cold Start Router Actions ---")
    for c in cold_action_breakdown:
        print(f"L={c['History Length L']:02d} | Collab: {c['Collab (%)']}% | Semantic: {c['Semantic (%)']}% | Hybrid: {c['Hybrid (%)']}%")

if __name__ == "__main__":
    compute_extended_diagnostics()
