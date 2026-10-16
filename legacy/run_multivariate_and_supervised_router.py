"""
Empirical Script to Address Key Reviewer Concerns:
1. Oracle-RR vs Oracle-HR@10 equivalence proof and empirical validation.
2. Multivariate C vs S separability test (Logistic Regression, MLP, Random Forest/Gradient Boosting) with 5-fold Stratified CV.
3. Supervised Action Router directly trained on Oracle labels (H1 vs H2 arbitration).
4. Functional 1D mixture analysis: Eff_C distribution and metric impact.
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
from run_enhanced_ablation_suite import EnrichedEvidenceExtractor

device = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def main():
    print("=" * 80)
    print(" EMPIRICAL INVESTIGATION: ORACLE, MULTIVARIATE SEPARABILITY & SUPERVISED ROUTING")
    print("=" * 80)

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

    print("[1/4] Extracting validation and test evidence features & expert distributions...")
    val_x, val_pc, val_ps, val_targets = extract_split(val_data)
    test_x, test_pc, test_ps, test_targets = extract_split(test_data)
    N_test = len(test_targets)
    N_val = len(val_targets)

    val_ph = 0.5 * (val_pc + val_ps)
    test_ph = 0.5 * (test_pc + test_ps)

    # -------------------------------------------------------------
    # 1. ORACLE-RR vs ORACLE-HR@10 VALIDATION
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [EXPERIMENT 1] ORACLE-RR VS. ORACLE-HR@10 EMPIRICAL PROOF")
    print("-" * 70)

    p_c_np = test_pc.numpy()
    p_s_np = test_ps.numpy()
    p_h_np = test_ph.numpy()

    oracle_rr_hits_10 = []
    oracle_hr10_hits_10 = []
    oracle_rr_hits_5 = []
    oracle_hr5_hits_5 = []
    oracle_rr_hits_20 = []
    oracle_hr20_hits_20 = []
    oracle_actions = []

    mismatches_10 = 0
    mismatches_5 = 0
    mismatches_20 = 0

    for u in range(N_test):
        tgt_idx = test_targets[u] - 1
        rc = int(np.sum(p_c_np[u] >= p_c_np[u, tgt_idx]))
        rs = int(np.sum(p_s_np[u] >= p_s_np[u, tgt_idx]))
        rh = int(np.sum(p_h_np[u] >= p_h_np[u, tgt_idx]))

        # RR Oracle action
        rr_c, rr_s, rr_h = 1.0 / rc, 1.0 / rs, 1.0 / rh
        if rr_h >= max(rr_c, rr_s):
            best_act = "H"
            best_rank = rh
        elif rr_c > rr_s:
            best_act = "C"
            best_rank = rc
        else:
            best_act = "S"
            best_rank = rs
        oracle_actions.append(best_act)

        # Hit@K from RR Oracle
        hit_rr_10 = 1.0 if best_rank <= 10 else 0.0
        hit_rr_5 = 1.0 if best_rank <= 5 else 0.0
        hit_rr_20 = 1.0 if best_rank <= 20 else 0.0

        # Hit@K from True Max-Hit Oracle
        hit_hr_10 = max(1.0 if rc <= 10 else 0.0, 1.0 if rs <= 10 else 0.0, 1.0 if rh <= 10 else 0.0)
        hit_hr_5 = max(1.0 if rc <= 5 else 0.0, 1.0 if rs <= 5 else 0.0, 1.0 if rh <= 5 else 0.0)
        hit_hr_20 = max(1.0 if rc <= 20 else 0.0, 1.0 if rs <= 20 else 0.0, 1.0 if rh <= 20 else 0.0)

        if hit_rr_10 != hit_hr_10:
            mismatches_10 += 1
        if hit_rr_5 != hit_hr_5:
            mismatches_5 += 1
        if hit_rr_20 != hit_hr_20:
            mismatches_20 += 1

        oracle_rr_hits_10.append(hit_rr_10)
        oracle_hr10_hits_10.append(hit_hr_10)
        oracle_rr_hits_5.append(hit_rr_5)
        oracle_hr5_hits_5.append(hit_hr_5)
        oracle_rr_hits_20.append(hit_rr_20)
        oracle_hr20_hits_20.append(hit_hr_20)

    mean_rr_hr10 = np.mean(oracle_rr_hits_10)
    mean_true_hr10 = np.mean(oracle_hr10_hits_10)
    mean_rr_hr5 = np.mean(oracle_rr_hits_5)
    mean_true_hr5 = np.mean(oracle_hr5_hits_5)
    mean_rr_hr20 = np.mean(oracle_rr_hits_20)
    mean_true_hr20 = np.mean(oracle_hr20_hits_20)

    print(f"Total Test Users Evaluated: {N_test}")
    print(f"HR@10 - Oracle(RR): {mean_rr_hr10:.4f} | Oracle(HR@10): {mean_true_hr10:.4f} | Mismatches: {mismatches_10}/{N_test}")
    print(f"HR@5  - Oracle(RR): {mean_rr_hr5:.4f} | Oracle(HR@5):  {mean_true_hr5:.4f} | Mismatches: {mismatches_5}/{N_test}")
    print(f"HR@20 - Oracle(RR): {mean_rr_hr20:.4f} | Oracle(HR@20): {mean_true_hr20:.4f} | Mismatches: {mismatches_20}/{N_test}")
    print(f"-> Conclusion: Oracle-RR is IDENTICAL to True Max-Hit Oracle at all cutoffs (0 mismatches across {N_test} instances).")

    # -------------------------------------------------------------
    # 2. MULTIVARIATE SEPARABILITY TEST: C vs. S
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [EXPERIMENT 2] MULTIVARIATE SEPARABILITY OF C VS. S (5-FOLD CV)")
    print("-" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    # Extract C and S instances from test set
    spec_indices = [u for u, act in enumerate(oracle_actions) if act in ["C", "S"]]
    X_spec = test_x[spec_indices].numpy()
    y_spec = np.array([1 if oracle_actions[u] == "C" else 0 for u in spec_indices]) # 1 = C, 0 = S
    
    n_c = sum(y_spec == 1)
    n_s = sum(y_spec == 0)
    print(f"Specialist Cohort: N = {len(y_spec)} ({n_c} Collaborative, {n_s} Semantic)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=SEED),
        "MLP (2-layer: 32-32)": MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, early_stopping=True, random_state=SEED)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    multivariate_results = {}

    for name, clf in models.items():
        aucs, pr_aucs, baccs, f1s = [], [], [], []
        oof_preds = np.zeros(len(y_spec))
        oof_probs = np.zeros(len(y_spec))

        for train_idx, val_idx in skf.split(X_spec, y_spec):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_spec[train_idx])
            X_va = scaler.transform(X_spec[val_idx])

            clf.fit(X_tr, y_spec[train_idx])
            prob = clf.predict_proba(X_va)[:, 1]
            pred = clf.predict(X_va)

            oof_probs[val_idx] = prob
            oof_preds[val_idx] = pred

            aucs.append(roc_auc_score(y_spec[val_idx], prob))
            pr_aucs.append(average_precision_score(y_spec[val_idx], prob))
            baccs.append(balanced_accuracy_score(y_spec[val_idx], pred))
            f1s.append(f1_score(y_spec[val_idx], pred, zero_division=0))

        cm = confusion_matrix(y_spec, oof_preds)
        multivariate_results[name] = {
            "ROC_AUC": float(np.mean(aucs)),
            "PR_AUC": float(np.mean(pr_aucs)),
            "Balanced_Accuracy": float(np.mean(baccs)),
            "F1": float(np.mean(f1s)),
            "Confusion_Matrix": cm.tolist()
        }
        print(f"{name:<22} | ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f}) | Balanced Acc: {np.mean(baccs):.4f} | F1: {np.mean(f1s):.4f}")

    # -------------------------------------------------------------
    # 3. SUPERVISED ACTION ROUTER (H1 vs. H2 ARBITRATION)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [EXPERIMENT 3] SUPERVISED ACTION ROUTER (ORACLE SUPERVISION ON VALIDATION)")
    print("-" * 70)

    # Label validation set with Oracle actions
    val_oracle_labels = []
    val_pc_np = val_pc.numpy()
    val_ps_np = val_ps.numpy()
    val_ph_np = val_ph.numpy()

    for u in range(N_val):
        tgt = val_targets[u] - 1
        rc = int(np.sum(val_pc_np[u] >= val_pc_np[u, tgt]))
        rs = int(np.sum(val_ps_np[u] >= val_ps_np[u, tgt]))
        rh = int(np.sum(val_ph_np[u] >= val_ph_np[u, tgt]))
        rr_c, rr_s, rr_h = 1.0 / rc, 1.0 / rs, 1.0 / rh
        if rr_h >= max(rr_c, rr_s):
            val_oracle_labels.append(2) # 2 = H
        elif rr_c > rr_s:
            val_oracle_labels.append(0) # 0 = C
        else:
            val_oracle_labels.append(1) # 1 = S

    val_y_t = torch.tensor(val_oracle_labels, dtype=torch.long)
    print(f"Validation Oracle Action Proportions: C={sum(val_y_t==0).item()/N_val:.1%}, S={sum(val_y_t==1).item()/N_val:.1%}, H={sum(val_y_t==2).item()/N_val:.1%}")

    # Train a supervised neural action classifier: X -> 3 logits (C, S, H)
    class SupervisedActionRouter(torch.nn.Module):
        def __init__(self, in_features=13, hidden_dim=64):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, 3)
            )
        def forward(self, x):
            return self.net(x)

    sup_router = SupervisedActionRouter(in_features=val_x.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(sup_router.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Compute inverse class frequencies to test balanced cross entropy
    class_counts = torch.bincount(val_y_t, minlength=3).float()
    class_weights = (1.0 / class_counts) / (1.0 / class_counts).sum() * 3.0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Normalize features
    mean_x = val_x.mean(dim=0, keepdim=True)
    std_x = val_x.std(dim=0, keepdim=True).clamp(min=1e-5)
    val_x_norm = (val_x - mean_x) / std_x
    test_x_norm = (test_x - mean_x) / std_x

    # Train
    sup_router.train()
    for ep in range(35):
        optimizer.zero_grad()
        logits = sup_router(val_x_norm)
        loss = criterion(logits, val_y_t)
        loss.backward()
        optimizer.step()

    sup_router.eval()
    with torch.no_grad():
        test_logits = sup_router(test_x_norm)
        test_probs = torch.softmax(test_logits, dim=-1) # [N, 3] -> w_C, w_S, w_H
        test_preds = torch.argmax(test_probs, dim=-1).numpy()
        
        # Effective blend: p = w_C * p_C + w_S * p_S + w_H * p_H = (w_C + 0.5*w_H) p_C + (w_S + 0.5*w_H) p_S
        eff_c = test_probs[:, 0] + 0.5 * test_probs[:, 2]
        eff_c_np = eff_c.numpy()
        
        p_sup_routed = (eff_c.unsqueeze(-1) * test_pc + (1.0 - eff_c.unsqueeze(-1)) * test_ps).numpy()

    # Calculate test metrics for Supervised Action Router
    sup_hits_10 = []
    sup_ndcgs_10 = []
    for u in range(N_test):
        tgt = test_targets[u] - 1
        rank = int(np.sum(p_sup_routed[u] >= p_sup_routed[u, tgt]))
        sup_hits_10.append(1.0 if rank <= 10 else 0.0)
        sup_ndcgs_10.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)

    sup_hr10 = np.mean(sup_hits_10)
    sup_ndcg10 = np.mean(sup_ndcgs_10)

    pred_c_pct = 100.0 * np.sum(test_preds == 0) / N_test
    pred_s_pct = 100.0 * np.sum(test_preds == 1) / N_test
    pred_h_pct = 100.0 * np.sum(test_preds == 2) / N_test

    print(f"Supervised Action Router Test HR@10:   {sup_hr10:.4f}")
    print(f"Supervised Action Router Test NDCG@10: {sup_ndcg10:.4f}")
    print(f"Predicted Discrete Action Shares:      {pred_c_pct:.1f}% C / {pred_s_pct:.1f}% S / {pred_h_pct:.1f}% H")
    print(f"Effective Collaborative Weight Eff_C:  Mean = {np.mean(eff_c_np):.4f} +/- {np.std(eff_c_np):.4f} (Min={np.min(eff_c_np):.4f}, Max={np.max(eff_c_np):.4f})")
    print(f"Quantiles (P5, P25, P50, P75, P95):    {np.percentile(eff_c_np, [5, 25, 50, 75, 95])}")

    # Evaluate classification metrics against ground-truth test Oracle actions
    test_y_true = np.array([0 if a == "C" else (1 if a == "S" else 2) for a in oracle_actions])
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
    test_acc = accuracy_score(test_y_true, test_preds)
    test_bacc = balanced_accuracy_score(test_y_true, test_preds)
    test_cm = confusion_matrix(test_y_true, test_preds, labels=[0, 1, 2])
    test_report = classification_report(test_y_true, test_preds, labels=[0, 1, 2], target_names=["C", "S", "H"], output_dict=True)

    print(f"Action Classification Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Balanced Accuracy:               {test_bacc:.4f} ({test_bacc*100:.2f}%)")
    print(f"Per-Class F1 Scores:             F1_C={test_report['C']['f1-score']:.4f}, F1_S={test_report['S']['f1-score']:.4f}, F1_H={test_report['H']['f1-score']:.4f}")
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{test_cm}")

    # Output JSON summary
    summary = {
        "oracle_equivalence_verified": (mismatches_10 == 0 and mismatches_5 == 0 and mismatches_20 == 0),
        "oracle_hr10_exact": float(mean_true_hr10),
        "oracle_hr5_exact": float(mean_true_hr5),
        "oracle_hr20_exact": float(mean_true_hr20),
        "multivariate_c_vs_s": multivariate_results,
        "supervised_action_router": {
            "hr10": float(sup_hr10),
            "ndcg10": float(sup_ndcg10),
            "action_distribution": f"{pred_c_pct:.1f}% C / {pred_s_pct:.1f}% S / {pred_h_pct:.1f}% H",
            "eff_c_mean": float(np.mean(eff_c_np)),
            "eff_c_std": float(np.std(eff_c_np)),
            "eff_c_p5": float(np.percentile(eff_c_np, 5)),
            "eff_c_p95": float(np.percentile(eff_c_np, 95)),
            "action_accuracy": float(test_acc),
            "balanced_accuracy": float(test_bacc),
            "f1_c": float(test_report["C"]["f1-score"]),
            "precision_c": float(test_report["C"]["precision"]),
            "recall_c": float(test_report["C"]["recall"]),
            "f1_s": float(test_report["S"]["f1-score"]),
            "precision_s": float(test_report["S"]["precision"]),
            "recall_s": float(test_report["S"]["recall"]),
            "f1_h": float(test_report["H"]["f1-score"]),
            "precision_h": float(test_report["H"]["precision"]),
            "recall_h": float(test_report["H"]["recall"]),
            "macro_f1": float(test_report["macro avg"]["f1-score"]),
            "confusion_matrix": test_cm.tolist()
        }
    }

    with open("results/multivariate_and_supervised_router_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[OK] Saved results to results/multivariate_and_supervised_router_results.json")

if __name__ == "__main__":
    main()
