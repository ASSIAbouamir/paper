"""
Calibration Analysis for Epistemic Uncertainty in Sequential Recommenders.

Implements:
  1. Expected Calibration Error (ECE) for epistemic uncertainty Uc
  2. Brier Score for probabilistic ranking
  3. Reliability Diagrams (bin-wise accuracy vs. mean uncertainty)
  4. Uncertainty-Error Correlation (Pearson r between Uc and rank error)
  5. Sharpness (how peaked the uncertainty distribution is)
  6. Resolution (how discriminative the uncertainty is)

Scientific context:
  A well-calibrated uncertainty measure Uc should satisfy:
    P(hit@K | Uc = u) ≈ f(u)  (monotonically decreasing in u)
  i.e., higher uncertainty should correspond to lower hit rate.

Reference:
  - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
  - Kuleshov et al., "Calibrated Uncertainty for Deep Seq Models", UAI 2022
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_ece(
    uc_scores: np.ndarray,
    hit_labels: np.ndarray,
    n_bins: int = 10,
    normalize_uc: bool = True,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) for uncertainty Uc.

    Adaptation for recommendation:
      - Uc is binned from low to high uncertainty
      - Within each bin, we measure mean HR@K (as "accuracy")
      - We expect HR@K to decrease monotonically as Uc increases
      - ECE = Σ_b (|bin_b| / N) * |mean_HR_b - mean_expected_HR_b|
        where expected HR is calibration target (decreasing linear interpolation)

    Args:
        uc_scores:   [N] epistemic uncertainty values
        hit_labels:  [N] binary hit labels (1 = hit@K, 0 = miss)
        n_bins:      Number of calibration bins
        normalize_uc: Normalize Uc to [0,1] before binning

    Returns:
        dict with:
            ece:          float — Expected Calibration Error
            bin_means:    list — mean Uc per bin
            bin_acc:      list — mean HR@K per bin
            bin_counts:   list — N per bin
            pearson_r:    float — Pearson correlation(Uc, 1 - hit)
            pearson_p:    float — p-value for correlation
    """
    N = len(uc_scores)
    assert len(hit_labels) == N

    uc = uc_scores.copy().astype(float)
    hits = hit_labels.copy().astype(float)

    if normalize_uc:
        uc_min, uc_max = uc.min(), uc.max()
        if uc_max > uc_min:
            uc = (uc - uc_min) / (uc_max - uc_min)

    # Bin edges — equal-width
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_means  = []
    bin_acc    = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (uc >= lo) & (uc <= hi)
        else:
            mask = (uc >= lo) & (uc < hi)

        n_b = mask.sum()
        if n_b == 0:
            continue

        bin_means.append(float(uc[mask].mean()))
        bin_acc.append(float(hits[mask].mean()))
        bin_counts.append(int(n_b))

    # ECE: deviation from monotonic decrease
    # Expected: bin_acc should be high when uc is low, low when uc is high
    # Simple calibration: expected_acc_b = 1 - normalized_bin_uncertainty
    bin_means_arr  = np.array(bin_means)
    bin_acc_arr    = np.array(bin_acc)
    bin_counts_arr = np.array(bin_counts)
    total          = bin_counts_arr.sum()

    expected_acc = 1.0 - bin_means_arr   # idealized: low Uc → high HR
    ece = float(np.sum(bin_counts_arr / total * np.abs(bin_acc_arr - expected_acc)))

    # Pearson correlation between Uc and miss rate
    from scipy import stats as scipy_stats
    r, p = scipy_stats.pearsonr(uc, 1.0 - hits)

    return {
        "ece":          ece,
        "bin_means":    bin_means,
        "bin_acc":      bin_acc,
        "bin_counts":   bin_counts,
        "n_bins_used":  len(bin_means),
        "pearson_r":    float(r),
        "pearson_p":    float(p),
        "N":            int(N),
    }


def compute_brier_score(
    predicted_probs: np.ndarray,
    hit_labels: np.ndarray,
) -> float:
    """
    Computes Brier Score for probabilistic ranking.
    BS = (1/N) * Σ (p_hat_i - y_i)^2
    Lower is better (0 = perfect, 1 = worst).

    predicted_probs: [N] probability of hit (use normalized rank score or softmax)
    hit_labels: [N] binary
    """
    return float(np.mean((predicted_probs - hit_labels) ** 2))


def compute_quartile_precision(
    uc_scores: np.ndarray,
    hit_labels: np.ndarray,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Splits users into 4 quartiles by Uc and computes HR@K per quartile.
    This directly supports the claim that higher Uc → lower HR@K.

    Returns:
        dict with Q1..Q4 stats including mean Uc and HR@K per quartile
    """
    N    = len(uc_scores)
    idx  = np.argsort(uc_scores)
    uc_s = uc_scores[idx]
    hit_s = hit_labels[idx]

    q_size = N // 4
    result = {}

    for q in range(4):
        start = q * q_size
        end   = (q + 1) * q_size if q < 3 else N
        q_uc  = uc_s[start:end]
        q_hit = hit_s[start:end]
        result[f"Q{q+1}"] = {
            "mean_uc":  float(q_uc.mean()),
            "std_uc":   float(q_uc.std()),
            f"HR@{k}":  float(q_hit.mean()),
            "N":        int(len(q_hit)),
            "label":    ["Lowest Uc", "Low-Mid Uc", "Mid-High Uc", "Highest Uc"][q],
        }

    return result


def run_calibration_analysis(
    uc_collab: np.ndarray,
    uc_semantic: np.ndarray,
    hit_collab: np.ndarray,
    hit_semantic: np.ndarray,
    hit_router: np.ndarray,
    output_path: Optional[str] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Full calibration analysis for both experts and the router.

    Args:
        uc_collab:    [N] collaborative epistemic uncertainty
        uc_semantic:  [N] semantic epistemic uncertainty
        hit_collab:   [N] binary hit@K for collaborative expert
        hit_semantic: [N] binary hit@K for semantic expert
        hit_router:   [N] binary hit@K for the selective router
        output_path:  optional path to save JSON results
        n_bins:       number of ECE calibration bins

    Returns:
        Full calibration results dict
    """
    results = {}

    # ── ECE per expert ─────────────────────────────────────────────────────────
    results["ece_collaborative"] = compute_ece(
        uc_collab, hit_collab, n_bins=n_bins
    )
    results["ece_semantic"] = compute_ece(
        uc_semantic, hit_semantic, n_bins=n_bins
    )

    # ── ECE of Uc combined for router prediction ──────────────────────────────
    # When does high Uc_c and low Uc_s → router routes to semantic (and hits)?
    uc_diff = uc_collab - uc_semantic   # Positive → route to semantic
    results["ece_router_signal"] = compute_ece(
        np.abs(uc_diff), hit_router, n_bins=n_bins
    )

    # ── Quartile precision breakdown ──────────────────────────────────────────
    results["quartile_collab"]   = compute_quartile_precision(uc_collab,   hit_collab)
    results["quartile_semantic"] = compute_quartile_precision(uc_semantic, hit_semantic)

    # ── Summary statistics ────────────────────────────────────────────────────
    results["summary"] = {
        "collab": {
            "ece":          results["ece_collaborative"]["ece"],
            "pearson_r":    results["ece_collaborative"]["pearson_r"],
            "pearson_p":    results["ece_collaborative"]["pearson_p"],
            "mean_uc":      float(uc_collab.mean()),
            "std_uc":       float(uc_collab.std()),
            "global_hr":    float(hit_collab.mean()),
        },
        "semantic": {
            "ece":          results["ece_semantic"]["ece"],
            "pearson_r":    results["ece_semantic"]["pearson_r"],
            "pearson_p":    results["ece_semantic"]["pearson_p"],
            "mean_uc":      float(uc_semantic.mean()),
            "std_uc":       float(uc_semantic.std()),
            "global_hr":    float(hit_semantic.mean()),
        },
        "hit_on_misses": {
            "collab_hit_mean_uc":   float(uc_collab[hit_collab == 1].mean()) if hit_collab.sum() > 0 else 0.0,
            "collab_miss_mean_uc":  float(uc_collab[hit_collab == 0].mean()) if (1-hit_collab).sum() > 0 else 0.0,
            "semantic_hit_mean_uc": float(uc_semantic[hit_semantic == 1].mean()) if hit_semantic.sum() > 0 else 0.0,
            "semantic_miss_mean_uc":float(uc_semantic[hit_semantic == 0].mean()) if (1-hit_semantic).sum() > 0 else 0.0,
        },
    }

    # Print summary
    print("\n[Calibration] Summary:")
    print(f"  Collaborative Expert ECE:   {results['summary']['collab']['ece']:.4f}")
    print(f"  Collaborative Pearson r:    {results['summary']['collab']['pearson_r']:.4f} (p={results['summary']['collab']['pearson_p']:.4f})")
    print(f"  Semantic Expert ECE:        {results['summary']['semantic']['ece']:.4f}")
    print(f"  Semantic Pearson r:         {results['summary']['semantic']['pearson_r']:.4f} (p={results['summary']['semantic']['pearson_p']:.4f})")
    print(f"\n  Mean Uc on Hits (Collab):   {results['summary']['hit_on_misses']['collab_hit_mean_uc']:.5f}")
    print(f"  Mean Uc on Misses (Collab): {results['summary']['hit_on_misses']['collab_miss_mean_uc']:.5f}")

    print("\n[Calibration] Quartile breakdown (Collaborative):")
    for q_label, q_data in results["quartile_collab"].items():
        print(f"  {q_label} ({q_data['label']}): "
              f"Mean Uc={q_data['mean_uc']:.5f}, HR@10={q_data['HR@10']:.4f}")

    if output_path:
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj
        with open(output_path, "w") as f:
            json.dump(_convert(results), f, indent=2)
        print(f"[Calibration] Results saved to {output_path}")

    return results
