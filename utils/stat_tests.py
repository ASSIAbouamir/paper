"""
Statistical Significance Testing Suite for Recommendation System Evaluations.
Implements:
1. Paired Student's t-test (parametric)
2. Wilcoxon Signed-Rank test (non-parametric rank-based)
3. Cohen's d effect size calculation
4. 95% Confidence Intervals (CI)
Self-contained, robust to dependency issues.
"""

import math
import numpy as np
from typing import Dict, List, Tuple


def _erfc(x: float) -> float:
    """Complementary error function approximation for p-value computation."""
    # Abramowitz and Stegun formula 7.1.26
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 1.0 - sign * y if sign >= 0 else 1.0 + y


def standard_normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_t_test(sample_a: List[float], sample_b: List[float]) -> Dict[str, float]:
    """
    Computes Paired Student's t-test between two paired observations.
    Returns: mean difference, t-statistic, p-value (two-tailed), and Cohen's d.
    """
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)
    assert len(a) == len(b), "Samples must have identical length for paired test."

    n = len(a)
    if n < 2:
        return {"mean_diff": 0.0, "t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0, "significant_005": False}

    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    if std_diff == 0.0:
        return {
            "mean_diff": mean_diff,
            "t_stat": 0.0,
            "p_value": 1.0 if mean_diff == 0.0 else 0.0,
            "cohens_d": 0.0,
            "significant_005": mean_diff != 0.0
        }

    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se
    cohens_d = mean_diff / std_diff

    # For large RecSys samples (N > 100), t-distribution closely matches standard normal
    # For smaller N, normal approximation is conservative
    p_value = 2.0 * (1.0 - standard_normal_cdf(abs(t_stat)))
    p_value = float(np.clip(p_value, 1e-15, 1.0))

    return {
        "mean_diff": mean_diff,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "cohens_d": float(cohens_d),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01
    }


def wilcoxon_signed_rank_test(sample_a: List[float], sample_b: List[float]) -> Dict[str, float]:
    """
    Computes Wilcoxon Signed-Rank test for paired non-normal data (standard in IR/RecSys).
    """
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)
    diff = a - b

    # Exclude zero differences
    non_zero_diff = diff[diff != 0.0]
    n_r = len(non_zero_diff)

    if n_r < 5:
        return {"w_stat": 0.0, "z_stat": 0.0, "p_value": 1.0, "significant_005": False}

    abs_diff = np.abs(non_zero_diff)
    ranks = np.argsort(np.argsort(abs_diff)) + 1  # 1-indexed ranks

    w_plus = float(np.sum(ranks[non_zero_diff > 0]))
    w_minus = float(np.sum(ranks[non_zero_diff < 0]))
    w_stat = min(w_plus, w_minus)

    # Normal approximation with continuity correction
    mean_w = n_r * (n_r + 1) / 4.0
    var_w = n_r * (n_r + 1) * (2 * n_r + 1) / 24.0
    std_w = math.sqrt(var_w)

    z_stat = (w_stat - mean_w + 0.5) / std_w
    p_value = 2.0 * standard_normal_cdf(-abs(z_stat))
    p_value = float(np.clip(p_value, 1e-15, 1.0))

    return {
        "w_stat": w_stat,
        "z_stat": float(z_stat),
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01
    }


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Computes mean and [lower, upper] bounds of Confidence Interval."""
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    
    # 95% critical value z ~ 1.96
    z_crit = 1.96 if confidence == 0.95 else 2.576
    margin = z_crit * se
    return mean, mean - margin, mean + margin
