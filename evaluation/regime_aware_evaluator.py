"""
Regime-Aware Evaluation Protocol for Sequential Recommendation.

This module implements the formal Regime-Aware Evaluation (RAE) framework,
the methodological contribution of the paper.

Five user regimes are defined based on sequence properties:
  R1: Popularity Collapse  — L_u <= 3  (short history, collaborative models collapse)
  R2: Personalization      — 4 <= L_u <= 10 (peak routing benefit zone)
  R3: Warm                 — 11 <= L_u <= 20
  R4: Saturation           — L_u > 20 (collaborative recency saturation)
  R5: Behavioral Rupture   — Drift_u > theta (abrupt preference shift)

REGIME CLASSIFICATION CORRECTNESS (critical):
  Sequence length is measured on test_user_seq[uid][0] (= S_u[:-1]),
  i.e., the history PRESENTED to the model at test time — AFTER
  the Leave-One-Out split and AFTER cold-start truncation.
  Cold-start users (truncated to <= cold_start_threshold interactions)
  will naturally fall into R1 or R2 based on their truncated history.
  This is intentional: their regimes reflect their OBSERVABLE informational
  state, not their latent full history.

RAP@K WEIGHT CORRECTNESS (critical):
  Regime weights w_r = |regime_r| / |N_test| are computed ONLY
  over the test partition users. They must NOT include validation or
  training users, and must NOT be computed on the full dataset before
  the cold-start split.

Regime-Aware Precision (RAP@K):
  RAP@K = sum_r  w_r * HR@K(regime_r)
  where w_r = |regime_r ∩ test_users| / |test_users|  (population-weighted)

Cross-Dataset Regime Table:
  For each (dataset, model, regime) → HR@K, NDCG@K, delta vs. static fusion
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Regime Definitions ─────────────────────────────────────────────────────────

REGIME_DEFS = {
    "R1_popularity_collapse": {
        "label": "R1: Popularity Collapse (L≤3)",
        "short": "R1",
        "condition": "length <= 3",
    },
    "R2_personalization": {
        "label": "R2: Personalization Discovery (4≤L≤10)",
        "short": "R2",
        "condition": "4 <= length <= 10",
    },
    "R3_warm": {
        "label": "R3: Warm Users (11≤L≤20)",
        "short": "R3",
        "condition": "11 <= length <= 20",
    },
    "R4_saturation": {
        "label": "R4: Collaborative Saturation (L>20)",
        "short": "R4",
        "condition": "length > 20",
    },
    "R5_behavioral_rupture": {
        "label": "R5: Behavioral Rupture (Drift>θ)",
        "short": "R5",
        "condition": "drift > drift_threshold",
    },
}

DEFAULT_DRIFT_THRESHOLD = 0.1721   # Median-split threshold from ML-1M experiments


def classify_user_regime(
    seq_length: int,
    drift: float = 0.0,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> str:
    """
    Assigns a user to one of the 5 operational regimes based on sequence
    length and semantic drift.

    Note: R5 (Behavioral Rupture) can overlap with R1-R4. When overlap occurs,
    R5 takes priority (behavioral rupture is the dominant signal).

    Args:
        seq_length: Number of non-padding interactions
        drift:      Cosine drift between recent and historical centroid
        drift_threshold: Threshold above which rupture is declared

    Returns:
        regime_key: str, one of the keys in REGIME_DEFS
    """
    if drift > drift_threshold:
        return "R5_behavioral_rupture"
    if seq_length <= 3:
        return "R1_popularity_collapse"
    if seq_length <= 10:
        return "R2_personalization"
    if seq_length <= 20:
        return "R3_warm"
    return "R4_saturation"


def compute_regime_partition(
    test_user_seq: Dict[int, Tuple],
    user_drift_scores: Optional[Dict[int, float]] = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    cold_start_users: Optional[set] = None,
) -> Dict[str, List[int]]:
    """
    Partitions test users into regimes based on sequence length and drift.

    IMPORTANT — Correctness guarantee:
      seq_length is computed from test_user_seq[uid][0], which is
      S_u[:-1] (the history after LOO split and cold-start truncation).
      This correctly reflects the model's observable state at test time.
      Cold-start users (in cold_start_users set) are classified into
      R1/R2 based on their truncated history — this is intentional
      and should be reported explicitly in the paper.

    Args:
        test_user_seq:    {uid: (history_seq, target_item)}
                          history_seq is already the truncated test history.
        user_drift_scores:{uid: drift_value} (optional, computed on history_seq)
        drift_threshold:  Drift threshold for R5 classification
        cold_start_users: Optional set of cold-start uids (for reporting only,
                          does not affect regime assignment).

    Returns:
        regime_partition: {regime_key: [uid, uid, ...]}
                          All uids are strictly from test_user_seq.
    """
    partition: Dict[str, List[int]] = defaultdict(list)
    cold_start_users = cold_start_users or set()

    for uid, (history, _) in test_user_seq.items():
        # seq_length = length of test-time history (AFTER cold-start truncation)
        seq_length = len(history)
        drift      = user_drift_scores.get(uid, 0.0) if user_drift_scores else 0.0
        regime     = classify_user_regime(seq_length, drift, drift_threshold)
        partition[regime].append(uid)

    # Sanity check: all test users must be assigned to exactly one regime
    assigned = sum(len(v) for v in partition.values())
    total    = len(test_user_seq)
    assert assigned == total, (
        f"Regime partition error: {assigned} users assigned but {total} in test set. "
        f"Every user must map to exactly one regime."
    )

    if cold_start_users:
        cs_in_r1 = len([u for u in partition.get("R1_popularity_collapse", [])
                         if u in cold_start_users])
        cs_in_r2 = len([u for u in partition.get("R2_personalization", [])
                         if u in cold_start_users])
        cs_total  = len([u for u in test_user_seq if u in cold_start_users])
        print(f"[RAE] Cold-start users in test: {cs_total} "
              f"(R1: {cs_in_r1}, R2: {cs_in_r2}, other: {cs_total-cs_in_r1-cs_in_r2})")

    return dict(partition)


def compute_hit_at_k(rank: int, k: int) -> float:
    """Returns 1.0 if rank <= k, else 0.0."""
    return 1.0 if rank <= k else 0.0


def compute_ndcg_at_k(rank: int, k: int) -> float:
    """Returns NDCG@K discount for a given rank."""
    if rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


class RegimeAwareEvaluator:
    """
    Evaluates one or more models across all regimes and computes:
      - Per-regime HR@K, NDCG@K for each model
      - Regime-Aware Precision (RAP@K) — population-weighted composite
      - Δ vs. static fusion baseline per regime
      - Statistical significance markers (*, **, ***)
    """

    SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "")]

    def __init__(
        self,
        k: int = 10,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
        bootstrap_B: int = 5_000,
        seed: int = 42,
    ):
        self.k               = k
        self.drift_threshold = drift_threshold
        self.bootstrap_B     = bootstrap_B
        self.rng             = np.random.default_rng(seed)

    def _bootstrap_pvalue(
        self,
        delta_per_user: np.ndarray,
        B: int = 5_000,
    ) -> float:
        """
        One-sample bootstrap test for H0: mean(delta) = 0.
        Returns p-value (two-sided).
        """
        n        = len(delta_per_user)
        obs_mean = delta_per_user.mean()
        centered = delta_per_user - obs_mean
        boot_means = np.array([
            self.rng.choice(centered, size=n, replace=True).mean()
            for _ in range(B)
        ])
        p = (np.abs(boot_means) >= np.abs(obs_mean)).mean()
        return float(p)

    def _sig_marker(self, p: float) -> str:
        for threshold, marker in self.SIG_LEVELS:
            if p < threshold:
                return marker
        return ""

    def evaluate(
        self,
        model_predictions: Dict[str, Dict[int, int]],
        test_user_seq: Dict[int, Tuple],
        regime_partition: Dict[str, List[int]],
        baseline_name: str = "Static Fusion",
    ) -> Dict[str, Any]:
        """
        Full regime-aware evaluation.

        Args:
            model_predictions: {model_name: {uid: rank_of_target_item}}
                  rank_of_target_item is 1-indexed rank in the full-catalog ranking.
            test_user_seq: {uid: (history, target_item)}
            regime_partition: {regime_key: [uid, ...]}
            baseline_name: Name of the model to use as baseline for delta computation.

        Returns:
            results dict containing:
                - per_regime: {regime_key: {model_name: {"HR@K": ..., "NDCG@K": ..., "N": ...}}}
                - rap_k: {model_name: float}  (Regime-Aware Precision@K)
                - delta_vs_baseline: {regime_key: {model_name: {"delta_hr": ..., "p": ..., "sig": ...}}}
                - global: {model_name: {"HR@K": ..., "NDCG@K": ...}}
        """
        k         = self.k
        all_users = list(test_user_seq.keys())
        N_total   = len(all_users)

        results = {
            "k": k,
            "N_total": N_total,
            "per_regime": {},
            "rap_k": {},
            "delta_vs_baseline": {},
            "global": {},
        }

        # ── Global evaluation ─────────────────────────────────────────────────
        for model_name, uid_to_rank in model_predictions.items():
            hits  = []
            ndcgs = []
            for uid in all_users:
                rank = uid_to_rank.get(uid, k + 1)
                hits.append(compute_hit_at_k(rank, k))
                ndcgs.append(compute_ndcg_at_k(rank, k))
            results["global"][model_name] = {
                f"HR@{k}":   float(np.mean(hits)),
                f"NDCG@{k}": float(np.mean(ndcgs)),
                "N": N_total,
            }

        # ── Per-regime evaluation ─────────────────────────────────────────────
        for regime_key, regime_uids in regime_partition.items():
            if not regime_uids:
                continue

            regime_label = REGIME_DEFS.get(regime_key, {}).get("label", regime_key)
            results["per_regime"][regime_key] = {
                "label": regime_label,
                "N": len(regime_uids),
                "weight": len(regime_uids) / N_total,
                "models": {},
            }

            for model_name, uid_to_rank in model_predictions.items():
                hits  = []
                ndcgs = []
                for uid in regime_uids:
                    rank = uid_to_rank.get(uid, k + 1)
                    hits.append(compute_hit_at_k(rank, k))
                    ndcgs.append(compute_ndcg_at_k(rank, k))

                results["per_regime"][regime_key]["models"][model_name] = {
                    f"HR@{k}":   float(np.mean(hits)),
                    f"NDCG@{k}": float(np.mean(ndcgs)),
                    "N": len(regime_uids),
                    "_hits_array": hits,   # kept for bootstrap test
                }

        # ── Regime-Aware Precision (RAP@K) ────────────────────────────────────
        model_names = list(model_predictions.keys())
        for model_name in model_names:
            rap = 0.0
            for regime_key, regime_data in results["per_regime"].items():
                w       = regime_data["weight"]
                hr      = regime_data["models"].get(model_name, {}).get(f"HR@{k}", 0.0)
                rap    += w * hr
            results["rap_k"][model_name] = float(rap)

        # ── Delta vs. baseline + statistical significance ─────────────────────
        if baseline_name in model_predictions:
            for regime_key, regime_data in results["per_regime"].items():
                regime_uids  = regime_partition[regime_key]
                results["delta_vs_baseline"][regime_key] = {}

                baseline_hits = np.array(
                    regime_data["models"].get(baseline_name, {}).get("_hits_array", [])
                )

                for model_name in model_names:
                    if model_name == baseline_name:
                        continue
                    model_hits = np.array(
                        regime_data["models"].get(model_name, {}).get("_hits_array", [])
                    )

                    if len(model_hits) == 0 or len(baseline_hits) == 0:
                        continue

                    delta_per_user = model_hits - baseline_hits
                    delta_mean     = float(delta_per_user.mean())
                    p_val          = self._bootstrap_pvalue(delta_per_user, B=self.bootstrap_B)
                    sig            = self._sig_marker(p_val)
                    baseline_hr    = float(baseline_hits.mean())
                    rel_gain       = (delta_mean / (baseline_hr + 1e-9)) * 100

                    results["delta_vs_baseline"][regime_key][model_name] = {
                        "delta_hr":     delta_mean,
                        "rel_gain_pct": rel_gain,
                        "p_value":      p_val,
                        "sig":          sig,
                        "N":            len(regime_uids),
                    }

        # Clean up internal arrays
        for regime_data in results["per_regime"].values():
            for model_data in regime_data["models"].values():
                model_data.pop("_hits_array", None)

        return results

    def format_regime_table(
        self, results: Dict[str, Any], model_order: Optional[List[str]] = None
    ) -> str:
        """
        Formats evaluation results as a markdown table for the paper.

        Returns:
            Markdown-formatted table string
        """
        k = results["k"]
        model_names = model_order or list(results["global"].keys())

        header = (
            f"| Regime | N | Weight |"
            + "".join(f" {m} HR@{k} | {m} NDCG@{k} |" for m in model_names)
            + "\n"
        )
        sep = (
            "| :--- | :---: | :---: |"
            + " :---: | :---: |" * len(model_names)
            + "\n"
        )
        rows = ""

        # Global row
        row = f"| **Global** | {results['N_total']} | 1.000 |"
        for m in model_names:
            g = results["global"].get(m, {})
            row += f" {g.get(f'HR@{k}', 0):.4f} | {g.get(f'NDCG@{k}', 0):.4f} |"
        rows += row + "\n"

        # Per-regime rows
        for regime_key in REGIME_DEFS:
            if regime_key not in results["per_regime"]:
                continue
            rd = results["per_regime"][regime_key]
            label  = rd["label"]
            N      = rd["N"]
            weight = rd["weight"]
            row    = f"| {label} | {N} | {weight:.3f} |"

            for m in model_names:
                md = rd["models"].get(m, {})
                hr   = md.get(f"HR@{k}", 0.0)
                ndcg = md.get(f"NDCG@{k}", 0.0)
                # Add significance marker if vs. baseline
                delta_info = results.get("delta_vs_baseline", {}).get(regime_key, {}).get(m, {})
                sig = delta_info.get("sig", "")
                row += f" {hr:.4f}{sig} | {ndcg:.4f} |"
            rows += row + "\n"

        # RAP@K row
        row = f"| **RAP@{k} (weighted)** | — | — |"
        for m in model_names:
            rap = results["rap_k"].get(m, 0.0)
            row += f" **{rap:.4f}** | — |"
        rows += row + "\n"

        return header + sep + rows

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Saves results dict as JSON."""
        # Convert numpy types to Python native for JSON serialization
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
        print(f"[RAE] Results saved to {output_path}")


# ── Convenience function ───────────────────────────────────────────────────────

def run_regime_aware_evaluation(
    model_predictions: Dict[str, Dict[int, int]],
    test_user_seq: Dict[int, Tuple],
    user_drift_scores: Optional[Dict[int, float]] = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    baseline_name: str = "Static Fusion",
    k: int = 10,
    output_path: Optional[str] = None,
    model_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    One-call interface for regime-aware evaluation.

    Args:
        model_predictions: {model_name: {uid: rank_of_target}}
        test_user_seq: {uid: (history_seq, target_item)}
        user_drift_scores: {uid: float} optional per-user drift
        baseline_name: model name to use as comparison baseline
        k: top-K cutoff
        output_path: if given, saves JSON results to this path
        model_order: display order for table formatting

    Returns:
        results dict + prints formatted markdown table
    """
    evaluator = RegimeAwareEvaluator(k=k, drift_threshold=drift_threshold)

    regime_partition = compute_regime_partition(
        test_user_seq, user_drift_scores, drift_threshold
    )

    print("\n[RAE] User regime distribution:")
    total = sum(len(v) for v in regime_partition.values())
    for rk, uids in sorted(regime_partition.items()):
        label = REGIME_DEFS.get(rk, {}).get("label", rk)
        print(f"  {label}: {len(uids)} users ({len(uids)/total*100:.1f}%)")

    results = evaluator.evaluate(
        model_predictions=model_predictions,
        test_user_seq=test_user_seq,
        regime_partition=regime_partition,
        baseline_name=baseline_name,
    )

    print("\n[RAE] Regime-Aware Evaluation Results:")
    print(evaluator.format_regime_table(results, model_order=model_order))

    if output_path:
        evaluator.save_results(results, output_path)

    return results
