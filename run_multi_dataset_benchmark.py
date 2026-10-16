"""
Multi-Dataset Benchmark Runner.

Runs the complete Selective Evidence Routing pipeline across all supported datasets:
  - MovieLens-1M
  - Amazon Beauty
  - Amazon Sports & Outdoors
  - Steam Games

For each dataset:
  1. Preprocess / load data
  2. Train SASRec + Semantic Expert
  3. Extract LLM embeddings (MiniLM)
  4. Train the Selective Router (original 8D + enhanced 12D)
  5. Run Regime-Aware Evaluation across all regimes
  6. Run calibration analysis
  7. Compare against baselines: SASRec, BERT4Rec, GRU4Rec, Semantic, Static Fusion

All results are saved to results/<dataset_name>/ and compiled into
a cross-dataset summary table at results/cross_dataset_summary.json.

Usage:
  python run_multi_dataset_benchmark.py --datasets ml1m amazon_beauty steam
  python run_multi_dataset_benchmark.py --all --seeds 42 123 2024
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset_registry import get_dataset, DATASET_CONFIGS, list_datasets


def _ensure_results_dir(dataset_name: str) -> str:
    """Creates and returns the results directory for a dataset."""
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        dataset_name,
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def run_single_dataset_benchmark(
    dataset_name: str,
    seeds: List[int],
    use_enhanced_router: bool = True,
    run_bert4rec: bool = True,
    run_gru4rec: bool = True,
    run_regime_evaluation: bool = True,
    run_calibration: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Full benchmark pipeline for a single dataset.

    Args:
        dataset_name:           Dataset key from registry
        seeds:                  List of random seeds for multi-seed evaluation
        use_enhanced_router:    Also run the 12D EnhancedSelectiveRouter
        run_bert4rec:           Include BERT4Rec as baseline
        run_gru4rec:            Include GRU4Rec as baseline
        run_regime_evaluation:  Run Regime-Aware Evaluation (RAE)
        run_calibration:        Run calibration analysis
        device:                 "cuda" or "cpu"

    Returns:
        results dict
    """
    print(f"\n{'='*65}")
    print(f"  BENCHMARK: {DATASET_CONFIGS[dataset_name].display_name}")
    print(f"  Seeds: {seeds} | Device: {device}")
    print(f"{'='*65}\n")

    results_dir = _ensure_results_dir(dataset_name)

    # ── Step 1: Load / preprocess data ────────────────────────────────────────
    print("[Step 1/6] Loading dataset...")
    t0 = time.time()
    processed_data = get_dataset(dataset_name)
    print(f"  Users: {processed_data['num_users']:,} | "
          f"Items: {processed_data['num_items']:,} | "
          f"Cold-start: {len(processed_data['cold_start_users']):,} | "
          f"({time.time()-t0:.1f}s)")

    cfg = DATASET_CONFIGS[dataset_name]

    # ── Step 2: Extract LLM embeddings (MiniLM-L6) ───────────────────────────
    print("\n[Step 2/6] Extracting MiniLM embeddings...")
    emb_path = cfg.embeddings_path

    if os.path.exists(emb_path):
        print(f"  Cached embeddings found at {emb_path}")
    else:
        try:
            from models.llm_encoder import extract_llm_embeddings
            t0 = time.time()
            extract_llm_embeddings(
                item_metadata=processed_data["item_metadata"],
                output_path=emb_path,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                batch_size=256,
                text_field="text",
            )
            print(f"  Embeddings extracted in {time.time()-t0:.1f}s → {emb_path}")
        except Exception as e:
            print(f"  [Warning] Embedding extraction failed: {e}")
            print("  Continuing with random embeddings as fallback.")

    # ── Step 3: Multi-seed training and evaluation ────────────────────────────
    print(f"\n[Step 3/6] Multi-seed training ({len(seeds)} seeds)...")

    # Import after optional CUDA check
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    except ImportError:
        print("[Warning] PyTorch not available. Skipping training.")
        return {"dataset": dataset_name, "error": "PyTorch not available"}

    all_seed_results: Dict[int, Dict] = {}

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        seed_results = _train_and_eval_single_seed(
            processed_data=processed_data,
            cfg=cfg,
            seed=seed,
            device=device,
            emb_path=emb_path,
            results_dir=results_dir,
            run_bert4rec=run_bert4rec,
            run_gru4rec=run_gru4rec,
            use_enhanced_router=use_enhanced_router,
        )
        all_seed_results[seed] = seed_results

    # ── Step 4: Aggregate multi-seed statistics ───────────────────────────────
    print(f"\n[Step 4/6] Aggregating multi-seed results...")
    agg_results = _aggregate_seed_results(all_seed_results)

    agg_path = os.path.join(results_dir, "multi_seed_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg_results, f, indent=2)
    print(f"  Aggregate results saved to {agg_path}")

    # ── Step 5: Regime-Aware Evaluation ──────────────────────────────────────
    rae_results = None
    if run_regime_evaluation:
        print(f"\n[Step 5/6] Regime-Aware Evaluation...")
        try:
            from evaluation.regime_aware_evaluator import run_regime_aware_evaluation

            # Use last seed's predictions for regime analysis
            last_seed = seeds[-1]
            if last_seed in all_seed_results and "predictions" in all_seed_results[last_seed]:
                preds = all_seed_results[last_seed]["predictions"]
                drift_scores = all_seed_results[last_seed].get("drift_scores", None)

                rae_results = run_regime_aware_evaluation(
                    model_predictions=preds,
                    test_user_seq=processed_data["test_user_seq"],
                    user_drift_scores=drift_scores,
                    baseline_name="Static Fusion",
                    k=10,
                    output_path=os.path.join(results_dir, "regime_aware_results.json"),
                )
        except Exception as e:
            print(f"  [Warning] Regime evaluation failed: {e}")

    # ── Step 6: Calibration analysis ─────────────────────────────────────────
    calib_results = None
    if run_calibration:
        print(f"\n[Step 6/6] Calibration Analysis...")
        try:
            from analysis.calibration_analysis import run_calibration_analysis
            last_seed = seeds[-1]
            if last_seed in all_seed_results and "uncertainty_data" in all_seed_results[last_seed]:
                udata = all_seed_results[last_seed]["uncertainty_data"]
                calib_results = run_calibration_analysis(
                    uc_collab=udata["uc_collab"],
                    uc_semantic=udata["uc_semantic"],
                    hit_collab=udata["hit_collab"],
                    hit_semantic=udata["hit_semantic"],
                    hit_router=udata["hit_router"],
                    output_path=os.path.join(results_dir, "calibration_results.json"),
                )
        except Exception as e:
            print(f"  [Warning] Calibration analysis failed: {e}")

    return {
        "dataset":          dataset_name,
        "display_name":     cfg.display_name,
        "num_users":        processed_data["num_users"],
        "num_items":        processed_data["num_items"],
        "seeds":            seeds,
        "aggregate":        agg_results,
        "regime_aware":     rae_results,
        "calibration":      calib_results,
    }


def _train_and_eval_single_seed(
    processed_data: Dict,
    cfg,
    seed: int,
    device: str,
    emb_path: str,
    results_dir: str,
    run_bert4rec: bool,
    run_gru4rec: bool,
    use_enhanced_router: bool,
) -> Dict:
    """Train all models for a single seed and return per-model metrics."""
    import torch
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {}

    try:
        from train_and_evaluate_evidence_routing import run_full_evidence_routing_pipeline
        pipeline_results = run_full_evidence_routing_pipeline(
            processed_data=processed_data,
            cfg=cfg,
            seed=seed,
            device=device,
            emb_path=emb_path,
            checkpoint_dir=os.path.join(results_dir, f"checkpoints_seed{seed}"),
            run_bert4rec=run_bert4rec,
            run_gru4rec=run_gru4rec,
            use_enhanced_router=use_enhanced_router,
        )
        results.update(pipeline_results)
    except Exception as e:
        print(f"    [Warning] Pipeline failed for seed {seed}: {e}")
        import traceback
        traceback.print_exc()

    return results


def _aggregate_seed_results(all_seed_results: Dict[int, Dict]) -> Dict:
    """Computes mean ± std across seeds for all tracked metrics."""
    import numpy as np

    seeds = list(all_seed_results.keys())
    model_names = set()
    for seed_res in all_seed_results.values():
        if "metrics" in seed_res:
            model_names.update(seed_res["metrics"].keys())

    agg = {"seeds": seeds, "models": {}}

    for model in model_names:
        hr_vals   = []
        ndcg_vals = []
        for seed_res in all_seed_results.values():
            m = seed_res.get("metrics", {}).get(model, {})
            if "HR@10" in m:
                hr_vals.append(m["HR@10"])
            if "NDCG@10" in m:
                ndcg_vals.append(m["NDCG@10"])

        if hr_vals:
            agg["models"][model] = {
                "HR@10_mean":   float(np.mean(hr_vals)),
                "HR@10_std":    float(np.std(hr_vals)),
                "NDCG@10_mean": float(np.mean(ndcg_vals)) if ndcg_vals else None,
                "NDCG@10_std":  float(np.std(ndcg_vals))  if ndcg_vals else None,
                "per_seed_HR":  hr_vals,
            }

    return agg


def run_all_datasets(
    dataset_names: List[str],
    seeds: List[int],
    **kwargs,
) -> Dict[str, Any]:
    """
    Runs benchmarks across all specified datasets and compiles a summary.

    Returns:
        cross-dataset summary dict
    """
    all_results = {}

    for dataset_name in dataset_names:
        if dataset_name not in DATASET_CONFIGS:
            print(f"[Warning] Unknown dataset '{dataset_name}', skipping.")
            continue
        try:
            result = run_single_dataset_benchmark(
                dataset_name=dataset_name,
                seeds=seeds,
                **kwargs,
            )
            all_results[dataset_name] = result
        except Exception as e:
            print(f"[Error] Dataset '{dataset_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {"error": str(e)}

    # Compile cross-dataset summary table
    summary = _compile_cross_dataset_summary(all_results)
    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "cross_dataset_summary.json",
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  CROSS-DATASET SUMMARY")
    print(f"{'='*65}")
    _print_cross_dataset_table(summary)
    print(f"\nSaved to: {summary_path}")

    return summary


def _compile_cross_dataset_summary(all_results: Dict) -> Dict:
    """Compiles results from all datasets into a cross-dataset summary."""
    summary = {"datasets": {}}
    for dataset_name, result in all_results.items():
        if "error" in result:
            summary["datasets"][dataset_name] = {"error": result["error"]}
            continue
        agg = result.get("aggregate", {}).get("models", {})
        summary["datasets"][dataset_name] = {
            "display_name": result.get("display_name", dataset_name),
            "num_users":    result.get("num_users", 0),
            "num_items":    result.get("num_items", 0),
            "models":       agg,
        }
    return summary


def _print_cross_dataset_table(summary: Dict) -> None:
    """Prints a cross-dataset comparison table."""
    datasets = list(summary["datasets"].keys())
    if not datasets:
        return

    # Collect all model names
    model_names = set()
    for ds_data in summary["datasets"].values():
        if "models" in ds_data:
            model_names.update(ds_data["models"].keys())
    model_names = sorted(model_names)

    print(f"\n| Dataset | Users | Items |" + "".join(f" {m} HR@10 |" for m in model_names))
    print(f"| :--- | :---: | :---: |" + " :---: |" * len(model_names))

    for ds_name in datasets:
        ds = summary["datasets"][ds_name]
        if "error" in ds:
            print(f"| {ds_name} | ERROR | — |" + " — |" * len(model_names))
            continue
        row = f"| {ds.get('display_name', ds_name)} | {ds.get('num_users', '?'):,} | {ds.get('num_items', '?'):,} |"
        for m in model_names:
            m_data = ds.get("models", {}).get(m, {})
            hr = m_data.get("HR@10_mean", None)
            std = m_data.get("HR@10_std", None)
            if hr is not None:
                row += f" {hr:.4f}±{std:.4f} |" if std is not None else f" {hr:.4f} |"
            else:
                row += " — |"
        print(row)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Selective Evidence Routing Benchmark"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["ml1m"],
        help="Dataset names to run (e.g., ml1m amazon_beauty steam)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all registered datasets"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 2024, 7, 99],
        help="Random seeds for multi-seed evaluation"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device (cuda or cpu)"
    )
    parser.add_argument(
        "--no-bert4rec", action="store_true",
        help="Skip BERT4Rec baseline (saves training time)"
    )
    parser.add_argument(
        "--no-gru4rec", action="store_true",
        help="Skip GRU4Rec baseline"
    )
    parser.add_argument(
        "--no-enhanced-router", action="store_true",
        help="Skip EnhancedRouter (12D features)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available datasets and exit"
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    dataset_names = (
        list(DATASET_CONFIGS.keys()) if args.all
        else args.datasets
    )

    print(f"\n[Multi-Dataset Benchmark]")
    print(f"  Datasets: {dataset_names}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Device:   {args.device}")

    run_all_datasets(
        dataset_names=dataset_names,
        seeds=args.seeds,
        device=args.device,
        run_bert4rec=not args.no_bert4rec,
        run_gru4rec=not args.no_gru4rec,
        use_enhanced_router=not args.no_enhanced_router,
    )


if __name__ == "__main__":
    main()
