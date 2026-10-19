"""
Unified Master Pipeline Runner for Selective Evidence Routing.
Provides CLI interface to run, verify, and reproduce:
- Symmetric multi-seed benchmark across 5 deterministic seeds [42, 123, 2024, 7, 99]
- Informational regimes across prefix lengths L in [1, 2, 3, 5, 10, 20]
- Controlled vs. natural drift detection
- Publication figures generation (Figures 1, 2, 3, 4)
- Comprehensive failure and diagnostic analysis
"""

import os
import sys
import argparse
import subprocess

def run_command_stream(cmd):
    print(f"\n[RUNNING] {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"[ERROR] Command failed with exit code {ret.returncode}: {cmd}")
        sys.exit(ret.returncode)

def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for Selective Evidence Routing")
    parser.add_argument("--multiseed", action="store_true", help="Display / verify symmetric 5-seed benchmark")
    parser.add_argument("--regimes", action="store_true", help="Display sequence length regimes and popularity collapse")
    parser.add_argument("--drift", action="store_true", help="Display controlled vs. natural drift analysis")
    parser.add_argument("--figures", action="store_true", help="Regenerate all 4 publication figures")
    parser.add_argument("--diagnostics", action="store_true", help="Run comprehensive failure analysis")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline and verify all artifacts")

    args = parser.parse_args()

    # If no flags passed, display help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    py_exe = os.path.join("venv", "Scripts", "python.exe") if os.path.exists(os.path.join("venv", "Scripts", "python.exe")) else sys.executable

    if args.figures or args.all:
        print("\n" + "=" * 70)
        print(" GENERATING PUBLICATION-QUALITY FIGURES (FIG 1-4)")
        print("=" * 70)
        run_command_stream(f'"{py_exe}" generate_fig1.py')
        run_command_stream(f'"{py_exe}" generate_publication_figures.py')

    if args.diagnostics or args.regimes or args.drift or args.all:
        print("\n" + "=" * 70)
        print(" RUNNING DIAGNOSTICS & REGIME FAILURE ANALYSIS")
        print("=" * 70)
        run_command_stream(f'"{py_exe}" diagnostics_and_failure_analysis.py')

    if args.multiseed or args.all:
        print("\n" + "=" * 70)
        print(" VERIFYING SYMMETRIC 5-SEED REPLICATION BENCHMARK")
        print("=" * 70)
        sym_path = "symmetric_multiseed_results.json" if os.path.exists("symmetric_multiseed_results.json") else os.path.join("results", "symmetric_multiseed_results.json")
        if os.path.exists(sym_path):
            import json
            with open(sym_path, "r") as f:
                data = json.load(f)
            table = data["symmetric_multiseed_benchmark"]
            print(f"{'Model':<30} | {'HR@10 (Mean +- Std)':<22} | {'NDCG@10 (Mean +- Std)':<22} | {'Oracle Gap'}")
            print("-" * 90)
            for row in table:
                print(f"{row['Model']:<30} | {row['HR@10 (Mean +- Std)']:<22} | {row['NDCG@10 (Mean +- Std)']:<22} | {row['NDCG@10 Oracle Gap (%)']}")
        else:
            print("[INFO] symmetric_multiseed_results.json not found in root or results/. Running full 5-seed training...")
            run_command_stream(f'"{py_exe}" run_symmetric_multiseed.py')

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    print(" All artifacts are available in:")
    print(" - Figures: figures/fig1_architecture.png, fig2_length_regimes.png, fig3_drift_detection.png, fig4_popularity_vs_coverage.png")
    print(" - Benchmarks: results/symmetric_multiseed_results.json, results/final_validation_results.json")
    print(" - Paper LaTeX: main.tex, references.bib, sections/")
    print("=" * 70)

if __name__ == "__main__":
    main()
