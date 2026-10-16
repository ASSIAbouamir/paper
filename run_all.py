"""
Master Orchestration Script for Selective Evidence Routing Reproducibility.
Executes the unified verification pipeline:
1. Symmetric 5-Seed Benchmark Verification
2. Sequence Length Regimes & Popularity vs. Coverage Analysis
3. Controlled & Natural Preference Drift Detection
4. Publication Figures Generation (Figures 1-4)
5. Comprehensive Failure Diagnostics
"""

import sys
import subprocess

def main():
    print("=" * 80)
    print(" RUNNING AUTHORITATIVE MASTER REPRODUCIBILITY SUITE")
    print("=" * 80)
    cmd = [sys.executable, "run_pipeline.py", "--all"]
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()
