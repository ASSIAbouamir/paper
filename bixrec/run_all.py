"""Resumable experiment grid. Skips any run whose metrics.json already exists.

    python -m bixrec.run_all --data_root data --runs runs                 # everything, priority order
    python -m bixrec.run_all --datasets ml1m --stage core                 # a subset

Stages (run in this order):
    core     : sasrec, text, hybrid, dual_late, bix, then poe (needs sasrec + text)   x 5 seeds x all datasets
    ablation : dual_late_pen, bix_nogate, bix_gate                                     x 5 seeds x all datasets
    audit    : sasrec_lastpos, sasrec_bce, bix_lastpos                                 x 3 seeds x ML-1M only
"""

import argparse
import os
import subprocess
import sys
import time

SEEDS = [42, 123, 2024, 7, 99]
STAGES = {
    "core": (["sasrec", "text", "hybrid", "dual_late", "bix", "poe"], SEEDS, None),
    "ablation": (["dual_late_pen", "bix_nogate", "bix_gate"], SEEDS, None),
    "audit": (["sasrec_lastpos", "sasrec_bce", "bix_lastpos"], SEEDS[:3], ["ml1m", "ml100k"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--datasets", nargs="+", default=["ml1m", "beauty", "sports"])
    ap.add_argument("--stage", nargs="+", default=list(STAGES))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--extra", default="", help="extra args passed to bixrec.train, e.g. '--epochs 100'")
    a = ap.parse_args()

    for stage in a.stage:
        variants, seeds, only = STAGES[stage]
        seeds = a.seeds or seeds
        for ds in a.datasets:
            if only and ds not in only:
                continue
            if not os.path.exists(os.path.join(a.data_root, ds, "dataset.pt")):
                print(f"[skip] {ds}: no dataset.pt")
                continue
            for seed in seeds:
                for v in variants:
                    done = os.path.join(a.runs, ds, "poe" if v == "poe" else v, f"seed{seed}", "metrics.json")
                    if os.path.exists(done):
                        continue
                    cmd = [sys.executable, "-m", "bixrec.train", "--data", os.path.join(a.data_root, ds),
                           "--variant", v, "--seed", str(seed), "--out", a.runs, "--quiet"] + a.extra.split()
                    t0 = time.time()
                    print(f"[run] {stage} {ds} {v} seed{seed}", flush=True)
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        print(res.stdout[-2000:], res.stderr[-4000:])
                        raise SystemExit(f"failed: {' '.join(cmd)}")
                    print(f"      done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
