"""Resumable experiment grid. Skips any run whose metrics.json already exists (after checking it is not stale).

    python -m bixrec.run_all --data_root data --runs runs --stage tune        # 1. validation-only tuning
    python -m bixrec.run_all --data_root data --runs runs                     # 2. core -> ablation -> audit
    python -m bixrec.run_all --datasets sports --stage core --seeds 42        # a subset

Stages:
    tune     : per dataset, seed 42, --no_test (the test split is never evaluated), selection on validation
               NDCG@10; writes <data>/hparams.json, which bixrec.train then reads:
               1. content encoder: the `text` model with each candidate encoder (attribute data: lsa, tfidf;
                  real text: lsa, sbert, sbert + whitening)
               2. backbone size d in {64, 128, 256}: SASRec, shared by every model
               3. learning rate in {1e-3, 1e-4}: the same two trials for EVERY model trained on that dataset
                  (equal tuning budget; model-specific settings such as aux = 0.5 are fixed, never tuned)
               Ties go to the earlier option in each list.
    core     : sasrec, text, hybrid, dual_late, bix, *_aux, *_ef_aux, then poe (needs sasrec + text)
    ablation : dual_late_pen, bix_nogate, bix_gate                  (beauty, sports, ml100k, ml1m)
    audit    : sasrec_lastpos, sasrec_bce, bix_lastpos               (ml100k, ml1m)
Training settings shared by all runs (epochs, patience, device, ...) go in --extra; they are fingerprinted, and a
finished run or tuning trial made with other settings is reported as stale instead of being silently reused.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

from .train import build_parser, pick_device

SEEDS = [42, 123, 2024]  # fixed before any v4 run (PR #3 decision, 2026-10-20)
STAGES = {
    "core": (["sasrec", "text", "hybrid", "dual_late", "bix", "dual_late_aux", "bix_aux", "dual_late_ef_aux",
              "bix_ef_aux", "poe"], SEEDS, None),
    "ablation": (["dual_late_pen", "bix_nogate", "bix_gate"], SEEDS, ["beauty", "sports", "ml100k", "ml1m"]),
    "audit": (["sasrec_lastpos", "sasrec_bce", "bix_lastpos"], SEEDS[:3], ["ml100k", "ml1m"]),
}
DATASETS = ["ml100k", "beauty", "sports", "beauty_text", "sports_text", "toys"]
ATTRIBUTE_DATASETS = {"beauty", "sports", "toys"}  # S3-Rec release: anonymised attribute ids as item "text"
ENCODERS = {  # name -> bixrec.embed arguments
    "lsa": ["--encoder", "lsa"],
    "tfidf": ["--encoder", "tfidf"],
    "sbert": ["--encoder", "sbert"],
    "sbert_w": ["--encoder", "sbert", "--whiten"],
}
TUNE_D = [64, 128, 256]
TUNE_LR = [1e-3, 1e-4]
TUNE_SEED = 42
# flags that run_all sets itself; allowing them in --extra would bypass tuning or break the skip logic
FORBIDDEN_EXTRA = {"--d", "--lr", "--emb", "--tag", "--out", "--seed", "--variant", "--no_test", "--data",
                   "--allow_dirty", "--dropout", "--max_len"}
FINGERPRINT = ("epochs", "patience", "min_epochs", "batch", "wd", "layers", "heads", "max_users", "device")


def encoder_candidates(ds: str) -> List[str]:
    return ["lsa", "tfidf"] if ds in ATTRIBUTE_DATASETS else ["lsa", "sbert", "sbert_w"]


def fingerprint(extra: str) -> Dict:
    toks = extra.split()
    bad = [t for t in toks if t.split("=")[0] in FORBIDDEN_EXTRA]
    if bad:
        raise SystemExit(f"--extra may not contain {bad}: run_all sets these itself")
    a = build_parser().parse_args(["--data", "-", "--variant", "sasrec"] + toks)
    fp = {k: getattr(a, k) for k in FINGERPRINT}
    fp["device"] = pick_device(a.device).type  # the device actually used, not the request ("auto")
    return fp


def check_fresh(metrics_path: str, fp: Dict, hp: Dict = None, variant: str = None):
    """A finished run is reused only if it was made with the current settings."""
    with open(metrics_path) as f:
        m = json.load(f)
    c = dict(m["config"], device=m.get("env", {}).get("device"))
    diff = {k: (c.get(k), v) for k, v in fp.items() if c.get(k) != v}
    if hp:
        want = {"d": hp["d"], "emb": f"text_emb_{hp['encoder']}.pt"}
        if variant in hp.get("lr", {}):
            want["lr"] = hp["lr"][variant]
        diff.update({k: (c.get(k), v) for k, v in want.items() if c.get(k) != v})
    if diff:
        raise SystemExit(f"stale run {metrics_path}: (was, now) = {diff}; delete it to rerun with current settings")


def run(cmd, label):
    t0 = time.time()
    print(f"[run] {label}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-4000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    print(f"      done in {time.time() - t0:.0f}s", flush=True)


def val_score(run_dir: str) -> float:
    """Validation NDCG@10 of a tuning run. Refuses runs that evaluated the test split."""
    with open(os.path.join(run_dir, f"seed{TUNE_SEED}", "metrics.json")) as f:
        m = json.load(f)
    if any(k.startswith("test") for k in m):
        raise SystemExit(f"{run_dir}: tuning run has test metrics; tuning must use --no_test")
    return m["val"]["NDCG@10"]


def pick(scores: Dict, order: List) -> object:
    """Best key by score; ties go to the earliest key in `order`."""
    return max(order, key=lambda k: (scores[k], -order.index(k)))


def stage_variants(ds: str) -> List[str]:
    out = []
    for variants, _, only in STAGES.values():
        if only is None or ds in only:
            out += [v for v in variants if v != "poe" and v not in out]
    return out


def write_json(path: str, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)


def tune(a, ds: str, fp: Dict):
    data = os.path.join(a.data_root, ds)
    hp_path = os.path.join(data, "hparams.json")
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            old = json.load(f).get("fingerprint")
        if old != fp:
            raise SystemExit(f"{hp_path} was tuned with {old}, now {fp}: delete it and {a.tune_runs}/{ds}")
        print(f"[skip] {ds}: {hp_path} exists (delete it and {a.tune_runs}/{ds} to re-tune)")
        return
    root = os.path.join(a.tune_runs, ds)
    train = [sys.executable, "-m", "bixrec.train", "--data", data, "--seed", str(TUNE_SEED), "--out", a.tune_runs,
             "--no_test", "--quiet"] + a.extra.split()

    def trial(variant, tag, *args):
        done = os.path.join(root, f"{variant}@{tag}", f"seed{TUNE_SEED}", "metrics.json")
        if os.path.exists(done):
            check_fresh(done, fp)
        else:
            run(train + ["--variant", variant, "--tag", tag, *args], f"tune {ds} {variant}@{tag}")
        return val_score(os.path.join(root, f"{variant}@{tag}"))

    trials = {}
    # 1. content encoder, judged by the text-only model (favours no fusion design)
    enc_scores = {}
    for e in encoder_candidates(ds):
        if not os.path.exists(os.path.join(data, f"text_emb_{e}.pt")):
            run([sys.executable, "-m", "bixrec.embed", "--data", data, "--name", e] + ENCODERS[e], f"embed {ds} {e}")
        enc_scores[e] = trials[f"text@emb_{e}"] = trial("text", f"emb_{e}", "--emb", f"text_emb_{e}.pt",
                                                       "--d", str(TUNE_D[0]), "--lr", str(TUNE_LR[0]))
    enc = pick(enc_scores, encoder_candidates(ds))
    emb = ["--emb", f"text_emb_{enc}.pt"]
    # 2. shared backbone size, judged by SASRec
    d_scores = {d: trial("sasrec", f"d{d}", "--d", str(d), "--lr", str(TUNE_LR[0]), *emb) for d in TUNE_D}
    trials.update({f"sasrec@d{d}": s for d, s in d_scores.items()})
    d = pick(d_scores, TUNE_D)
    # 3. learning rate: the same two trials for every model
    lrs = {}
    for v in stage_variants(ds):
        scores = {}
        for lr in TUNE_LR:
            if v == "sasrec" and lr == TUNE_LR[0]:
                scores[lr] = d_scores[d]  # identical configuration, already trained in step 2
                continue
            tag = f"d{d}_lr{lr:g}_{enc}"  # the encoder is part of the trial's identity
            scores[lr] = trials[f"{v}@{tag}"] = trial(v, tag, "--d", str(d), "--lr", str(lr), *emb)
        lrs[v] = pick(scores, TUNE_LR)
    write_json(hp_path, {"d": d, "lr": lrs, "encoder": enc, "fingerprint": fp,
                         "selected_on": f"validation NDCG@10, seed {TUNE_SEED}; test split never evaluated",
                         "trials": trials})
    print(f"[tuned] {ds}: d={d} encoder={enc} lr={lrs}", flush=True)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tune_runs", default="runs_tune")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=["tune"] + list(STAGES))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--extra", default="", help="settings shared by all runs, e.g. '--device mps --epochs 100'")
    ap.add_argument("--allow_defaults", action="store_true", help="train without hparams.json (not tuned)")
    a = ap.parse_args()
    fp = fingerprint(a.extra)

    for stage in a.stage:
        for ds in a.datasets:
            data = os.path.join(a.data_root, ds)
            if not os.path.exists(os.path.join(data, "dataset.pt")):
                print(f"[skip] {ds}: no dataset.pt")
                continue
            if stage == "tune":
                tune(a, ds, fp)
                continue
            variants, seeds, only = STAGES[stage]
            if only and ds not in only:
                continue
            hp = None
            if os.path.exists(os.path.join(data, "hparams.json")):
                with open(os.path.join(data, "hparams.json")) as f:
                    hp = json.load(f)
                if hp.get("fingerprint") != fp:
                    raise SystemExit(f"{ds}: hparams.json was tuned with {hp.get('fingerprint')}, now {fp}")
            elif not a.allow_defaults:
                raise SystemExit(f"{ds}: no hparams.json -- run --stage tune first (or --allow_defaults)")
            for seed in a.seeds or seeds:
                for v in variants:
                    done = os.path.join(a.runs, ds, v, f"seed{seed}", "metrics.json")
                    if v == "poe" and os.path.exists(done):  # PoE is stale if an expert was retrained after it
                        experts = [os.path.join(a.runs, ds, e, f"seed{seed}", "metrics.json") for e in ("sasrec", "text")]
                        if any(os.path.getmtime(e) > os.path.getmtime(done) for e in experts if os.path.exists(e)):
                            raise SystemExit(f"stale run {done}: an expert was retrained after it; delete it")
                    if os.path.exists(done):
                        check_fresh(done, fp, hp, v)
                        continue
                    run([sys.executable, "-m", "bixrec.train", "--data", data, "--variant", v, "--seed", str(seed),
                         "--out", a.runs, "--quiet"] + a.extra.split(), f"{stage} {ds} {v} seed{seed}")


if __name__ == "__main__":
    main()
