"""Train / evaluate one (dataset, variant, seed) run.

    python -m bixrec.train --data data/ml1m --variant bix --seed 42 --out runs
    python -m bixrec.train --data data/ml1m --variant poe --seed 42 --out runs   # needs sasrec + text runs

Each run writes runs/<dataset>/<variant>[@tag]/seed<k>/ with
    ranks_*.npy (val, val_masked, test, test_masked), model.pt, then metrics.json (written last = "done")

Protocol (v4):
* Ranking is over the full catalogue. The primary metric keeps the user's history items in the ranking, as the
  published side-information results do (RecBole's full-sort evaluation does not mask history for sequential
  models; DIFF states it does not penalise repeated items). "masked" variants remove the history items in the
  input window (BSARec / DLFSRec-style evaluation) and are reported as a secondary view.
* Ties between equal scores are broken by a fixed random order of the items (= random tie-breaking, but
  deterministic), so models whose item vectors coincide (identical content) are not penalised by the tie rule.
* Early stopping, the PoE alpha and every tuning decision use validation NDCG@10. --no_test skips the test split
  entirely (tuning stage). Hyperparameters: command line > <data>/hparams.json (run_all --stage tune) > defaults;
  if hparams.json exists, every variant must be in it (equal tuning budget) unless --lr is given explicitly.
"""

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import summarize
from .models import SeqRec, count_params, trim_left

VARIANTS: Dict[str, dict] = {
    # single-stream
    "sasrec":         dict(kind="id"),
    "sasrec_bce":     dict(kind="id", loss="bce"),         # v1 expert loss (1 sampled negative)
    "sasrec_lastpos": dict(kind="id", positions="last"),   # v1 BiX-GDR training bug
    "text":           dict(kind="text"),
    "hybrid":         dict(kind="hybrid"),                 # early fusion: ID + text item embedding
    # dual-stream, jointly trained
    "dual_late":      dict(kind="dual"),                                   # no cross-attention
    "dual_late_pen":  dict(kind="dual", penalty=True),
    "bix_nogate":     dict(kind="dual", cross=True),
    "bix_gate":       dict(kind="dual", cross=True, gate=True),
    "bix":            dict(kind="dual", cross=True, gate=True, penalty=True),  # full model
    "bix_lastpos":    dict(kind="dual", cross=True, gate=True, penalty=True, positions="last"),
    # + per-stream auxiliary CE (each stream must also predict the next item on its own)
    "dual_late_aux":  dict(kind="dual", aux=0.5),
    "bix_aux":        dict(kind="dual", cross=True, gate=True, penalty=True, aux=0.5),
    # + early-fused content stream (content stream items = ID + text), v4
    "dual_late_ef_aux": dict(kind="dual", ef=True, aux=0.5),
    "bix_ef_aux":     dict(kind="dual", cross=True, gate=True, penalty=True, ef=True, aux=0.5),
}
POSTHOC = {"poe"}  # late fusion of separately trained sasrec + text experts

DATASET_DEFAULTS = {
    # max_len, dropout
    "ml1m": dict(max_len=200, dropout=0.2),
    "ml100k": dict(max_len=200, dropout=0.2),
    "default": dict(max_len=50, dropout=0.5),
}
DEFAULT_D, DEFAULT_LR = 64, 1e-3
MPS_FREE_EVERY = 25  # steps; batch shapes vary, and the MPS allocator otherwise keeps every freed block
TIE_SEED = 0
# settings that must not change between a checkpoint and its resumed run
RESUME_KEYS = ("d", "lr", "dropout", "max_len", "emb", "aux", "loss", "positions", "batch", "wd", "layers", "heads",
               "epochs", "patience", "min_epochs", "max_users", "git_sha")


# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def mem_gb(dev: torch.device) -> float:
    """Device memory in use (MPS), peak allocated (CUDA) or peak process RSS (CPU), in GB."""
    if dev.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


def git_sha() -> str:
    """Commit of the code that produced a run; '-dirty' if tracked files were modified."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def left_pad(x: List[int], L: int) -> List[int]:
    x = x[-L:]
    return [0] * (L - len(x)) + x


def make_splits(user_seqs: List[List[int]], L: int):
    tr_in, tr_tg = [], []
    for s in user_seqs:
        t = s[:-2]
        if len(t) >= 2:
            tr_in.append(left_pad(t[:-1], L))
            tr_tg.append(left_pad(t[1:], L))
    val = (torch.tensor([left_pad(s[:-2], L) for s in user_seqs]), torch.tensor([s[-2] for s in user_seqs]))
    test = (torch.tensor([left_pad(s[:-1], L) for s in user_seqs]), torch.tensor([s[-1] for s in user_seqs]))
    return (torch.tensor(tr_in), torch.tensor(tr_tg)), val, test


_TIE_KEYS: Dict[Tuple[int, str], torch.Tensor] = {}


def tie_key(n: int, device) -> torch.Tensor:
    """Fixed random order of the n item columns, used only to break exact score ties."""
    k = (n, str(device))
    if k not in _TIE_KEYS:
        _TIE_KEYS[k] = torch.randperm(n, generator=torch.Generator().manual_seed(TIE_SEED)).to(device)
    return _TIE_KEYS[k]


def ranks_of(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rank (1 = best) of the target among all items. Exact ties are broken by a fixed random item order;
    a non-finite target score gets the worst rank."""
    st = scores.gather(1, target[:, None])
    key = tie_key(scores.shape[1], scores.device)
    r = (scores > st).sum(1) + ((scores == st) & (key[None, :] < key[target][:, None])).sum(1) + 1
    return torch.where(torch.isfinite(st[:, 0]), r, torch.full_like(r, scores.shape[1]))


def free_cache(dev):
    """Return cached, unused device memory (memory management only; no effect on any computed value).
    Without it, the many batch shapes leave the MPS allocator holding ~18 GB of freed blocks."""
    if dev.type == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def score_batches(model: SeqRec, seqs: torch.Tensor, dev, bs: int = 256, head: str = None):
    """Yields (row indices, batch_seq, final-position scores (B, N+1)) in length-sorted order.
    head='c' / 's' scores with one stream of a dual model only (analysis)."""
    model.eval()
    tables = model.item_tables()
    order = torch.argsort(seqs.ne(0).sum(1), stable=True)  # length-sorted so trimming is effective
    for i in range(0, len(seqs), bs):
        idx = order[i:i + bs]
        b = seqs[idx].to(dev)
        bt, = trim_left(b)
        h = model.encode(bt, tables)
        last = {k: v[:, -1] for k, v in h.items()}
        if head is None:
            yield idx, b, model.logits(last, tables)
        else:
            yield idx, b, model.stream_logits(last, tables)[0 if head == "c" else 1]


@torch.no_grad()
def evaluate(model, split, dev, masked: bool = False, head: str = None) -> np.ndarray:
    seqs, tgt = split
    out = torch.zeros(len(seqs), dtype=torch.long)
    for idx, b, sc in score_batches(model, seqs, dev, head=head):
        if masked:
            sc = sc.scatter(1, b, float("-inf"))
        out[idx] = ranks_of(sc, tgt[idx].to(dev)).cpu()
    free_cache(dev)
    return out.numpy().astype(np.int32)


# ---------------------------------------------------------------------------
def train_model(model: SeqRec, train, val, dev, cfg) -> Tuple[dict, int, float, float]:
    """Returns (best state, best epoch, seconds, max memory GB). Selection: validation NDCG@10."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    inp, tgt = train
    lens = inp.ne(0).sum(1)
    g = torch.Generator().manual_seed(cfg["seed"])
    best, best_state, best_ep, bad, peak = -1.0, None, 0, 0, 0.0
    t0 = time.time()
    sig = {k: cfg.get(k) for k in RESUME_KEYS}
    start_ep, ckpt = 1, cfg.get("ckpt")
    if ckpt and os.path.exists(ckpt):  # resume an interrupted run exactly where it stopped
        c = torch.load(ckpt, weights_only=False)
        if c.get("sig", sig) != sig:
            raise SystemExit(f"{ckpt} was written with different settings {c.get('sig')} != {sig}; delete it")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        g.set_state(c["gen"])
        torch.set_rng_state(c["torch_rng"])
        np.random.set_state(c["np_rng"])
        random.setstate(c["py_rng"])
        if c.get("cuda_rng") is not None and torch.cuda.is_available():  # dropout / BCE negatives on GPU
            torch.cuda.set_rng_state_all(c["cuda_rng"])
        if c.get("mps_rng") is not None and dev.type == "mps":
            torch.mps.set_rng_state(c["mps_rng"])
        best, best_state, best_ep, bad, start_ep = c["best"], c["best_state"], c["best_ep"], c["bad"], c["ep"] + 1
        peak = c.get("peak", 0.0)
        t0 -= c["elapsed"]
        print(f"resumed from epoch {c['ep']}", flush=True)
    loss_kind = cfg.get("loss", "ce")
    positions = cfg.get("positions", "all")
    for ep in range(start_ep, cfg["epochs"] + 1):
        model.train()
        # length-bucketed batches: shuffle, sort within chunks of 50 batches by length, shuffle batch order
        perm = torch.randperm(len(inp), generator=g)
        chunk = 50 * cfg["batch"]
        batches = []
        for c in range(0, len(perm), chunk):
            part = perm[c:c + chunk]
            part = part[torch.argsort(lens[part], stable=True)]
            batches += [part[i:i + cfg["batch"]] for i in range(0, len(part), cfg["batch"])]
        order = torch.randperm(len(batches), generator=g)
        tot, nb = 0.0, 0
        for bi in order.tolist():
            idx = batches[bi]
            x, y = trim_left(inp[idx], tgt[idx])
            x, y = x.to(dev), y.to(dev)
            if positions == "last":
                sel = torch.zeros_like(y, dtype=torch.bool)
                sel[:, -1] = True
                sel &= y.ne(0)
            else:
                sel = y.ne(0)
            tables = model.item_tables()
            h = model.encode(x, tables)
            hs = {k: v[sel] for k, v in h.items()}
            ys = y[sel]
            if loss_kind == "ce" and cfg.get("aux"):
                out, sc, ss = model.logits(hs, tables, streams=True)
                loss = F.cross_entropy(out, ys) + cfg["aux"] * (
                    F.cross_entropy(sc[:, 1:], ys - 1) + F.cross_entropy(ss[:, 1:], ys - 1))
            elif loss_kind == "ce":
                loss = F.cross_entropy(model.logits(hs, tables), ys)
            else:  # BCE with one uniformly sampled negative per position
                neg = torch.randint(1, model.num_items + 1, ys.shape, device=dev, generator=None)
                pos_s = model.pair_scores(hs, tables, ys)
                neg_s = model.pair_scores(hs, tables, neg)
                loss = F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s)) + \
                    F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
            if nb % MPS_FREE_EVERY == 0:
                free_cache(dev)
        peak = max(peak, mem_gb(dev))
        m = summarize(evaluate(model, val, dev))
        if m["NDCG@10"] > best:
            best, best_ep, bad = m["NDCG@10"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if cfg["verbose"]:
            print(f"ep {ep:3d} loss {tot / max(nb, 1):.4f} val HR@10 {m['HR@10']:.4f} "
                  f"NDCG@10 {m['NDCG@10']:.4f} best@{best_ep} ({time.time() - t0:.0f}s, {peak:.1f} GB)", flush=True)
        if bad >= cfg["patience"] and ep >= cfg.get("min_epochs", 0):
            break
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(),
                        "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                        "py_rng": random.getstate(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "mps_rng": torch.mps.get_rng_state() if dev.type == "mps" else None,
                        "best": best, "best_state": best_state, "peak": peak, "sig": sig,
                        "best_ep": best_ep, "bad": bad, "ep": ep, "elapsed": time.time() - t0}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    model.load_state_dict(best_state)
    return best_state, best_ep, time.time() - t0, peak


# ---------------------------------------------------------------------------
def load_data(data_dir: str, emb_file: str = "text_emb.pt"):
    ds = torch.load(os.path.join(data_dir, "dataset.pt"), weights_only=False)
    p = os.path.join(data_dir, emb_file)
    emb = torch.load(p, weights_only=False)["emb"] if os.path.exists(p) else None
    return ds, emb


def emb_description(data_dir: str, emb_file: str) -> str:
    p = os.path.join(data_dir, emb_file)
    return torch.load(p, weights_only=False).get("model", "unknown") if os.path.exists(p) else "none"


def load_hparams(data_dir: str) -> dict:
    p = os.path.join(data_dir, "hparams.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def resolve_hparams(cli: dict, hp: dict, variant: str, dflt: dict) -> dict:
    """d / dropout / lr / emb with precedence: command line > hparams.json > defaults.
    With an hparams.json, a trained variant missing from its lr table is an error (equal tuning budget)."""
    lr_file = hp.get("lr", {})
    lr = cli.get("lr") or lr_file.get(variant)
    if lr is None and hp and variant in VARIANTS:
        raise SystemExit(f"'{variant}' was not tuned (not in hparams.json lr table); run the tune stage for it "
                         f"or pass --lr explicitly")
    return {
        "d": cli.get("d") or hp.get("d") or DEFAULT_D,
        "dropout": cli["dropout"] if cli.get("dropout") is not None else hp.get("dropout", dflt["dropout"]),
        "lr": lr or DEFAULT_LR,
        "emb": cli.get("emb") or (f"text_emb_{hp['encoder']}.pt" if hp.get("encoder") else "text_emb.pt"),
    }


def build_model(ds, emb, variant_cfg, cfg):
    return SeqRec(ds["num_items"], variant_cfg["kind"], emb, d=cfg["d"], layers=cfg["layers"],
                  heads=cfg["heads"], max_len=cfg["max_len"], dropout=cfg["dropout"],
                  cross=variant_cfg.get("cross", False), gate=variant_cfg.get("gate", False),
                  penalty=variant_cfg.get("penalty", False), ef=variant_cfg.get("ef", False))


def _atomic(path: str, write):
    write(path + ".tmp")
    os.replace(path + ".tmp", path)


def save_run(out: str, metrics: dict, ranks: Dict[str, np.ndarray], state=None):
    """Ranks and checkpoint first, metrics.json last: its presence means the run is complete."""
    os.makedirs(out, exist_ok=True)
    for k, v in ranks.items():
        def save_np(p, v=v):
            with open(p, "wb") as f:
                np.save(f, v)
        _atomic(os.path.join(out, f"ranks_{k}.npy"), save_np)
    if state is not None:
        _atomic(os.path.join(out, "model.pt"), lambda p: torch.save(state, p))

    def dump(p):
        with open(p, "w") as f:
            json.dump(metrics, f, indent=2)
    _atomic(os.path.join(out, "metrics.json"), dump)


def run_poe(ds, emb, cfg, run_root, dev, env):
    """Static product of experts of separately trained SASRec and text experts.
    log p = a * log p_C + (1 - a) * log p_S ; a tuned on validation NDCG@10 (grid 0..1 step .05),
    plus the fixed a = 0.5 rule ("poe50")."""
    _, val, test = make_splits(ds["user_seqs"], cfg["max_len"])
    experts = {}
    for name in ("sasrec", "text"):
        d = os.path.join(run_root, name, f"seed{cfg['seed']}")
        if not os.path.exists(os.path.join(d, "metrics.json")):
            raise FileNotFoundError(f"PoE needs a finished '{name}' run in {d}; train it with the same seed first")
        with open(os.path.join(d, "metrics.json")) as f:
            ecfg = json.load(f)
        if name == "text" and ecfg.get("env", {}).get("emb_model", env["emb_model"]) != env["emb_model"]:
            raise SystemExit(f"text expert was trained on {ecfg['env']['emb_model']}, now {env['emb_model']}")
        m = build_model(ds, emb, VARIANTS[name], dict(cfg, d=ecfg["config"]["d"])).to(dev)
        m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev))
        experts[name] = m
    alphas = torch.linspace(0, 1, 21)

    @torch.no_grad()
    def fused_ranks(split, alpha_list, masked=False):
        seqs, tgt = split
        outs = [torch.zeros(len(seqs), dtype=torch.long) for _ in alpha_list]
        gen_c = score_batches(experts["sasrec"], seqs, dev)
        gen_s = score_batches(experts["text"], seqs, dev)
        for (idx, b, sc), (idx2, _, ss) in zip(gen_c, gen_s):
            assert torch.equal(idx, idx2)
            lc, ls = F.log_softmax(sc, 1), F.log_softmax(ss, 1)
            t = tgt[idx].to(dev)
            for k, a in enumerate(alpha_list):
                f = a * lc + (1 - a) * ls
                f[:, 0] = float("-inf")
                if masked:
                    f = f.scatter(1, b, float("-inf"))
                outs[k][idx] = ranks_of(f, t).cpu()
        return [o.numpy().astype(np.int32) for o in outs]

    val_r = fused_ranks(val, alphas.tolist())
    nd = [summarize(r)["NDCG@10"] for r in val_r]
    best_a = float(alphas[int(np.argmax(nd))])
    results = {}
    for tag, a in (("poe50", 0.5), ("poe", best_a)):  # poe last: its metrics.json marks the pair done
        ranks = {"val": val_r[int(round(a * 20))]}
        ranks["val_masked"], = fused_ranks(val, [a], masked=True)
        metrics = {"variant": tag, "seed": cfg["seed"], "alpha": a, "params": 0, "env": env,
                   "config": {k2: v for k2, v in cfg.items() if k2 != "verbose"}}
        if not cfg["no_test"]:
            ranks["test"], = fused_ranks(test, [a])
            ranks["test_masked"], = fused_ranks(test, [a], masked=True)
        metrics.update({k: summarize(v) for k, v in ranks.items()})
        save_run(os.path.join(run_root, tag, f"seed{cfg['seed']}"), metrics, ranks)
        results[tag] = metrics
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(allow_abbrev=False)  # no prefix tricks around run_all's flag guard
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS) + sorted(POSTHOC))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="", help="run dir suffix: <variant>@<tag> (tuning trials)")
    ap.add_argument("--emb", default=None, help="text embedding file inside --data (default: hparams encoder)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--no_test", action="store_true", help="never evaluate the test split (tuning)")
    ap.add_argument("--allow_dirty", action="store_true", help="allow running uncommitted code (tests only)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_epochs", type=int, default=20, help="no early stopping before this epoch")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="default: hparams.json, else 1e-3")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--d", type=int, default=None, help="default: hparams.json, else 64")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--max_users", type=int, default=0, help="debug: subsample users")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main():
    a = build_parser().parse_args()
    sha = git_sha()
    if sha.endswith("-dirty") and not a.allow_dirty:
        raise SystemExit("uncommitted changes in tracked files: commit first so every run maps to one commit")
    name = os.path.basename(os.path.normpath(a.data))
    dflt = DATASET_DEFAULTS.get(name, DATASET_DEFAULTS["default"])
    cfg = dict(vars(a))
    cfg.update(resolve_hparams(vars(a), load_hparams(a.data), a.variant, dflt))
    cfg["max_len"] = a.max_len or dflt["max_len"]
    cfg["verbose"] = not a.quiet
    cfg["dataset"] = name
    cfg["git_sha"] = sha  # part of the resume signature: a resumed run never mixes code versions
    dev = pick_device(a.device)
    set_seed(a.seed)
    env = {"device": dev.type, "git_sha": sha, "torch": torch.__version__,
           "emb": cfg["emb"], "emb_model": emb_description(a.data, cfg["emb"]), "tie_seed": TIE_SEED}

    ds, emb = load_data(a.data, cfg["emb"])
    if a.max_users:
        ds = dict(ds)
        ds["user_seqs"] = ds["user_seqs"][:a.max_users]
    run_root = os.path.join(a.out, name)

    if a.variant == "poe":
        res = run_poe(ds, emb, cfg, run_root, dev, env)
        for k, v in res.items():
            print(k, "alpha", v["alpha"], "val", v["val"]["NDCG@10"])
        return

    vcfg = VARIANTS[a.variant]
    cfg.update({k: v for k, v in vcfg.items() if k in ("loss", "positions", "aux")})
    train, val, test = make_splits(ds["user_seqs"], cfg["max_len"])
    model = build_model(ds, emb, vcfg, cfg).to(dev)
    run_dir = os.path.join(run_root, a.variant + (f"@{a.tag}" if a.tag else ""), f"seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    cfg["ckpt"] = os.path.join(run_dir, "ckpt.pt")
    state, best_ep, secs, peak = train_model(model, train, val, dev, cfg)
    ranks = {"val": evaluate(model, val, dev), "val_masked": evaluate(model, val, dev, masked=True)}
    if not a.no_test:
        ranks["test"] = evaluate(model, test, dev)
        ranks["test_masked"] = evaluate(model, test, dev, masked=True)
    metrics = {"variant": a.variant, "seed": a.seed, "params": count_params(model), "best_epoch": best_ep,
               "train_seconds": secs, "env": dict(env, max_mem_gb=round(peak, 2)),
               **{k: summarize(v) for k, v in ranks.items()},
               "config": {k: v for k, v in cfg.items() if k not in ("verbose", "ckpt")}}
    if model.kind == "dual":  # analysis only, validation only
        w, lam = model.fusion_params()
        metrics["fusion_w"], metrics["fusion_lambda"] = w.item(), lam.item()
        metrics["streams"] = {h: {"val": summarize(evaluate(model, val, dev, head=h))} for h in ("c", "s")}
    save_run(run_dir, metrics, ranks, state)
    if os.path.exists(cfg["ckpt"]):
        os.remove(cfg["ckpt"])
    shown = "val" if a.no_test else "test"
    print(json.dumps({"variant": a.variant, "seed": a.seed, shown: metrics[shown]}, indent=1))


if __name__ == "__main__":
    main()
