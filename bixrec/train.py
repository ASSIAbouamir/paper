"""Train / evaluate one (dataset, variant, seed) run.

    python -m bixrec.train --data data/ml1m --variant bix --seed 42 --out runs/ml1m
    python -m bixrec.train --data data/ml1m --variant poe --seed 42 --out runs/ml1m   # needs sasrec + text runs

Each run writes runs/<dataset>/<variant>/seed<k>/ with
    metrics.json, ranks_val.npy, ranks_test.npy, ranks_test_masked.npy, model.pt
"""

import argparse
import json
import math
import os
import random
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
}
POSTHOC = {"poe"}  # late fusion of separately trained sasrec + text experts

DATASET_DEFAULTS = {
    # max_len, dropout
    "ml1m": dict(max_len=200, dropout=0.2),
    "ml100k": dict(max_len=200, dropout=0.2),
    "default": dict(max_len=50, dropout=0.5),
}


# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def ranks_of(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pessimistic rank of the target among all items (ties count against it)."""
    st = scores.gather(1, target[:, None])
    return (scores > st).sum(1) + (scores == st).sum(1)


@torch.no_grad()
def score_batches(model: SeqRec, seqs: torch.Tensor, dev, bs: int = 256):
    """Yields (row indices, batch_seq, final-position scores (B, N+1)) in length-sorted order."""
    model.eval()
    tables = model.item_tables()
    order = torch.argsort(seqs.ne(0).sum(1), stable=True)  # length-sorted so trimming is effective
    for i in range(0, len(seqs), bs):
        idx = order[i:i + bs]
        b = seqs[idx].to(dev)
        bt, = trim_left(b)
        h = model.encode(bt, tables)
        yield idx, b, model.logits({k: v[:, -1] for k, v in h.items()}, tables)


@torch.no_grad()
def evaluate(model, split, dev, masked: bool = False) -> np.ndarray:
    seqs, tgt = split
    out = torch.zeros(len(seqs), dtype=torch.long)
    for idx, b, sc in score_batches(model, seqs, dev):
        if masked:
            sc = sc.scatter(1, b, float("-inf"))
        out[idx] = ranks_of(sc, tgt[idx].to(dev)).cpu()
    return out.numpy().astype(np.int32)


# ---------------------------------------------------------------------------
def train_model(model: SeqRec, train, val, dev, cfg) -> Tuple[dict, int, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    inp, tgt = train
    lens = inp.ne(0).sum(1)
    g = torch.Generator().manual_seed(cfg["seed"])
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.time()
    loss_kind = cfg.get("loss", "ce")
    positions = cfg.get("positions", "all")
    for ep in range(1, cfg["epochs"] + 1):
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
            if loss_kind == "ce":
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
        r = evaluate(model, val, dev)
        m = summarize(r)
        if m["NDCG@10"] > best:
            best, best_ep, bad = m["NDCG@10"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if cfg["verbose"]:
            print(f"ep {ep:3d} loss {tot / max(nb, 1):.4f} val HR@10 {m['HR@10']:.4f} "
                  f"NDCG@10 {m['NDCG@10']:.4f} best@{best_ep} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return best_state, best_ep, time.time() - t0


# ---------------------------------------------------------------------------
def load_data(data_dir: str):
    ds = torch.load(os.path.join(data_dir, "dataset.pt"), weights_only=False)
    p = os.path.join(data_dir, "text_emb.pt")
    emb = torch.load(p, weights_only=False)["emb"] if os.path.exists(p) else None
    return ds, emb


def build_model(ds, emb, variant_cfg, cfg):
    return SeqRec(ds["num_items"], variant_cfg["kind"], emb, d=cfg["d"], layers=cfg["layers"],
                  heads=cfg["heads"], max_len=cfg["max_len"], dropout=cfg["dropout"],
                  cross=variant_cfg.get("cross", False), gate=variant_cfg.get("gate", False),
                  penalty=variant_cfg.get("penalty", False))


def save_run(out: str, metrics: dict, ranks: Dict[str, np.ndarray], state=None):
    os.makedirs(out, exist_ok=True)
    for k, v in ranks.items():
        np.save(os.path.join(out, f"ranks_{k}.npy"), v)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if state is not None:
        torch.save(state, os.path.join(out, "model.pt"))


def run_poe(ds, emb, cfg, run_root, dev):
    """Static product of experts of separately trained SASRec and text experts.
    log p = a * log p_C + (1 - a) * log p_S ; a tuned on validation NDCG@10 (grid 0..1 step .05),
    plus the fixed a = 0.5 rule ("poe50")."""
    _, val, test = make_splits(ds["user_seqs"], cfg["max_len"])
    experts = {}
    for name in ("sasrec", "text"):
        p = os.path.join(run_root, name, f"seed{cfg['seed']}", "model.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"PoE needs {p}; train '{name}' with the same seed first")
        m = build_model(ds, emb, VARIANTS[name], cfg).to(dev)
        m.load_state_dict(torch.load(p, map_location=dev))
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
    for tag, a in (("poe", best_a), ("poe50", 0.5)):
        r_test, = fused_ranks(test, [a])
        r_test_m, = fused_ranks(test, [a], masked=True)
        r_val = val_r[int(round(a * 20))]
        metrics = {"variant": tag, "seed": cfg["seed"], "alpha": a, "params": 0,
                   "val": summarize(r_val), "test": summarize(r_test), "test_masked": summarize(r_test_m),
                   "config": {k: v for k, v in cfg.items() if k != "verbose"}}
        save_run(os.path.join(run_root, tag, f"seed{cfg['seed']}"), metrics,
                 {"val": r_val, "test": r_test, "test_masked": r_test_m})
        results[tag] = metrics
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS) + sorted(POSTHOC))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--max_users", type=int, default=0, help="debug: subsample users")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    name = os.path.basename(os.path.normpath(a.data))
    dflt = DATASET_DEFAULTS.get(name, DATASET_DEFAULTS["default"])
    cfg = dict(vars(a))
    cfg["max_len"] = a.max_len or dflt["max_len"]
    cfg["dropout"] = a.dropout if a.dropout is not None else dflt["dropout"]
    cfg["verbose"] = not a.quiet
    cfg["dataset"] = name
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(a.seed)

    ds, emb = load_data(a.data)
    if a.max_users:
        ds = dict(ds)
        ds["user_seqs"] = ds["user_seqs"][:a.max_users]
    run_root = os.path.join(a.out, name)

    if a.variant == "poe":
        res = run_poe(ds, emb, cfg, run_root, dev)
        for k, v in res.items():
            print(k, "alpha", v["alpha"], "test", v["test"])
        return

    vcfg = VARIANTS[a.variant]
    cfg.update({k: v for k, v in vcfg.items() if k in ("loss", "positions")})
    train, val, test = make_splits(ds["user_seqs"], cfg["max_len"])
    model = build_model(ds, emb, vcfg, cfg).to(dev)
    state, best_ep, secs = train_model(model, train, val, dev, cfg)
    r_val = evaluate(model, val, dev)
    r_test = evaluate(model, test, dev)
    r_test_m = evaluate(model, test, dev, masked=True)
    metrics = {"variant": a.variant, "seed": a.seed, "params": count_params(model), "best_epoch": best_ep,
               "train_seconds": secs, "val": summarize(r_val), "test": summarize(r_test),
               "test_masked": summarize(r_test_m),
               "config": {k: v for k, v in cfg.items() if k != "verbose"}}
    if model.kind == "dual":
        w, lam = model.fusion_params()
        metrics["fusion_w"], metrics["fusion_lambda"] = w.item(), lam.item()
    save_run(os.path.join(run_root, a.variant, f"seed{a.seed}"), metrics,
             {"val": r_val, "test": r_test, "test_masked": r_test_m}, state)
    print(json.dumps({"variant": a.variant, "seed": a.seed, "test": metrics["test"]}, indent=1))


if __name__ == "__main__":
    main()
