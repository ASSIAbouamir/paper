"""Ranking metrics from per-user target ranks (1 = best), full-catalog, no negative sampling."""

import numpy as np


def summarize(ranks) -> dict:
    r = np.asarray(ranks, dtype=np.float64)
    out = {"users": int(len(r))}
    for k in (5, 10, 20):
        hit = r <= k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float((hit / np.log2(r + 1)).mean())
        out[f"hits@{k}"] = int(hit.sum())
    out["MRR"] = float((1.0 / r).mean())
    return out
