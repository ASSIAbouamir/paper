"""Paired statistics for comparing two recommenders evaluated on the same users."""

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as st


def mcnemar_exact(hit_a: np.ndarray, hit_b: np.ndarray) -> Dict[str, float]:
    """Exact two-sided McNemar test on per-user hit indicators."""
    a, b = np.asarray(hit_a, bool), np.asarray(hit_b, bool)
    n01 = int((~a & b).sum())  # only b hits
    n10 = int((a & ~b).sum())  # only a hits
    n = n01 + n10
    p = 1.0 if n == 0 else float(st.binomtest(n10, n, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "p": p}


def paired_bootstrap(x: np.ndarray, y: np.ndarray, B: int = 10000, seed: int = 0) -> Dict[str, float]:
    """95% CI of mean(x - y) over users (e.g. per-user NDCG)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "p": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))}


def wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, float) - np.asarray(y, float)
    if np.allclose(d, 0):
        return 1.0
    return float(st.wilcoxon(d, zero_method="wilcox").pvalue)


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (monotone)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def seed_summary(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, float)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": float(v.mean()), "sd": sd, "lo": float(v.mean() - half), "hi": float(v.mean() + half), "n": n}


def paired_seed_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired t-test over seeds (a_i, b_i share seed i)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a - b, (a - b)[0]):
        return {"t": float("nan"), "p": float("nan")}
    r = st.ttest_rel(a, b)
    return {"t": float(r.statistic), "p": float(r.pvalue)}


def ndcg_per_user(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    r = np.asarray(ranks, float)
    return np.where(r <= k, 1.0 / np.log2(r + 1), 0.0)
