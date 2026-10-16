"""
Comprehensive Evaluation Metrics for Sequential Recommendation Systems.
Includes:
1. Hit Rate (HR@K), Normalized Discounted Cumulative Gain (NDCG@K), Mean Reciprocal Rank (MRR@K)
2. Strict Cold-Start Performance Splits
3. Support for both Sampled-100 and Full-Catalog (All-Items) Ranking protocols
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Set

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def evaluate_batch_ranking(
    scores,
    candidates,
    uids,
    cold_start_users: Set[int],
    k: int = 10
) -> Dict[str, List[float]]:
    """
    Evaluates top-K recommendations for a batch of users.
    Candidate at index 0 is the true ground-truth positive item.
    """
    if TORCH_AVAILABLE and isinstance(scores, torch.Tensor):
        topk_indices = torch.topk(scores, k=k, dim=-1).indices.cpu().tolist()
        uids_list = uids.cpu().tolist()
    else:
        scores_np = np.array(scores)
        topk_indices = np.argsort(scores_np, axis=-1)[:, ::-1][:, :k].tolist()
        uids_list = uids.tolist() if hasattr(uids, "tolist") else list(uids)

    hr_list, ndcg_list, mrr_list = [], [], []
    cold_hr, cold_ndcg, cold_mrr = [], [], []
    warm_hr, warm_ndcg, warm_mrr = [], [], []

    batch_size = len(uids_list)

    for i in range(batch_size):
        uid = int(uids_list[i])
        is_cold = uid in cold_start_users
        topk = topk_indices[i]

        if 0 in topk:
            rank = topk.index(0) + 1
            hr = 1.0
            ndcg = 1.0 / math.log2(rank + 1)
            mrr = 1.0 / rank
        else:
            hr, ndcg, mrr = 0.0, 0.0, 0.0

        hr_list.append(hr)
        ndcg_list.append(ndcg)
        mrr_list.append(mrr)

        if is_cold:
            cold_hr.append(hr)
            cold_ndcg.append(ndcg)
            cold_mrr.append(mrr)
        else:
            warm_hr.append(hr)
            warm_ndcg.append(ndcg)
            warm_mrr.append(mrr)

    return {
        "hr": hr_list,
        "ndcg": ndcg_list,
        "mrr": mrr_list,
        "cold_hr": cold_hr,
        "cold_ndcg": cold_ndcg,
        "cold_mrr": cold_mrr,
        "warm_hr": warm_hr,
        "warm_ndcg": warm_ndcg,
        "warm_mrr": warm_mrr,
    }


def aggregate_metrics(metric_dict_list: List[Dict[str, List[float]]]) -> Dict[str, float]:
    """Aggregates batch metric lists into mean metrics across dataset."""
    all_hr, all_ndcg, all_mrr = [], [], []
    cold_hr, cold_ndcg, cold_mrr = [], [], []
    warm_hr, warm_ndcg, warm_mrr = [], [], []

    for m in metric_dict_list:
        all_hr.extend(m["hr"])
        all_ndcg.extend(m["ndcg"])
        all_mrr.extend(m["mrr"])
        cold_hr.extend(m["cold_hr"])
        cold_ndcg.extend(m["cold_ndcg"])
        cold_mrr.extend(m["cold_mrr"])
        warm_hr.extend(m["warm_hr"])
        warm_ndcg.extend(m["warm_ndcg"])
        warm_mrr.extend(m["warm_mrr"])

    return {
        "HR@10": float(np.mean(all_hr)) if len(all_hr) > 0 else 0.0,
        "NDCG@10": float(np.mean(all_ndcg)) if len(all_ndcg) > 0 else 0.0,
        "MRR@10": float(np.mean(all_mrr)) if len(all_mrr) > 0 else 0.0,
        "Cold_HR@10": float(np.mean(cold_hr)) if len(cold_hr) > 0 else 0.0,
        "Cold_NDCG@10": float(np.mean(cold_ndcg)) if len(cold_ndcg) > 0 else 0.0,
        "Cold_MRR@10": float(np.mean(cold_mrr)) if len(cold_mrr) > 0 else 0.0,
        "Warm_HR@10": float(np.mean(warm_hr)) if len(warm_hr) > 0 else 0.0,
        "Warm_NDCG@10": float(np.mean(warm_ndcg)) if len(warm_ndcg) > 0 else 0.0,
        "Warm_MRR@10": float(np.mean(warm_mrr)) if len(warm_mrr) > 0 else 0.0,
    }
