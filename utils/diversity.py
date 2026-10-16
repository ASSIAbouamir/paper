"""
Advanced Diversity Module for Recommendation Systems.
Implements:
1. Adaptive Calibrated MMR (AC-MMR): Modulates diversity trade-off by user taste entropy.
2. Relevance Tolerance Pruning: Restricts diversity exploration to near-optimal items.
3. Popularity De-biasing: Boosts long-tail item coverage without harming precision.
4. Intra-List Diversity (ILD) and Catalog & Long-Tail Coverage metrics.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Set

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def compute_user_entropy(user_genres_dict: Dict[str, int]) -> float:
    """
    Computes normalized Shannon entropy of user historical genre preferences.
    H(u) in [0, 1]: 0 = concentrated single-genre user, 1 = perfectly eclectic user.
    """
    if not user_genres_dict:
        return 0.5
    total = sum(user_genres_dict.values())
    if total <= 1:
        return 0.2
    
    probs = [count / total for count in user_genres_dict.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy = math.log(len(user_genres_dict) + 1e-12)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
    return float(np.clip(norm_entropy, 0.0, 1.0))


def apply_popularity_debiasing(scores: np.ndarray, candidate_ids: List[int], item_frequencies: np.ndarray, gamma: float = 0.15) -> np.ndarray:
    """
    Applies inverse popularity penalization to boost catalog coverage of long-tail items.
    score_debiased = score - gamma * log(freq + 1) / max_log_freq
    """
    freqs = np.array([item_frequencies[cid] if cid < len(item_frequencies) else 1 for cid in candidate_ids])
    log_freqs = np.log(freqs + 1.0)
    max_log = np.max(log_freqs) if len(log_freqs) > 0 and np.max(log_freqs) > 0 else 1.0
    penalty = gamma * (log_freqs / max_log)
    return scores - penalty


def apply_adaptive_calibrated_mmr(
    scores,
    candidate_ids,
    item_embeddings,
    user_genre_dict: Dict[str, int] = None,
    item_frequencies: np.ndarray = None,
    top_k: int = 10,
    base_lambda: float = 0.75,
    relevance_tolerance: float = 0.20,
    enable_debiasing: bool = True
) -> List[int]:
    """
    Adaptive Calibrated MMR (AC-MMR):
    1. Computes user-specific lambda_u from historical genre entropy.
    2. Optional popularity debiasing for tail items.
    3. Prunes candidates outside the top relevance tolerance band (prevents precision collapse).
    4. Balances relevance and diversity dynamically.
    """
    cand_ids_list = candidate_ids.tolist() if hasattr(candidate_ids, "tolist") else list(candidate_ids)
    
    if hasattr(scores, "detach"):
        scores_np = scores.detach().cpu().numpy().copy()
    else:
        scores_np = np.array(scores, dtype=np.float32).copy()

    # 1. Popularity De-biasing
    if enable_debiasing and item_frequencies is not None:
        scores_np = apply_popularity_debiasing(scores_np, cand_ids_list, item_frequencies, gamma=0.12)

    # 2. Dynamic Lambda based on user genre entropy
    user_entropy = compute_user_entropy(user_genre_dict) if user_genre_dict else 0.5
    # Eclectic users get more diversity (lower lambda), focused users get higher lambda (pure accuracy)
    lambda_u = base_lambda + (1.0 - base_lambda) * (1.0 - user_entropy) * 0.5
    lambda_u = float(np.clip(lambda_u, 0.60, 0.95))

    # 3. Softmax relevance scores
    exp_scores = np.exp(scores_np - np.max(scores_np))
    rel_scores = exp_scores / np.sum(exp_scores)

    # 4. Filter candidates
    valid_cands = []
    valid_scores = []
    valid_embs = []
    for c_id, score in zip(cand_ids_list, rel_scores):
        if c_id < len(item_embeddings):
            valid_cands.append(c_id)
            valid_scores.append(score)
            valid_embs.append(item_embeddings[c_id])

    if not valid_cands:
        return []

    # Relevance tolerance pruning: protect precision by disallowing completely irrelevant items
    max_score = np.max(valid_scores)
    min_allowable_score = max_score * (1.0 - relevance_tolerance)

    embs_array = np.array(valid_embs)
    norms = np.linalg.norm(embs_array, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-9
    norm_embs = embs_array / norms
    sim_matrix = np.dot(norm_embs, norm_embs.T)

    selected_indices = []
    unselected_indices = list(range(len(valid_cands)))

    # Always pick the top-1 relevance item first
    first_idx = int(np.argmax(valid_scores))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    # Greedily select top-k
    while len(selected_indices) < min(top_k, len(valid_cands)) and unselected_indices:
        best_mmr = -float("inf")
        best_idx = -1

        for idx in unselected_indices:
            rel = valid_scores[idx]
            # If item falls below tolerance threshold, penalize it heavily to preserve accuracy
            if rel < min_allowable_score:
                rel_penalty = 0.5
            else:
                rel_penalty = 1.0

            max_sim = np.max([sim_matrix[idx, s_idx] for s_idx in selected_indices])
            mmr = (lambda_u * rel * rel_penalty) - ((1.0 - lambda_u) * max_sim)

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx != -1:
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
        else:
            break

    return [valid_cands[idx] for idx in selected_indices]


def calculate_ild(recommended_item_ids: List[int], item_embeddings) -> float:
    """
    Computes Intra-List Diversity (ILD) for recommended items.
    Strictly bounded in [0.0, 1.0] using normalized cosine distance:
    d(i, j) = (1 - CosineSimilarity(e_i, e_j)) / 2.0
    """
    k = len(recommended_item_ids)
    if k <= 1:
        return 0.0

    valid_ids = [idx for idx in recommended_item_ids if idx < len(item_embeddings)]
    if len(valid_ids) <= 1:
        return 0.0

    if TORCH_AVAILABLE and isinstance(item_embeddings, torch.Tensor):
        embs = item_embeddings[valid_ids]
        norm_embs = F.normalize(embs, p=2, dim=-1)
        sim_matrix = torch.matmul(norm_embs, norm_embs.T)
        # Normalized distance in [0, 1]
        dist_matrix = (1.0 - torch.clamp(sim_matrix, -1.0, 1.0)) / 2.0
        triu_indices = torch.triu_indices(len(valid_ids), len(valid_ids), offset=1)
        pairwise_distances = dist_matrix[triu_indices[0], triu_indices[1]]
        ild = float(torch.mean(pairwise_distances).item())
        return float(np.clip(ild, 0.0, 1.0))
    else:
        embs = np.array([item_embeddings[idx] for idx in valid_ids])
        norms = np.linalg.norm(embs, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-9
        norm_embs = embs / norms
        sim_matrix = np.clip(np.dot(norm_embs, norm_embs.T), -1.0, 1.0)
        # Normalized distance in [0, 1]
        dist_matrix = (1.0 - sim_matrix) / 2.0
        triu_indices = np.triu_indices(len(valid_ids), k=1)
        ild = float(np.mean(dist_matrix[triu_indices]))
        return float(np.clip(ild, 0.0, 1.0))


def calculate_catalog_coverage(all_recommended_lists: List[List[int]], total_items: int) -> float:
    """Computes catalog coverage ratio."""
    unique_items = set(item for r in all_recommended_lists for item in r)
    return len(unique_items) / float(total_items) if total_items > 0 else 0.0


def calculate_tail_coverage(all_recommended_lists: List[List[int]], item_frequencies: np.ndarray, percentile: float = 80.0) -> float:
    """Computes coverage percentage of long-tail items (items below the 80th frequency percentile)."""
    threshold = np.percentile(item_frequencies[1:], percentile)
    tail_items = set(idx for idx in range(1, len(item_frequencies)) if item_frequencies[idx] <= threshold)
    recommended_items = set(item for r in all_recommended_lists for item in r)
    covered_tail = tail_items.intersection(recommended_items)
    return len(covered_tail) / float(len(tail_items)) if len(tail_items) > 0 else 0.0
