"""
Cross-Modal Contrastive Alignment Module (InfoNCE Loss).
Aligns learned collaborative embeddings with semantic LLM textual representations.
Addresses the frozen embedding limitation without requiring full LLM fine-tuning.
"""

import math
import numpy as np
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class InfoNCEAlignmentLoss(nn.Module if TORCH_AVAILABLE else object):
    """
    Computes bidirectional cross-modal InfoNCE contrastive alignment loss:
    Aligns collaborative item embedding e_item with projected LLM embedding e_llm.
    L_align = -log( exp(sim(e_item, e_llm)/tau) / sum_j exp(sim(e_item, e_llm_j)/tau) )
    """

    def __init__(self, temperature: float = 0.07):
        if TORCH_AVAILABLE:
            super(InfoNCEAlignmentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, item_embs: torch.Tensor, llm_proj_embs: torch.Tensor) -> torch.Tensor:
        """
        item_embs: [BatchSize, Dim]
        llm_proj_embs: [BatchSize, Dim]
        """
        if not TORCH_AVAILABLE:
            return 0.0

        item_norm = F.normalize(item_embs, p=2, dim=-1)
        llm_norm = F.normalize(llm_proj_embs, p=2, dim=-1)

        # Batch similarity matrix [B, B]
        logits = torch.matmul(item_norm, llm_norm.T) / self.temperature
        batch_size = item_embs.shape[0]
        labels = torch.arange(batch_size, device=item_embs.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0


def compute_contrastive_alignment_numpy(item_embs: np.ndarray, llm_proj: np.ndarray, temperature: float = 0.07) -> float:
    """NumPy implementation of cross-modal InfoNCE loss."""
    norm_item = item_embs / (np.linalg.norm(item_embs, axis=-1, keepdims=True) + 1e-9)
    norm_llm = llm_proj / (np.linalg.norm(llm_proj, axis=-1, keepdims=True) + 1e-9)

    sim = np.dot(norm_item, norm_llm.T) / temperature
    exp_sim = np.exp(sim - np.max(sim, axis=-1, keepdims=True))
    probs = exp_sim / np.sum(exp_sim, axis=-1, keepdims=True)

    b = len(item_embs)
    diag_probs = np.diag(probs)
    loss = -np.mean(np.log(diag_probs + 1e-12))
    return float(loss)
