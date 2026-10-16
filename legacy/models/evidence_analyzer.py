"""
Evidence Analyzer for Dual-View Sequential Recommendation.
Extracts predictive epistemic uncertainty, semantic uncertainty, and cross-view disagreement:
1. U_c: Predictive Epistemic Uncertainty via Mutual Information
   U_c = H(mean(p_c)) - (1/M) * sum_m H(p_c^(m))
2. U_s: Semantic Uncertainty via dispersion / predictive entropy
3. D_JS: Jensen-Shannon Disagreement JS(p_c || p_s)
4. Contextual signals: sequence length L_u, entropy H_c, H_s, recency R_u, session drift Drift_u
Output: Evidence Feature Vector x_u in R^8
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class EvidenceAnalyzer:
    """
    Extracts calibrated uncertainty and disagreement features from
    collaborative distributions p_c and semantic distributions p_s.
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def compute_predictive_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Computes Shannon entropy H(p) = - sum_i p_i * log(p_i)
        probs: [..., C] valid probability distribution summing to 1
        Returns: [...] scalar entropy
        """
        p_clamped = torch.clamp(probs, min=self.eps, max=1.0)
        entropy = -torch.sum(p_clamped * torch.log(p_clamped), dim=-1)
        return entropy

    def compute_mutual_information_uncertainty(self, mc_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Predictive Epistemic Uncertainty via Mutual Information:
        U_c = H( (1/M) sum_m p_m ) - (1/M) sum_m H(p_m)
        mc_probs: [BatchSize, M, NumCandidates]
        Returns:
            mean_probs: [BatchSize, NumCandidates]
            U_c: [BatchSize] predictive epistemic uncertainty >= 0
        """
        # 1. Expected distribution (ensemble average)
        mean_probs = torch.mean(mc_probs, dim=1) # [B, C]

        # 2. Total predictive uncertainty H(mean_p)
        total_uncertainty = self.compute_predictive_entropy(mean_probs) # [B]

        # 3. Expected aleatoric uncertainty (1/M) sum_m H(p_m)
        individual_entropies = self.compute_predictive_entropy(mc_probs) # [B, M]
        expected_aleatoric = torch.mean(individual_entropies, dim=1) # [B]

        # 4. Epistemic uncertainty = Total - Aleatoric
        U_c = torch.clamp(total_uncertainty - expected_aleatoric, min=0.0) # [B]
        return mean_probs, U_c

    def compute_jensen_shannon_divergence(self, p_c: torch.Tensor, p_s: torch.Tensor) -> torch.Tensor:
        """
        Computes symmetric Jensen-Shannon divergence:
        JS(p_c || p_s) = 0.5 * KL(p_c || M) + 0.5 * KL(p_s || M)
        where M = 0.5 * (p_c + p_s)
        p_c, p_s: [BatchSize, NumCandidates]
        Returns:
            D_JS: [BatchSize] in [0, log(2)]
        """
        p_c = torch.clamp(p_c, min=self.eps)
        p_s = torch.clamp(p_s, min=self.eps)
        m = 0.5 * (p_c + p_s)

        kl_cm = torch.sum(p_c * (torch.log(p_c) - torch.log(m)), dim=-1)
        kl_sm = torch.sum(p_s * (torch.log(p_s) - torch.log(m)), dim=-1)

        d_js = 0.5 * kl_cm + 0.5 * kl_sm
        return torch.clamp(d_js, min=0.0)

    def compute_session_drift(self, item_sequence: torch.Tensor, semantic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes drift magnitude delta_u between the recent window (last 3 items)
        and the historical centroid (preceding items).
        item_sequence: [BatchSize, SeqLen]
        semantic_embeddings: [NumItems + 1, Dim]
        Returns:
            drift: [BatchSize] cosine distance in [0, 2]
        """
        batch_size, seq_len = item_sequence.shape
        drift_scores = torch.zeros(batch_size, device=item_sequence.device)

        for b in range(batch_size):
            valid_items = item_sequence[b][item_sequence[b] != 0]
            if len(valid_items) < 4:
                drift_scores[b] = 0.0
                continue

            recent_items = valid_items[-3:]
            past_items = valid_items[:-3]

            recent_embs = semantic_embeddings[recent_items] # [3, D]
            past_embs = semantic_embeddings[past_items] # [K, D]

            recent_centroid = F.normalize(recent_embs.mean(dim=0, keepdim=True), dim=-1)
            past_centroid = F.normalize(past_embs.mean(dim=0, keepdim=True), dim=-1)

            cos_sim = (recent_centroid * past_centroid).sum().item()
            drift_scores[b] = 1.0 - cos_sim

        return drift_scores

    def extract_evidence_features(
        self,
        mc_probs_c: torch.Tensor,
        mc_probs_s: torch.Tensor,
        input_seqs: torch.Tensor,
        semantic_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts the 8-dimensional evidence feature vector x_u:
        [ U_c, U_s, D_JS, H_c, H_s, L_u / 50.0, R_u, Drift_u ]
        Returns:
            features: [BatchSize, 8]
            mean_pc: [BatchSize, NumCandidates]
            mean_ps: [BatchSize, NumCandidates]
        """
        # 1. Epistemic uncertainties U_c and U_s
        mean_pc, U_c = self.compute_mutual_information_uncertainty(mc_probs_c)
        mean_ps, U_s = self.compute_mutual_information_uncertainty(mc_probs_s)

        # 2. Cross-expert disagreement D_JS
        D_JS = self.compute_jensen_shannon_divergence(mean_pc, mean_ps)

        # 3. Entropies of mean distributions
        H_c = self.compute_predictive_entropy(mean_pc)
        H_s = self.compute_predictive_entropy(mean_ps)

        # 4. Contextual sequence length and recency
        seq_lengths = (input_seqs != 0).sum(dim=-1).float()
        norm_len = torch.clamp(seq_lengths / 50.0, 0.0, 1.0)
        recency = torch.clamp(seq_lengths / 10.0, 0.0, 1.0)

        # 5. Semantic session drift
        drift = self.compute_session_drift(input_seqs, semantic_embeddings)

        # Assemble into x_u
        evidence_features = torch.stack([
            U_c,
            U_s,
            D_JS,
            H_c / 10.0,
            H_s / 10.0,
            norm_len,
            recency,
            drift
        ], dim=-1) # [B, 8]

        return evidence_features, mean_pc, mean_ps
