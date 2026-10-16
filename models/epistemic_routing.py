"""
Epistemic Routing for Uncertainty-Aware Sequential Recommendation.
Novel Theoretical Mechanism:
Estimates the epistemic uncertainty of the collaborative user representation via Monte-Carlo Dropout
and dynamically routes between collaborative representations (SASRec) and semantic LLM representations (LoRA)
proportionally to this uncertainty.

Theoretical Formulation:
1. Monte-Carlo stochastic views of user sequence:
   {h_u^(1), ..., h_u^(M)} ~ p(h_u | S_u)
2. Empirical mean and epistemic dispersion (trace of covariance):
   mu_u = (1 / M) * sum_{m=1}^M h_u^(m)
   u_u = (1 / d) * Tr( (1 / (M-1)) * sum_{m=1}^M (h_u^(m) - mu_u)(h_u^(m) - mu_u)^T )
3. Calibrated Log-Scaled Epistemic Gate:
   alpha_u = sigmoid( w * log(1 + u_u) + b )
4. Uncertainty-conditioned fused representation:
   e_i^(fused) = LayerNorm( (1 - alpha_u) * (W_c e_i^(collab)) + alpha_u * (W_s e_i^(LoRA)) )
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class EpistemicUncertaintyEstimator(nn.Module):
        """
        Estimates epistemic uncertainty u_u from M stochastic Monte-Carlo passes.
        """
        def __init__(self, hidden_dim: int, num_mc_samples: int = 5):
            super(EpistemicUncertaintyEstimator, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_mc_samples = num_mc_samples

        def compute_uncertainty(self, mc_representations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            mc_representations: [BatchSize, M, hidden_dim]
            Returns:
                mean_rep: [BatchSize, hidden_dim]
                uncertainty: [BatchSize] scalar epistemic uncertainty u_u >= 0
            """
            # Empirical mean across Monte-Carlo passes
            mean_rep = torch.mean(mc_representations, dim=1) # [B, D]

            # Sample variance per dimension across M passes
            # Var = sum( (x - mean)^2 ) / (M - 1)
            diff = mc_representations - mean_rep.unsqueeze(1) # [B, M, D]
            variance_per_dim = torch.sum(diff ** 2, dim=1) / max(self.num_mc_samples - 1, 1) # [B, D]

            # Scalar epistemic uncertainty: normalized trace of covariance matrix
            epistemic_uncertainty = torch.mean(variance_per_dim, dim=-1) # [B]
            return mean_rep, epistemic_uncertainty


    class EpistemicGate(nn.Module):
        """
        Calibrated Epistemic Gate mapping uncertainty u_u to routing weight alpha_u in (0, 1).
        When collaborative uncertainty is high (e.g. cold-start, session drift, sparse interactions),
        alpha_u -> 1 (prioritizes LLM semantic knowledge).
        When collaborative uncertainty is low (long, consistent history),
        alpha_u -> 0 (prioritizes collaborative sequential transitions).
        """
        def __init__(self, hidden_dim: int = 64, initial_w: float = 2.0, initial_b: float = -1.0):
            super(EpistemicGate, self).__init__()
            self.hidden_dim = hidden_dim
            
            # Learnable calibration parameters for log(1 + u_u)
            self.weight = nn.Parameter(torch.tensor([initial_w], dtype=torch.float32))
            self.bias = nn.Parameter(torch.tensor([initial_b], dtype=torch.float32))

            # Contextual refinement projection conditioned on [log(1+u_u), mean_rep]
            self.context_gate = nn.Linear(hidden_dim + 1, hidden_dim)
            self.gate_norm = nn.LayerNorm(hidden_dim)

        def forward(self, epistemic_uncertainty: torch.Tensor, user_mean_rep: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            epistemic_uncertainty: [BatchSize] or [BatchSize, 1]
            user_mean_rep: Optional [BatchSize, hidden_dim]
            Returns:
                alpha_u: [BatchSize, 1] or [BatchSize, hidden_dim] routing coefficient in (0, 1)
            """
            if epistemic_uncertainty.dim() == 1:
                epistemic_uncertainty = epistemic_uncertainty.unsqueeze(-1) # [B, 1]

            # Log-stabilized transformation: prevents explosion for outlier variances
            log_u = torch.log1p(torch.clamp(epistemic_uncertainty, min=0.0)) # [B, 1]
            scalar_gate = torch.sigmoid(self.weight * log_u + self.bias) # [B, 1]

            if user_mean_rep is not None:
                # Dimension-wise adaptive gating conditioned on semantic state
                gate_input = torch.cat([log_u, user_mean_rep], dim=-1) # [B, D + 1]
                vector_gate = torch.sigmoid(self.context_gate(gate_input)) # [B, D]
                # Modulate vector gate by calibrated scalar uncertainty
                alpha_u = 0.5 * (scalar_gate + vector_gate)
            else:
                alpha_u = scalar_gate

            return alpha_u


    class EpistemicFusedRecommendation(nn.Module):
        """
        Complete End-to-End Epistemic Routing Recommendation Module.
        Fuses collaborative sequence embeddings (SASRec) and LLM textual representations (LoRA)
        governed by epistemic uncertainty u_u.
        """
        def __init__(
            self,
            hidden_dim: int = 64,
            llm_dim: int = 64,
            num_mc_samples: int = 5,
            dropout_rate: float = 0.20
        ):
            super(EpistemicFusedRecommendation, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_mc_samples = num_mc_samples
            self.dropout_rate = dropout_rate

            self.uncertainty_estimator = EpistemicUncertaintyEstimator(hidden_dim, num_mc_samples)
            self.epistemic_gate = EpistemicGate(hidden_dim=hidden_dim)

            # Projections for collaborative and semantic spaces
            self.W_collab = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_semantic = nn.Linear(llm_dim, hidden_dim, bias=False)
            self.fused_layernorm = nn.LayerNorm(hidden_dim, eps=1e-8)
            self.dropout = nn.Dropout(p=dropout_rate)

        def fuse_representations(
            self,
            collab_rep: torch.Tensor,
            semantic_rep: torch.Tensor,
            alpha_u: torch.Tensor
        ) -> torch.Tensor:
            """
            Fuses representations using uncertainty gate alpha_u:
            z_fused = LayerNorm( (1 - alpha_u) * W_c(collab) + alpha_u * W_s(semantic) )
            """
            # Align shapes
            if alpha_u.dim() < collab_rep.dim():
                while alpha_u.dim() < collab_rep.dim():
                    alpha_u = alpha_u.unsqueeze(1)

            proj_collab = self.W_collab(collab_rep)
            proj_semantic = self.W_semantic(semantic_rep)

            fused = (1.0 - alpha_u) * proj_collab + alpha_u * proj_semantic
            return self.fused_layernorm(self.dropout(fused))

        def forward(
            self,
            mc_user_reps: torch.Tensor,
            user_llm_rep: torch.Tensor,
            cand_collab_embs: torch.Tensor,
            cand_llm_embs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            mc_user_reps: [B, M, D] from SASRec with MC-Dropout
            user_llm_rep: [B, D] historical semantic centroid or LLM query representation
            cand_collab_embs: [B, C, D] or [N_items, D]
            cand_llm_embs: [B, C, D] or [N_items, D]
            
            Returns:
                scores: [B, C] or [B, N_items]
                alpha_u: [B, 1] routing gate values
                u_u: [B] epistemic uncertainties
            """
            mean_user_rep, u_u = self.uncertainty_estimator.compute_uncertainty(mc_user_reps)
            alpha_u = self.epistemic_gate(u_u, mean_user_rep) # [B, D] or [B, 1]

            # Fuse user representation
            fused_user = self.fuse_representations(mean_user_rep, user_llm_rep, alpha_u) # [B, D]

            # Fuse candidate representations
            fused_cands = self.fuse_representations(cand_collab_embs, cand_llm_embs, alpha_u) # [B, C, D] or [N, D]

            # Compute inner-product ranking scores
            if fused_cands.dim() == 3:
                scores = torch.bmm(fused_cands, fused_user.unsqueeze(-1)).squeeze(-1) # [B, C]
            else:
                scores = torch.matmul(fused_user, fused_cands.T) # [B, N_items]

            return scores, alpha_u, u_u


# ==============================================================================
# Pure NumPy Implementation (Testing Harness & Fallback)
# ==============================================================================
class NumPyEpistemicRouter:
    """
    Exact mathematical NumPy implementation of Epistemic Routing for
    verification and CPU execution environments.
    """
    def __init__(self, hidden_dim: int = 64, num_mc_samples: int = 5, w: float = 2.0, b: float = -1.0):
        self.hidden_dim = hidden_dim
        self.num_mc_samples = num_mc_samples
        self.w = w
        self.b = b

    def compute_uncertainty(self, mc_reps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        mc_reps: [B, M, D]
        Returns:
            mean_rep: [B, D]
            u_u: [B] scalar epistemic uncertainty
        """
        mean_rep = np.mean(mc_reps, axis=1) # [B, D]
        diff = mc_reps - np.expand_dims(mean_rep, 1)
        var_per_dim = np.sum(diff ** 2, axis=1) / max(self.num_mc_samples - 1, 1) # [B, D]
        u_u = np.mean(var_per_dim, axis=-1) # [B]
        return mean_rep, u_u

    def compute_gate(self, u_u: np.ndarray) -> np.ndarray:
        """
        alpha_u = sigmoid( w * log(1 + u_u) + b )
        """
        log_u = np.log1p(np.maximum(u_u, 0.0))
        z = self.w * log_u + self.b
        alpha_u = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        return alpha_u

    def fuse(self, collab: np.ndarray, semantic: np.ndarray, alpha_u: np.ndarray) -> np.ndarray:
        if alpha_u.ndim == 1:
            alpha_u = alpha_u[:, np.newaxis]
        fused = (1.0 - alpha_u) * collab + alpha_u * semantic
        # LayerNorm simulation
        mean = np.mean(fused, axis=-1, keepdims=True)
        std = np.std(fused, axis=-1, keepdims=True) + 1e-8
        return (fused - mean) / std


def test_epistemic_routing():
    """Unit test validating mathematical properties of Epistemic Routing."""
    np.random.seed(42)
    router = NumPyEpistemicRouter(hidden_dim=64, num_mc_samples=5, w=2.0, b=-0.5)

    # Simulate Cold-start user (high variance across MC passes)
    B = 10
    mc_cold = np.random.randn(B, 5, 64) * 1.5
    mean_cold, u_cold = router.compute_uncertainty(mc_cold)
    alpha_cold = router.compute_gate(u_cold)

    # Simulate Long-history user (consistent representations across MC passes)
    base_rep = np.random.randn(B, 1, 64)
    mc_warm = base_rep + np.random.randn(B, 5, 64) * 0.1
    mean_warm, u_warm = router.compute_uncertainty(mc_warm)
    alpha_warm = router.compute_gate(u_warm)

    assert np.all(u_cold > u_warm), "Cold-start epistemic uncertainty must exceed warm-user uncertainty"
    assert np.all(alpha_cold > alpha_warm), "Cold-start routing gate must assign higher weight to LLM"
    print(f"[Epistemic Routing Test Passed] Mean Cold-start alpha: {alpha_cold.mean():.3f} vs Warm alpha: {alpha_warm.mean():.3f}")


if __name__ == "__main__":
    test_epistemic_routing()
