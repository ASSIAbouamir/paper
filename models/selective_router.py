"""
Selective Router for Dual-View Sequential Recommendation under Epistemic Uncertainty and Disagreement.

Maps the 8-dimensional evidence feature vector:
x_u = [ U_c, U_s, D_JS, H_c, H_s, L_u, R_u, Drift_u ]
to action preferences over:
a in { C: Collaborative Expert, S: Semantic Expert, H: Hybrid Fusion }

Architecture:
Lightweight 2-layer MLP f_theta(x_u) -> R^3
Trained with supervision from an offline Oracle target a*(u) = argmax { Y_C(u), Y_S(u), Y_H(u) }.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SelectiveRouter(nn.Module):
    """
    Learns selective routing between Collaborative (SASRec), Semantic (MiniLM),
    and Hybrid evidence streams conditioned on predictive uncertainty and cross-view disagreement.
    """
    ACTION_NAMES = {0: "Collaborative", 1: "Semantic", 2: "Hybrid"}

    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 32,
        num_actions: int = 3,
        dropout: float = 0.10
    ):
        super(SelectiveRouter, self).__init__()
        self.in_features = in_features
        self.num_actions = num_actions

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, evidence_features: torch.Tensor) -> torch.Tensor:
        """
        evidence_features: [BatchSize, 8]
        Returns:
            logits: [BatchSize, 3] representing unnormalized log-evidence / utility V_theta(a | u)
        """
        return self.mlp(evidence_features)

    def predict_action(self, evidence_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        evidence_features: [BatchSize, 8]
        Returns:
            selected_actions: [BatchSize] in {0, 1, 2}
            action_probs: [BatchSize, 3] softmax distribution
        """
        logits = self.forward(evidence_features)
        probs = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)
        return actions, probs

    def route(
        self,
        p_c: torch.Tensor,
        p_s: torch.Tensor,
        p_h: torch.Tensor,
        evidence_features: torch.Tensor,
        hard_selection: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes selective evidence routing.
        p_c: [BatchSize, NumItems] Collaborative distribution
        p_s: [BatchSize, NumItems] Semantic distribution
        p_h: [BatchSize, NumItems] Hybrid distribution
        evidence_features: [BatchSize, 8]
        hard_selection: If True, uses argmax hard routing; else uses soft mixture-of-experts weighting.

        Returns:
            p_routed: [BatchSize, NumItems] Final routed recommendation distribution
            actions: [BatchSize] Selected actions (argmax)
            probs: [BatchSize, 3] Routing weights [w_C, w_S, w_H]
        """
        logits = self.forward(evidence_features)
        probs = F.softmax(logits, dim=-1) # [B, 3]
        actions = torch.argmax(probs, dim=-1) # [B]

        if hard_selection:
            batch_size, num_items = p_c.shape
            p_routed = torch.zeros_like(p_c)
            for b in range(batch_size):
                act = actions[b].item()
                if act == 0:
                    p_routed[b] = p_c[b]
                elif act == 1:
                    p_routed[b] = p_s[b]
                else:
                    p_routed[b] = p_h[b]
        else:
            w_c = probs[:, 0:1] # [B, 1]
            w_s = probs[:, 1:2] # [B, 1]
            w_h = probs[:, 2:3] # [B, 1]
            p_routed = w_c * p_c + w_s * p_s + w_h * p_h

        return p_routed, actions, probs


class ContinuousAlphaRouter(nn.Module):
    """
    Direct Continuous Gate Baseline:
    Predicts a scalar alpha_u in [0, 1] conditioned on the evidence vector x_u:
    p_routed = alpha_u * p_c + (1 - alpha_u) * p_s
    """
    def __init__(self, in_features: int = 8, hidden_dim: int = 32, dropout: float = 0.10):
        super(ContinuousAlphaRouter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, evidence_features: torch.Tensor) -> torch.Tensor:
        """Returns alpha_u in [0, 1] of shape [BatchSize, 1]."""
        return torch.sigmoid(self.mlp(evidence_features))

    def route(
        self,
        p_c: torch.Tensor,
        p_s: torch.Tensor,
        evidence_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.forward(evidence_features) # [B, 1]
        p_routed = alpha * p_c + (1.0 - alpha) * p_s
        return p_routed, alpha


class TwoActionRouter(nn.Module):
    """
    Reduced 2-Action Router ({C, S} only):
    Ablates the explicit Hybrid action H to directly assess whether H is necessary or redundant.
    """
    def __init__(self, in_features: int = 8, hidden_dim: int = 32, dropout: float = 0.10):
        super(TwoActionRouter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, evidence_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(evidence_features)

    def route(
        self,
        p_c: torch.Tensor,
        p_s: torch.Tensor,
        evidence_features: torch.Tensor,
        hard_selection: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(evidence_features)
        probs = F.softmax(logits, dim=-1) # [B, 2]
        actions = torch.argmax(probs, dim=-1) # [B]

        if hard_selection:
            batch_size = p_c.shape[0]
            p_routed = torch.zeros_like(p_c)
            for b in range(batch_size):
                p_routed[b] = p_c[b] if actions[b].item() == 0 else p_s[b]
        else:
            w_c = probs[:, 0:1]
            w_s = probs[:, 1:2]
            p_routed = w_c * p_c + w_s * p_s

        return p_routed, actions, probs

