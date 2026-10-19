"""
Enhanced Selective Router — Improved evidence routing architecture.

Upgrades over the original SelectiveRouter (v1):
  1. Richer 12-dimensional evidence feature vector (from 8D)
  2. 3-layer MLP with residual connection (from 2-layer)
  3. Feature-level attention for interpretability (which features drive routing)
  4. Curriculum training support (hard Oracle → soft Oracle → REINFORCE)
  5. Feature importance tracking for SHAP-style analysis

New features in x_u ∈ ℝ¹²:
  f1  : U_c  — Collaborative epistemic uncertainty (MI)
  f2  : U_s  — Semantic epistemic uncertainty (MI)
  f3  : D_JS — Jensen-Shannon divergence (cross-view disagreement)
  f4  : H_c  — Collaborative predictive entropy (normalized)
  f5  : H_s  — Semantic predictive entropy (normalized)
  f6  : L_u  — Sequence length (normalized by max_seq_len)
  f7  : R_u  — Recency density
  f8  : Drift_u — Semantic cosine drift (recent vs. history)
  f9  : PopRatio_u — Popularity ratio (recent items / history items)
  f10 : UcVariance — Variance of Uc across MC passes (temporal instability)
  f11 : GenreDiversity_u — Normalized unique genre count in sequence
  f12 : TailRatio_u — Fraction of long-tail items in recent interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FeatureAttention(nn.Module):
    """
    Learnable per-feature attention weights.
    Allows the router to focus on the most discriminative features,
    and provides feature importance for interpretability.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, in_features),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, F]
        Returns:
            attended: [B, F] attention-weighted features
            weights:  [B, F] attention weights (for importance analysis)
        """
        weights  = self.attention(x)   # [B, F]
        attended = x * weights         # element-wise scaling
        return attended, weights


class EnhancedSelectiveRouter(nn.Module):
    """
    3-layer MLP with:
      - Feature attention for interpretability
      - Residual connection from input to hidden layer
      - LayerNorm after each hidden layer
      - Curriculum-compatible forward (returns both logits and routing weights)
    """
    ACTION_NAMES = {0: "Collaborative", 1: "Semantic", 2: "Hybrid"}

    def __init__(
        self,
        in_features: int = 12,
        hidden_dim: int = 64,
        num_actions: int = 3,
        dropout: float = 0.10,
        use_feature_attention: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.use_feature_attention = use_feature_attention

        if use_feature_attention:
            self.feat_attention = FeatureAttention(in_features)

        # Layer 1: in_features → hidden_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Layer 2: hidden_dim → hidden_dim (with residual projection if dims differ)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Layer 3: hidden_dim → hidden_dim // 2
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Output head
        self.head = nn.Linear(hidden_dim // 2, num_actions)

        # Residual projection: input → hidden (to add skip connection)
        self.residual_proj = nn.Linear(in_features, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, 12] evidence feature vector
        Returns:
            logits:   [B, 3] routing logits
            attn_weights: [B, 12] feature attention weights (if return_attention=True)
        """
        attn_weights = None

        if self.use_feature_attention:
            x_attended, attn_weights = self.feat_attention(x)
        else:
            x_attended = x

        # Layer 1 + residual from input
        h1 = self.layer1(x_attended) + self.residual_proj(x_attended)

        # Layer 2 (with self-residual)
        h2 = self.layer2(h1) + h1

        # Layer 3
        h3 = self.layer3(h2)

        logits = self.head(h3)

        if return_attention:
            return logits, attn_weights
        return logits, None

    def predict_action(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns hard-selected action (argmax) and softmax weights.

        x: [B, 12]
        Returns:
            actions: [B] in {0, 1, 2}
            probs:   [B, 3] softmax probabilities
        """
        logits, _ = self.forward(x)
        probs   = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)
        return actions, probs

    def route(
        self,
        p_c: torch.Tensor,
        p_s: torch.Tensor,
        p_h: torch.Tensor,
        x: torch.Tensor,
        hard_selection: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Full soft-routing (or hard-routing) of evidence distributions.

        Args:
            p_c: [B, N] collaborative score distribution
            p_s: [B, N] semantic score distribution
            p_h: [B, N] hybrid score distribution
            x:   [B, 12] evidence feature vector
            hard_selection: if True, use argmax instead of soft mixture

        Returns:
            p_routed:     [B, N] final recommendation distribution
            actions:      [B] hard action selections
            probs:        [B, 3] routing weights [w_C, w_S, w_H]
            attn_weights: [B, 12] feature attention weights (or None)
        """
        logits, attn_weights = self.forward(x, return_attention=return_attention)
        probs   = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)

        if hard_selection:
            B, N     = p_c.shape
            p_routed = torch.zeros_like(p_c)
            for b in range(B):
                act = actions[b].item()
                if act == 0:
                    p_routed[b] = p_c[b]
                elif act == 1:
                    p_routed[b] = p_s[b]
                else:
                    p_routed[b] = p_h[b]
        else:
            w_c = probs[:, 0:1]
            w_s = probs[:, 1:2]
            w_h = probs[:, 2:3]
            p_routed = w_c * p_c + w_s * p_s + w_h * p_h

        return p_routed, actions, probs, attn_weights

    def get_feature_importance(
        self, x_batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes average feature importance across a batch by running
        the feature attention module.

        Args:
            x_batch: [B, 12] evidence feature vectors

        Returns:
            dict with 'mean_importance' [12] and 'std_importance' [12]
        """
        if not self.use_feature_attention:
            raise RuntimeError("Feature attention is disabled (use_feature_attention=False).")

        with torch.no_grad():
            _, attn_weights = self.forward(x_batch, return_attention=True)

        return {
            "mean_importance": attn_weights.mean(dim=0),   # [12]
            "std_importance":  attn_weights.std(dim=0),    # [12]
        }


# ── Feature names for interpretability ────────────────────────────────────────

FEATURE_NAMES_12D = [
    "U_c (Collab Uncertainty)",
    "U_s (Semantic Uncertainty)",
    "D_JS (Cross-View Disagreement)",
    "H_c (Collab Entropy, norm)",
    "H_s (Semantic Entropy, norm)",
    "L_u (Sequence Length, norm)",
    "R_u (Recency Density)",
    "Drift_u (Semantic Drift)",
    "PopRatio_u (Popularity Ratio)",
    "UcVariance (MC Instability)",
    "GenreDiversity_u (Taste Breadth)",
    "TailRatio_u (Long-Tail Fraction)",
]

FEATURE_NAMES_8D = [
    "U_c (Collab Uncertainty)",
    "U_s (Semantic Uncertainty)",
    "D_JS (Cross-View Disagreement)",
    "H_c (Collab Entropy, norm)",
    "H_s (Semantic Entropy, norm)",
    "L_u (Sequence Length, norm)",
    "R_u (Recency Density)",
    "Drift_u (Semantic Drift)",
]


# ── Enhanced Evidence Analyzer — 12D features ─────────────────────────────────

class EnhancedEvidenceAnalyzer:
    """
    Extended evidence analyzer producing 12-dimensional feature vectors.
    Adds 4 new features on top of the original 8:
      f9:  Popularity ratio (recent vs. full history)
      f10: Variance of Uc across MC seeds (instability signal)
      f11: Genre diversity (unique genres / total genres)
      f12: Long-tail fraction (items below median popularity)
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(probs, min=self.eps)
        return -(p * torch.log(p)).sum(dim=-1)

    def compute_mutual_information(
        self, mc_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """mc_probs: [B, M, C] → (mean_probs [B,C], U_c [B])"""
        mean_p = mc_probs.mean(dim=1)
        total  = self.compute_entropy(mean_p)
        aleat  = self.compute_entropy(mc_probs).mean(dim=1)
        return mean_p, torch.clamp(total - aleat, min=0.0)

    def compute_js_divergence(
        self, p_c: torch.Tensor, p_s: torch.Tensor
    ) -> torch.Tensor:
        p_c = torch.clamp(p_c, min=self.eps)
        p_s = torch.clamp(p_s, min=self.eps)
        m   = 0.5 * (p_c + p_s)
        kl1 = (p_c * (p_c.log() - m.log())).sum(dim=-1)
        kl2 = (p_s * (p_s.log() - m.log())).sum(dim=-1)
        return torch.clamp(0.5 * kl1 + 0.5 * kl2, min=0.0)

    def compute_session_drift(
        self, item_sequence: torch.Tensor, semantic_embeddings: torch.Tensor
    ) -> torch.Tensor:
        B, T = item_sequence.shape
        drift = torch.zeros(B, device=item_sequence.device)
        for b in range(B):
            valid = item_sequence[b][item_sequence[b] != 0]
            if len(valid) < 4:
                continue
            recent = semantic_embeddings[valid[-3:]]
            past   = semantic_embeddings[valid[:-3]]
            r_cent = F.normalize(recent.mean(0, keepdim=True), dim=-1)
            p_cent = F.normalize(past.mean(0, keepdim=True), dim=-1)
            drift[b] = 1.0 - (r_cent * p_cent).sum().clamp(-1, 1)
        return drift

    def compute_popularity_ratio(
        self,
        item_sequence: torch.Tensor,
        item_frequencies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recent (last 3) item popularity vs. full history popularity.
        High ratio → recent items are more popular than history → possible popularity bias.
        """
        B = item_sequence.shape[0]
        ratio = torch.zeros(B, device=item_sequence.device)
        for b in range(B):
            valid = item_sequence[b][item_sequence[b] != 0]
            if len(valid) < 2:
                continue
            freq = item_frequencies.to(item_sequence.device, dtype=torch.float32)
            history_pop = freq[valid[:-3]].mean() if len(valid) > 3 else freq[valid].mean()
            recent_pop  = freq[valid[-3:]].mean()
            denom = history_pop + self.eps
            ratio[b] = torch.clamp(recent_pop / denom, 0.0, 5.0) / 5.0  # Normalize to [0,1]
        return ratio

    def compute_uc_variance(self, mc_probs_c: torch.Tensor) -> torch.Tensor:
        """
        Variance of per-seed Uc across MC passes.
        High variance → temporal instability in uncertainty estimates.
        mc_probs_c: [B, M, C]
        """
        M = mc_probs_c.shape[1]
        per_seed_entropy = []
        for m in range(M):
            p_m = torch.clamp(mc_probs_c[:, m, :], min=self.eps)
            h_m = -(p_m * torch.log(p_m)).sum(dim=-1)
            per_seed_entropy.append(h_m)
        entropies = torch.stack(per_seed_entropy, dim=1)  # [B, M]
        # Normalize variance by mean to get coefficient of variation
        mean_h = entropies.mean(dim=1) + self.eps
        return (entropies.var(dim=1) / mean_h).clamp(0.0, 1.0)

    def compute_genre_diversity(
        self,
        item_sequence: torch.Tensor,
        item_genre_lookup: Optional[Dict[int, List[str]]],
    ) -> torch.Tensor:
        """
        Fraction of unique genres over all genres in sequence.
        High → broad taste (hard to model with collaborative filtering alone).
        """
        B = item_sequence.shape[0]
        diversity = torch.zeros(B)
        if item_genre_lookup is None:
            return diversity
        for b in range(B):
            valid = item_sequence[b][item_sequence[b] != 0].tolist()
            genres = []
            for item_id in valid:
                genres.extend(item_genre_lookup.get(item_id, []))
            if not genres:
                continue
            diversity[b] = len(set(genres)) / len(genres)
        return diversity.to(item_sequence.device)

    def compute_tail_ratio(
        self,
        item_sequence: torch.Tensor,
        item_frequencies: torch.Tensor,
        num_recent: int = 3,
    ) -> torch.Tensor:
        """
        Fraction of recent items below median global popularity.
        High → user recently explored the long tail → semantic signal more reliable.
        """
        B = item_sequence.shape[0]
        ratio = torch.zeros(B, device=item_sequence.device)
        freq  = item_frequencies.to(item_sequence.device, dtype=torch.float32)
        median_pop = freq[freq > 0].median()
        for b in range(B):
            valid  = item_sequence[b][item_sequence[b] != 0]
            recent = valid[-num_recent:]
            if len(recent) == 0:
                continue
            below = (freq[recent] < median_pop).float().mean()
            ratio[b] = below
        return ratio

    def extract_evidence_features_12d(
        self,
        mc_probs_c: torch.Tensor,
        mc_probs_s: torch.Tensor,
        input_seqs: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        item_frequencies: torch.Tensor,
        item_genre_lookup: Optional[Dict[int, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts the 12-dimensional evidence feature vector x_u ∈ ℝ¹².

        Returns:
            features: [B, 12]
            mean_pc:  [B, C]
            mean_ps:  [B, C]
        """
        # Core 8 features
        mean_pc, U_c = self.compute_mutual_information(mc_probs_c)
        mean_ps, U_s = self.compute_mutual_information(mc_probs_s)
        D_JS = self.compute_js_divergence(mean_pc, mean_ps)
        H_c  = self.compute_entropy(mean_pc)
        H_s  = self.compute_entropy(mean_ps)

        lengths  = (input_seqs != 0).sum(dim=-1).float()
        norm_len = torch.clamp(lengths / 50.0, 0.0, 1.0)
        recency  = torch.clamp(lengths / 10.0, 0.0, 1.0)
        drift    = self.compute_session_drift(input_seqs, semantic_embeddings)

        # 4 new features
        pop_ratio     = self.compute_popularity_ratio(input_seqs, item_frequencies)
        uc_variance   = self.compute_uc_variance(mc_probs_c)
        genre_div     = self.compute_genre_diversity(input_seqs, item_genre_lookup)
        tail_ratio    = self.compute_tail_ratio(input_seqs, item_frequencies)

        features = torch.stack([
            U_c,
            U_s,
            D_JS,
            H_c / 10.0,
            H_s / 10.0,
            norm_len,
            recency,
            drift,
            pop_ratio,
            uc_variance,
            genre_div,
            tail_ratio,
        ], dim=-1)  # [B, 12]

        return features, mean_pc, mean_ps
