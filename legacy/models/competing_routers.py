"""
Competing Routing Mechanisms — Ablation Suite.

Implements three alternative routing mechanisms to compare against the
Selective Evidence Router (MLP-based):

  1. MMoE (Multi-gate Mixture-of-Experts): Ma et al., KDD 2018
     - Shared expert networks + per-task gating networks
     - Adapted here: gating on evidence x_u → soft expert mixture

  2. AttentionGateRouter: Attention-based soft routing
     - x_u attends over [p_c; p_s] via cross-attention
     - Routing weights derived from attention scores

  3. BanditRouter: Thompson Sampling UCB router
     - Maintains per-action Beta(α, β) posterior
     - Selects action via Thompson sampling at inference
     - Updated with binary reward signal (hit@K or not)
     - Designed for online/streaming settings

Scientific purpose:
  Showing that our MLP router outperforms these alternatives validates
  that (a) the evidence features x_u carry routing signal, and (b)
  the MLP parameterization is appropriate for the task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# 1. MMoE Router (Multi-gate Mixture-of-Experts)
# ══════════════════════════════════════════════════════════════════════════════

class MMoERouter(nn.Module):
    """
    Multi-gate Mixture-of-Experts routing adapted for evidence-based selection.

    Architecture:
      - K shared expert networks (each maps x_u → hidden)
      - 1 gating network: softmax over K experts conditioned on x_u
      - Final head: gated mixture → routing logits over {C, S, H}

    Key difference from our MLP router:
      - The MLP router directly maps x_u → logits
      - MMoE routes through K intermediate expert representations,
        theoretically allowing disentangled feature specialization
    """

    def __init__(
        self,
        in_features: int = 8,
        num_experts: int = 4,
        expert_hidden: int = 32,
        num_actions: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_actions = num_actions

        # Shared experts: each maps x_u → expert_hidden
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, expert_hidden),
                nn.LayerNorm(expert_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, expert_hidden),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        ])

        # Single gating network: x_u → softmax over K experts
        self.gate = nn.Sequential(
            nn.Linear(in_features, num_experts),
        )

        # Output head: aggregated expert output → routing logits
        self.head = nn.Linear(expert_hidden, num_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_features]
        Returns: logits [B, num_actions]
        """
        # Gate: [B, K] soft weights over experts
        gate_weights = F.softmax(self.gate(x), dim=-1)   # [B, K]

        # Expert outputs: [K, B, expert_hidden]
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # [B, K, expert_hidden]

        # Gated mixture: Σ_k gate_k * expert_k(x)
        g = gate_weights.unsqueeze(-1)           # [B, K, 1]
        mixture = (g * expert_outs).sum(dim=1)   # [B, expert_hidden]

        return self.head(mixture)  # [B, num_actions]

    def route(
        self,
        p_c: torch.Tensor,
        p_s: torch.Tensor,
        p_h: torch.Tensor,
        x: torch.Tensor,
        hard_selection: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits  = self.forward(x)
        probs   = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)

        if hard_selection:
            p_routed = torch.zeros_like(p_c)
            for b in range(p_c.shape[0]):
                act = actions[b].item()
                p_routed[b] = [p_c, p_s, p_h][act][b]
        else:
            p_routed = probs[:, 0:1] * p_c + probs[:, 1:2] * p_s + probs[:, 2:3] * p_h

        return p_routed, actions, probs


# ══════════════════════════════════════════════════════════════════════════════
# 2. Attention Gate Router
# ══════════════════════════════════════════════════════════════════════════════

class AttentionGateRouter(nn.Module):
    """
    Cross-attention routing: the evidence vector x_u acts as a query
    that attends over the expert score vectors [p_c, p_s, p_h].

    Unlike the MLP router which uses only evidence meta-features,
    this mechanism directly incorporates the score distributions into routing.

    Architecture:
      Query:  Q = W_q(x_u)                      [B, d_k]
      Keys:   K_a = W_k(top50(p_a))             [B, 3, d_k]  for a in {C, S, H}
      Weights: α = softmax(Q K^T / sqrt(d_k))   [B, 3]
      Scores:  p_routed = Σ_a α_a * p_a         [B, N_items]

    KEY DESIGN DECISION — Fixed top-K projection:
      We project each expert distribution to its top-K (default K=50) scores
      before computing attention keys. This is ALWAYS fixed at K=50 regardless
      of num_items, ensuring a stable key representation. If num_items < K,
      we pad with zeros (which contribute no attention signal).
      This avoids variable-length keys and unstable gradients.

    IMPORTANT: This router uses score distributions as keys, which means it
    requires p_c, p_s, p_h at routing time (unlike the MLP router which only
    needs x_u). This is by design — it's a different inductive bias.
    """

    def __init__(
        self,
        in_features: int = 8,
        top_k_proj: int = 50,      # Fixed K — always use exactly 50 top scores
        d_key: int = 32,
        num_actions: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.top_k_proj  = top_k_proj   # FIXED at construction time
        self.d_key       = d_key
        self.num_actions = num_actions
        self.scale       = d_key ** -0.5

        # Query projection: evidence features → query vector
        self.W_q = nn.Sequential(
            nn.Linear(in_features, d_key),
            nn.LayerNorm(d_key),
            nn.Tanh(),
        )

        # Key projection: FIXED top_k_proj scores → key vector
        # Note: input dim is always top_k_proj regardless of catalog size
        self.W_k = nn.Linear(top_k_proj, d_key)

        # Auxiliary MLP for regularization (blended in with weight 0.1)
        self.aux_mlp = nn.Sequential(
            nn.Linear(in_features, num_actions),
        )

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _project_scores(self, p: torch.Tensor) -> torch.Tensor:
        """
        Project score distribution to a fixed-K compact key representation.

        Always produces a tensor of exactly [B, top_k_proj] regardless of
        the catalog size N_items. This guarantees stable key dimensions:
          - If N_items >= top_k_proj: take the top-K scores (sorted descending)
          - If N_items < top_k_proj:  take all scores and zero-pad to top_k_proj

        Zero-padding contributes no information-theoretic signal because
        the key network W_k is linear: zeros map to the zero vector in key space.

        p: [B, N_items] — un-normalized or softmax scores
        Returns: [B, top_k_proj]
        """
        N_items = p.shape[-1]
        K       = min(self.top_k_proj, N_items)

        # Take top-K values (deterministic: largest K scores)
        top_vals, _ = torch.topk(p, K, dim=-1, largest=True, sorted=True)  # [B, K]

        if K < self.top_k_proj:
            # Pad with zeros up to top_k_proj
            pad = torch.zeros(
                p.shape[0], self.top_k_proj - K,
                device=p.device, dtype=p.dtype
            )
            top_vals = torch.cat([top_vals, pad], dim=-1)   # [B, top_k_proj]

        return top_vals   # always [B, top_k_proj]

    def forward(
        self,
        p_c: torch.Tensor,
        p_s: torch.Tensor,
        p_h: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs:   [B, 3] attention weights over {C, S, H}
            logits:  [B, 3] raw (pre-softmax) attention scores
        """
        # Query from evidence features
        Q = self.W_q(x)  # [B, d_key]

        # Keys from projected score distributions (always top_k_proj dim)
        K_c = self.W_k(self._project_scores(p_c))  # [B, d_key]
        K_s = self.W_k(self._project_scores(p_s))  # [B, d_key]
        K_h = self.W_k(self._project_scores(p_h))  # [B, d_key]

        # Stack keys: [B, 3, d_key]
        K = torch.stack([K_c, K_s, K_h], dim=1)

        # Attention scores: [B, 3]
        Q_exp  = Q.unsqueeze(1)                            # [B, 1, d_key]
        scores = (Q_exp * K).sum(dim=-1) * self.scale     # [B, 3]

        # Stability clamp before aux blend (avoids exploding logits)
        scores = torch.clamp(scores, min=-10.0, max=10.0)

        # Mix in auxiliary MLP signal for regularization
        aux_scores = self.aux_mlp(x)                       # [B, 3]
        logits     = scores + 0.1 * aux_scores             # blend

        probs = F.softmax(self.drop(logits), dim=-1)
        return probs, logits

    def route(
        self,
        p_c: torch.Tensor,
        p_s: torch.Tensor,
        p_h: torch.Tensor,
        x: torch.Tensor,
        hard_selection: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, logits = self.forward(p_c, p_s, p_h, x)
        actions       = torch.argmax(probs, dim=-1)

        if hard_selection:
            p_routed = torch.zeros_like(p_c)
            for b in range(p_c.shape[0]):
                p_routed[b] = [p_c, p_s, p_h][actions[b].item()][b]
        else:
            p_routed = probs[:, 0:1] * p_c + probs[:, 1:2] * p_s + probs[:, 2:3] * p_h

        return p_routed, actions, probs


# ══════════════════════════════════════════════════════════════════════════════
# 3. Bandit Router (Thompson Sampling)
# ══════════════════════════════════════════════════════════════════════════════

class BanditRouter:
    """
    Online Bandit Router using Thompson Sampling over Beta posteriors.

    Each action a ∈ {C, S, H} maintains a Beta(α_a, β_a) distribution.
    At inference:
      1. Sample θ_a ~ Beta(α_a, β_a) for each action
      2. Select action a* = argmax θ_a
      3. Route to a*
      4. Observe reward r = HR@K (1 if hit, 0 if miss)
      5. Update: α_a += r, β_a += (1 − r)

    ⚠️  PAPER DISCLAIMER (mandatory in any publication):
      The offline_replay_evaluation() method simulates Thompson Sampling
      using a REPLAY approach on pre-collected offline rankings. This is
      NOT equivalent to a true online bandit deployment where:
        (a) users arrive sequentially in real time,
        (b) the router observes real feedback immediately after each recommendation,
        (c) exploration affects which users are served next.

      In offline replay, we iterate over shuffled test users, route each user
      using the current posterior, observe the pre-computed rank (not a real
      user click), and update the posterior. This provides a lower-bound
      estimate of Thompson Sampling performance and is a standard offline
      evaluation methodology for bandits (Li et al., 2011; Precup et al., 2000).

      The reported HR@K represents the cumulative hit rate over the full
      offline replay trajectory, averaged over multiple shuffle orderings.

    Scientific purpose:
      Demonstrates that exploration-exploitation tradeoff via bandit routing
      is suboptimal compared to learned offline routing (MLP router) when
      sufficient training data is available, but may be preferable in
      cold-start or non-stationary settings.

    Reference:
      Li et al., "A Contextual-Bandit Approach to Personalized News Article
      Recommendation", WWW 2010.
    """

    def __init__(
        self,
        num_actions: int = 3,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        seed: int = 42,
    ):
        self.num_actions = num_actions
        self.rng         = np.random.default_rng(seed)
        # Beta distribution parameters: [α_C, α_S, α_H] and [β_C, β_S, β_H]
        self.alpha = np.full(num_actions, alpha_init, dtype=np.float64)
        self.beta  = np.full(num_actions, beta_init,  dtype=np.float64)
        self.counts = np.zeros(num_actions, dtype=np.int64)  # selection counts

    def select_action(self) -> int:
        """Thompson sampling: draw θ_a ~ Beta(α_a, β_a), pick argmax."""
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        """
        Bayesian update of Beta posterior.
        reward: 1.0 = hit, 0.0 = miss (can be soft: NDCG score ∈ [0,1])
        """
        self.alpha[action] += reward
        self.beta[action]  += (1.0 - reward)
        self.counts[action] += 1

    def get_posterior_means(self) -> np.ndarray:
        """Returns α / (α + β) for each action."""
        return self.alpha / (self.alpha + self.beta)

    def get_routing_stats(self) -> Dict[str, float]:
        means = self.get_posterior_means()
        return {
            "alpha":          self.alpha.tolist(),
            "beta":           self.beta.tolist(),
            "posterior_means": means.tolist(),
            "selection_counts": self.counts.tolist(),
            "selection_pct":   (self.counts / (self.counts.sum() + 1e-9)).tolist(),
        }

    def offline_replay_evaluation(
        self,
        uid_to_ranks: Dict[int, Dict[str, int]],
        action_names: Optional[Dict[int, str]] = None,
        k: int = 10,
        shuffle_seed: int = 42,
    ) -> Dict:
        """
        Simulates online Thompson Sampling on offline rankings via replay.

        Args:
            uid_to_ranks: {uid: {"Collaborative": rank, "Semantic": rank, "Hybrid": rank}}
            action_names: {0: "Collaborative", 1: "Semantic", 2: "Hybrid"}
            k: HR@K cutoff
            shuffle_seed: random order for user traversal

        Returns:
            results dict with cumulative HR@K, selection stats, etc.
        """
        if action_names is None:
            action_names = {0: "Collaborative", 1: "Semantic", 2: "Hybrid"}

        # Reset posterior
        self.alpha = np.ones(self.num_actions, dtype=np.float64)
        self.beta  = np.ones(self.num_actions, dtype=np.float64)
        self.counts = np.zeros(self.num_actions, dtype=np.int64)

        rng       = np.random.default_rng(shuffle_seed)
        user_ids  = np.array(list(uid_to_ranks.keys()))
        rng.shuffle(user_ids)

        hits        = []
        action_hist = []

        for uid in user_ids:
            ranks = uid_to_ranks[uid]
            action = self.select_action()
            action_name = action_names[action]

            rank   = ranks.get(action_name, k + 1)
            reward = 1.0 if rank <= k else 0.0

            self.update(action, reward)
            hits.append(reward)
            action_hist.append(action)

        action_hist_arr = np.array(action_hist)
        return {
            "HR@K":             float(np.mean(hits)),
            "routing_stats":    self.get_routing_stats(),
            "cumulative_HR":    np.cumsum(hits).tolist(),
            "action_histogram": {
                action_names[a]: int((action_hist_arr == a).sum())
                for a in range(self.num_actions)
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Routing Mechanism Comparison Ablation
# ══════════════════════════════════════════════════════════════════════════════

ROUTING_MECHANISMS = {
    "mlp_8d":        "Selective Router (MLP, 8D)     [Proposed v1]",
    "mlp_12d":       "Enhanced Router  (MLP, 12D)    [Proposed v2]",
    "mmoe":          "MMoE Router      (K=4 experts) [Competitor]",
    "attention_gate":"Attention Gate   (Cross-Attn)  [Competitor]",
    "bandit_ts":     "Bandit Router    (Thompson TS) [Online baseline]",
    "static_50_50":  "Static Fusion    (α=0.5)       [Non-adaptive baseline]",
    "length_gate":   "Length Gate      (L_u/50)      [Heuristic baseline]",
    "oracle":        "Oracle Router    (Per-user OPT)[Upper bound]",
}


def build_router(
    mechanism: str,
    in_features: int = 8,
    **kwargs,
) -> nn.Module:
    """
    Factory function to instantiate any routing mechanism by name.

    Args:
        mechanism: key from ROUTING_MECHANISMS
        in_features: feature dimension (8 for v1, 12 for enhanced)

    Returns:
        Router module (or BanditRouter for bandit_ts)
    """
    if mechanism in ("mlp_8d", "mlp_12d"):
        from models.selective_router import SelectiveRouter
        return SelectiveRouter(in_features=in_features, **kwargs)

    elif mechanism == "mmoe":
        return MMoERouter(in_features=in_features, **kwargs)

    elif mechanism == "attention_gate":
        return AttentionGateRouter(in_features=in_features, **kwargs)

    elif mechanism == "bandit_ts":
        return BanditRouter(**kwargs)

    else:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. "
            f"Available: {list(ROUTING_MECHANISMS.keys())}"
        )
