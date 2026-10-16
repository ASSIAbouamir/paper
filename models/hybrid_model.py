"""
Proposed Approach: Hybrid SASRec + LLM Textual Embeddings Model.
Features:
- Length-Conditioned Adaptive Gated Fusion (tailored for Cold-Start)
- LLM semantic projection
- Multi-head self-attention with causal masking
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sasrec import SASRecBlock


class MultiScaleSemanticGatedFusion(nn.Module):
    """
    Multi-Scale Semantic-Consistent Gated Fusion (MSC-Gate).
    Dynamically balances learned collaborative embeddings and fine-tuned LLM representations
    conditioned on:
    1. Interaction sequence length (L_u / tau_L);
    2. User historical semantic coherence (kappa_u based on semantic variance);
    3. Multi-scale candidate-to-history semantic alignment.
    """

    def __init__(self, item_dim: int, llm_dim: int, projected_dim: int = 64, gamma_0: float = 0.50):
        super(MultiScaleSemanticGatedFusion, self).__init__()
        self.item_dim = item_dim
        self.gamma_0 = gamma_0
        self.llm_proj = nn.Linear(llm_dim, projected_dim)
        self.proj_match = nn.Linear(projected_dim, item_dim) if projected_dim != item_dim else nn.Identity()

        # Multi-scale gate layer taking [item_emb, llm_proj, candidate_alignment]
        self.gate_layer = nn.Linear(item_dim * 2 + 1, item_dim)
        self.layernorm = nn.LayerNorm(item_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        item_emb: torch.Tensor,
        llm_emb: torch.Tensor,
        seq_lengths: torch.Tensor = None,
        history_semantic_mean: torch.Tensor = None,
        semantic_coherence: torch.Tensor = None
    ) -> torch.Tensor:
        """
        item_emb: [..., item_dim]
        llm_emb: [..., llm_dim]
        seq_lengths: Optional [BatchSize] interaction counts
        history_semantic_mean: Optional [BatchSize, item_dim] centroid of past consumed items
        semantic_coherence: Optional [BatchSize] kappa_u in [0, 1]
        """
        proj_llm = self.proj_match(self.llm_proj(llm_emb)) # [..., item_dim]

        # Candidate alignment feature
        if history_semantic_mean is not None and history_semantic_mean.dim() > 0:
            while history_semantic_mean.dim() < proj_llm.dim():
                history_semantic_mean = history_semantic_mean.unsqueeze(1)
            cand_align = F.cosine_similarity(proj_llm, history_semantic_mean, dim=-1, eps=1e-8).unsqueeze(-1)
        else:
            cand_align = torch.zeros((*proj_llm.shape[:-1], 1), device=item_emb.device)

        concat = torch.cat([item_emb, proj_llm, cand_align], dim=-1)
        gate = torch.sigmoid(self.gate_layer(concat)) # [..., item_dim]

        # 1. Length-conditioned saturation
        if seq_lengths is not None and seq_lengths.dim() > 0:
            length_factor = torch.clamp(seq_lengths.float() / 10.0, 0.1, 1.0)
            while length_factor.dim() < gate.dim():
                length_factor = length_factor.unsqueeze(-1)
            gate = gate * length_factor

        # 2. Semantic coherence modulation: high coherence allows collaborative reliance
        if semantic_coherence is not None and semantic_coherence.dim() > 0:
            while semantic_coherence.dim() < gate.dim():
                semantic_coherence = semantic_coherence.unsqueeze(-1)
            coherence_mod = self.gamma_0 + (1.0 - self.gamma_0) * semantic_coherence
            gate = gate * coherence_mod

        fused = gate * item_emb + (1.0 - gate) * proj_llm
        fused = self.layernorm(fused)
        return self.dropout(fused)


# Backward compatibility alias
LengthAdaptiveGatedFusion = MultiScaleSemanticGatedFusion


class SASRecLLMHybrid(nn.Module):
    """
    Hybrid RecSys model combining SASRec sequential transformer encoder with

    pre-computed LLM textual embeddings and Length-Adaptive Gated Fusion.
    """

    def __init__(
        self,
        num_items: int,
        llm_embeddings: torch.Tensor,
        max_seq_len: int = 50,
        item_embed_dim: int = 64,
        projected_llm_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2
    ):
        super(SASRecLLMHybrid, self).__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.item_embed_dim = item_embed_dim

        # Frozen LLM embedding table
        self.register_buffer("llm_embeddings", llm_embeddings)
        llm_dim = llm_embeddings.shape[1]

        # Learnable classic item embeddings & positional embeddings
        self.item_emb = nn.Embedding(num_items + 1, item_embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, item_embed_dim)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Length-Adaptive Gated Fusion Module
        self.fusion_module = LengthAdaptiveGatedFusion(
            item_dim=item_embed_dim,
            llm_dim=llm_dim,
            projected_dim=projected_llm_dim
        )

        # Transformer Blocks
        self.last_layernorm = nn.LayerNorm(item_embed_dim, eps=1e-8)
        self.blocks = nn.ModuleList([
            SASRecBlock(item_embed_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'llm_embeddings' in name:
                continue
            try:
                if 'weight' in name and param.dim() > 1:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            except Exception:
                pass

    def get_fused_item_representations(self, item_indices: torch.Tensor, seq_lengths: torch.Tensor = None) -> torch.Tensor:
        learned_embs = self.item_emb(item_indices)
        llm_embs = self.llm_embeddings[item_indices]
        return self.fusion_module(learned_embs, llm_embs, seq_lengths)

    def get_causal_mask(self, seq_len: int, device: torch.device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        device = log_seqs.device
        seq_len = log_seqs.shape[1]

        # Non-zero count gives sequence length per user
        seq_lengths = (log_seqs != 0).sum(dim=1)

        seqs = self.get_fused_item_representations(log_seqs, seq_lengths)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= (~timeline_mask).unsqueeze(-1)

        attn_mask = self.get_causal_mask(seq_len, device)
        for block in self.blocks:
            seqs = block(seqs, attn_mask=attn_mask)

        return self.last_layernorm(seqs)

    def forward(self, log_seqs: torch.Tensor, pos_seqs: torch.Tensor, neg_seqs: torch.Tensor):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.get_fused_item_representations(pos_seqs)
        neg_embs = self.get_fused_item_representations(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, log_seqs: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(log_seqs)
        final_user_rep = log_feats[:, -1, :]

        item_embs = self.get_fused_item_representations(item_indices)
        if item_embs.dim() == 3:
            scores = torch.bmm(item_embs, final_user_rep.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.matmul(final_user_rep, item_embs.T)
        return scores


class EpistemicSASRecLLMHybrid(SASRecLLMHybrid):
    """
    Epistemic Routing Hybrid Recommender:
    Quantifies collaborative representation uncertainty via Monte-Carlo Dropout (M passes)
    and dynamically routes ranking decisions between collaborative transitions and LLM semantic space.
    """
    def __init__(
        self,
        num_items: int,
        llm_embeddings: torch.Tensor,
        max_seq_len: int = 50,
        item_embed_dim: int = 64,
        projected_llm_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        num_mc_samples: int = 5,
        gate_w: float = 2.0,
        gate_b: float = -1.0
    ):
        super(EpistemicSASRecLLMHybrid, self).__init__(
            num_items=num_items,
            llm_embeddings=llm_embeddings,
            max_seq_len=max_seq_len,
            item_embed_dim=item_embed_dim,
            projected_llm_dim=projected_llm_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.num_mc_samples = num_mc_samples
        
        # Epistemic Gate parameters: alpha_u = sigmoid(w * log(1 + u_u) + b)
        self.epistemic_w = nn.Parameter(torch.tensor([gate_w], dtype=torch.float32))
        self.epistemic_b = nn.Parameter(torch.tensor([gate_b], dtype=torch.float32))
        
        # Linear semantic adapter for item representation
        self.semantic_proj = nn.Linear(llm_embeddings.shape[1], item_embed_dim)
        self.fused_norm = nn.LayerNorm(item_embed_dim)

    def compute_mc_representations(self, log_seqs: torch.Tensor, num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs M stochastic passes with active dropout to sample from posterior representation.
        Returns:
            mean_rep: [B, D]
            u_u: [B] scalar epistemic uncertainty
        """
        M = num_samples or self.num_mc_samples
        dropout_modules = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        orig_states = [m.training for m in dropout_modules]
        for m in dropout_modules:
            m.train(True)

        samples = []
        for _ in range(M):
            feats = self.log2feats(log_seqs) # [B, T, D]
            samples.append(feats[:, -1, :])  # [B, D]

        for m, state in zip(dropout_modules, orig_states):
            m.train(state)

        mc_tensor = torch.stack(samples, dim=1) # [B, M, D]
        mean_rep = torch.mean(mc_tensor, dim=1) # [B, D]
        diff = mc_tensor - mean_rep.unsqueeze(1) # [B, M, D]
        var_per_dim = torch.sum(diff ** 2, dim=1) / max(M - 1, 1) # [B, D]
        u_u = torch.mean(var_per_dim, dim=-1) # [B]
        return mean_rep, u_u

    def get_epistemic_gate(self, u_u: torch.Tensor) -> torch.Tensor:
        """Computes routing coefficient alpha_u in (0, 1)."""
        log_u = torch.log1p(torch.clamp(u_u, min=0.0)).unsqueeze(-1) # [B, 1]
        alpha_u = torch.sigmoid(self.epistemic_w * log_u + self.epistemic_b) # [B, 1]
        return alpha_u

    def predict_with_uncertainty(self, log_seqs: torch.Tensor, item_indices: torch.Tensor, num_mc_samples: int = 5):
        """
        Predicts ranking scores conditioned on epistemic uncertainty.
        Returns:
            scores: [B, num_candidates]
            alpha_u: [B, 1] LLM routing proportion
            u_u: [B] epistemic uncertainty values
        """
        mean_user_rep, u_u = self.compute_mc_representations(log_seqs, num_samples=num_mc_samples)
        alpha_u = self.get_epistemic_gate(u_u) # [B, 1]

        # Candidate embeddings: collaborative vs LLM
        collab_item_embs = self.item_emb(item_indices) # [..., D]
        raw_llm = self.llm_embeddings[item_indices]
        semantic_item_embs = self.semantic_proj(raw_llm) # [..., D]

        # Uncertainty routing fusion for candidates
        while alpha_u.dim() < collab_item_embs.dim():
            alpha_u = alpha_u.unsqueeze(1)

        fused_items = (1.0 - alpha_u) * collab_item_embs + alpha_u * semantic_item_embs
        fused_items = self.fused_norm(fused_items)

        # Candidate scoring
        if fused_items.dim() == 3:
            scores = torch.bmm(fused_items, mean_user_rep.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.matmul(mean_user_rep, fused_items.T)

        return scores, alpha_u.squeeze(-1), u_u

