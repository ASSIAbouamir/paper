"""
Dual-View Sequential Recommendation with Epistemic Uncertainty and Disagreement Routing.

Architecture:
1. Collaborative Expert (SASRec):
   - Item embedding table E_c in R^(N+1 x D)
   - Causal Multi-Head Self-Attention Blocks with Dropout
   - M Monte-Carlo Dropout passes -> mc_probs_c in R^(B x M x N)

2. Semantic Expert (MiniLM Sequential Encoder):
   - Precomputed frozen MiniLM item text embeddings S in R^(N+1 x D_sem)
   - Trainable semantic projection W_s: D_sem -> D
   - Causal Multi-Head Self-Attention Blocks with Dropout
   - M Monte-Carlo Dropout passes -> mc_probs_s in R^(B x M x N)

3. Evidence Analyzer:
   - U_c: Collaborative predictive epistemic uncertainty (Mutual Information)
   - U_s: Semantic predictive epistemic uncertainty (Mutual Information)
   - D_JS: Cross-view Jensen-Shannon disagreement JS(p_c || p_s)
   - Contextual signals: H_c, H_s, L_u, R_u, Drift_u
   - Evidence vector x_u in R^8

4. Selective Router:
   - f_theta(x_u) -> logits over actions { C: Collaborative, S: Semantic, H: Hybrid }
   - Supervised by offline Oracle targets on validation set
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from models.evidence_analyzer import EvidenceAnalyzer
from models.selective_router import SelectiveRouter


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.attn_ln = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn_ln = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.attn_ln(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x_norm = self.ffn_ln(x)
        ffn_out = self.fc2(F.relu(self.fc1(x_norm)))
        return x + self.dropout(ffn_out)


class CollaborativeExpert(nn.Module):
    """
    SASRec Sequential Recommender with Monte-Carlo Dropout for Epistemic Uncertainty Estimation.
    """
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        max_seq_len: int = 50,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.last_ln = nn.LayerNorm(hidden_dim, eps=1e-8)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        B, T = log_seqs.shape
        seqs = self.item_emb(log_seqs)
        positions = torch.arange(T, device=log_seqs.device).unsqueeze(0).repeat(B, 1)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions))

        timeline_mask = (log_seqs == 0)
        seqs = seqs * (~timeline_mask).unsqueeze(-1)

        causal_mask = torch.triu(torch.ones(T, T, device=log_seqs.device), diagonal=1).bool()
        for block in self.blocks:
            seqs = block(seqs, mask=causal_mask)
        return self.last_ln(seqs)

    def forward(self, in_seq: torch.Tensor, pos_seq: torch.Tensor, neg_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_feats = self.log2feats(in_seq)
        pos_embs = self.item_emb(pos_seq)
        neg_embs = self.item_emb(neg_seq)

        pos_scores = (log_feats * pos_embs).sum(dim=-1)
        neg_scores = (log_feats * neg_embs).sum(dim=-1)
        return pos_scores, neg_scores

    def get_user_rep(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """Returns deterministic final user representation [B, D]."""
        return self.log2feats(log_seqs)[:, -1, :]

    def mc_predict(
        self,
        log_seqs: torch.Tensor,
        num_samples: int = 5,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes M Monte-Carlo Dropout passes to produce candidate probability distributions:
        mc_probs: [B, M, N]
        mean_scores: [B, N]
        """
        # Save training state and activate dropouts
        dropouts = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        orig_states = [m.training for m in dropouts]
        for m in dropouts:
            m.train(True)

        all_ids = torch.arange(1, self.num_items + 1, device=log_seqs.device)
        cand_embs = self.item_emb(all_ids) # [N, D]

        mc_prob_list = []
        mc_score_list = []
        for _ in range(num_samples):
            user_rep = self.log2feats(log_seqs)[:, -1, :] # [B, D]
            scores = user_rep @ cand_embs.T # [B, N]
            probs = F.softmax(scores / temperature, dim=-1)
            mc_prob_list.append(probs)
            mc_score_list.append(scores)

        # Restore dropout states
        for m, s in zip(dropouts, orig_states):
            m.train(s)

        mc_probs = torch.stack(mc_prob_list, dim=1) # [B, M, N]
        mean_scores = torch.stack(mc_score_list, dim=1).mean(dim=1) # [B, N]
        return mc_probs, mean_scores


class SemanticExpert(nn.Module):
    """
    Semantic Sequential Recommender leveraging frozen MiniLM item embeddings
    with a trainable projection and causal Transformer sequence encoder.
    """
    def __init__(
        self,
        num_items: int,
        semantic_table: torch.Tensor,
        hidden_dim: int = 64,
        max_seq_len: int = 50,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("semantic_table", semantic_table)

        sem_dim = semantic_table.shape[1]
        self.sem_proj = nn.Linear(sem_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.last_ln = nn.LayerNorm(hidden_dim, eps=1e-8)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "semantic_table" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def encode_semantic_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        raw_sem = self.semantic_table[item_ids]
        proj_sem = self.sem_proj(raw_sem)
        return F.normalize(proj_sem, p=2, dim=-1)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        B, T = log_seqs.shape
        seqs = self.encode_semantic_items(log_seqs)
        positions = torch.arange(T, device=log_seqs.device).unsqueeze(0).repeat(B, 1)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions))

        timeline_mask = (log_seqs == 0)
        seqs = seqs * (~timeline_mask).unsqueeze(-1)

        causal_mask = torch.triu(torch.ones(T, T, device=log_seqs.device), diagonal=1).bool()
        for block in self.blocks:
            seqs = block(seqs, mask=causal_mask)
        return self.last_ln(seqs)

    def forward(self, in_seq: torch.Tensor, pos_seq: torch.Tensor, neg_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_feats = self.log2feats(in_seq)
        pos_embs = self.encode_semantic_items(pos_seq)
        neg_embs = self.encode_semantic_items(neg_seq)

        pos_scores = (log_feats * pos_embs).sum(dim=-1)
        neg_scores = (log_feats * neg_embs).sum(dim=-1)
        return pos_scores, neg_scores

    def get_user_rep(self, log_seqs: torch.Tensor) -> torch.Tensor:
        return self.log2feats(log_seqs)[:, -1, :]

    def mc_predict(
        self,
        log_seqs: torch.Tensor,
        num_samples: int = 5,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes M Monte-Carlo Dropout passes to produce semantic probability distributions:
        mc_probs: [B, M, N]
        mean_scores: [B, N]
        """
        dropouts = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        orig_states = [m.training for m in dropouts]
        for m in dropouts:
            m.train(True)

        all_ids = torch.arange(1, self.num_items + 1, device=log_seqs.device)
        cand_embs = self.encode_semantic_items(all_ids) # [N, D]

        mc_prob_list = []
        mc_score_list = []
        for _ in range(num_samples):
            user_rep = self.log2feats(log_seqs)[:, -1, :]
            scores = user_rep @ cand_embs.T
            probs = F.softmax(scores / temperature, dim=-1)
            mc_prob_list.append(probs)
            mc_score_list.append(scores)

        for m, s in zip(dropouts, orig_states):
            m.train(s)

        mc_probs = torch.stack(mc_prob_list, dim=1)
        mean_scores = torch.stack(mc_score_list, dim=1).mean(dim=1)
        return mc_probs, mean_scores


class DualExpertEvidenceRecSys(nn.Module):
    """
    End-to-End Dual-Expert System with Selective Evidence Router.
    """
    def __init__(
        self,
        num_items: int,
        semantic_table: torch.Tensor,
        hidden_dim: int = 64,
        max_seq_len: int = 50,
        num_mc_samples: int = 5,
        router_hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_items = num_items
        self.num_mc_samples = num_mc_samples
        self.register_buffer("semantic_table", semantic_table)

        self.collab_expert = CollaborativeExpert(
            num_items=num_items,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        self.semantic_expert = SemanticExpert(
            num_items=num_items,
            semantic_table=semantic_table,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        self.evidence_analyzer = EvidenceAnalyzer()
        self.selective_router = SelectiveRouter(
            in_features=8,
            hidden_dim=router_hidden_dim,
            num_actions=3,
            dropout=0.10
        )

    def extract_evidence(self, log_seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts evidence features x_u and distributions p_c, p_s, p_h:
        Returns:
            evidence_features: [B, 8]
            p_c: [B, N]
            p_s: [B, N]
            p_h: [B, N]
        """
        mc_probs_c, _ = self.collab_expert.mc_predict(log_seqs, num_samples=self.num_mc_samples)
        mc_probs_s, _ = self.semantic_expert.mc_predict(log_seqs, num_samples=self.num_mc_samples)

        evidence_features, p_c, p_s = self.evidence_analyzer.extract_evidence_features(
            mc_probs_c=mc_probs_c,
            mc_probs_s=mc_probs_s,
            input_seqs=log_seqs,
            semantic_embeddings=self.semantic_table
        )
        p_h = 0.5 * (p_c + p_s)
        return evidence_features, p_c, p_s, p_h
