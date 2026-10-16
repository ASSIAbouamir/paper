"""
BiX-GDR: Gated Bilateral Cross-Attention with Geometric Discrepancy Regularization
for Dual-Expert Sequential Recommendation.

Integrates:
1. Dual-stream Transformer backbone (Collaborative item IDs + Frozen Semantic Text Embeddings)
2. Causal Bilateral Cross-Attention with Adaptive Information Injection Gates
3. Candidate-Level Geometric Discrepancy Regularization (- lambda * |Sc - Ss|)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedCrossAttentionBlock(nn.Module):
    """
    Transformer layer combining causal self-attention, bilateral cross-attention,
    and dynamic information injection gating to filter cross-modal noise.
    """
    def __init__(self, hidden_dim: int = 64, num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        # Self-attention per modality
        self.self_attn_c = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_s = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_c1 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ln_s1 = nn.LayerNorm(hidden_dim, eps=1e-8)

        # Bilateral cross-attention: Collab queries Semantic, Semantic queries Collab
        self.cross_attn_c2s = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_s2c = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_s = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_c2 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ln_s2 = nn.LayerNorm(hidden_dim, eps=1e-8)

        # Feed-forward networks
        self.ffn_c = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln_c3 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ln_s3 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_c: torch.Tensor,
        h_s: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Causal Self-Attention
        c_norm = self.ln_c1(h_c)
        s_norm = self.ln_s1(h_s)
        ac, _ = self.self_attn_c(c_norm, c_norm, c_norm, attn_mask=causal_mask, need_weights=False)
        as_out, _ = self.self_attn_s(s_norm, s_norm, s_norm, attn_mask=causal_mask, need_weights=False)
        h_c = h_c + self.dropout(ac)
        h_s = h_s + self.dropout(as_out)

        # 2. Bilateral Cross-Attention
        c_norm2 = self.ln_c2(h_c)
        s_norm2 = self.ln_s2(h_s)
        cross_c, _ = self.cross_attn_c2s(query=c_norm2, key=s_norm2, value=s_norm2, attn_mask=causal_mask, need_weights=False)
        cross_s, _ = self.cross_attn_s2c(query=s_norm2, key=c_norm2, value=c_norm2, attn_mask=causal_mask, need_weights=False)

        # 3. Adaptive Information Injection Gating
        g_c = torch.sigmoid(self.gate_c(torch.cat([c_norm2, cross_c], dim=-1)))
        g_s = torch.sigmoid(self.gate_s(torch.cat([s_norm2, cross_s], dim=-1)))
        h_c = h_c + self.dropout(g_c * cross_c)
        h_s = h_s + self.dropout(g_s * cross_s)

        # 4. FFN
        h_c = h_c + self.ffn_c(self.ln_c3(h_c))
        h_s = h_s + self.ffn_s(self.ln_s3(h_s))
        return h_c, h_s


class BiXGDR(nn.Module):
    """
    BiX-GDR Sequential Recommender Architecture.
    """
    def __init__(
        self,
        num_items: int,
        semantic_table: torch.Tensor,
        hidden_dim: int = 64,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("semantic_table", semantic_table)

        # Item & Positional Embeddings
        self.collab_item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.sem_proj = nn.Linear(semantic_table.shape[1], hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Gated Bilateral Transformer Blocks
        self.blocks = nn.ModuleList([
            GatedCrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final_c = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ln_final_s = nn.LayerNorm(hidden_dim, eps=1e-8)

        # Adaptive modal balancing and geometric discrepancy penalty parameters
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.log_lambda = nn.Parameter(torch.tensor([-1.5]))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "semantic_table" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def encode_semantic_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        raw_sem = self.semantic_table[item_ids]
        return F.normalize(self.sem_proj(raw_sem), p=2, dim=-1)

    def forward_seq(self, seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = seqs.shape
        pos = torch.arange(T, device=seqs.device).unsqueeze(0).repeat(B, 1)

        # Sequence representations
        h_c = self.emb_dropout(self.collab_item_emb(seqs) + self.pos_emb(pos))
        h_s = self.emb_dropout(self.encode_semantic_items(seqs) + self.pos_emb(pos))

        # Padding mask
        timeline_mask = (seqs == 0)
        h_c = h_c * (~timeline_mask).unsqueeze(-1)
        h_s = h_s * (~timeline_mask).unsqueeze(-1)

        # Causal timeline attention mask
        causal_mask = torch.triu(torch.ones(T, T, device=seqs.device), diagonal=1).bool()
        for block in self.blocks:
            h_c, h_s = block(h_c, h_s, causal_mask=causal_mask)

        u_c = self.ln_final_c(h_c)[:, -1, :]
        u_s = self.ln_final_s(h_s)[:, -1, :]
        return u_c, u_s

    def compute_all_scores(self, u_c: torch.Tensor, u_s: torch.Tensor) -> torch.Tensor:
        """
        Computes candidate logits over full item catalog with geometric discrepancy regularization.
        """
        all_ids = torch.arange(1, self.num_items + 1, device=u_c.device)
        i_c = self.collab_item_emb(all_ids)
        i_s = self.encode_semantic_items(all_ids)

        sc = u_c @ i_c.T
        ss = u_s @ i_s.T

        w = torch.sigmoid(self.alpha)
        base_score = w * sc + (1.0 - w) * ss

        # Candidate-level Geometric Discrepancy Regularization
        lam = torch.sigmoid(self.log_lambda) * 0.35
        discrepancy = torch.abs(sc - ss)
        return base_score - lam * discrepancy
