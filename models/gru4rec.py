"""
GRU4Rec: Session-based Recommendations with Recurrent Neural Networks.

Reference: Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks",
ICLR 2016.

Architecture:
  - GRU-based sequential encoder
  - Item embedding lookup
  - TOP-1 or BPR loss during training
  - Last hidden state as user representation at inference
  - MC-Dropout for epistemic uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GRU4Rec(nn.Module):
    """
    GRU-based sequential recommendation model.

    This serves as an important RNN-family baseline showing that
    attention-based models (SASRec, BERT4Rec) are needed for
    long-range dependency capture.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items   = num_items
        self.hidden_units = hidden_units
        self.num_layers  = num_layers

        # Item embedding (padding_idx=0)
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.emb_drop = nn.Dropout(embedding_dropout)

        # Stacked GRU
        self.gru = nn.GRU(
            input_size=hidden_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Output dropout (applied to last hidden state before scoring)
        self.output_drop = nn.Dropout(dropout_rate)
        self.layer_norm  = nn.LayerNorm(hidden_units, eps=1e-8)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)

    def encode(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """
        Encode user history through GRU.

        Args:
            log_seqs: [B, T] item ID sequences (0-padded)

        Returns:
            hidden: [B, T, D] all hidden states
            last_h: [B, D] last valid hidden state (before padding)
        """
        B, T = log_seqs.shape
        device = log_seqs.device

        # Embed
        seqs = self.emb_drop(self.item_emb(log_seqs))  # [B, T, D]

        # Pack padded sequence for efficiency
        lengths = (log_seqs != 0).sum(dim=1).cpu().clamp(min=1)  # [B]
        packed  = nn.utils.rnn.pack_padded_sequence(
            seqs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        hidden, _     = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )  # [B, T, D]

        hidden = self.layer_norm(hidden)

        # Extract last valid hidden state
        idx    = (lengths - 1).long().to(device)  # [B]
        last_h = hidden[torch.arange(B, device=device), idx]  # [B, D]
        last_h = self.output_drop(last_h)

        return hidden, last_h

    def forward(
        self,
        log_seqs: torch.Tensor,
        pos_seqs: torch.Tensor,
        neg_seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass using BPR-style Binary Cross-Entropy.

        Args:
            log_seqs: [B, T] input sequences
            pos_seqs: [B, T] positive item targets
            neg_seqs: [B, T] negative item samples

        Returns:
            pos_logits: [B, T] positive scores
            neg_logits: [B, T] negative scores
        """
        hidden, _ = self.encode(log_seqs)  # [B, T, D]

        pos_embs = self.item_emb(pos_seqs)  # [B, T, D]
        neg_embs = self.item_emb(neg_seqs)  # [B, T, D]

        pos_logits = (hidden * pos_embs).sum(dim=-1)  # [B, T]
        neg_logits = (hidden * neg_embs).sum(dim=-1)  # [B, T]

        return pos_logits, neg_logits

    def predict(
        self,
        log_seqs: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference scoring of candidate items.

        Args:
            log_seqs:     [B, T] user history
            item_indices: [B, C] candidate items OR [N] all items

        Returns:
            scores: [B, C] or [B, N] ranking scores
        """
        _, last_h = self.encode(log_seqs)  # [B, D]

        if item_indices.dim() == 2:
            cand_embs = self.item_emb(item_indices)  # [B, C, D]
            scores    = torch.bmm(cand_embs, last_h.unsqueeze(-1)).squeeze(-1)  # [B, C]
        else:
            cand_embs = self.item_emb(item_indices)  # [N, D]
            scores    = torch.matmul(last_h, cand_embs.T)  # [B, N]

        return scores

    def mc_forward(self, log_seqs: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
        """
        Monte-Carlo Dropout for epistemic uncertainty estimation.

        Returns:
            mc_reps: [B, num_samples, D] stochastic last hidden states
        """
        dropout_modules = [
            m for m in self.modules()
            if isinstance(m, (nn.Dropout, nn.GRU))
        ]
        # For GRU, enable training mode to activate internal dropout
        orig_states = {}
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                orig_states[m] = m.training
                m.train(True)

        samples = []
        for _ in range(num_samples):
            _, last_h = self.encode(log_seqs)
            samples.append(last_h)

        for m, state in orig_states.items():
            m.train(state)

        return torch.stack(samples, dim=1)  # [B, num_samples, D]
