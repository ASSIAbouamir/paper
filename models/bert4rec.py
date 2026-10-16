"""
BERT4Rec: Bidirectional Encoder Representations from Transformers for Sequential Recommendation.

Reference: Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations
from Transformer", CIKM 2019.

Key differences from SASRec:
  - Bidirectional self-attention (no causal masking during training)
  - Cloze task: random item masking during training (MLM-style)
  - At inference: the last item in the sequence is replaced with [MASK], and
    the model predicts the masked position.

Architecture:
  - Item embedding + Positional embedding
  - N Transformer blocks (bidirectional attention)
  - Linear head over item vocabulary for masked position prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BERT4RecBlock(nn.Module):
    """Single Transformer block with bidirectional attention (no causal mask)."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, 4 * hidden_units),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_units, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        seqs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = seqs
        seqs = self.attention_norm(seqs)
        attn_out, _ = self.attention(
            seqs, seqs, seqs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        seqs = residual + attn_out

        # Pre-norm FFN
        residual = seqs
        seqs = residual + self.ffn(self.ffn_norm(seqs))
        return seqs


class BERT4Rec(nn.Module):
    """
    BERT4Rec sequential recommendation model.

    Training: MLM Cloze task — randomly mask item positions and predict them.
    Inference: Mask the target position (last item) and predict from full context.
    """

    MASK_TOKEN = -1  # Sentinel; actual mask embedding index = num_items + 1

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        mask_prob: float = 0.15,       # MLM masking probability during training
    ):
        super().__init__()
        self.num_items   = num_items
        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units
        self.mask_prob   = mask_prob
        self.mask_token  = num_items + 1   # Special [MASK] token index
        self.pad_token   = 0              # Special [PAD] token index

        # +2: 0=PAD, num_items+1=MASK
        self.item_emb = nn.Embedding(num_items + 2, hidden_units, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len, hidden_units)
        self.emb_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.emb_drop = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([
            BERT4RecBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.head_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        # Output projection to item vocabulary (tied weights with item_emb)
        self.out_proj = nn.Linear(hidden_units, num_items + 2, bias=False)
        # Weight tying
        self.out_proj.weight = self.item_emb.weight

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1 and "emb" not in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _embed(self, seqs: torch.Tensor) -> torch.Tensor:
        """Embed items + positions, apply dropout and norm."""
        B, T = seqs.shape
        device = seqs.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(seqs) + self.pos_emb(positions)
        x = self.emb_drop(self.emb_norm(x))
        return x

    def encode(
        self,
        seqs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a sequence through all Transformer blocks.
        seqs: [B, T] item indices (may contain MASK tokens)
        key_padding_mask: [B, T] True where positions should be ignored (PAD)
        Returns: [B, T, D] contextual representations
        """
        x = self._embed(seqs)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.head_norm(x)

    def forward(
        self,
        log_seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass with random MLM masking.

        Args:
            log_seqs: [B, T] original item sequences (0-padded)

        Returns:
            masked_seqs:   [B, T] sequences with random positions replaced by MASK
            mask_labels:   [B, T] original items at masked positions, 0 elsewhere
            logits:        [B, T, V] prediction logits over vocabulary
        """
        B, T = log_seqs.shape
        device = log_seqs.device

        # Build MLM mask: only mask non-padding positions
        non_pad = (log_seqs != self.pad_token)
        rand    = torch.rand(B, T, device=device)
        to_mask = non_pad & (rand < self.mask_prob)

        # Ensure at least 1 position is masked per sequence
        for b in range(B):
            if not to_mask[b].any() and non_pad[b].any():
                valid_positions = non_pad[b].nonzero(as_tuple=True)[0]
                chosen = valid_positions[torch.randint(len(valid_positions), (1,))]
                to_mask[b, chosen] = True

        masked_seqs  = log_seqs.clone()
        mask_labels  = torch.zeros_like(log_seqs)

        mask_labels[to_mask]  = log_seqs[to_mask]
        masked_seqs[to_mask]  = self.mask_token

        key_padding_mask = ~non_pad  # True = ignore (PAD positions)

        hidden = self.encode(masked_seqs, key_padding_mask=key_padding_mask)  # [B, T, D]
        logits = self.out_proj(hidden)                                         # [B, T, V]

        return masked_seqs, mask_labels, logits

    def predict(self, log_seqs: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Inference: insert [MASK] at the last position and predict from full context.

        Args:
            log_seqs:     [B, T] historical item sequences (0-padded)
            item_indices: [B, C] candidate item indices (C candidates per user)
                          OR [N] flat item indices for full-catalog ranking

        Returns:
            scores: [B, C] ranking scores for candidate items
        """
        B, T = log_seqs.shape
        device = log_seqs.device

        # Append [MASK] at the last valid position
        masked_seqs = log_seqs.clone()
        # Find last non-padding position and replace with MASK
        lengths = (log_seqs != self.pad_token).sum(dim=1)  # [B]
        for b in range(B):
            pos = min(int(lengths[b].item()) - 1, T - 1)
            masked_seqs[b, pos] = self.mask_token

        key_padding_mask = (masked_seqs == self.pad_token)  # [B, T]

        with torch.no_grad():
            hidden = self.encode(masked_seqs, key_padding_mask=key_padding_mask)  # [B, T, D]

        # Take representation at the [MASK] position
        mask_pos = torch.clamp(lengths - 1, 0, T - 1)  # [B]
        final_rep = hidden[torch.arange(B, device=device), mask_pos]  # [B, D]

        # Score candidates
        if item_indices.dim() == 2:
            # [B, C, D]
            cand_embs = self.item_emb(item_indices)
            scores = torch.bmm(cand_embs, final_rep.unsqueeze(-1)).squeeze(-1)  # [B, C]
        else:
            # Full catalog: [N, D]
            cand_embs = self.item_emb(item_indices)
            scores = torch.matmul(final_rep, cand_embs.T)  # [B, N]

        return scores

    def mc_forward(self, log_seqs: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
        """
        Monte-Carlo Dropout forward passes for epistemic uncertainty estimation.
        Activates dropout during inference for M stochastic passes.

        Returns:
            mc_reps: [B, num_samples, D] stochastic hidden representations
        """
        dropout_modules = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        orig_states     = [m.training for m in dropout_modules]

        for m in dropout_modules:
            m.train(True)

        B, T     = log_seqs.shape
        device   = log_seqs.device
        lengths  = (log_seqs != self.pad_token).sum(dim=1)
        mask_pos = torch.clamp(lengths - 1, 0, T - 1)

        masked_seqs = log_seqs.clone()
        for b in range(B):
            masked_seqs[b, mask_pos[b]] = self.mask_token

        key_padding_mask = (masked_seqs == self.pad_token)

        samples = []
        for _ in range(num_samples):
            hidden = self.encode(masked_seqs, key_padding_mask=key_padding_mask)
            rep    = hidden[torch.arange(B, device=device), mask_pos]
            samples.append(rep)

        for m, state in zip(dropout_modules, orig_states):
            m.train(state)

        return torch.stack(samples, dim=1)  # [B, num_samples, D]


def bert4rec_training_loss(
    logits: torch.Tensor,
    mask_labels: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Computes cross-entropy loss only over masked positions.

    Args:
        logits:      [B, T, V] prediction logits
        mask_labels: [B, T] true item IDs at masked positions (0 = not masked)
        ignore_index: label value to ignore (PAD = 0)

    Returns:
        loss: scalar
    """
    B, T, V = logits.shape
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    # Reshape: [B*T, V] and [B*T]
    loss = loss_fn(logits.view(B * T, V), mask_labels.view(B * T))
    return loss
