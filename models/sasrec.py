"""
SASRec (Self-Attentive Sequential Recommendation) model implementation in PyTorch.
Reference: Kang & McAuley (ICDM 2018) "Self-Attentive Sequential Recommendation"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRecBlock(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super(SASRecBlock, self).__init__()
        self.attention_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.forward_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.forward_layer = PointWiseFeedForward(hidden_units, dropout_rate)

    def forward(self, seqs, attn_mask=None):
        Q = self.attention_layernorm(seqs)
        # MultiheadAttention with causal mask
        attn_out, _ = self.attention_layer(Q, Q, Q, attn_mask=attn_mask, need_weights=False)
        seqs = seqs + attn_out
        
        # Feed-forward block
        seqs_norm = self.forward_layernorm(seqs)
        seqs = self.forward_layer(seqs_norm)
        return seqs


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_seq_len: int = 50, hidden_units: int = 64, num_blocks: int = 2, num_heads: int = 2, dropout_rate: float = 0.2):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units

        # Embeddings
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            try:
                if 'weight' in name and param.dim() > 1:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            except Exception:
                pass

    def get_causal_mask(self, seq_len: int, device: torch.device):
        """Creates causal upper triangular mask for multi-head self-attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def log2feats(self, log_seqs):
        """Encodes user input interaction sequence into user context representation."""
        device = log_seqs.device
        seq_len = log_seqs.shape[1]

        # Item + Position embedding
        seqs = self.item_emb(log_seqs)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Padding mask (padding items set to 0)
        timeline_mask = (log_seqs == 0)
        seqs *= (~timeline_mask).unsqueeze(-1)

        # Causal mask for attention
        attn_mask = self.get_causal_mask(seq_len, device)

        # Pass through Transformer blocks
        for block in self.blocks:
            seqs = block(seqs, attn_mask=attn_mask)

        seqs = self.last_layernorm(seqs)
        return seqs

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        """Computes training loss using Binary Cross-Entropy with positive and negative targets."""
        log_feats = self.log2feats(log_seqs) # [B, T, D]

        pos_embs = self.item_emb(pos_seqs) # [B, T, D]
        neg_embs = self.item_emb(neg_seqs) # [B, T, D]

        pos_logits = (log_feats * pos_embs).sum(dim=-1) # [B, T]
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # [B, T]

        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):
        """Predicts ranking scores for candidate item_indices given user log_seqs."""
        log_feats = self.log2feats(log_seqs) # [B, T, D]
        final_user_rep = log_feats[:, -1, :] # [B, D] (take last position)

        item_embs = self.item_emb(item_indices) # [B, C, D] or [N_items, D]
        if item_embs.dim() == 3:
            scores = torch.bmm(item_embs, final_user_rep.unsqueeze(-1)).squeeze(-1) # [B, C]
        else:
            scores = torch.matmul(final_user_rep, item_embs.T) # [B, N_items]
        return scores

    def mc_forward(self, log_seqs, num_samples: int = 5):
        """
        Executes M stochastic forward passes with active dropout (Monte-Carlo Dropout)
        to draw samples from the posterior representation distribution p(h_u | S_u).
        Returns:
            mc_reps: [BatchSize, num_samples, hidden_units]
        """
        dropout_modules = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        orig_states = [m.training for m in dropout_modules]

        for m in dropout_modules:
            m.train(True)

        samples = []
        for _ in range(num_samples):
            log_feats = self.log2feats(log_seqs)
            samples.append(log_feats[:, -1, :])

        for m, state in zip(dropout_modules, orig_states):
            m.train(state)

        return torch.stack(samples, dim=1)

