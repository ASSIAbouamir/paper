"""Sequential recommenders used in BiXRec v2.

All models share one transformer block implementation (pre-LN, causal, with a
proper key-padding mask) so that differences between rows of the result tables
come from the fusion design, not from implementation details.

Single-stream models (one transformer):
    id      : SASRec, items = learned ID embeddings
    text    : semantic expert, items = projected frozen text embeddings (no IDs)
    hybrid  : early fusion, items = ID embedding + projected text embedding

Dual-stream model (two transformers, one per modality) with switches:
    cross   : bilateral cross-attention between the streams at every layer
    gate    : sigmoid injection gate on the cross-attention output
    penalty : candidate-level discrepancy penalty  - lambda * |s_c - s_s|
With cross=False the dual model is a *jointly trained late fusion* control: the
two streams never exchange information before scoring, but are trained end to
end through the fused score. Comparing it with cross=True isolates the effect of
cross-attention from the effect of joint training.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_mask(seq: torch.Tensor, heads: int) -> torch.Tensor:
    """Boolean mask (B*heads, T, T), True = blocked. Causal + key padding.
    Each position may always attend to itself so fully padded rows stay finite."""
    B, T = seq.shape
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=seq.device), 1)
    pad = seq.eq(0)
    m = causal.unsqueeze(0) | pad.unsqueeze(1)
    eye = torch.eye(T, dtype=torch.bool, device=seq.device).unsqueeze(0)
    m = m & ~eye
    return m.repeat_interleave(heads, dim=0)


class FFN(nn.Module):
    def __init__(self, d: int, p: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d), nn.Dropout(p))

    def forward(self, x):
        return self.net(x)


class SelfBlock(nn.Module):
    def __init__(self, d: int, h: int, p: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ffn = FFN(d, p)
        self.drop = nn.Dropout(p)

    def attend(self, x, mask):
        y = self.ln1(x)
        a, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        return x + self.drop(a)

    def forward(self, x, mask):
        x = self.attend(x, mask)
        return x + self.ffn(self.ln2(x))


class CrossBlock(nn.Module):
    """Self-attention per stream, then bilateral cross-attention, then FFN."""

    def __init__(self, d: int, h: int, p: float, cross: bool, gate: bool):
        super().__init__()
        self.c, self.s = SelfBlock(d, h, p), SelfBlock(d, h, p)
        self.cross, self.gate = cross, gate
        if cross:
            self.ln_xc, self.ln_xs = nn.LayerNorm(d), nn.LayerNorm(d)
            self.x_c = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)  # c queries s
            self.x_s = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)  # s queries c
            if gate:
                self.g_c, self.g_s = nn.Linear(2 * d, d), nn.Linear(2 * d, d)
                for g in (self.g_c, self.g_s):
                    nn.init.zeros_(g.bias)
            self.drop = nn.Dropout(p)

    def forward(self, c, s, mask):
        c, s = self.c.attend(c, mask), self.s.attend(s, mask)
        if self.cross:
            cn, sn = self.ln_xc(c), self.ln_xs(s)
            xc, _ = self.x_c(cn, sn, sn, attn_mask=mask, need_weights=False)
            xs, _ = self.x_s(sn, cn, cn, attn_mask=mask, need_weights=False)
            if self.gate:
                xc = torch.sigmoid(self.g_c(torch.cat([cn, xc], -1))) * xc
                xs = torch.sigmoid(self.g_s(torch.cat([sn, xs], -1))) * xs
            c, s = c + self.drop(xc), s + self.drop(xs)
        c = c + self.c.ffn(self.c.ln2(c))
        s = s + self.s.ffn(self.s.ln2(s))
        return c, s


class TextProjector(nn.Module):
    def __init__(self, text_emb: torch.Tensor, d: int):
        super().__init__()
        self.register_buffer("table", text_emb.float(), persistent=False)
        self.proj = nn.Sequential(nn.Linear(text_emb.shape[1], 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self) -> torch.Tensor:  # (N+1, d)
        out = self.proj(self.table)
        return torch.cat([torch.zeros_like(out[:1]), out[1:]], 0)


class SeqRec(nn.Module):
    def __init__(self, num_items: int, kind: str, text_emb: Optional[torch.Tensor] = None,
                 d: int = 64, layers: int = 2, heads: int = 2, max_len: int = 200, dropout: float = 0.2,
                 cross: bool = False, gate: bool = False, penalty: bool = False, lambda_max: float = 0.35):
        super().__init__()
        assert kind in {"id", "text", "hybrid", "dual"}
        self.kind, self.num_items, self.d, self.heads, self.max_len = kind, num_items, d, heads, max_len
        self.cross, self.gate, self.penalty, self.lambda_max = cross, gate, penalty, lambda_max
        needs_id = kind in {"id", "hybrid", "dual"}
        needs_text = kind in {"text", "hybrid", "dual"}
        if needs_text and text_emb is None:
            raise ValueError(f"model kind '{kind}' needs text embeddings")
        if needs_id:
            self.item_emb = nn.Embedding(num_items + 1, d, padding_idx=0)
            nn.init.normal_(self.item_emb.weight, std=d ** -0.5)
            with torch.no_grad():
                self.item_emb.weight[0].zero_()
        if needs_text:
            self.text = TextProjector(text_emb, d)
        n_pos_tables = 2 if kind == "dual" else 1
        self.pos = nn.ModuleList([nn.Embedding(max_len, d) for _ in range(n_pos_tables)])
        self.emb_drop = nn.Dropout(dropout)
        if kind == "dual":
            self.blocks = nn.ModuleList([CrossBlock(d, heads, dropout, cross, gate) for _ in range(layers)])
            self.ln_c, self.ln_s = nn.LayerNorm(d), nn.LayerNorm(d)
            self.mix = nn.Parameter(torch.zeros(1))          # w = sigmoid(mix), starts at 0.5
            self.pen = nn.Parameter(torch.full((1,), -1.5))  # lambda = lambda_max * sigmoid(pen)
        else:
            self.blocks = nn.ModuleList([SelfBlock(d, heads, dropout) for _ in range(layers)])
            self.ln = nn.LayerNorm(d)

    # ------------------------------------------------------------------
    def item_tables(self) -> Dict[str, torch.Tensor]:
        if self.kind == "id":
            return {"x": self.item_emb.weight}
        if self.kind == "text":
            return {"x": self.text()}
        if self.kind == "hybrid":
            return {"x": self.item_emb.weight + self.text()}
        return {"c": self.item_emb.weight, "s": self.text()}

    def _embed(self, seq, table, pos_table):
        T = seq.shape[1]
        # right-aligned positions: the most recent item always has index max_len - 1, so trimming
        # leading padding columns (see trim_left) leaves the computation unchanged
        pos = torch.arange(self.max_len - T, self.max_len, device=seq.device).unsqueeze(0)
        x = table[seq] * math.sqrt(self.d) + pos_table(pos)
        return self.emb_drop(x) * seq.ne(0).unsqueeze(-1)

    def encode(self, seq: torch.Tensor, tables: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """seq (B, T) left-padded -> hidden states (B, T, d) per stream."""
        mask = attention_mask(seq, self.heads)
        keep = seq.ne(0).unsqueeze(-1)
        if self.kind == "dual":
            c = self._embed(seq, tables["c"], self.pos[0])
            s = self._embed(seq, tables["s"], self.pos[1])
            for b in self.blocks:
                c, s = b(c, s, mask)
                c, s = c * keep, s * keep
            return {"c": self.ln_c(c), "s": self.ln_s(s)}
        x = self._embed(seq, tables["x"], self.pos[0])
        for b in self.blocks:
            x = b(x, mask) * keep
        return {"x": self.ln(x)}

    def fusion_params(self):
        w = torch.sigmoid(self.mix)
        lam = self.lambda_max * torch.sigmoid(self.pen) if self.penalty else torch.zeros_like(w)
        return w, lam

    def logits(self, h: Dict[str, torch.Tensor], tables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """h: hidden rows (M, d) per stream -> scores (M, N+1), column 0 = -inf."""
        if self.kind == "dual":
            sc = h["c"] @ tables["c"].T
            ss = h["s"] @ tables["s"].T
            w, lam = self.fusion_params()
            out = w * sc + (1 - w) * ss - lam * (sc - ss).abs()
        else:
            out = h["x"] @ tables["x"].T
        out[:, 0] = float("-inf")
        return out

    def stream_logits(self, h, tables):
        """Per-stream scores of a dual model (for analysis)."""
        sc = h["c"] @ tables["c"].T
        ss = h["s"] @ tables["s"].T
        sc[:, 0] = float("-inf")
        ss[:, 0] = float("-inf")
        return sc, ss

    def pair_scores(self, h, tables, items: torch.Tensor) -> torch.Tensor:
        """Scores of specific items (M,) for single-stream models (BCE ablation)."""
        return (h["x"] * tables["x"][items]).sum(-1)


def trim_left(*xs: torch.Tensor):
    """Drop leading columns that are padding (0) in every row of the first tensor."""
    nz = xs[0].ne(0).any(0)
    if not nz.any():
        return xs
    first = int(nz.nonzero()[0])
    return tuple(x[:, first:] for x in xs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
