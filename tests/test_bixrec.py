"""Sanity tests: python -m pytest tests -q"""
import numpy as np
import torch

from bixrec.models import SeqRec, attention_mask
from bixrec.train import left_pad, make_splits, ranks_of
from bixrec.stats import holm, mcnemar_exact


def _model(kind, **kw):
    torch.manual_seed(0)
    emb = torch.nn.functional.normalize(torch.randn(51, 16), dim=1)
    return SeqRec(50, kind, emb, d=16, layers=2, heads=2, max_len=10, dropout=0.0, **kw)


def test_no_nan_with_left_padding():
    seq = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    for kind, kw in [("id", {}), ("text", {}), ("hybrid", {}), ("dual", dict(cross=True, gate=True, penalty=True))]:
        m = _model(kind, **kw).eval()
        t = m.item_tables()
        h = m.encode(seq, t)
        for v in h.values():
            assert torch.isfinite(v).all()


def test_causality():
    """Changing a future item must not change the state of an earlier position."""
    m = _model("dual", cross=True, gate=True, penalty=True).eval()
    t = m.item_tables()
    a = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 5, 6, 7]])
    with torch.no_grad():
        ha = m.encode(a, t)
    b = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 5, 6, 9]])
    with torch.no_grad():
        hb = m.encode(b, t)
    for k in ha:
        assert torch.allclose(ha[k][0, 8], hb[k][0, 8], atol=1e-5)


def test_mask_blocks_future_and_padding():
    seq = torch.tensor([[0, 1, 2]])
    m = attention_mask(seq, 1)[0]
    assert m[1, 0] and m[1, 2] and not m[1, 1] and not m[2, 1]


def test_splits_no_leak():
    seqs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    (tr_in, tr_tg), (v_in, v_t), (te_in, te_t) = make_splits(seqs, 4)
    assert v_t.tolist() == [4, 10] and te_t.tolist() == [5, 11]
    for row_in, row_tg in zip(tr_in.tolist(), tr_tg.tolist()):
        assert 4 not in row_tg and 5 not in row_tg and 10 not in row_tg and 11 not in row_tg
    assert 5 not in te_in[0].tolist() and 4 in te_in[0].tolist()


def test_masking_never_hurts():
    torch.manual_seed(1)
    sc = torch.randn(8, 30)
    sc[:, 0] = float("-inf")
    seq = torch.randint(1, 30, (8, 5))
    tgt = torch.randint(1, 30, (8,))
    seq[seq == tgt[:, None]] = 0  # target never in history (dedup guarantee)
    r = ranks_of(sc, tgt)
    rm = ranks_of(sc.scatter(1, seq, float("-inf")), tgt)
    assert (rm <= r).all()


def test_stats():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 0, 1, 1], bool)
    assert mcnemar_exact(a, b)["n10"] == 1
    assert left_pad([1, 2], 4) == [0, 0, 1, 2]


def test_aux_stream_logits():
    """streams=True returns the unchanged fused score plus the per-stream scores."""
    m = _model("dual", cross=True, gate=True, penalty=True).eval()
    t = m.item_tables()
    h = {k: v[:, -1] for k, v in m.encode(torch.tensor([[0, 0, 0, 0, 0, 0, 0, 5, 6, 7]]), t).items()}
    out, sc, ss = m.logits(h, t, streams=True)
    assert torch.equal(out, m.logits(h, t)) and sc.shape == ss.shape == out.shape


def test_aux_loss_changes_training():
    """The aux branch of train_model is live: same seed and data, aux off vs on -> different weights."""
    from bixrec.train import VARIANTS, train_model
    seqs = [[(u + i) % 50 + 1 for i in range(8)] for u in range(20)]
    train, val, _ = make_splits(seqs, 10)
    cfg = dict(lr=1e-2, wd=0.0, seed=0, epochs=1, batch=8, patience=5, min_epochs=0, verbose=False)
    weights = []
    for aux in (0.0, VARIANTS["bix_aux"]["aux"]):
        m = _model("dual", cross=True, gate=True, penalty=True)
        train_model(m, train, val, torch.device("cpu"), dict(cfg, aux=aux))
        weights.append(m.text.proj[0].weight.detach().clone())
    assert not torch.equal(*weights)
