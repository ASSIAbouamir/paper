"""Sanity tests: python -m pytest tests -q"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from bixrec.models import SeqRec, attention_mask
from bixrec.train import left_pad, make_splits, ranks_of
from bixrec.stats import holm, mcnemar_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = [("id", {}), ("text", {}), ("hybrid", {}), ("dual", dict(cross=True, gate=True, penalty=True)),
         ("dual", dict(ef=True)), ("dual", dict(cross=True, gate=True, penalty=True, ef=True))]


def _model(kind, **kw):
    torch.manual_seed(0)
    emb = torch.nn.functional.normalize(torch.randn(51, 16), dim=1)
    return SeqRec(50, kind, emb, d=16, layers=2, heads=2, max_len=10, dropout=0.0, **kw)


def test_no_nan_with_left_padding():
    seq = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    for kind, kw in KINDS:
        m = _model(kind, **kw).eval()
        t = m.item_tables()
        h = m.encode(seq, t)
        for v in h.values():
            assert torch.isfinite(v).all()


def test_causality():
    """Changing a future item must not change the state of an earlier position (every model kind)."""
    a = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 5, 6, 7]])
    b = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 5, 6, 9]])
    for kind, kw in KINDS:
        m = _model(kind, **kw).eval()
        t = m.item_tables()
        with torch.no_grad():
            ha, hb = m.encode(a, t), m.encode(b, t)
        for k in ha:
            assert torch.allclose(ha[k][0, 8], hb[k][0, 8], atol=1e-5), (kind, kw)


def test_padding_invariance():
    """Extra leading padding never changes the last state (right-aligned positions, masked padding)."""
    for kind, kw in KINDS:
        m = _model(kind, **kw).eval()
        t = m.item_tables()
        with torch.no_grad():
            long = m.encode(torch.tensor([[0, 0, 0, 0, 0, 0, 0, 5, 6, 7]]), t)
            short = m.encode(torch.tensor([[5, 6, 7]]), t)
        for k in long:
            assert torch.allclose(long[k][0, -1], short[k][0, -1], atol=1e-5), (kind, kw)


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


def test_masked_evaluation_never_worse():
    from bixrec.train import evaluate
    seqs = [list(dict.fromkeys((u * 7 + i * 3) % 50 + 1 for i in range(9))) for u in range(30)]
    _, val, test = make_splits(seqs, 10)
    for kind, kw in KINDS:
        m = _model(kind, **kw)
        for split in (val, test):
            r, rm = evaluate(m, split, torch.device("cpu")), evaluate(m, split, torch.device("cpu"), masked=True)
            assert (rm <= r).all(), (kind, kw)


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


def test_logits_without_penalty_is_the_plain_mix():
    """Skipping the zero penalty term is exact: out = w*s_c + (1-w)*s_s."""
    m = _model("dual", cross=True, gate=True).eval()
    t = m.item_tables()
    h = {k: v[:, -1] for k, v in m.encode(torch.tensor([[0, 0, 0, 0, 0, 0, 0, 5, 6, 7]]), t).items()}
    out, sc, ss = m.logits(h, t, streams=True)
    w, _ = m.fusion_params()
    ref = w * sc + (1 - w) * ss
    ref[:, 0] = float("-inf")
    assert torch.equal(out, ref)


def test_ef_content_stream_sees_ids():
    m = _model("dual", ef=True)
    t = m.item_tables()
    assert torch.equal(t["s"], m.item_emb.weight + m.text()) and torch.equal(t["c"], m.item_emb.weight)
    with pytest.raises(AssertionError):
        _model("hybrid", ef=True)


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


def test_hparams_precedence():
    from bixrec.train import resolve_hparams
    none = {"d": None, "dropout": None, "lr": None, "emb": None}
    assert resolve_hparams(none, {}, "bix", {"dropout": 0.5}) == \
        {"d": 64, "dropout": 0.5, "lr": 1e-3, "emb": "text_emb.pt"}
    hp = {"d": 128, "dropout": 0.3, "lr": {"bix": 1e-4}, "encoder": "lsa"}
    assert resolve_hparams(none, hp, "bix", {"dropout": 0.5}) == \
        {"d": 128, "dropout": 0.3, "lr": 1e-4, "emb": "text_emb_lsa.pt"}
    with pytest.raises(SystemExit):  # tuned dataset, untuned variant: refuse (equal tuning budget)
        resolve_hparams(none, hp, "sasrec", {"dropout": 0.5})
    assert resolve_hparams(none, hp, "poe", {"dropout": 0.5})["lr"] == 1e-3  # PoE trains nothing
    cli = {"d": 32, "dropout": 0.1, "lr": 5e-4, "emb": "x.pt"}
    assert resolve_hparams(cli, hp, "bix", {"dropout": 0.5}) == cli


def test_ties_broken_in_fixed_random_order():
    """All-equal scores: every target position gets a distinct rank (not the worst one), deterministically."""
    sc = torch.zeros(100, 101)
    sc[:, 0] = float("-inf")
    tgt = torch.arange(1, 101)
    r = ranks_of(sc, tgt)
    assert sorted(r.tolist()) == list(range(1, 101)) and torch.equal(r, ranks_of(sc, tgt))
    sc2 = torch.randn(4, 30)
    assert torch.equal(ranks_of(sc2, torch.tensor([1, 2, 3, 4])),
                       (sc2 > sc2.gather(1, torch.tensor([[1], [2], [3], [4]]))).sum(1) + 1)  # no ties: plain rank


def test_non_finite_target_gets_worst_rank():
    sc = torch.randn(2, 30)
    sc[0, 5] = float("nan")
    r = ranks_of(sc, torch.tensor([5, 6]))
    assert r[0] == 30 and r[1] < 30


def test_resume_refuses_changed_settings(tmp_path):
    from bixrec.train import train_model
    seqs = [[(u + i) % 50 + 1 for i in range(8)] for u in range(20)]
    train, val, _ = make_splits(seqs, 10)
    cfg = dict(lr=1e-2, wd=0.0, seed=0, epochs=2, patience=5, min_epochs=0, verbose=False, batch=8, d=16,
               ckpt=str(tmp_path / "ckpt.pt"), git_sha="abc")
    train_model(_model("id"), train, val, torch.device("cpu"), cfg)  # leaves its checkpoint behind
    assert os.path.exists(cfg["ckpt"])
    for changed in (dict(lr=1e-3), dict(epochs=1), dict(git_sha="def")):  # settings, stopping rule, code
        with pytest.raises(SystemExit):
            train_model(_model("id"), train, val, torch.device("cpu"), dict(cfg, **changed))
    train_model(_model("id"), train, val, torch.device("cpu"), cfg)  # same settings: resumes


def test_run_all_guards(tmp_path):
    from bixrec.run_all import check_fresh, fingerprint
    for bad in ("--lr 1e-4", "--d 128", "--emb x.pt", "--tag t", "--seed 1", "--no_test", "--dropout 0.1",
                "--max_len 5", "--no_te", "--allow_d", "--see 1"):  # incl. argparse prefix abbreviations
        with pytest.raises(SystemExit):
            fingerprint(bad)
    fp = fingerprint("--device cpu --epochs 100 --patience 10 --min_epochs 30")
    assert fingerprint("--device cpu --epochs 100 --patience 10 --min_epochs 30 --max_users 300") != fp
    hp = {"d": 64, "lr": {"bix": 1e-3}, "encoder": "lsa"}
    run = tmp_path / "metrics.json"
    run.write_text(json.dumps({"config": dict(fp, d=64, lr=1e-3, emb="text_emb_lsa.pt", device="auto"),
                               "env": {"device": "cpu"}}))
    check_fresh(str(run), fp, hp, "bix")  # current: fine
    check_fresh(str(run), fp, hp, "poe")  # variants without a tuned lr still get d / encoder checks
    for stale_fp, stale_hp in ((dict(fp, epochs=200), None), (fp, dict(hp, d=128)), (fp, dict(hp, encoder="sbert")),
                               (dict(fp, device="mps"), None)):
        with pytest.raises(SystemExit):
            check_fresh(str(run), stale_fp, stale_hp, "bix")


def _tiny_dataset(path):
    os.makedirs(path, exist_ok=True)
    seqs = [list(dict.fromkeys((u * 7 + i * 3) % 50 + 1 for i in range(9))) for u in range(30)]
    torch.save({"num_items": 50, "user_seqs": seqs, "item_text": [""] + [f"item {i}" for i in range(1, 51)],
                "stats": {}}, os.path.join(path, "dataset.pt"))
    torch.manual_seed(0)
    torch.save({"emb": torch.cat([torch.zeros(1, 8), torch.randn(50, 8)]), "model": "test"},
               os.path.join(path, "text_emb.pt"))


def test_no_test_flag_never_touches_test(tmp_path):
    data, out = str(tmp_path / "tiny"), str(tmp_path / "runs")
    _tiny_dataset(data)
    for v in ("sasrec", "bix_ef_aux"):
        subprocess.run([sys.executable, "-m", "bixrec.train", "--data", data, "--variant", v, "--epochs", "1",
                        "--min_epochs", "0", "--no_test", "--out", out, "--device", "cpu", "--quiet", "--d", "16",
                        "--tag", "t", "--allow_dirty"], cwd=ROOT, check=True, capture_output=True)
        run_dir = os.path.join(out, "tiny", f"{v}@t", "seed42")
        m = json.load(open(os.path.join(run_dir, "metrics.json")))
        assert not any(k.startswith("test") for k in m) and "val" in m and "val_masked" in m
        assert not any(f.startswith("ranks_test") for f in os.listdir(run_dir))
        assert m["config"]["d"] == 16 and m["env"]["device"] == "cpu" and "git_sha" in m["env"]
        if "streams" in m:
            assert all(set(s) == {"val"} for s in m["streams"].values())


def test_tuning_reads_validation_only(tmp_path):
    from bixrec.run_all import pick, val_score
    run = tmp_path / "sasrec@d64" / "seed42"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"val": {"NDCG@10": 0.1}}))
    assert val_score(str(tmp_path / "sasrec@d64")) == 0.1
    (run / "metrics.json").write_text(json.dumps({"val": {"NDCG@10": 0.1}, "test": {"NDCG@10": 0.9}}))
    with pytest.raises(SystemExit):
        val_score(str(tmp_path / "sasrec@d64"))
    assert pick({64: 0.1, 128: 0.1}, [64, 128]) == 64        # ties -> earlier (v3 default)
    assert pick({1e-3: 0.1, 1e-4: 0.2}, [1e-3, 1e-4]) == 1e-4


def test_every_model_gets_the_same_lr_trials():
    from bixrec.run_all import STAGES, stage_variants
    for ds in ("ml100k", "beauty", "beauty_text"):
        vs = stage_variants(ds)
        assert "poe" not in vs and "bix_ef_aux" in vs and "sasrec" in vs
        assert all(v in vs for v in STAGES["core"][0] if v != "poe")


def test_same_interactions():
    from bixrec.data import same_interactions
    a = {"user_seqs": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [1, 3, 5, 6, 2]]}
    perm = {i: 7 - i for i in range(1, 7)}  # relabel items
    b = {"user_seqs": [[perm[i] for i in s] for s in a["user_seqs"]]}
    assert same_interactions(a, b) == 1.0
    c = {"user_seqs": [s[:] for s in b["user_seqs"]]}
    c["user_seqs"][0][-2:] = c["user_seqs"][0][-1:-3:-1]  # swap the val / test targets of one user
    assert same_interactions(a, c) < 1.0


def test_whiten():
    from bixrec.embed import whiten
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 8)) @ rng.normal(size=(8, 8)) + 3.0
    w = whiten(z)
    assert np.allclose(w.mean(0), 0, atol=1e-8) and np.allclose(w.T @ w / len(w), np.eye(w.shape[1]), atol=1e-6)


def test_table_rows_and_captions():
    from bixrec import make_tables as mt
    from bixrec.train import VARIANTS
    for v in mt.MAIN_ROWS + mt.ABL_ROWS + mt.AUDIT_ROWS:
        assert v in mt.NAMES
    for v, cfg in VARIANTS.items():  # a different objective is always flagged, and never sits under "same loss"
        if cfg.get("aux"):
            assert mt.DAG in mt.NAMES[v] and v not in mt.ABL_ROWS and v not in mt.AUDIT_ROWS


def test_selected_model_uses_validation_only(monkeypatch):
    from bixrec import make_tables as mt
    monkeypatch.setattr(mt, "DATASETS", [("d1", "D1"), ("d2", "D2")])
    monkeypatch.setattr(mt, "MAIN_ROWS", ["a", "b"])
    run = lambda val, test: {42: {"val": {"NDCG@10": val}, "test": {"NDCG@10": test}}}
    runs = {"d1": {"a": run(0.30, 0.10), "b": run(0.20, 0.90)}, "d2": {"a": run(0.05, 0.10), "b": run(0.04, 0.90)}}
    assert mt.choose_proposed(runs)[0] == "a"  # best on validation, although b is far better on test
    del runs["d2"]["b"]  # incomplete grid: no silent choice among the finished models
    assert mt.choose_proposed(runs)[0] is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")
def test_mps_matches_cpu():
    seq = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    for kind, kw in KINDS:
        m = _model(kind, **kw).eval()
        with torch.no_grad():
            hc = m.encode(seq, m.item_tables())
            mm = m.to("mps")
            hm = mm.encode(seq.to("mps"), mm.item_tables())
        for k in hc:
            assert torch.allclose(hc[k], hm[k].cpu(), atol=1e-4), (kind, kw)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")
def test_mps_training_step_matches_cpu():
    """One training epoch on MPS (with cache freeing) gives the same weights as on CPU, within float tolerance."""
    from bixrec.train import train_model
    seqs = [[(u + i) % 50 + 1 for i in range(8)] for u in range(60)]
    train, val, _ = make_splits(seqs, 10)
    cfg = dict(lr=1e-2, wd=0.0, seed=0, epochs=1, batch=8, patience=5, min_epochs=0, verbose=False)
    w = {}
    for dev in ("cpu", "mps"):
        m = _model("dual", cross=True, gate=True, penalty=True, ef=True).to(dev)
        train_model(m, train, val, torch.device(dev), dict(cfg, aux=0.5))
        w[dev] = m.item_emb.weight.detach().cpu()
    assert torch.allclose(w["cpu"], w["mps"], atol=1e-4)


@pytest.mark.skipif(not os.environ.get("V3_RUNS"), reason="set V3_RUNS=<v3 runs dir> and V3_DATA=<v3 data dir>")
def test_v3_replay():
    """v3 checkpoints reproduce their saved validation ranks with v4 code (numerical equivalence)."""
    from bixrec.train import VARIANTS, build_model, evaluate, load_data
    ds_name = os.path.basename(os.path.normpath(os.environ["V3_DATA"]))
    ds, emb = load_data(os.environ["V3_DATA"])
    for v in ("sasrec", "hybrid", "bix"):
        run = os.path.join(os.environ["V3_RUNS"], ds_name, v, "seed42")
        cfg = json.load(open(os.path.join(run, "metrics.json")))["config"]
        m = build_model(ds, emb, VARIANTS[v], cfg)
        m.load_state_dict(torch.load(os.path.join(run, "model.pt")))
        _, val, _ = make_splits(ds["user_seqs"], cfg["max_len"])
        same = (evaluate(m, val, torch.device("cpu")) == np.load(os.path.join(run, "ranks_val.npy"))).mean()
        assert same >= 0.999, (v, same)
