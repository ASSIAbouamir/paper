# BiXRec: score routing vs. bilateral cross-attention for ID + text sequential recommendation

Code for the paper *"Where Should Collaborative and Semantic Experts Meet? Score Routing versus Bilateral
Cross-Attention in Sequential Recommendation"* (`paper/main.tex`).

**Status: the v2 experiments have not been run yet.** Every number in the paper is generated from `runs/` by
`bixrec/make_tables.py`; until the runs exist, tables show `--`. Red `[TODO]` notes in the paper mark sentences to
write once results are in.

## What is compared

All models share one transformer block, one loss (full-softmax cross-entropy at every position), one training
budget and one early-stopping rule; only the fusion design changes.

| variant | description |
|---|---|
| `sasrec` | ID stream only |
| `text` | frozen MiniLM text embeddings + MLP projection, no IDs |
| `poe`, `poe50` | static product of the two separately trained experts (α tuned on validation / α = 0.5) |
| `hybrid` | early fusion: item embedding = ID + text |
| `dual_late` | two streams, no interaction, trained jointly through the fused score (control) |
| `dual_late_pen` | + disagreement penalty |
| `bix_nogate`, `bix_gate` | + bilateral cross-attention (without / with gate) |
| `bix` | full model: cross-attention + gate + penalty |
| `sasrec_lastpos`, `sasrec_bce`, `bix_lastpos` | protocol audit: loss on last position only / BCE with one negative |

Protocol: 5-core, chronological leave-one-out, repeated user–item pairs removed, full-catalog ranking without
negative sampling, ties counted against the target, 5 seeds, Holm-corrected paired tests.

## Run

Easiest: open `run_experiments_colab.ipynb` in Google Colab with a GPU and run all cells (results are saved to
Google Drive and the grid resumes where it stopped).

Manually:

```bash
pip install -r requirements.txt
python -m bixrec.download --root data/raw --datasets ml1m beauty sports
python -m bixrec.data ml1m   --raw data/raw/ml-1m --out data/ml1m
python -m bixrec.data amazon --raw data/raw/reviews_Beauty_5.json.gz --meta data/raw/meta_Beauty.json.gz --out data/beauty
python -m bixrec.data amazon --raw data/raw/reviews_Sports_and_Outdoors_5.json.gz \
                             --meta data/raw/meta_Sports_and_Outdoors.json.gz --out data/sports
for d in ml1m beauty sports; do python -m bixrec.embed --data data/$d; done

python -m bixrec.run_all --data_root data --runs runs          # core -> ablation -> audit, resumable
python -m bixrec.make_tables --runs runs --data_root data --out paper/generated
cd paper && latexmk -pdf main.tex
```

One run: `python -m bixrec.train --data data/ml1m --variant bix --seed 42 --out runs`.
Each run saves `metrics.json`, per-user ranks (`ranks_*.npy`) and the checkpoint.

Tests: `python -m pytest tests -q` (causality, padding, no split leakage, masking can only help, statistics).

## Layout

```
bixrec/      data prep, text embedding, models, training/eval, statistics, table generation
paper/       main.tex, references.bib, generated/tables.tex (auto-generated)
tests/       sanity tests
legacy/      v1 exploratory code, kept for transparency; not used for any v2 number (see legacy/README.md)
data/        small raw samples from v1 (Amazon/Yelp/KuaiRec); ML-1M is downloaded by bixrec.download
```
