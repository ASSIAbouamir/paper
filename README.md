# BiXRec: score routing vs. bilateral cross-attention for ID + text sequential recommendation

Code for the paper *"Where Should Collaborative and Semantic Experts Meet? Score Routing versus Bilateral
Cross-Attention in Sequential Recommendation"* (`paper/main.tex`).

**Status:** v2 results (ML-100K, Beauty, Sports; CPU) are in `paper/`. v4 (below) is implemented and tested;
its runs have not started. Every number in the paper is generated from `runs/` by `bixrec/make_tables.py`.

## What is compared

All models share one transformer block, one loss (full-softmax cross-entropy at every position), one training
budget and one early-stopping rule; only the fusion design changes (one exception, the `*_aux` objective study below).

| variant | description |
|---|---|
| `sasrec` | ID stream only |
| `text` | frozen content embeddings (encoder chosen on validation, see v4) + MLP projection, no IDs |
| `poe`, `poe50` | static product of the two separately trained experts (α tuned on validation / α = 0.5) |
| `hybrid` | early fusion: item embedding = ID + text |
| `dual_late` | two streams, no interaction, trained jointly through the fused score (control) |
| `dual_late_pen` | + disagreement penalty |
| `bix_nogate`, `bix_gate` | + bilateral cross-attention (without / with gate) |
| `bix` | full model: cross-attention + gate + penalty |
| `sasrec_lastpos`, `sasrec_bce`, `bix_lastpos` | protocol audit: loss on last position only / BCE with one negative |
| `dual_late_aux`, `bix_aux` | objective study: `dual_late` / `bix` + a per-stream auxiliary loss (see below) |
| `dual_late_ef_aux`, `bix_ef_aux` | as `*_aux`, but the content stream uses early-fused item vectors (ID + text) |

**Exception to "one loss":** the `*_aux` variants train with `CE(fused) + 0.5 · (CE(ID stream) + CE(content
stream))`, so each stream must also predict the next item on its own. They change the objective, not only the
fusion design: they are marked † in every table, and each BiXRec `*_aux` model has a joint-late-fusion control
trained with the same loss. In v4 they are part of the `core` stage and of the main table.

## v4: protocol for the strongest honest results

v4 adds `dual_late_ef_aux` / `bix_ef_aux` (the content stream scores items with **ID + content** vectors, so it
can separate items whose content is identical; 81% of Sports items share their exact attribute set with another
item), real-text Amazon datasets, a held-out dataset (Toys), validation-only tuning and Apple-GPU (MPS) support.
Rules, fixed before any v4 run:

1. **Validation only.** Early stopping, the PoE α, the content encoder, the backbone size, the learning rate and
   the selected model are chosen on validation NDCG@10. Tuning runs use `--no_test`, so the test split is never
   evaluated during tuning; `run_all` refuses a tuning run that has test metrics, refuses to train without
   `hparams.json`, and reports runs made with other settings as stale instead of reusing them.
2. **Equal tuning budget.** Per dataset, the backbone size d ∈ {64, 128, 256} is chosen with SASRec and shared by
   all models; every trained model then gets the same two learning-rate trials {1e-3, 1e-4}; `bixrec.train`
   refuses a variant that is missing from the tuned table. Model-specific settings (aux weight 0.5, learnable
   fusion weight) are fixed, never tuned. Ties go to the earlier option in each list. (PoE's α is a fusion weight
   picked on validation from 21 values; this favours PoE's validation score, not its test score.)
3. **Fixed seeds, everything reported.** The seed set ({42, 123, 2024, 7, 99} or its first three) is fixed before
   the first v4 run and stated in the v4 pull request; all runs of all variants appear in the tables (cells with
   fewer seeds are marked); `make_tables` selects a model only once every main-table model exists on every dataset,
   from validation scores only, and records any `--exclude` in `summary.json`.
4. **No leakage.** Content encoders are fitted on item text/attributes only; Amazon metadata fields that encode
   other users' behaviour (`related`, also-bought/viewed, sales rank) are never read; tests check causality,
   padding invariance and split leakage for every model kind.
5. **Comparable metrics.** The primary metric keeps the user's history items in the full-catalogue ranking. This
   is the protocol of the published side-information results shown for reference on the same Beauty/Sports/Toys
   data (DIF-SR, DIFF): RecBole's full-sort evaluation passes no history for sequential models, and DIFF states
   that it does not penalise repeated items. Removing the history items in the input window (BSARec/DLFSRec-style)
   is reported separately and never compared with those published rows. Exact score ties are broken in a fixed
   random item order (RecBole's `topk` breaks them arbitrarily); counting ties against the target would penalise
   only models whose item vectors coincide, i.e. the content expert.
6. **Disclosure of what was seen before v4.** Test results of v2/v3 were seen for every core and ablation variant
   on ML-100K, Beauty and Sports (3 seeds, both with and without history masking), for `bix_aux`/`dual_late_aux`
   on Sports (seeds 42 and 123), and, in a Colab run with real product text, for ML-1M and Beauty. The `*_aux`
   loss, the `ef` stream and the v4 protocol were designed after that. `beauty_text` / `sports_text` have the same
   users, sequences and test targets as Beauty/Sports (only the item content differs), so they are **not** unseen
   data. **Toys (S3-Rec release) is the only untouched dataset**: it is a held-out confirmation and should not be
   inspected before the protocol and model list are frozen. Its validation split is used only to tune its own
   hyperparameters; it is excluded from selecting the model (`make_tables.HELDOUT`). Every run records its device, git commit (runs refuse
   uncommitted code) and memory; `make_tables` warns when a dataset mixes devices or commits.

```bash
R=data_v4/raw
python -m bixrec.download --root $R --datasets s3rec_beauty s3rec_sports s3rec_toys ml100k beauty sports
python -m bixrec.data s3rec   --raw $R/Beauty.txt --meta $R/Beauty_item2attributes.json --out data_v4/beauty
python -m bixrec.data s3rec   --raw $R/Sports_and_Outdoors.txt --meta $R/Sports_and_Outdoors_item2attributes.json --out data_v4/sports
python -m bixrec.data s3rec   --raw $R/Toys_and_Games.txt --meta $R/Toys_and_Games_item2attributes.json --out data_v4/toys
python -m bixrec.data recbole --raw $R/ml-100k.inter --meta $R/ml-100k.item --out data_v4/ml100k
python -m bixrec.data amazon  --raw $R/reviews_Beauty_5.json.gz --meta $R/meta_Beauty.json.gz --out data_v4/beauty_text
python -m bixrec.data amazon  --raw $R/reviews_Sports_and_Outdoors_5.json.gz --meta $R/meta_Sports_and_Outdoors.json.gz --out data_v4/sports_text
python -m bixrec.data compare --raw data_v4/beauty --meta data_v4/beauty_text    # same interactions? (1.0)

X="--device mps --epochs 100 --patience 10 --min_epochs 30"
python -m bixrec.run_all --data_root data_v4 --runs runs_v4 --stage tune --extra "$X"     # writes hparams.json
python -m bixrec.run_all --data_root data_v4 --runs runs_v4 --stage core ablation audit --extra "$X"
python -m bixrec.make_tables --runs runs_v4 --data_root data_v4 --out paper/generated
```

## Run (v2/v3 layout: `data/`, `runs/`; for v4 use the commands above)

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
