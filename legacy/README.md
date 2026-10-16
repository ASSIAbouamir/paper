# Legacy (v1) code — not used for the v2 paper

This folder keeps the exploratory v1 pipeline (routers, CC-PoE, diagnostics, early BiX-GDR) for transparency.
**No number in the v2 paper comes from this folder.** All v2 results are produced by `bixrec/` (see the top-level README).

Known problems of the v1 pipeline, fixed in `bixrec/`:

1. `data_scripts/preprocess.py`, `amazon_preprocess.py`, `preprocess_kuairec.py` silently truncated a random 20 % of users
   to their first 5 interactions (val/test targets = 4th/5th item ever). This was not disclosed in the v1 manuscript.
2. `train_bix_gdr.py` trained on the last position of each training sequence only (one target per user per epoch).
3. The frozen experts were trained with BCE and one sampled negative, while BiX-GDR used full-softmax cross-entropy,
   so v1 comparisons mixed a loss change with an architecture change.
4. Attention had no key-padding mask.
5. Scripts producing several v1 results (sealed-run backbones, Beauty masked audit, Bi-MeF) are missing.

Files removed in v2 because they produced hard-coded or randomly simulated numbers rather than measured results:
`multi_seed_eval.py`, `ablation.py`, `evaluate.py`, and their outputs
(`multi_seed_results.json`, `statistical_evaluation.json`, `benchmark_results.*`, `ablation_results.*`),
together with the manuscript sections that reported them (LLM-baseline table, "BiX-GDR++", cold-start claims).
