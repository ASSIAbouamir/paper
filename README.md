# Beyond the Simplex Bottleneck: Gated Bilateral Cross-Attention and Geometric Discrepancy Regularization for Dual-Expert Sequential Recommendation

This repository provides the official PyTorch implementation and reproducibility suite for the paper:
**"Beyond the Simplex Bottleneck: Gated Bilateral Cross-Attention and Geometric Discrepancy Regularization for Dual-Expert Sequential Recommendation"** (Target: ACM TOIS / IEEE TKDE).

---

## 📌 Overview

Sequential recommender systems predominantly rely on behavioral sequential transition dynamics (e.g., SASRec) or semantic textual representations (e.g., MiniLM encoders). While existing approaches attempt to fuse these streams via late score gating or static blends, this work reveals:
1. **The Arithmetic Simplex Bottleneck:** Late sequence-level routing collapses to stationary 50/50 fusion due to an absence of item-aligned directional discriminability, creating statistical selection bias (Clark's bivariate extreme formula) and an information-theoretic classification barrier (Fano's inequality, $P_e \ge 53\%$).
2. **BiX-GDR Architecture:** Rather than isolating representations until the final scoring layer, BiX-GDR introduces:
   - **Bilateral Cross-Attention:** Deep cross-modal query contextualization between collaborative behavioral sequences and semantic text manifolds.
   - **Adaptive Information Injection Gating:** Dynamic sigmoid gates $\mathbf{g}$ to prevent semantic noise injection on sparse trajectories.
   - **Geometric Discrepancy Regularization (GDR):** Candidate-level divergence penalization ($-\lambda |S_c - S_s|$) to eliminate false-positive tail intrusions.

### Key Results
- **MovieLens-1M (Full-Catalog $M=3{,}628$, Unsampled):** Reaches $\text{HR@10} = \mathbf{0.1403}$ ($847.3$ hits), achieving a **+172.4% relative gain over SASRec** ($0.0515$) and **+9.7% over calibrated Static PoE** ($p < 10^{-6}$), surpassing the post-hoc action Oracle ($825.8$ hits).
- **Amazon Beauty (Full-Catalog $M=11{,}707$, Unsampled):** Reaches $\text{HR@10} = \mathbf{0.0382}$ ($377.0$ hits), securing **+131.7% over SASRec** and **+25.5% over Static PoE** ($p < 10^{-6}$), decisively reversing the failure of late routers.
- **SOTA 100-Negative Benchmark:** Achieves $\text{HR@10} = \mathbf{0.3890}$ (BiX-GDR++), outperforming recent LLM-based recommenders like **LlamaRec ($0.2980$, +30.5%)** and **TALLRec ($0.2850$, +36.5%)**.
- **Cold-Start Robustness ($L \le 5$):** Delivers $\text{HR@10} = \mathbf{0.3420}$ (**+175.8% gain over SASRec** at $0.1240$).

---

## 📁 Repository Structure

```text
├── main.tex                    # Master LaTeX document for submission
├── references.bib              # Complete bibliography
├── sections/                   # Modular LaTeX paper sections
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_system_architecture.tex
│   ├── 04_empirical_evaluation.tex
│   ├── 05_discussion_and_implications.tex
│   ├── 06_threats_and_limitations.tex
│   └── 07_conclusion.tex
├── figures/                    # Vector and high-res publication figures
├── models/                     # PyTorch model definitions
│   ├── bix_gdr.py              # BiX-GDR & Gated Bilateral Cross-Attention
│   ├── dual_expert_recsys.py   # Dual-stream baseline models
│   ├── sasrec.py               # Collaborative self-attentive backbone
│   ├── bert4rec.py             # BERT4Rec sequential baseline
│   └── gru4rec.py              # GRU4Rec recurrent baseline
├── data/                       # Dataset loaders and preprocessors
│   ├── dataset_registry.py     # Unified registry across benchmarks
│   ├── ml-1m/                  # MovieLens-1M dataset and embeddings
│   ├── amazon_beauty/          # Amazon Beauty 5-core dataset
│   └── yelp/                   # Yelp dataset
├── results/                    # Sealed benchmark result logs (.json)
├── config.py                   # Global hyperparameters and experiment configs
├── train.py                    # Dual-expert backbone training
├── train_bix_gdr.py            # End-to-end BiX-GDR training & evaluation
├── evaluate.py                 # Multi-metric evaluation suite
└── requirements.txt            # Environment dependencies
```

---

## 🛠️ Environment Setup

```bash
# Clone the repository
git clone https://github.com/anonymous-recsys/bix-gdr.git
cd bix-gdr

# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Reproduction & Training

### 1. Train and Evaluate BiX-GDR
```bash
# Train BiX-GDR across 3 seeds on MovieLens-1M
python train_bix_gdr.py --dataset ml1m --seeds 42 123 2024 --epochs 8

# Train BiX-GDR on Amazon Beauty
python train_bix_gdr.py --dataset amazon_beauty --seeds 42 123 2024 --epochs 8
```

### 2. Run Comprehensive Full-Catalog Evaluation
```bash
# Evaluates all models under unsampled full-catalog protocol
python evaluate.py --dataset ml1m --eval_mode full_catalog
```

---

## 📊 Citation

```bibtex
@article{anonymous2026bixgdr,
  title     = {Beyond the Simplex Bottleneck: Gated Bilateral Cross-Attention and Geometric Discrepancy Regularization for Dual-Expert Sequential Recommendation},
  author    = {Anonymous Author(s)},
  journal   = {ACM Transactions on Information Systems (TOIS)},
  year      = {2026}
}
```
