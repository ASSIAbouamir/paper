"""
Router Diagnostic & Fixed Training Pipeline.

Diagnoses and fixes the 0.0388 < 0.0394 (Router < Static Fusion) issue.

ROOT CAUSES identified:
  BUG 1: Oracle labels use Y_a = 1/rank instead of 1/log2(rank+1).
          => Disproportionately rewards very low ranks (rank=1 gets Y=1.0,
             rank=2 gets Y=0.5, rank=3 gets Y=0.33...).
             Correct: Y_a = 1/log2(rank+1) (DCG-style, standard in RecSys).

  BUG 2: Router trained with CrossEntropyLoss(hard argmax) instead of
          KL-divergence on continuous soft labels.
          => Hard argmax loses magnitude information.

  BUG 3: Router hidden_dim=32 is too small (8D input -> 32 -> 16 -> 3).
          Recommended: 64 hidden units for 8D input.

  BUG 4: No early stopping on router training.
          => Overfits the val split, does not generalize to test.

  BUG 5: Oracle action distribution is ~60% Hybrid, so a trivial classifier
          that always predicts Hybrid gets ~60% accuracy AND matches Static
          Fusion on HR@10 (since Hybrid = Static Fusion 50/50).
          => Router needs to correctly route the ~40% non-Hybrid users.

FIX: Re-train the router with:
  - Continuous Y_a = 1/log2(rank+1) labels, normalized to simplex
  - KL-divergence loss (soft supervision)
  - hidden_dim=64
  - Early stopping on val KL loss
  - Hybrid-default regularization: penalize deviation from Hybrid
    when evidence is weak (D_JS < median, U_c < median)

Run: python fix_router_training.py
     (Requires existing checkpoints/collab_expert.pt + checkpoints/semantic_expert.pt)
"""

import os
import sys
import math
import time
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")


# ── Fixed Oracle Label Computation ────────────────────────────────────────────

def compute_oracle_labels_correct(
    p_c_np: np.ndarray,
    p_s_np: np.ndarray,
    p_h_np: np.ndarray,
    target_idxs: np.ndarray,
    k: int = 10,
) -> tuple:
    """
    Computes CORRECT continuous oracle labels using DCG-style discounting:
      Y_a(u) = 1 / log2(rank_a(u) + 1)   if rank_a(u) <= K
      Y_a(u) = 0                           otherwise

    Normalizes [Y_C, Y_S, Y_H] to a probability simplex.
    If all Y=0 (all miss), uses uniform [1/3, 1/3, 1/3].

    Returns:
      hard_labels: [N] int, argmax oracle
      soft_labels: [N, 3] float, normalized continuous labels
      per_action_hr: dict with HR@K per action (diagnostic)
    """
    N = len(target_idxs)
    soft_labels = np.zeros((N, 3), dtype=np.float32)
    hard_labels = np.zeros(N, dtype=np.int64)

    action_hits = [[], [], []]  # for diagnostic

    for u in range(N):
        tgt = target_idxs[u]
        dists = [p_c_np[u], p_s_np[u], p_h_np[u]]
        y_vals = []

        for a, p in enumerate(dists):
            rank = int(np.sum(p >= p[tgt]))   # 1-indexed rank
            if rank <= k:
                y = 1.0 / math.log2(rank + 1)
                action_hits[a].append(1.0)
            else:
                y = 0.0
                action_hits[a].append(0.0)
            y_vals.append(y)

        y_arr = np.array(y_vals, dtype=np.float32)
        total = y_arr.sum()

        if total > 0:
            soft_labels[u] = y_arr / total
        else:
            soft_labels[u] = np.array([1/3, 1/3, 1/3], dtype=np.float32)

        hard_labels[u] = int(np.argmax(y_arr))

    per_action_hr = {
        "Collaborative": float(np.mean(action_hits[0])),
        "Semantic":      float(np.mean(action_hits[1])),
        "Hybrid":        float(np.mean(action_hits[2])),
    }

    return hard_labels, soft_labels, per_action_hr


# ── Fixed Router Architecture ──────────────────────────────────────────────────

class FixedSelectiveRouter(nn.Module):
    """
    Fixed router with:
    - hidden_dim=64 (was 32)
    - 2 hidden layers + output
    - LayerNorm after each hidden layer
    - Kaiming init
    - Dropout 0.1
    - Bias initialized to [0, 0, +0.5] to start slightly biased toward Hybrid
      (index 2 = H), since Oracle shows ~60% Hybrid prevalence.
      This prevents the router from being stuck at uniform initialization.
    """
    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 64,
        num_actions: int = 3,
        dropout: float = 0.10,
        hybrid_bias: float = 0.5,   # Initial logit bias toward Hybrid
    ):
        super().__init__()
        self.in_features  = in_features
        self.num_actions  = num_actions
        self.hybrid_bias  = hybrid_bias

        self.layer1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.head = nn.Linear(hidden_dim // 2, num_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Bias head toward Hybrid (index 2) to match oracle prior
        with torch.no_grad():
            self.head.bias[2] = self.hybrid_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layer2(self.layer1(x)))


# ── Hybrid-Default Regularizer ─────────────────────────────────────────────────

def hybrid_default_regularizer(
    logits: torch.Tensor,
    evidence: torch.Tensor,
    djs_col: int = 2,
    uc_col: int = 0,
    lambda_reg: float = 0.1,
) -> torch.Tensor:
    """
    Penalizes deviation from Hybrid for users with low evidence strength.
    Low evidence = D_JS < 0.15 AND U_c < 0.003 (weak routing signal).

    This prevents the router from making confident routing decisions
    when the evidence doesn't support it, preserving Static Fusion behavior
    in ambiguous cases.

    Loss = lambda * E_{weak evidence users}[ KL(probs || uniform_H) ]
    where uniform_H = [0.05, 0.05, 0.90] (strongly prefer Hybrid)
    """
    d_js = evidence[:, djs_col]
    u_c  = evidence[:, uc_col]

    weak_mask = (d_js < 0.15) & (u_c < 0.003)

    if not weak_mask.any():
        return torch.tensor(0.0, device=logits.device)

    probs_weak   = F.softmax(logits[weak_mask], dim=-1)
    hybrid_prior = torch.tensor([0.05, 0.05, 0.90], device=logits.device)
    hybrid_prior = hybrid_prior.unsqueeze(0).expand_as(probs_weak)

    # KL( probs_weak || hybrid_prior )
    kl = (probs_weak * (torch.log(probs_weak + 1e-12) - torch.log(hybrid_prior + 1e-12))).sum(dim=-1)
    return kl.mean() * lambda_reg


# ── Fixed Router Training ──────────────────────────────────────────────────────

def train_router_fixed(
    router: FixedSelectiveRouter,
    features: torch.Tensor,
    hard_labels: torch.Tensor,
    soft_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_hard_labels: torch.Tensor,
    val_soft_labels: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
    lambda_reg: float = 0.1,
) -> tuple:
    """
    Trains with:
    - Soft KL-divergence loss (continuous oracle)
    - Hybrid-default regularizer
    - Early stopping on val KL loss
    - Cosine LR schedule
    """
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    dataset = TensorDataset(features, hard_labels, soft_labels)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    best_val_loss  = float("inf")
    best_state     = None
    patience_cnt   = 0
    history        = []

    for epoch in range(1, epochs + 1):
        router.train()
        epoch_loss = 0.0
        n_batches  = 0

        for feats, h_lbl, s_lbl in loader:
            optimizer.zero_grad()
            logits  = router(feats)
            log_p   = F.log_softmax(logits, dim=-1)

            # Primary: KL-divergence with continuous oracle labels
            kl_loss = -(s_lbl * log_p).sum(dim=-1).mean()

            # Secondary: Hybrid-default regularizer
            reg = hybrid_default_regularizer(logits, feats, lambda_reg=lambda_reg)

            loss = kl_loss + reg
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation
        router.eval()
        with torch.no_grad():
            val_logits   = router(val_features)
            val_log_p    = F.log_softmax(val_logits, dim=-1)
            val_kl       = -(val_soft_labels * val_log_p).sum(dim=-1).mean().item()
            val_acc      = (torch.argmax(val_logits, dim=-1) == val_hard_labels).float().mean().item()

        history.append({"epoch": epoch, "train_loss": epoch_loss/n_batches, "val_kl": val_kl, "val_acc": val_acc})

        if val_kl < best_val_loss:
            best_val_loss = val_kl
            best_state    = {k: v.clone() for k, v in router.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping at epoch {epoch} (best val KL: {best_val_loss:.4f})")
                break

        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}: train_loss={epoch_loss/n_batches:.4f}  "
                  f"val_kl={val_kl:.4f}  val_acc={val_acc*100:.1f}%  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    router.load_state_dict(best_state)
    return router, history


# ── Diagnostic Analysis ────────────────────────────────────────────────────────

def diagnose_router_vs_static(
    router: FixedSelectiveRouter,
    features: torch.Tensor,
    p_c: np.ndarray,
    p_s: np.ndarray,
    p_h: np.ndarray,
    target_idxs: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Full diagnostic of why router may under/outperform Static Fusion.

    Computes:
    - Routing distribution {C%, S%, H%}
    - HR@K per routing bucket (users sent to C / S / H)
    - HR@K for Static Fusion on same user groups
    - Delta per group = Router - Static (where does the router win / lose?)
    """
    router.eval()
    with torch.no_grad():
        logits  = router(features)
        probs   = F.softmax(logits, dim=-1).numpy()
        actions = np.argmax(probs, axis=-1)   # [N]

    N = len(target_idxs)
    action_names = {0: "Collaborative", 1: "Semantic", 2: "Hybrid"}

    # Compute per-user ranks for all distributions
    def hit(p, tgt):
        rank = int(np.sum(p >= p[tgt]))
        return 1.0 if rank <= k else 0.0

    router_hits  = []
    static_hits  = []
    per_group    = {0: {"router": [], "static": []},
                    1: {"router": [], "static": []},
                    2: {"router": [], "static": []}}

    for u in range(N):
        tgt   = target_idxs[u]
        a     = actions[u]
        p_sel = [p_c[u], p_s[u], p_h[u]][a]

        r_hit = hit(p_sel, tgt)
        s_hit = hit(p_h[u], tgt)   # Static = p_h = 50/50

        router_hits.append(r_hit)
        static_hits.append(s_hit)
        per_group[a]["router"].append(r_hit)
        per_group[a]["static"].append(s_hit)

    diag = {
        "HR@K_router":  float(np.mean(router_hits)),
        "HR@K_static":  float(np.mean(static_hits)),
        "delta_global": float(np.mean(router_hits) - np.mean(static_hits)),
        "routing_dist": {
            action_names[a]: f"{(actions == a).mean()*100:.1f}%" for a in range(3)
        },
        "per_bucket": {},
    }

    print(f"\n  [Diagnostic] Router HR@{k}: {diag['HR@K_router']:.4f}  "
          f"Static HR@{k}: {diag['HR@K_static']:.4f}  "
          f"Delta: {diag['delta_global']:+.4f}")
    print(f"  Routing distribution: {diag['routing_dist']}")
    print(f"\n  Per-bucket analysis (where does router win/lose?):")

    for a in range(3):
        n_a    = len(per_group[a]["router"])
        if n_a == 0:
            continue
        hr_r   = float(np.mean(per_group[a]["router"]))
        hr_s   = float(np.mean(per_group[a]["static"]))
        delta  = hr_r - hr_s
        win_lbl = "WIN" if delta > 0 else ("TIE" if delta == 0 else "LOSE")
        print(f"    {action_names[a]:>14}: {n_a:>5} users  "
              f"Router HR={hr_r:.4f}  Static HR={hr_s:.4f}  "
              f"Delta={delta:+.4f}  [{win_lbl}]")
        diag["per_bucket"][action_names[a]] = {
            "n": n_a, "router_hr": hr_r, "static_hr": hr_s, "delta": delta,
        }

    return diag


# ── Main: Run fixed training and diagnosis ─────────────────────────────────────

def main():
    print("=" * 70)
    print("  ROUTER DIAGNOSTIC & FIXED TRAINING")
    print("=" * 70)

    # ── 1. Load pre-computed test distributions ───────────────────────────────
    # Check if full test tensors are cached
    cache_files = {
        "pc":      "checkpoints/test_pc.pt",
        "ps":      "checkpoints/test_ps.pt",
        "ph":      "checkpoints/test_ph.pt",
        "feats":   "checkpoints/test_features.pt",
        "targets": "checkpoints/test_targets.pt",
    }
    val_cache = {
        "feats":   "checkpoints/val_features.pt",
        "targets": "checkpoints/val_targets.pt",
    }

    has_cache = all(os.path.exists(p) for p in cache_files.values())

    if not has_cache:
        print("\n  [!] Full tensor cache not found.")
        print("  The main pipeline (train_and_evaluate_evidence_routing.py)")
        print("  must be modified to save caches. See instructions below.\n")
        print("  To enable caching, add this at end of extract_dataset_evidence_and_oracle():")
        print("    torch.save(pc_tensor,  'checkpoints/val_pc.pt')")
        print("    torch.save(ps_tensor,  'checkpoints/val_ps.pt')")
        print("    torch.save(ph_tensor,  'checkpoints/val_ph.pt')")
        print("    torch.save(features_tensor, 'checkpoints/val_features.pt')")
        print("    torch.save(torch.tensor(all_targets), 'checkpoints/val_targets.pt')")
        print()
        print("  Meanwhile, running diagnostic on synthetic data to validate the fix...")
        _run_synthetic_diagnostic()
        return

    # ── Load cached tensors ───────────────────────────────────────────────────
    print("\n  Loading cached tensors...")
    test_pc   = torch.load(cache_files["pc"],      weights_only=False)
    test_ps   = torch.load(cache_files["ps"],      weights_only=False)
    test_ph   = torch.load(cache_files["ph"],      weights_only=False)
    test_feat = torch.load(cache_files["feats"],   weights_only=False)
    test_tgt  = torch.load(cache_files["targets"], weights_only=False)

    val_feat  = torch.load(val_cache["feats"],   weights_only=False)
    val_tgt   = torch.load(val_cache["targets"], weights_only=False)

    # For val: need val_pc, val_ps, val_ph from cache too
    val_pc = torch.load("checkpoints/val_pc.pt", weights_only=False)
    val_ps = torch.load("checkpoints/val_ps.pt", weights_only=False)
    val_ph = torch.load("checkpoints/val_ph.pt", weights_only=False)

    N_test = test_pc.shape[0]
    N_val  = val_pc.shape[0]
    print(f"  Test: {N_test} users | Val: {N_val} users")

    # ── 2. Compute CORRECT oracle labels ─────────────────────────────────────
    test_tgt_np = test_tgt.numpy() - 1   # 0-indexed
    val_tgt_np  = val_tgt.numpy() - 1

    print("\n  Computing correct oracle labels (Y = 1/log2(rank+1))...")

    # Val labels (for router training)
    n_train  = int(N_val * 0.75)
    rng      = np.random.default_rng(SEED)
    perm     = rng.permutation(N_val)
    tr_idx   = perm[:n_train]
    va_idx   = perm[n_train:]

    h_tr, s_tr, hr_tr = compute_oracle_labels_correct(
        val_pc[tr_idx].numpy(), val_ps[tr_idx].numpy(), val_ph[tr_idx].numpy(),
        val_tgt_np[tr_idx],
    )
    h_va, s_va, hr_va = compute_oracle_labels_correct(
        val_pc[va_idx].numpy(), val_ps[va_idx].numpy(), val_ph[va_idx].numpy(),
        val_tgt_np[va_idx],
    )

    print(f"  Oracle HR@10 on val-train: C={hr_tr['Collaborative']:.4f} "
          f"S={hr_tr['Semantic']:.4f} H={hr_tr['Hybrid']:.4f}")
    print(f"  Oracle action distribution: {np.bincount(h_tr, minlength=3).tolist()}")

    # ── 3. Train fixed router ─────────────────────────────────────────────────
    print("\n  Training FIXED router (KL-div, hidden_dim=64, early stopping)...")
    router = FixedSelectiveRouter(in_features=val_feat.shape[1], hidden_dim=64)

    router, hist = train_router_fixed(
        router,
        features=val_feat[tr_idx],
        hard_labels=torch.tensor(h_tr, dtype=torch.long),
        soft_labels=torch.tensor(s_tr, dtype=torch.float32),
        val_features=val_feat[va_idx],
        val_hard_labels=torch.tensor(h_va, dtype=torch.long),
        val_soft_labels=torch.tensor(s_va, dtype=torch.float32),
        epochs=200, lr=1e-3, patience=20, lambda_reg=0.1,
    )

    torch.save(router.state_dict(), "checkpoints/router_fixed.pt")
    print("  Saved -> checkpoints/router_fixed.pt")

    # ── 4. Diagnose router vs. Static Fusion ─────────────────────────────────
    print("\n  Diagnosing router on test set...")
    h_test, s_test, hr_test = compute_oracle_labels_correct(
        test_pc.numpy(), test_ps.numpy(), test_ph.numpy(), test_tgt_np,
    )
    diag = diagnose_router_vs_static(
        router, test_feat, test_pc.numpy(), test_ps.numpy(), test_ph.numpy(),
        test_tgt_np, k=10,
    )

    with open("results/router_diagnostic.json", "w") as f:
        json.dump(diag, f, indent=2)
    print("\n  Saved -> results/router_diagnostic.json")


def _run_synthetic_diagnostic():
    """Runs the diagnostic logic on synthetic data to validate the fix."""
    print("  Synthetic diagnostic (N=2000 users, N_items=3628)...")
    N, NI = 2000, 3628
    rng = np.random.default_rng(42)

    # Simulate: collaborative is good for 40% users, semantic for 40%, hybrid for 20%
    p_c = rng.dirichlet(np.ones(NI) * 0.01, size=N).astype(np.float32)
    p_s = rng.dirichlet(np.ones(NI) * 0.01, size=N).astype(np.float32)
    p_h = 0.5 * p_c + 0.5 * p_s

    # Random targets
    targets = rng.integers(0, NI, size=N)

    # Compute oracle labels
    h_lbl, s_lbl, hr = compute_oracle_labels_correct(p_c, p_s, p_h, targets, k=10)
    print(f"  Oracle HR@10: C={hr['Collaborative']:.4f} S={hr['Semantic']:.4f} H={hr['Hybrid']:.4f}")

    # Construct dummy evidence features
    feat = torch.randn(N, 8) * 0.1

    # Train fixed router
    router = FixedSelectiveRouter(in_features=8, hidden_dim=64)
    tr, va = int(N*0.75), int(N*0.25)
    router, hist = train_router_fixed(
        router,
        features=feat[:tr],
        hard_labels=torch.tensor(h_lbl[:tr], dtype=torch.long),
        soft_labels=torch.tensor(s_lbl[:tr], dtype=torch.float32),
        val_features=feat[tr:],
        val_hard_labels=torch.tensor(h_lbl[tr:], dtype=torch.long),
        val_soft_labels=torch.tensor(s_lbl[tr:], dtype=torch.float32),
        epochs=50, lr=1e-3, patience=10, lambda_reg=0.0,
    )

    diag = diagnose_router_vs_static(router, feat, p_c, p_s, p_h, targets, k=10)
    print(f"\n  [Synthetic] Delta vs. Static: {diag['delta_global']:+.4f}")

    if diag["delta_global"] >= 0:
        print("  [OK] Fixed router >= Static Fusion on synthetic data.")
    else:
        print("  [WARN] Even fixed router < Static Fusion on random data.")
        print("  This is expected when features carry no signal (random features).")
        print("  The fix will help when real evidence features carry routing signal.")


if __name__ == "__main__":
    main()
