"""
Generate Figure 1: CC-PoE Architecture and Candidate Disagreement Penalty Mechanism
Publication-quality architecture diagram for ACM RecSys.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
ax.axis("off")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.0)

def draw_box(ax, x, y, w, h, text, color="#eaf2f8", border="#2980b9", title=""):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                  facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 0.035, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color="#1a5276")
        ax.text(x + w/2, y + (h - 0.035)/2, text, ha="center", va="center", fontsize=8, color="#2c3e50")
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#2c3e50")

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.5))
    if label:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.015, label, ha="center", va="bottom", fontsize=7.5, color="#555555", fontweight="bold")

# 1. Input Sequence
draw_box(ax, 0.03, 0.62, 0.20, 0.24, 
         "Sequence S_u = [v_1, ..., v_t]\nFull Catalog |I| Items\nStrict History Masking", 
         "#fef9e7", "#f39c12", "Input Trajectory")

# 2. Collaborative Expert (SASRec)
draw_box(ax, 0.28, 0.74, 0.24, 0.21, 
         "Self-Attention Transformer\nDiscrete Item Transitions\nOutput: log P_C(i | u)", 
         "#ebf5fb", "#2980b9", "Collaborative Expert (C)")

# 3. Semantic Expert (Text + Attention)
draw_box(ax, 0.28, 0.49, 0.24, 0.21, 
         "Sentence Transformer (MiniLM)\nEnriched Title + Brand + Cat\nOutput: log P_S(i | u)", 
         "#f4ecf7", "#8e44ad", "Semantic Expert (S)")

# Arrows Input -> Experts
draw_arrow(ax, 0.23, 0.78, 0.28, 0.82)
draw_arrow(ax, 0.23, 0.68, 0.28, 0.60)

# 4. Evidence Descriptors Analyzer
draw_box(ax, 0.57, 0.53, 0.24, 0.42, 
         "• Collaborative Entropy H_c\n• Semantic Entropy H_s\n• Margin Distances M_c, M_s\n• Divergence D_JS(P_C || P_S)\n• Epistemic Variance U_c, U_s\n• Sequence Length L_u / 50\n• Semantic Drift Delta_u\n\nx_u in R^13", 
         "#e8f8f5", "#16a085", "13D Evidence Vector")

# Arrows Experts -> Evidence Analyzer
draw_arrow(ax, 0.52, 0.83, 0.57, 0.78, "log P_C")
draw_arrow(ax, 0.52, 0.59, 0.57, 0.64, "log P_S")

# 5. Adaptive Disagreement Penalty Policy
draw_box(ax, 0.57, 0.12, 0.24, 0.28, 
         "Compact MLP (578 params):\nLinear(13, 32) -> LN -> ReLU\n• delta(x_u) in [-0.10, +0.10]\n• lambda(x_u) in [0, 0.34]\n(A priori safety bound:\nmin effective weight >= 0.06)", 
         "#fbeee6", "#d35400", "Disagreement Policy Head")

# Arrow Evidence Analyzer -> Parameter Head
draw_arrow(ax, 0.69, 0.53, 0.69, 0.40, "x_u")

# 6. Candidate-Level Disagreement Penalty
draw_box(ax, 0.28, 0.08, 0.24, 0.32, 
         "Item Discrepancy:\nd(u, i) = log P_C(i) - log P_S(i)\n\nScore Formula:\nS(u, i) = (0.5 + delta) log P_C\n        + (0.5 - delta) log P_S\n        - lambda · |log P_C - log P_S|\n\nCandidate-Level Regularization", 
         "#eafaf1", "#27ae60", "Candidate Arbitration")

# Arrows from Experts and Policy to Candidate Arbitration
ax.annotate("", xy=(0.40, 0.40), xytext=(0.40, 0.49),
            arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.5))
ax.text(0.41, 0.44, "Expert Logits", fontsize=7.5, color="#555555")

draw_arrow(ax, 0.57, 0.22, 0.52, 0.22, "delta, lambda")

# 7. Final Top-10 Slate
draw_box(ax, 0.03, 0.12, 0.20, 0.24, 
         "Full-Catalog Ranking\nSuppresses Incompatible\nFalse-Positive Intrusions\n\nEvaluated Top-10 Slate", 
         "#eaeded", "#7f8c8d", "Final Recommendation")

draw_arrow(ax, 0.28, 0.24, 0.23, 0.24, "Rank S(u, ·)")

plt.tight_layout()
plt.savefig("figures/fig1_architecture.png", bbox_inches="tight")
plt.savefig("figures/fig1_architecture.pdf", bbox_inches="tight")
plt.close()
print("Saved figures/fig1_architecture.png and .pdf successfully!")
