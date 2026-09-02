#!/usr/bin/env python3
"""Updated Figure 2 — hybrid variants with complementarity story."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figures"

variants = json.loads((RESULTS / "phase6_iter2_hybrid_variants.json").read_text())

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.fontsize": 9})

# Order: spectral-only first, then DSGE+spec variants (poly weak, frac/trig strong)
order = [
    "spectral-only",
    "poly + spec",
    "frac + spec",
    "trig + spec",
    "poly + frac + spec",
    "all 4 bases + spec",
]

# Build accuracy + p-value tables for both tasks
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for ax, task_key, task_label in zip(
        (ax1, ax2),
        ("task2", "task3"),
        ("Task 2: DJI vs non-DJI", "Task 3: Mavic Pro vs Inspire 2 (intra-OcuSync)")):
    task = variants[task_key]
    accs = []
    p_vals = []
    sig = []
    feat_dims = []
    for v in order:
        if v == "spectral-only":
            accs.append(task["spectral_only"]["accuracy"])
            p_vals.append(None)
            sig.append(False)
            feat_dims.append(task["spectral_only"]["feature_dim"])
        else:
            r = task["variants"][v]
            accs.append(r["accuracy"])
            p_vals.append(r["bootstrap_vs_spectral"]["p_value"])
            sig.append(r["bootstrap_vs_spectral"]["significant"])
            feat_dims.append(r["feature_dim"])

    colors = []
    for v, s in zip(order, sig):
        if v == "spectral-only":
            colors.append("#888888")
        elif s:
            colors.append("#2ca02c")  # green for significant
        else:
            colors.append("#ff7f0e")  # orange for non-significant

    x = np.arange(len(order))
    bars = ax.bar(x, accs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar, acc, p, s, dim in zip(bars, accs, p_vals, sig, feat_dims):
        ax.annotate(f"{acc:.4f}", xy=(bar.get_x() + bar.get_width()/2, acc),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, weight="bold")
        if p is not None:
            star = "★" if s else ""
            text = f"p={p:.3f}{star}\ndim={dim}"
            ax.annotate(text, xy=(bar.get_x() + bar.get_width()/2, 0.02),
                        ha="center", fontsize=8, color="black")
        else:
            ax.annotate(f"baseline\ndim={dim}", xy=(bar.get_x() + bar.get_width()/2, 0.02),
                        ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_title(task_label)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.4, lw=1)

ax1.set_ylabel("Accuracy")

# Custom legend
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#888888", edgecolor="black", label="Spectral baseline"),
    Patch(facecolor="#2ca02c", edgecolor="black", label="Significant boost (★)"),
    Patch(facecolor="#ff7f0e", edgecolor="black", label="Not significant"),
]
fig.legend(handles=legend_elems, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))

fig.suptitle("Hybrid variants: which DSGE basis adds complementary information to Spectral PSD?",
             fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIGS / "fig2_hybrid_variants.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIGS / 'fig2_hybrid_variants.pdf'}")
