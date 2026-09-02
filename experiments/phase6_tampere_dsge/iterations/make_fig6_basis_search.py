#!/usr/bin/env python3
"""Figure 6 — Extended basis search for 5.8 GHz."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figures"

data = json.loads((RESULTS / "phase6_5G_basis_search.json").read_text())

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.fontsize": 9})

candidates = list(data["candidates"].keys())
labels_pretty = {
    "cumulant_explicit": "Cumulant\n(γ₃, γ₄ explicit)",
    "envelope_stats": "Envelope stats\n(|y(t)|)",
    "phase_derivative": "Phase derivative\n(d arg/dn)",
    "hermite_orth": "Hermite\n(H₂, H₃, H₄)",
    "mixed_exponents": "Mixed exponents\n({0.25, 0.75, 1.5, 2.5})",
}

# For each candidate: 3 deltas (one per seed)
deltas_by_cand = {c: [r["delta"] for r in data["candidates"][c]["per_seed"]]
                  for c in candidates}
sig_by_cand = {c: [r["significant"] for r in data["candidates"][c]["per_seed"]]
               for c in candidates}
seeds = [r["seed"] for r in data["candidates"][candidates[0]]["per_seed"]]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(candidates))
width = 0.25
seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, (seed, color) in enumerate(zip(seeds, seed_colors)):
    deltas = [deltas_by_cand[c][i] for c in candidates]
    sigs = [sig_by_cand[c][i] for c in candidates]
    bars = ax.bar(x + (i - 1) * width, deltas, width,
                   label=f"seed {seed}", color=color, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    for bar, sig in zip(bars, sigs):
        if sig:
            h = bar.get_height()
            ax.annotate("★", xy=(bar.get_x() + bar.get_width() / 2, h),
                          xytext=(0, 3), textcoords="offset points",
                          ha="center", color="darkgreen",
                          fontsize=14, weight="bold")

ax.axhline(0, ls="-", color="#888", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([labels_pretty[c] for c in candidates], fontsize=9)
ax.set_ylabel("Δ Accuracy vs Spectral-only baseline")
ax.set_title("Extended basis search on 5.8 GHz Task 2 — 3-seed replication test\n"
             "★ = significant (bootstrap p<0.05, R=500)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Annotate "reproducible" / "single-seed" labels
for i, c in enumerate(candidates):
    n_sig = sum(sig_by_cand[c])
    text = f"{n_sig}/3"
    color = "darkgreen" if n_sig >= 2 else "#888"
    ax.text(i, -0.001, text, ha="center", fontsize=10,
             color=color, weight="bold")

plt.tight_layout()
plt.savefig(FIGS / "fig6_basis_search_5G.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIGS / 'fig6_basis_search_5G.pdf'}")
