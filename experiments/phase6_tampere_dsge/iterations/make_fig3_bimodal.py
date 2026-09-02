#!/usr/bin/env python3
"""Updated Fig 3 — bimodal PATP α-sweep with degenerate-α=0.5 dip."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figures"

data = json.loads((RESULTS / "phase6_patp_complementarity.json").read_text())

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.fontsize": 10})

alphas = np.array(data["alphas"])
accs = np.array([r["accuracy"] for r in data["per_alpha"]])
deltas = np.array([r["delta_vs_spectral"] for r in data["per_alpha"]])
sig = np.array([r["significant"] for r in data["per_alpha"]])
spec_baseline = data["spectral_baseline"]["accuracy"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Panel 1: accuracy vs alpha
ax1.plot(alphas, accs, "o-", color="#1f77b4", lw=2, markersize=9, zorder=3)
# Mark significant points
ax1.plot(alphas[sig], accs[sig], "*", color="#2ca02c", markersize=15,
         label="Significant boost (p<0.05)", zorder=4)
ax1.axhline(spec_baseline, ls="--", color="#888", lw=1.5,
            label=f"Spectral baseline ({spec_baseline:.4f})")
ax1.axvline(0.5, ls=":", color="red", alpha=0.6, lw=1.5,
            label="α=0.5: degenerate (basis collapses to identity)")
ax1.fill_between([0, 0.4], 0.94, 1.0, alpha=0.1, color="green",
                  label="Frac branch (best)")
ax1.fill_between([0.6, 0.8], 0.94, 1.0, alpha=0.1, color="orange",
                  label="Poly branch (partial)")

# Annotate boundary peaks
ax1.annotate(f"α=0.0\nacc={accs[0]:.4f}",
              xy=(0.0, accs[0]), xytext=(0.05, accs[0]-0.02),
              fontsize=9, color="darkgreen", weight="bold")
ax1.annotate(f"α=0.5\n(degenerate)",
              xy=(0.5, accs[5]), xytext=(0.4, 0.93),
              fontsize=9, color="darkred",
              arrowprops=dict(arrowstyle="->", color="darkred", alpha=0.6))

ax1.set_ylabel("Accuracy (Hybrid: PATP + Spectral)")
ax1.set_title("PATP α-sweep — Task 2 (DJI vs non-DJI)\nBimodal curve reveals two distinct DSGE mechanisms")
ax1.set_ylim(0.93, 1.005)
ax1.legend(loc="lower right", fontsize=8.5)
ax1.grid(alpha=0.3)

# Panel 2: Δ accuracy vs spectral baseline
colors = ["#2ca02c" if s else "#d62728" if d > 0.001 else "#888"
          for s, d in zip(sig, deltas)]
ax2.bar(alphas, deltas, width=0.06, color=colors, alpha=0.85,
        edgecolor="black", linewidth=0.5)
ax2.axhline(0, ls="-", color="#888", lw=0.5)
ax2.axvline(0.5, ls=":", color="red", alpha=0.4, lw=1)
ax2.set_xlabel("PATP α (transition parameter)")
ax2.set_ylabel("Δ Accuracy vs Spectral-only\n(★ = bootstrap p<0.05)")
ax2.set_title("Complementarity boost over Spectral baseline")
ax2.set_xticks(alphas)
ax2.set_xticklabels([f"{a:.1f}" for a in alphas])
ax2.grid(axis="y", alpha=0.3)

# Annotate sig points
for a, d, s in zip(alphas, deltas, sig):
    if s:
        ax2.annotate("★", xy=(a, d), xytext=(0, 8),
                      textcoords="offset points", ha="center",
                      fontsize=14, color="darkgreen", weight="bold")

plt.tight_layout()
plt.savefig(FIGS / "fig3_patp_bimodal.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIGS / 'fig3_patp_bimodal.pdf'}")
