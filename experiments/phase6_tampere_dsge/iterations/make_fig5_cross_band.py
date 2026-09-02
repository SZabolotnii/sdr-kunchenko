#!/usr/bin/env python3
"""Cross-band comparison figure — 2.4 GHz vs 5.8 GHz PATP α-sweeps.

Shows the inverted complementarity pattern between bands.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figures"

data_24 = json.loads((RESULTS / "phase6_patp_complementarity.json").read_text())
data_58 = json.loads((RESULTS / "phase6_5G_patp_complementarity.json").read_text())

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.fontsize": 10})

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

for col, (data, band, fs_label) in enumerate([
        (data_24, "2.4 GHz", "fs=120 MS/s, 4 protocols"),
        (data_58, "5.8 GHz", "fs=200 MS/s, 4 protocols")]):

    alphas = np.array(data["alphas"])
    accs = np.array([r["accuracy"] for r in data["per_alpha"]])
    deltas = np.array([r["delta_vs_spectral"] for r in data["per_alpha"]])
    sig = np.array([r["significant"] for r in data["per_alpha"]])
    spec_baseline = data["spectral_baseline"]["accuracy"]

    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    # Top: accuracy curve
    ax_top.plot(alphas, accs, "o-", color="#1f77b4", lw=2, markersize=9, zorder=3)
    if any(sig):
        ax_top.plot(alphas[sig], accs[sig], "*", color="#2ca02c", markersize=15,
                     zorder=4, label="Significant boost (p<0.05)")
    ax_top.axhline(spec_baseline, ls="--", color="#888", lw=1.5,
                    label=f"Spectral baseline ({spec_baseline:.4f})")
    ax_top.axvline(0.5, ls=":", color="red", alpha=0.5, lw=1.2)

    # Highlight winning region
    if data is data_24:
        ax_top.fill_between([0, 0.4], 0.94, 1.005, alpha=0.15, color="green",
                             label="Frac branch (winner)")
    else:
        ax_top.fill_between([0.7, 1.0], 0.96, 1.0, alpha=0.15, color="orange",
                             label="Poly branch (winner)")

    best_idx = int(np.argmax(accs))
    ax_top.annotate(f"best α={alphas[best_idx]:.2f}\nacc={accs[best_idx]:.4f}",
                     xy=(alphas[best_idx], accs[best_idx]),
                     xytext=(alphas[best_idx] + 0.05, accs[best_idx] - 0.005),
                     fontsize=9, weight="bold", color="darkred")

    ax_top.set_title(f"Tampere {band} ({fs_label})", fontsize=12)
    ax_top.set_ylabel("Accuracy (PATP+spec hybrid)")
    ax_top.set_ylim(min(accs.min(), spec_baseline) - 0.005, 1.005)
    ax_top.legend(loc="lower right", fontsize=9)
    ax_top.grid(alpha=0.3)

    # Bottom: Δ accuracy bars
    colors = ["#2ca02c" if s else "#aaaaaa" for s in sig]
    ax_bot.bar(alphas, deltas, width=0.06, color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.5)
    ax_bot.axhline(0, ls="-", color="#888", lw=0.5)
    ax_bot.axvline(0.5, ls=":", color="red", alpha=0.4)
    ax_bot.set_xlabel("PATP α (transition parameter)")
    ax_bot.set_ylabel("Δ Accuracy vs Spectral")
    ax_bot.set_xticks(alphas)
    ax_bot.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax_bot.grid(axis="y", alpha=0.3)
    for a, d, s in zip(alphas, deltas, sig):
        if s:
            ax_bot.annotate("★", xy=(a, d), xytext=(0, 6),
                              textcoords="offset points", ha="center",
                              fontsize=14, color="darkgreen", weight="bold")

# Big title
fig.suptitle("Band-dependent DSGE-spectral complementarity in OcuSync classification",
              fontsize=14, y=1.0)

plt.tight_layout()
plt.savefig(FIGS / "fig5_cross_band_patp.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIGS / 'fig5_cross_band_patp.pdf'}")

# Summary table
summary = {
    "2.4_GHz": {
        "best_alpha": data_24["best_alpha"],
        "best_accuracy": data_24["best_accuracy"],
        "spectral_baseline": data_24["spectral_baseline"]["accuracy"],
        "boost_pp": (data_24["best_accuracy"] - data_24["spectral_baseline"]["accuracy"]) * 100,
        "best_branch": "fractional (α=0, boundary)",
    },
    "5.8_GHz": {
        "best_alpha": data_58["best_alpha"],
        "best_accuracy": data_58["best_accuracy"],
        "spectral_baseline": data_58["spectral_baseline"]["accuracy"],
        "boost_pp": (data_58["best_accuracy"] - data_58["spectral_baseline"]["accuracy"]) * 100,
        "best_branch": "polynomial (α=1, boundary)",
    },
}
print("\n=== Cross-band summary ===")
for band, info in summary.items():
    print(f"{band}:")
    for k, v in info.items():
        print(f"  {k}: {v}")

with open(RESULTS / "phase6_cross_band_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
