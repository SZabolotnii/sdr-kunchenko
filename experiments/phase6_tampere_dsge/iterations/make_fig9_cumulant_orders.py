#!/usr/bin/env python3
"""Fig 9 — cumulant explicit, order 4 vs 6, on both bands."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figures"
data = json.loads((RESULTS / "phase6_cumulant_4vs6.json").read_text())

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.fontsize": 10})

bands = ["2.4_GHz", "5.8_GHz"]
configs = ["cumulant_order4", "cumulant_order6"]
config_labels = {"cumulant_order4": "γ₃, γ₄ (6 dims)",
                 "cumulant_order6": "γ₃, γ₄, γ₅, γ₆ (12 dims)"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

for ax_idx, band in enumerate(bands):
    ax = axes[ax_idx]
    seeds = [s["seed"] for s in data[band]["per_seed"]]
    spec_accs = [s["spectral_acc"] for s in data[band]["per_seed"]]

    width = 0.35
    x = np.arange(len(seeds))
    for c_idx, cfg in enumerate(configs):
        accs = [s["configurations"][cfg]["accuracy"] for s in data[band]["per_seed"]]
        sigs = [s["configurations"][cfg]["significant"] for s in data[band]["per_seed"]]
        deltas = [s["configurations"][cfg]["delta"] for s in data[band]["per_seed"]]
        bars = ax.bar(x + (c_idx - 0.5) * width, accs, width,
                       label=config_labels[cfg],
                       color=f"C{c_idx+2}", alpha=0.85,
                       edgecolor="black", linewidth=0.5)
        for bar, acc, sig, d in zip(bars, accs, sigs, deltas):
            text = f"{acc:.4f}\nΔ={d:+.4f}"
            ax.annotate(text, xy=(bar.get_x() + bar.get_width()/2, acc),
                          xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=8, weight="bold" if sig else "normal")
            if sig:
                ax.annotate("★", xy=(bar.get_x() + bar.get_width()/2, acc + 0.005),
                              ha="center", fontsize=14, color="darkgreen")

    # Spectral baselines
    for i, s in enumerate(spec_accs):
        ax.hlines(s, i - 0.5, i + 0.5, ls="--", color="gray", lw=1.5)
    ax.plot([], [], ls="--", color="gray", label="spectral baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Accuracy on Task 2 (DJI vs non-DJI)")

    band_label = band.replace("_", " ")
    fs = data[band]["per_seed"][0].get("spectral_acc", 0)
    ax.set_title(f"{band_label} — Task 2")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ymin = min(min(spec_accs) - 0.005, 0.94)
    ax.set_ylim(ymin, 1.005)

# Overall summary
summary_text = []
for band in bands:
    s4 = data[band]["summary"]["cumulant_order4"]
    s6 = data[band]["summary"]["cumulant_order6"]
    summary_text.append(f"{band.replace('_', ' ')}: γ₃₄ mean Δ={s4['mean_delta']:+.3f} → "
                         f"γ₃-γ₆ mean Δ={s6['mean_delta']:+.3f}")

fig.suptitle("Explicit cumulant features as complement to spectral PSD\n"
              "Order 4 (γ₃,γ₄) vs Order 6 (γ₃,γ₄,γ₅,γ₆) — 3-seed replication",
              fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig(FIGS / "fig9_cumulant_orders.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIGS / 'fig9_cumulant_orders.pdf'}")

# Also: order-increment figure
fig2, ax2 = plt.subplots(figsize=(8, 5))
band_labels = ["2.4 GHz", "5.8 GHz"]
order4_deltas = [data[b]["summary"]["cumulant_order4"]["mean_delta"] for b in bands]
order6_deltas = [data[b]["summary"]["cumulant_order6"]["mean_delta"] for b in bands]
order4_std = [data[b]["summary"]["cumulant_order4"]["std_delta"] for b in bands]
order6_std = [data[b]["summary"]["cumulant_order6"]["std_delta"] for b in bands]

x = np.arange(len(band_labels))
width = 0.35
b1 = ax2.bar(x - width/2, order4_deltas, width, yerr=order4_std,
              label="Order 4 (γ₃, γ₄)", color="#1f77b4", capsize=5,
              edgecolor="black")
b2 = ax2.bar(x + width/2, order6_deltas, width, yerr=order6_std,
              label="Order 6 (γ₃, γ₄, γ₅, γ₆)", color="#ff7f0e", capsize=5,
              edgecolor="black")

for bars, vals in zip((b1, b2), (order4_deltas, order6_deltas)):
    for bar, v in zip(bars, vals):
        ax2.annotate(f"{v:+.4f}",
                       xy=(bar.get_x() + bar.get_width()/2, v),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=10, weight="bold")

# Annotate increment from γ₅, γ₆
for i, band in enumerate(bands):
    incr = order6_deltas[i] - order4_deltas[i]
    text = f"γ₅,γ₆ adds:\n{incr:+.4f}"
    ax2.annotate(text, xy=(i, max(order4_deltas[i], order6_deltas[i]) + 0.005),
                  xytext=(0, 30), textcoords="offset points",
                  ha="center", fontsize=10, color="darkred",
                  arrowprops=dict(arrowstyle="<->", color="darkred", alpha=0.6))

ax2.set_xticks(x)
ax2.set_xticklabels(band_labels)
ax2.set_ylabel("Δ accuracy vs spectral baseline (mean over 3 seeds)")
ax2.set_title("Cumulant order vs band: γ₅,γ₆ matter on 2.4 GHz, not on 5.8 GHz")
ax2.legend(loc="upper right")
ax2.grid(axis="y", alpha=0.3)
ax2.axhline(0, color="black", lw=0.5)
plt.tight_layout()
plt.savefig(FIGS / "fig10_order_increment.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIGS / 'fig10_order_increment.pdf'}")
