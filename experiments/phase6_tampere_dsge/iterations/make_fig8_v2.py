#!/usr/bin/env python3
"""Fig 8 v2 — corrected with window-level block bootstrap CIs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS = Path(__file__).parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figures"
data = json.loads((RESULTS / "phase6_distribution_diagnostics_v2.json").read_text())

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.fontsize": 9})

cumulant_names = ["gamma3", "gamma4", "gamma5", "gamma6"]
cumulant_labels = [r"$\gamma_3$ (skewness)", r"$\gamma_4$ (excess kurt.)",
                    r"$\gamma_5$", r"$\gamma_6$"]

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=False)
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for c_idx, (cum_name, cum_label) in enumerate(zip(cumulant_names, cumulant_labels)):
    ax = axes[positions[c_idx]]
    bar_pos, bar_vals, bar_lo, bar_hi, bar_labels, bar_colors = [], [], [], [], [], []
    bands_iter = [(b, info) for b, info in data.items()
                  if b != "config" and "classes" in info]
    for b_idx, (band_name, band) in enumerate(bands_iter):
        for cls_idx, (cls_name, info) in enumerate(band["classes"].items()):
            x_pos = b_idx * 5 + cls_idx
            val = info["pooled_cumulants_Re"][cum_name]
            ci = info["Re_bootstrap_95CI_window_level"][cum_name]
            bar_pos.append(x_pos)
            bar_vals.append(val)
            bar_lo.append(val - ci[0])
            bar_hi.append(ci[1] - val)
            bar_labels.append(f"{cls_name}\n({band_name})")
            color = "tab:blue" if "DJI" in info["protocol"] else "tab:orange"
            bar_colors.append(color)

    ax.bar(bar_pos, bar_vals, yerr=[bar_lo, bar_hi], color=bar_colors, alpha=0.85,
           edgecolor="black", linewidth=0.5, capsize=4)
    ax.axhline(0, ls="-", color="#888", lw=0.5)
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(cum_label)
    ax.set_title(f"Pooled-sample {cum_label}, **window-level** 95% bootstrap CI (R=100)")
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(3.5, ls=":", color="black", alpha=0.4, lw=1)

legend = [Patch(facecolor="tab:blue", label="DJI (OcuSync)"),
          Patch(facecolor="tab:orange", label="WiFi-based")]
fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Higher-order cumulants γ₃–γ₆ across protocols and bands\n"
             "(corrected v2: window-level bootstrap respects within-window correlation)",
             fontsize=12, y=1.06)
plt.tight_layout(rect=[0, 0, 1, 1.0])
plt.savefig(FIGS / "fig8_cumulants_v2.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIGS / 'fig8_cumulants_v2.pdf'}")
