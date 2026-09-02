#!/usr/bin/env python3
"""Generate figures for Phase 6 draft paper.

Figures:
    fig1_basis_comparison.pdf — bar chart of accuracy per basis × task
    fig2_dsge_vs_spectral_vs_hybrid.pdf — three-way comparison Task 2 + 3
    fig3_patp_alpha_sweep.pdf — Task 3 PATP α-sweep curve
    fig4_tsne_features.pdf — t-SNE 2D projection of DSGE features (4-class)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Vendored DSGE toolkit modules live next to this file (see toolkit/README.md).
_TOOLKIT = Path(__file__).resolve().parent / "toolkit"
if not _TOOLKIT.is_dir():                      # scripts under iterations/
    _TOOLKIT = Path(__file__).resolve().parent.parent / "toolkit"
sys.path.insert(0, str(_TOOLKIT))
from complex_features import ComplexDSGEFeatureExtractor

RESULTS = Path(__file__).parent / "results"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)

# Load results
task1 = json.loads((RESULTS / "phase6_iter2_task1.json").read_text())
task23 = json.loads((RESULTS / "phase6_iter2_task23.json").read_text())

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
})

# ---- Figure 1: Basis comparison across 3 tasks ----
print("Fig 1: basis comparison...")
bases = ["polynomial", "fractional", "trigonometric", "robust"]
task_data = {
    "Task 1: 4-class\nmulti-protocol": [task1["bases"][b]["accuracy"] for b in bases],
    "Task 2: DJI vs\nnon-DJI (binary)": [task23["task2"]["bases"][b]["accuracy"] for b in bases],
    "Task 3: Mavic vs\nInspire (intra-OcuSync)": [task23["task3"]["bases"][b]["accuracy"] for b in bases],
}
random_baselines = [0.25, 0.50, 0.50]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(bases))
width = 0.25
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i, (task, vals) in enumerate(task_data.items()):
    bars = ax.bar(x + i*width - width, vals, width, label=task, color=colors[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
# Random baselines
for i, baseline in enumerate(random_baselines):
    ax.axhline(baseline, ls="--", color=colors[i], alpha=0.4, lw=1)

ax.set_xticks(x)
ax.set_xticklabels(bases)
ax.set_ylabel("Accuracy on test set")
ax.set_title("DSGE basis comparison across 3 tasks (Tampere Zenodo, iter-2)")
ax.set_ylim(0.0, 1.0)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "fig1_basis_comparison.pdf", bbox_inches="tight")
plt.close()
print(f"  saved {FIGS / 'fig1_basis_comparison.pdf'}")


# ---- Figure 2: DSGE-only vs Spectral-only vs Hybrid ----
print("Fig 2: DSGE / spectral / hybrid...")
tasks_with_hybrid = ["task2", "task3"]
task_labels = ["DJI vs non-DJI\n(Task 2)", "Mavic vs Inspire\n(Task 3, intra-OcuSync)"]

dsge_only = [
    max(task23["task2"]["bases"][b]["accuracy"] for b in bases),
    max(task23["task3"]["bases"][b]["accuracy"] for b in bases),
]
spec_only = [task23["task2"]["spectral_only"]["accuracy"],
             task23["task3"]["spectral_only"]["accuracy"]]
hybrid = [task23["task2"]["hybrid"]["accuracy"],
          task23["task3"]["hybrid"]["accuracy"]]

x = np.arange(len(task_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width, dsge_only, width, label="DSGE-only (best basis)", color="#1f77b4")
bars2 = ax.bar(x, spec_only, width, label="Spectral PSD only (16 bins)", color="#ff7f0e")
bars3 = ax.bar(x + width, hybrid, width, label="Hybrid (DSGE + Spectral)", color="#2ca02c")

for bars in (bars1, bars2, bars3):
    for bar in bars:
        v = bar.get_height()
        ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=9)

ax.axhline(0.5, ls="--", color="gray", alpha=0.5, lw=1, label="Random (binary)")
ax.set_xticks(x)
ax.set_xticklabels(task_labels)
ax.set_ylabel("Accuracy")
ax.set_title("DSGE vs Spectral PSD vs Hybrid — Tampere Zenodo")
ax.set_ylim(0.0, 1.05)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Annotate bootstrap p-values
bs2 = task23["task2"]["bootstrap_hybrid_vs_spectral"]
bs3 = task23["task3"]["bootstrap_hybrid_vs_spectral"]
ax.text(0, 0.05, f"Bootstrap p={bs2['p_value']:.3f}\nΔ={bs2['delta']:+.4f}", fontsize=8, ha="center")
ax.text(1, 0.05, f"Bootstrap p={bs3['p_value']:.3f}\nΔ={bs3['delta']:+.4f}", fontsize=8, ha="center")

plt.tight_layout()
plt.savefig(FIGS / "fig2_dsge_vs_spectral_vs_hybrid.pdf", bbox_inches="tight")
plt.close()
print(f"  saved {FIGS / 'fig2_dsge_vs_spectral_vs_hybrid.pdf'}")


# ---- Figure 3: PATP α-sweep ----
print("Fig 3: PATP α-sweep...")
patp = task23["task3"]["patp"]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(patp["alphas"], patp["macro_f1"], "o-", color="#1f77b4", lw=2, markersize=8)
ax.axvline(patp["best_alpha"], ls="--", color="red", alpha=0.6,
           label=f"α_opt = {patp['best_alpha']:.2f}")
ax.axhline(0.5, ls=":", color="gray", alpha=0.5, label="Random baseline")

# Annotate boundary status
boundary_msg = patp["boundary_check"]["recommendation"]
ax.text(0.5, 0.02, boundary_msg, transform=ax.transAxes,
        ha="center", fontsize=9, style="italic", color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.set_xlabel("PATP α (transition parameter)")
ax.set_ylabel("Macro F1 (Mavic vs Inspire)")
ax.set_title("PATP α-sweep on Task 3 (intra-OcuSync fingerprinting)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.4, 0.75)
plt.tight_layout()
plt.savefig(FIGS / "fig3_patp_alpha_sweep.pdf", bbox_inches="tight")
plt.close()
print(f"  saved {FIGS / 'fig3_patp_alpha_sweep.pdf'}")


# ---- Figure 4: t-SNE on DSGE features ----
print("Fig 4: t-SNE…")
# Need to recompute features — load data and apply best basis
WIN = 4096
N_PER_CLASS = 1000
# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
CLASSES = [
    ("Mavic Pro",   DATA_DIR / "DJI_mavic_pro_2G.bin"),
    ("Inspire 2",   DATA_DIR / "DJI_inspire_2_2G.bin"),
    ("Parrot Disco",DATA_DIR / "Parrot_disco_2G.bin"),
    ("Yuneec H",    DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin"),
]


def load_burst(path, n, w, rng):
    data = np.fromfile(path, dtype="<i2", count=2 * n * w * 2)
    sig = (data.astype(np.float32).reshape(-1, 2)[:, 0]
           + 1j * data.astype(np.float32).reshape(-1, 2)[:, 1]).astype(np.complex64)
    nc = len(sig) // w
    chunks = sig[:nc * w].reshape(nc, w)
    energies = (chunks.real ** 2 + chunks.imag ** 2).mean(axis=1)
    high = np.where(energies >= np.percentile(energies, 40))[0]
    if len(high) >= n:
        sel = rng.choice(high, n, replace=False)
    else:
        sel = np.concatenate([high, rng.choice(np.where(energies < np.percentile(energies, 40))[0],
                                                 n - len(high), replace=False)])
    sel = np.sort(sel)
    w_arr = chunks[sel]
    m, s = w_arr.mean(axis=1, keepdims=True), w_arr.std(axis=1, keepdims=True) + 1e-9
    return ((w_arr - m) / s).astype(np.complex64)


# Use a subset for t-SNE speed
TSNE_N_PER_CLASS = 300
rng = np.random.default_rng(42)
X_all, y_all = [], []
for c, (name, path) in enumerate(CLASSES):
    w = load_burst(path, TSNE_N_PER_CLASS, WIN, rng)
    X_all.append(w); y_all.append(np.full(TSNE_N_PER_CLASS, c))
X_all = np.vstack(X_all); y_all = np.concatenate(y_all)

# Best basis from Task 1 = trigonometric
best_basis = max(task1["bases"], key=lambda k: task1["bases"][k]["accuracy"])
print(f"  using {best_basis} basis for t-SNE")
ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=best_basis, n=3, ridge=0.01).fit(X_all, y_all)
features = ext.transform(X_all)

from sklearn.manifold import TSNE
print("  running t-SNE...")
embed = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features)

fig, ax = plt.subplots(figsize=(8, 6.5))
class_names = [c[0] for c in CLASSES]
colors_tsne = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for c, (name, color) in enumerate(zip(class_names, colors_tsne)):
    mask = y_all == c
    ax.scatter(embed[mask, 0], embed[mask, 1], c=color, label=name,
               alpha=0.6, s=30, edgecolors="none")
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
ax.set_title(f"t-SNE of DSGE features ({best_basis} basis, re_im mode)\nTampere Zenodo, 4 drone protocols")
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIGS / "fig4_tsne_dsge_features.pdf", bbox_inches="tight")
plt.close()
print(f"  saved {FIGS / 'fig4_tsne_dsge_features.pdf'}")

print("\nAll 4 figures generated.")
