"""Generate Exp 3 figures — grayscale-readable (distinct markers + linestyles + hatches)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# Grayscale-safe style palette: distinct markers + linestyles for each series
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "|||", "---"]
GRAYS = ["0.0", "0.35", "0.55", "0.70", "0.85"]  # distinct gray shades


def fig_accuracy_vs_snr(df: pd.DataFrame, out: Path) -> None:
    """Fig. 5: Accuracy vs SNR for three feature-set types (grayscale-safe)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    types = ["baseline", "dsge", "hybrid"]
    labels = ["Spectral-only", "DSGE-only", "Hybrid (DSGE+Spectral)"]
    for i, (typ, label) in enumerate(zip(types, labels, strict=False)):
        sub = df[df["type"] == typ]
        if sub.empty:
            continue
        best = sub.groupby("snr_db")["accuracy"].max().sort_index()
        ax.plot(
            best.index, best.values,
            marker=MARKERS[i],
            linestyle=LINESTYLES[i],
            lw=2, ms=8,
            color=GRAYS[i],
            label=label,
        )
    ax.axhline(0.70, color="0.0", linestyle=(0, (4, 2, 1, 2)), lw=1.5, alpha=0.7,
               label="70% threshold (Spec §6.8 #1)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 3: noise-type classification accuracy vs SNR")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    fig.savefig(out)
    plt.close(fig)


def fig_basis_comparison(df: pd.DataFrame, out: Path) -> None:
    """Fig. 2: Bar chart — mean accuracy per (basis, type) for RF classifier (grayscale-safe)."""
    rf = df[df["classifier"] == "rf"]
    pivot = rf.groupby(["basis", "type"])["accuracy"].mean().unstack("type")
    # Reorder columns for consistent legend
    col_order = [c for c in ["baseline", "dsge", "hybrid"] if c in pivot.columns]
    pivot = pivot[col_order]
    col_labels = {"baseline": "Spectral-only", "dsge": "DSGE-only", "hybrid": "Hybrid"}

    n_groups = len(pivot)
    n_bars = len(col_order)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, col in enumerate(col_order):
        ax.bar(
            x + j * width - (n_bars - 1) * width / 2,
            pivot[col],
            width=width * 0.9,
            label=col_labels.get(col, col),
            color=GRAYS[j],
            hatch=HATCHES[j],
            edgecolor="black",
            linewidth=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_xlabel("Basis")
    ax.set_ylabel("Mean accuracy across SNR")
    ax.set_title("Exp 3: basis comparison (RF classifier)")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.70, color="0.0", linestyle="--", lw=1.2, alpha=0.7,
               label="70% threshold")
    ax.legend(title="Feature set")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out)
    plt.close(fig)


def fig_classifier_comparison(df: pd.DataFrame, out: Path) -> None:
    """Best DSGE+hybrid by classifier (grayscale-safe)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    clf_list = ["lr", "rf", "svm_rbf"]
    clf_labels = {"lr": "LR", "rf": "RF", "svm_rbf": "SVM-RBF"}
    type_list = ["dsge", "hybrid"]
    type_labels = {"dsge": "DSGE-only", "hybrid": "Hybrid"}
    idx = 0
    for typ in type_list:
        sub = df[df["type"] == typ]
        for clf in clf_list:
            sub_clf = sub[sub["classifier"] == clf]
            if sub_clf.empty:
                continue
            best = sub_clf.groupby("snr_db")["accuracy"].max()
            ax.plot(
                best.index, best.values,
                marker=MARKERS[idx % len(MARKERS)],
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                color=GRAYS[idx % len(GRAYS)],
                lw=1.5, ms=7,
                label=f"{type_labels[typ]}-{clf_labels[clf]}",
            )
            idx += 1
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (best basis)")
    ax.set_title("Exp 3: classifier × type comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_patp_alpha_sweep(patp: pd.DataFrame, out: Path) -> None:
    """Fig. 3: Accuracy vs α for each SNR (grayscale-safe — distinct markers + linestyles)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    snr_values = sorted(patp["snr_db"].unique())
    for i, snr in enumerate(snr_values):
        grp = patp[patp["snr_db"] == snr].sort_values("alpha")
        ax.plot(
            grp["alpha"], grp["accuracy"],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            color=GRAYS[i % len(GRAYS)],
            lw=2, ms=7,
            label=f"SNR={snr:+.0f} dB",
        )
    ax.axvline(0.5, color="0.5", linestyle=":", lw=1.5, alpha=0.8,
               label="α=0.5 (degenerate linear)")
    ax.axvline(0.6, color="0.0", linestyle="--", lw=1.5, alpha=0.8,
               label="α*=0.6 (optimal)")
    ax.set_xlabel("α (PATP parameter)")
    ax.set_ylabel("DSGE+LR accuracy")
    ax.set_title("Exp 3: PATP α-sweep — α*=0.6 across all SNR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_alpha_optimal(patp: pd.DataFrame, out: Path) -> None:
    """Diagnostic plot: α* per SNR (grayscale-safe)."""
    rows = []
    for snr, grp in patp.groupby("snr_db"):
        best = grp.loc[grp["accuracy"].idxmax()]
        rows.append({"snr_db": snr, "alpha_opt": best["alpha"],
                      "accuracy": best["accuracy"]})
    df = pd.DataFrame(rows).sort_values("snr_db")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["snr_db"], df["alpha_opt"],
            marker="D", linestyle="-", lw=2, ms=12,
            color="0.0", label="α* (optimal)")
    ax.axhspan(0.05, 0.95, color="0.85", alpha=0.5, label="strictly interior region")
    ax.axhline(0.5, color="0.5", linestyle=":", lw=1.5, alpha=0.8,
               label="α=0.5 (degenerate)")
    ax.axhline(0.6, color="0.0", linestyle="--", lw=1.5, alpha=0.8,
               label="α*=0.6 reference")
    for _, r in df.iterrows():
        ax.annotate(f"acc={r['accuracy']:.3f}", xy=(r["snr_db"], r["alpha_opt"]),
                     xytext=(8, 12), textcoords="offset points", fontsize=9)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("α* (DSGE PATP)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Exp 3 diagnostic: selected PATP α by SNR")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_dsge_vs_spectral_scatter(df: pd.DataFrame, out: Path) -> None:
    """Fig. 1: DSGE acc vs Spectral acc — grayscale-safe (markers per SNR, no colormap)."""
    base = df[df["type"] == "baseline"][["snr_db", "classifier", "accuracy"]]
    base = base.rename(columns={"accuracy": "spectral_acc"})
    dsge = df[df["type"] == "dsge"]
    merged = dsge.merge(base, on=["snr_db", "classifier"])

    snr_values = sorted(merged["snr_db"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, snr in enumerate(snr_values):
        sub = merged[merged["snr_db"] == snr]
        ax.scatter(
            sub["spectral_acc"], sub["accuracy"],
            marker=MARKERS[i % len(MARKERS)],
            color=GRAYS[i % len(GRAYS)],
            s=70, alpha=0.85,
            label=f"SNR={snr:+.0f} dB",
            edgecolors="black", linewidths=0.5,
        )
    diag = np.linspace(0.5, 1.0, 100)
    ax.plot(diag, diag, "k--", alpha=0.5, lw=1.5, label="DSGE = Spectral")
    ax.set_xlabel("Spectral baseline accuracy")
    ax.set_ylabel("DSGE accuracy")
    ax.set_title("Exp 3: DSGE-only vs spectral-only baseline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def main(results_dir: Path) -> None:
    df = pd.read_csv(results_dir / "exp3_basis_comparison.csv")
    patp = pd.read_csv(results_dir / "exp3_patp_alpha_sweep.csv")
    fig_dir = results_dir.parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_accuracy_vs_snr(df, fig_dir / "fig3_accuracy_vs_snr.pdf")
    fig_basis_comparison(df, fig_dir / "fig3_basis_comparison_bar.pdf")
    fig_classifier_comparison(df, fig_dir / "fig3_classifier_comparison.pdf")
    fig_patp_alpha_sweep(patp, fig_dir / "fig3_patp_alpha_sweep.pdf")
    fig_alpha_optimal(patp, fig_dir / "fig3_alpha_optimal_per_snr.pdf")
    fig_dsge_vs_spectral_scatter(df, fig_dir / "fig3_dsge_vs_spectral.pdf")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "experiments" / "exp3_dsge_classification" / "results")
    args = p.parse_args()
    main(args.results_dir)
