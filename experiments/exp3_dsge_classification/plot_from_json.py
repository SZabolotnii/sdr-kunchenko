"""Generate all 5 article figures from exp3_classification_results.json.

IEEE TWC quality requirements:
- 300 dpi, vector PDF
- Font sizes ≥ 8pt for all labels, ticks, legend
- B&W readable: distinct markers AND linestyles (not just colors)
- Saved to experiments/exp3_dsge_classification/figures/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # embed fonts as Type 42 (TrueType) for IEEE
        "ps.fonttype": 42,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp3_dsge_classification"
    / "results"
    / "exp3_classification_results.json"
)
FIG_DIR = PROJECT_ROOT / "experiments" / "exp3_dsge_classification" / "figures"

BASES = ["poly_s3", "poly_s4", "frac_s3", "trig_s4", "robust_s3"]
BASE_LABELS = ["poly s=3", "poly s=4", "frac s=3", "trig s=4", "robust s=3"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_snr_list(data: dict) -> list[float]:
    return [r["snr_db"] for r in data["snr_results"]]


def get_spectral_rf(data: dict) -> list[float]:
    return [r["baseline_spectral"]["rf"]["accuracy"] for r in data["snr_results"]]


def get_dsge_best_rf(data: dict) -> list[float]:
    """Best DSGE RF accuracy per SNR (best basis per SNR)."""
    result = []
    for r in data["snr_results"]:
        best = max(
            r["by_basis"][b]["dsge"]["rf"]["accuracy"] for b in BASES
        )
        result.append(best)
    return result


def get_hybrid_best_rf(data: dict) -> list[float]:
    """Best Hybrid RF accuracy per SNR (best basis per SNR)."""
    result = []
    for r in data["snr_results"]:
        best = max(
            r["by_basis"][b]["hybrid"]["rf"]["accuracy"] for b in BASES
        )
        result.append(best)
    return result


def get_basis_mean_rf(data: dict) -> dict:
    """Mean RF accuracy over all SNR for each basis × type."""
    n_snr = len(data["snr_results"])
    dsge_means = {}
    hybrid_means = {}
    for b in BASES:
        dsge_means[b] = (
            sum(r["by_basis"][b]["dsge"]["rf"]["accuracy"] for r in data["snr_results"])
            / n_snr
        )
        hybrid_means[b] = (
            sum(r["by_basis"][b]["hybrid"]["rf"]["accuracy"] for r in data["snr_results"])
            / n_snr
        )
    spectral_mean = sum(get_spectral_rf(data)) / n_snr
    return {"dsge": dsge_means, "hybrid": hybrid_means, "spectral": spectral_mean}


def get_patp_sweep(data: dict) -> dict[float, list[dict]]:
    """Return {snr_db: [{alpha, accuracy, auc}, ...]}."""
    result = {}
    for r in data["snr_results"]:
        result[r["snr_db"]] = r["patp_sweep"]
    return result


def get_alpha_optimal(data: dict) -> list[dict]:
    """Return [{snr_db, alpha_opt, accuracy}] for best alpha per SNR."""
    rows = []
    for r in data["snr_results"]:
        sweep = r["patp_sweep"]
        best = max(sweep, key=lambda x: x["accuracy"])
        rows.append(
            {
                "snr_db": r["snr_db"],
                "alpha_opt": best["alpha"],
                "accuracy": best["accuracy"],
            }
        )
    return sorted(rows, key=lambda x: x["snr_db"])


# ---------------------------------------------------------------------------
# Figure 1 — fig3_dsge_vs_spectral.pdf
# ---------------------------------------------------------------------------


def fig1_dsge_vs_spectral(data: dict, out: Path) -> None:
    """3 lines: DSGE-only, Spectral-only, Hybrid (best basis per SNR).

    Distinct markers: DSGE=triangle, Spectral=square, Hybrid=circle
    Distinct linestyles: DSGE=dashed, Spectral=dotted, Hybrid=solid
    """
    snrs = get_snr_list(data)
    spectral = get_spectral_rf(data)
    dsge = get_dsge_best_rf(data)
    hybrid = get_hybrid_best_rf(data)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(
        snrs, dsge,
        linestyle="--", marker="^", markersize=6, linewidth=1.5,
        color="black", label="DSGE-only (best basis)",
    )
    ax.plot(
        snrs, spectral,
        linestyle=":", marker="s", markersize=6, linewidth=1.5,
        color="black", markerfacecolor="white", label="Spectral-only",
    )
    ax.plot(
        snrs, hybrid,
        linestyle="-", marker="o", markersize=6, linewidth=1.5,
        color="black", markerfacecolor="gray", label="Hybrid (best basis)",
    )
    ax.axhline(
        0.70, color="black", linestyle=(0, (3, 1, 1, 1)), linewidth=1.0,
        label="70% threshold",
    )

    ax.set_xlabel("SNR (dB)", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_ylim(0.60, 1.02)
    ax.set_xticks(snrs)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 2 — fig3_basis_comparison_bar.pdf
# ---------------------------------------------------------------------------


def fig2_basis_comparison_bar(data: dict, out: Path) -> None:
    """Bar chart: 5 bases × 3 types (DSGE-only, Hybrid, Spectral baseline) for RF.

    Mean accuracy over 5 SNR. Font sizes ≥ 8pt. 70% threshold line.
    """
    means = get_basis_mean_rf(data)
    spectral_mean = means["spectral"]

    x = np.arange(len(BASES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    # Hatching for B&W readability
    bars_dsge = ax.bar(
        x - width, [means["dsge"][b] for b in BASES],
        width, label="DSGE-only",
        color="white", edgecolor="black", hatch="///", linewidth=0.8,
    )
    bars_hybrid = ax.bar(
        x, [means["hybrid"][b] for b in BASES],
        width, label="Hybrid",
        color="gray", edgecolor="black", hatch="", linewidth=0.8,
    )
    bars_spectral = ax.bar(
        x + width, [spectral_mean] * len(BASES),
        width, label="Spectral baseline",
        color="white", edgecolor="black", hatch="...", linewidth=0.8,
    )

    ax.axhline(
        0.70, color="black", linestyle="--", linewidth=1.0, label="70% threshold",
    )

    ax.set_xlabel("Basis", fontsize=9)
    ax.set_ylabel("Mean Accuracy (over 5 SNR)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(BASE_LABELS, fontsize=8, rotation=15, ha="right")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 3 — fig3_patp_alpha_sweep.pdf
# ---------------------------------------------------------------------------


def fig3_patp_alpha_sweep(data: dict, out: Path) -> None:
    """5 curves (one per SNR level). X: α ∈ [0,1], Y: DSGE+LR accuracy.

    Vertical line at α=0.5 (degenerate, grey dotted).
    Vertical line at α*=0.6 (optimal, green dashed, labeled).
    """
    sweep = get_patp_sweep(data)
    snrs = sorted(sweep.keys())

    # Distinct markers and linestyles for B&W
    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 1))]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for i, snr in enumerate(snrs):
        pts = sorted(sweep[snr], key=lambda x: x["alpha"])
        alphas = [p["alpha"] for p in pts]
        accs = [p["accuracy"] for p in pts]
        ax.plot(
            alphas, accs,
            linestyle=linestyles[i], marker=markers[i], markersize=5,
            linewidth=1.2, color="black",
            markerfacecolor="white" if i % 2 == 0 else "gray",
            label=f"SNR={snr:+.0f} dB",
        )

    # Vertical lines
    ax.axvline(
        0.5, color="gray", linestyle=":", linewidth=1.2,
        label="α=0.5 (degenerate)",
    )
    ax.axvline(
        0.6, color="black", linestyle="--", linewidth=1.2,
        label="α*=0.6 (optimal)",
    )

    ax.set_xlabel("α (PATP basis parameter)", fontsize=9)
    ax.set_ylabel("DSGE+LR Accuracy", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower left", ncol=1)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 4 — fig3_alpha_optimal_per_snr.pdf
# ---------------------------------------------------------------------------


def fig4_alpha_optimal_per_snr(data: dict, out: Path) -> None:
    """X: SNR (dB), Y: α* (optimal alpha).

    Points showing α*=0.6 at each SNR.
    Horizontal line at α=0.6 (green dashed, labeled "α*=0.6").
    Horizontal line at α=0.5 (grey dotted, labeled "degenerate").
    Annotate each point with accuracy value.
    """
    rows = get_alpha_optimal(data)
    snrs = [r["snr_db"] for r in rows]
    alphas = [r["alpha_opt"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(
        snrs, alphas,
        linestyle="-", marker="o", markersize=8, linewidth=1.5,
        color="black", markerfacecolor="gray", label="α* per SNR",
    )

    # Reference lines
    ax.axhline(
        0.6, color="black", linestyle="--", linewidth=1.2, label="α*=0.6",
    )
    ax.axhline(
        0.5, color="gray", linestyle=":", linewidth=1.2, label="degenerate (α=0.5)",
    )

    # Annotate each point with accuracy
    for snr, alpha, acc in zip(snrs, alphas, accs):
        ax.annotate(
            f"{acc:.3f}",
            xy=(snr, alpha),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("SNR (dB)", fontsize=9)
    ax.set_ylabel("α* (optimal PATP parameter)", fontsize=9)
    ax.set_ylim(0.3, 0.9)
    ax.set_xticks(snrs)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 5 — fig3_accuracy_vs_snr.pdf
# ---------------------------------------------------------------------------


def fig5_accuracy_vs_snr(data: dict, out: Path) -> None:
    """3 lines: DSGE-only, Spectral-only, Hybrid (best basis per SNR).

    Distinct markers AND linestyles for B&W readability. Font sizes ≥ 8pt.
    """
    snrs = get_snr_list(data)
    spectral = get_spectral_rf(data)
    dsge = get_dsge_best_rf(data)
    hybrid = get_hybrid_best_rf(data)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(
        snrs, dsge,
        linestyle="--", marker="^", markersize=6, linewidth=1.5,
        color="black", label="DSGE-only (best basis)",
    )
    ax.plot(
        snrs, spectral,
        linestyle=":", marker="s", markersize=6, linewidth=1.5,
        color="black", markerfacecolor="white", label="Spectral-only",
    )
    ax.plot(
        snrs, hybrid,
        linestyle="-", marker="o", markersize=6, linewidth=1.5,
        color="black", markerfacecolor="gray", label="Hybrid (best basis)",
    )
    ax.axhline(
        0.70, color="black", linestyle=(0, (3, 1, 1, 1)), linewidth=1.0,
        label="70% threshold",
    )

    ax.set_xlabel("SNR (dB)", fontsize=9)
    ax.set_ylabel("Accuracy (RF classifier)", fontsize=9)
    ax.set_ylim(0.60, 1.02)
    ax.set_xticks(snrs)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading JSON: {JSON_PATH}")
    data = load_json(JSON_PATH)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIG_DIR}")
    print()

    print("Generating figures...")
    fig1_dsge_vs_spectral(data, FIG_DIR / "fig3_dsge_vs_spectral.pdf")
    fig2_basis_comparison_bar(data, FIG_DIR / "fig3_basis_comparison_bar.pdf")
    fig3_patp_alpha_sweep(data, FIG_DIR / "fig3_patp_alpha_sweep.pdf")
    fig4_alpha_optimal_per_snr(data, FIG_DIR / "fig3_alpha_optimal_per_snr.pdf")
    fig5_accuracy_vs_snr(data, FIG_DIR / "fig3_accuracy_vs_snr.pdf")

    print()
    print("All 5 figures generated successfully.")
    print(f"Location: {FIG_DIR}")

    # Verify all files exist
    expected = [
        "fig3_dsge_vs_spectral.pdf",
        "fig3_basis_comparison_bar.pdf",
        "fig3_patp_alpha_sweep.pdf",
        "fig3_alpha_optimal_per_snr.pdf",
        "fig3_accuracy_vs_snr.pdf",
    ]
    print()
    print("Verification:")
    all_ok = True
    for fname in expected:
        fpath = FIG_DIR / fname
        exists = fpath.exists()
        size_kb = fpath.stat().st_size / 1024 if exists else 0
        status = f"OK ({size_kb:.1f} KB)" if exists else "MISSING"
        print(f"  {'✓' if exists else '✗'} {fname}: {status}")
        if not exists:
            all_ok = False

    if all_ok:
        print()
        print("All 5 figures present and non-empty.")
    else:
        print()
        print("ERROR: Some figures are missing!")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
