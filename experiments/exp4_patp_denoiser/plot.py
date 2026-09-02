"""Generate Exp 4 figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def fig_psnr_vs_snr(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, alias in zip(axes, ["primary", "realnoise", "generalization"]):
        sub = df[df["dataset_alias"] == alias]
        if sub.empty:
            ax.set_title(f"{alias} (no data)")
            continue
        for method, color, lw in [
            ("noisy", "lightgrey", 1),
            ("median_w5", "C3", 1.5),
            ("wiener", "C2", 1.5),
            ("projection_K", "C1", 1.5),
            ("fixed_patp_05", "C4", 1.5),
            ("adaptive_patp", "C0", 2.5),
        ]:
            grp = sub[sub["method"] == method].groupby("snr_db")["psnr"].mean().sort_index()
            ax.plot(grp.index, grp.values, "o-", label=method, color=color, lw=lw)
        ax.set_title(alias)
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PSNR (dB)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Exp 4: denoiser PSNR vs input SNR")
    fig.savefig(out)
    plt.close(fig)


def fig_alpha_mapping(mapping: dict, out: Path) -> None:
    """Show training γ̂₄ ↔ α* scatter + fitted polynomial."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, alias in zip(axes, ["primary", "realnoise"]):
        if alias not in mapping:
            continue
        abs_corrs = []
        for snr_db, m in mapping[alias].items():
            g4 = np.asarray(m["training_gamma4"])
            a = np.asarray(m["training_alpha"])
            abs_corrs.append(abs(float(m["correlation"])))
            ax.scatter(g4, a, alpha=0.3, s=8, label=f"SNR={snr_db}dB")
            poly = np.poly1d(np.asarray(m["poly_coefs"]))
            xs = np.linspace(g4.min(), g4.max(), 100)
            ax.plot(xs, np.clip(poly(xs), 0, 1), lw=1.5)
        if abs_corrs:
            ax.text(0.03, 0.95, f"max |corr|={max(abs_corrs):.3f}",
                    transform=ax.transAxes, va="top",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"})
        ax.set_xlabel("γ̂₄")
        ax.set_title(alias)
        ax.grid(alpha=0.3)
        ax.set_xlim(-2, 8)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("α* (per-block optimal)")
    axes[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Exp 4: α*(γ̂₄) calibration mapping (KEY plot)")
    fig.savefig(out)
    plt.close(fig)


def fig_alpha_distribution(mapping: dict, out: Path) -> None:
    """Distribution of calibration α* (per-block optimal)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    all_alpha = []
    for alias in ["primary", "realnoise"]:
        values = []
        for m in mapping.get(alias, {}).values():
            values.extend(m["training_alpha"])
        if values:
            a = np.asarray(values, dtype=float)
            all_alpha.extend(values)
            ax.hist(a, bins=21, alpha=0.5, label=alias, density=True)
    frac_at_one = float(np.mean(np.isclose(np.asarray(all_alpha, dtype=float), 1.0))) if all_alpha else np.nan
    ax.axvline(1.0, color="black", linestyle="--", lw=1.2)
    if np.isfinite(frac_at_one):
        ax.text(0.98, 0.95, f"α=1 mass ≈ {frac_at_one:.0%}",
                transform=ax.transAxes, ha="right", va="top",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    ax.set_xlabel("α* (calibration optimum)")
    ax.set_ylabel("density")
    ax.set_title("Exp 4: distribution of calibration α*")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_psnr_gain_over_baselines(df: pd.DataFrame, out: Path) -> None:
    """Per-SNR PSNR gain of adaptive over each baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, alias in zip(axes, ["primary", "realnoise", "generalization"]):
        sub = df[df["dataset_alias"] == alias]
        if sub.empty:
            continue
        adaptive = sub[sub["method"] == "adaptive_patp"].groupby("snr_db")["psnr"].mean()
        for baseline, color in [("noisy", "lightgrey"), ("median_w5", "C3"),
                                  ("wiener", "C2"), ("fixed_patp_05", "C4")]:
            base = sub[sub["method"] == baseline].groupby("snr_db")["psnr"].mean()
            common = adaptive.index.intersection(base.index)
            ax.plot(common, (adaptive.loc[common] - base.loc[common]).values, "o-",
                     label=f"vs {baseline}", color=color, lw=1.5)
        ax.axvspan(-5, 5, color="0.9", alpha=0.35, label="SNR≤5 dB" if alias == "primary" else None)
        ax.set_title(alias)
        ax.set_xlabel("SNR (dB)")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PSNR gain of adaptive PATP (dB)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Exp 4: PATP-adaptive gain over baselines")
    fig.savefig(out)
    plt.close(fig)


def fig_denoised_examples(df: pd.DataFrame, results_dir: Path, out: Path) -> None:
    """Visual: pick one block per (dataset, SNR), show clean / noisy / methods overlay.

    Skipped — would need raw signals from CSV (only metrics stored).
    Lightweight placeholder produced.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, "(time-domain examples not stored in CSV; available\n"
             "via custom run with --save-signals flag — future work)",
             ha="center", va="center", fontsize=10)
    ax.axis("off")
    fig.savefig(out)
    plt.close(fig)


def main(results_dir: Path) -> None:
    df = pd.read_csv(results_dir / "exp4_denoiser_metrics.csv")
    mapping = json.loads((results_dir / "exp4_alpha_mapping.json").read_text())
    fig_dir = results_dir.parent / results_dir.name.replace("results", "figures")
    fig_dir.mkdir(exist_ok=True)
    fig_psnr_vs_snr(df, fig_dir / "fig4_psnr_vs_snr.pdf")
    fig_alpha_mapping(mapping, fig_dir / "fig4_alpha_optimal_vs_gamma4.pdf")
    fig_alpha_distribution(mapping, fig_dir / "fig4_alpha_distribution.pdf")
    fig_psnr_gain_over_baselines(df, fig_dir / "fig4_psnr_gain.pdf")
    fig_denoised_examples(df, results_dir, fig_dir / "fig4_denoised_examples.pdf")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "experiments" / "exp4_patp_denoiser" / "results")
    args = p.parse_args()
    main(args.results_dir)
