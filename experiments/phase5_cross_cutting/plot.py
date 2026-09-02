"""Phase 5 figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from sdr_kunchenko.patp.cross_domain import compute_pareto, fit_alpha_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

TASK_COLORS = {
    "detection": "#3274a1",       # blue
    "classification": "#3a923a",  # green
    "denoising": "#e1812c",       # orange
}


def fig_alpha_vs_gamma4(unified: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for task, grp in unified.groupby("task_type"):
        ax.scatter(grp["gamma4"], grp["alpha_opt"],
                    s=80, c=TASK_COLORS[task], label=task,
                    edgecolor="black", linewidth=0.5, alpha=0.8)
    # Add reference lines
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, label="α=0.5 (linear)")
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5, label="α=1.0 (poly)")
    ax.axhline(0.0, color="grey", linestyle="--", alpha=0.5, label="α=0.0 (frac)")
    ax.set_xlabel("γ₄ (excess kurtosis prior of dataset)")
    ax.set_ylabel("α* (optimal PATP)")
    ax.set_title("Phase 5: α* vs γ₄ across tasks (KEY plot for Spec §8)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_unified_mapping_3d(unified: pd.DataFrame, out: Path) -> None:
    """3D surface of fitted α(γ₄, SNR) for one task (detection)."""
    fit = fit_alpha_mapping(unified, include_task_dummies=True)
    g4_grid = np.linspace(unified["gamma4"].min(), unified["gamma4"].max(), 30)
    snr_grid = np.linspace(unified["snr_db"].min(), unified["snr_db"].max(), 30)
    g4_mesh, snr_mesh = np.meshgrid(g4_grid, snr_grid)
    z_mesh = np.vectorize(lambda g4, snr: fit["predict"](0.0, g4, snr, "detection"))(
        g4_mesh, snr_mesh
    )
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(g4_mesh, snr_mesh, z_mesh, cmap="viridis", alpha=0.8)
    # Overlay scatter of actual data
    for task, grp in unified.groupby("task_type"):
        ax.scatter(grp["gamma4"], grp["snr_db"], grp["alpha_opt"],
                    c=TASK_COLORS[task], s=50, edgecolor="k",
                    linewidth=0.5, label=task, depthshade=True)
    ax.set_xlabel("γ₄")
    ax.set_ylabel("SNR (dB)")
    ax.set_zlabel("α*")
    ax.set_title("Phase 5: fitted α surface (detection task) + cross-task scatter")
    ax.legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_transferability(verification: dict, out: Path) -> None:
    tr = verification["transferability"]
    records = tr.get("per_snr_records", [])
    if not records:
        return
    df = pd.DataFrame(records).sort_values("snr_db")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["snr_db"], df["ratio"], width=1.5,
           color=["#3a923a" if r >= 0.9 else "#cc4f4f" for r in df["ratio"]],
           edgecolor="black")
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.7,
                label="Spec §8.3 #2 target (0.90)")
    ax.axhline(tr["mean_ratio"], color="blue", linestyle=":", alpha=0.7,
                label=f"mean = {tr['mean_ratio']:.3f}")
    for _, r in df.iterrows():
        ax.text(r["snr_db"], r["ratio"] + 0.02,
                 f"α_pred={r['alpha_predicted']:.2f}\nα_true={r['alpha_native']:.2f}",
                 ha="center", fontsize=7)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("AUC ratio (predicted α / native α)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Phase 5: transferability {tr.get('train_on')}→{tr.get('test_on')}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out)
    plt.close(fig)


def fig_unimodal_alpha_gamma4(verification: dict, out: Path) -> None:
    """2D view of the monotonicity check: α(γ₄) is unimodal, not monotone."""
    mono = verification["monotonicity"]
    gamma4 = np.asarray(mono["gamma4_grid"], dtype=float)
    alphas = np.asarray(mono["alphas"], dtype=float)
    peak_idx = int(np.argmax(alphas))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gamma4, alphas, color="#3274a1", marker="o", linewidth=2)
    ax.scatter(
        [gamma4[peak_idx]],
        [alphas[peak_idx]],
        s=140,
        color="#e1812c",
        edgecolor="black",
        zorder=3,
        label=f"peak γ₄={gamma4[peak_idx]:+.2f}, α={alphas[peak_idx]:.3f}",
    )
    ax.axvline(0.0, color="grey", linestyle=":", alpha=0.6)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="α=0.5 linear")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("γ₄ (excess kurtosis)")
    ax.set_ylabel("predicted α*")
    ax.set_title("Phase 5: unimodal α(γ₄), not monotone")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_pareto(unified: pd.DataFrame, out: Path) -> None:
    pareto = compute_pareto(unified)
    fig, ax = plt.subplots(figsize=(8, 5))
    for task, grp in pareto.groupby("task_type"):
        opt = grp[grp["pareto_optimal"]]
        non = grp[~grp["pareto_optimal"]]
        ax.scatter(non["alpha_distance_to_05"], non["metric_normalized"],
                    s=40, c=TASK_COLORS[task], alpha=0.3,
                    label=f"{task} (dominated)")
        ax.scatter(opt["alpha_distance_to_05"], opt["metric_normalized"],
                    s=120, c=TASK_COLORS[task], edgecolor="black",
                    linewidth=1.2, marker="*", label=f"{task} (Pareto)")
    ax.set_xlabel("|α* − 0.5|  (distance from linear regime)")
    ax.set_ylabel("normalised primary metric")
    ax.set_title("Phase 5: Pareto frontier per task (Spec §8.3 #3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_residuals_per_task(unified: pd.DataFrame, out: Path) -> None:
    fit = fit_alpha_mapping(unified, include_task_dummies=True)
    df = fit["fitted_df"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, task in zip(axes, ["detection", "classification", "denoising"], strict=True):
        sub = df[df["task_type"] == task]
        if sub.empty:
            ax.set_title(f"{task} (no data)")
            continue
        ax.scatter(sub["alpha_opt"], sub["residual"],
                    s=60, c=TASK_COLORS[task], edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_title(f"{task}\nresid std={sub['residual'].std():.3f}")
        ax.set_xlabel("α* (observed)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("residual = α_obs − α_pred")
    fig.suptitle("Phase 5: residual diagnostics (poly₂ + task dummies)")
    fig.savefig(out)
    plt.close(fig)


def main(results_dir: Path) -> None:
    unified = pd.read_csv(results_dir / "phase5_unified_alpha.csv")
    verification = json.loads((results_dir / "phase5_verification.json").read_text())
    fig_dir = results_dir.parent / results_dir.name.replace("results", "figures")
    fig_dir.mkdir(exist_ok=True)
    fig_alpha_vs_gamma4(unified, fig_dir / "fig5_alpha_vs_gamma4_per_task.pdf")
    fig_unified_mapping_3d(unified, fig_dir / "fig5_unified_mapping_3d.pdf")
    fig_transferability(verification, fig_dir / "fig5_transferability_ratios.pdf")
    fig_transferability(verification, fig_dir / "fig5_transferability.pdf")
    fig_unimodal_alpha_gamma4(verification, fig_dir / "fig5_unimodal_alpha_gamma4.pdf")
    fig_pareto(unified, fig_dir / "fig5_pareto_frontier.pdf")
    fig_residuals_per_task(unified, fig_dir / "fig5_residuals_per_experiment.pdf")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "experiments" / "phase5_cross_cutting" / "results")
    args = p.parse_args()
    main(args.results_dir)
