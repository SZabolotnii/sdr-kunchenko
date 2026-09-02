"""Generate Exp 2 figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def fig_auc_vs_snr(per_det: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for det, grp in per_det.groupby("detector"):
        grp = grp.sort_values("snr_db")
        ax.plot(grp["snr_db"], grp["AUC"], "o-", label=det)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Exp 2: AUC vs SNR per detector")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_add_vs_snr(per_det: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for det, grp in per_det.groupby("detector"):
        grp = grp.sort_values("snr_db")
        ax.plot(grp["snr_db"], grp["ADD"], "o-", label=det)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("ADD (samples)")
    ax.set_yscale("log")
    ax.set_title("Exp 2: detection delay vs SNR")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_J_convergence(per_det: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    poly = per_det[per_det["detector"].str.startswith("gsa_poly_")].copy()
    poly["s"] = poly["detector"].str.extract(r"_s(\d+)").astype(int)
    for snr, grp in poly.groupby("snr_db"):
        grp = grp.sort_values("s")
        ax.plot(grp["s"], grp["J"], "o-", label=f"SNR={snr:+.0f} dB")
    ax.set_xlabel("polynomial degree s")
    ax.set_ylabel("J(s) — information functional")
    ax.set_title("Exp 2: J(s) monotonicity (Theorem 2b)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_patp_alpha_sweep(alpha_sw: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for snr, grp in alpha_sw.groupby("snr_db"):
        grp = grp.sort_values("alpha")
        axes[0].plot(grp["alpha"], grp["AUC"], "o-", label=f"{snr:+.0f} dB")
        axes[1].plot(grp["alpha"], grp["ADD"], "o-", label=f"{snr:+.0f} dB")
    axes[0].set_xlabel("α (PATP)")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("PATP α-sweep: AUC")
    axes[0].axvline(0.5, color="grey", linestyle=":", alpha=0.5, label="α=0.5 (linear)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("α (PATP)")
    axes[1].set_ylabel("ADD (samples)")
    axes[1].set_yscale("log")
    axes[1].set_title("PATP α-sweep: ADD")
    axes[1].axvline(0.5, color="grey", linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_alpha_optimal_per_snr(alpha_sw: pd.DataFrame, out: Path) -> None:
    rows = []
    for snr, grp in alpha_sw.groupby("snr_db"):
        ok = grp[grp["AUC"] > 0.5]
        if len(ok) == 0:
            ok = grp
        best = ok.loc[ok["ADD"].idxmin()]
        rows.append({"snr_db": snr, "alpha_opt": best["alpha"],
                      "AUC_opt": best["AUC"], "ADD_opt": best["ADD"]})
    df = pd.DataFrame(rows).sort_values("snr_db")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["snr_db"], df["alpha_opt"], "o-", lw=2, ms=10)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("α* (optimal PATP)")
    ax.set_title("Exp 2: PATP α* per SNR (KEY plot for paper)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhspan(0.05, 0.95, color="green", alpha=0.1, label="strictly interior")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    for _, r in df.iterrows():
        ax.annotate(f"AUC={r['AUC_opt']:.3f}\nADD={r['ADD_opt']:.0f}",
                     xy=(r["snr_db"], r["alpha_opt"]),
                     xytext=(5, 10), textcoords="offset points", fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_far_compliance(per_det: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for det, grp in per_det.groupby("detector"):
        grp = grp.sort_values("snr_db")
        ax.plot(grp["snr_db"], grp["FAR"], "o-", label=det)
    ax.axhline(0.01, color="red", linestyle="--", label="ε=0.01 target")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FAR (per-block)")
    ax.set_title("Exp 2: empirical FAR per detector vs SNR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def main(results_dir: Path) -> None:
    per_det = pd.read_csv(results_dir / "exp2_per_detector.csv")
    alpha_sw = pd.read_csv(results_dir / "exp2_patp_alpha_sweep.csv")
    fig_dir = results_dir.parent / results_dir.name.replace("results", "figures")
    fig_dir.mkdir(exist_ok=True)
    fig_auc_vs_snr(per_det, fig_dir / "fig2_auc_vs_snr.pdf")
    fig_add_vs_snr(per_det, fig_dir / "fig2_add_vs_snr.pdf")
    fig_J_convergence(per_det, fig_dir / "fig2_J_convergence.pdf")
    fig_patp_alpha_sweep(alpha_sw, fig_dir / "fig2_patp_alpha_sweep.pdf")
    fig_alpha_optimal_per_snr(alpha_sw, fig_dir / "fig2_alpha_optimal_per_snr.pdf")
    fig_far_compliance(per_det, fig_dir / "fig2_far_compliance.pdf")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "experiments" / "exp2_gsa_detection" / "results")
    args = p.parse_args()
    main(args.results_dir)
