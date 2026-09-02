"""Generate Exp 1A figures from results CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def fig_RE_vs_kurtosis(df: pd.DataFrame, out: Path) -> None:
    """Per-block RE = (A_OLS - A_true)² / (A_PMM3 - A_true)² vs γ₄."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df[df["pmm3_converged"]].copy()
    err_ols = (sub["A_OLS"] - sub["A_true"]) ** 2
    err_pmm3 = (sub["A_PMM3"] - sub["A_true"]) ** 2
    re = err_ols / err_pmm3.replace(0, np.nan)
    sc = ax.scatter(sub["gamma4_xi"], re, c=sub["snr_db"], cmap="viridis",
                     s=8, alpha=0.5)
    plt.colorbar(sc, label="SNR (dB)")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="RE=1 (no gain)")
    ax.axvline(-0.7, color="grey", linestyle=":", label="γ₄=−0.7 dispatch threshold")
    ax.set_xlabel("γ̂₄ (per-block estimate)")
    ax.set_ylabel("RE = (A_OLS − A_true)² / (A_PMM3 − A_true)²")
    ax.set_yscale("log")
    ax.set_xlim(-2, 6)
    ax.set_ylim(1e-3, 1e3)
    ax.set_title("Exp 1A: per-block PMM3 efficiency vs kurtosis")
    ax.legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)


def fig_g_emp_vs_theory(df: pd.DataFrame, out: Path) -> None:
    """Aggregate g_emp by γ₄-bin, compare to g_theory."""
    sub = df[df["pmm3_converged"]].copy()
    bins = np.arange(-2, 4.01, 0.5)
    sub["g4_bin"] = pd.cut(sub["gamma4_xi"], bins)
    rows = []
    for b, grp in sub.groupby("g4_bin", observed=True):
        if len(grp) < 20:
            continue
        var_ols = float(np.var(grp["A_OLS"] - grp["A_true"], ddof=1))
        var_pmm3 = float(np.var(grp["A_PMM3"] - grp["A_true"], ddof=1))
        if var_ols <= 0 or var_pmm3 <= 0:
            continue
        g_emp = var_pmm3 / var_ols
        rows.append({
            "g4_mid": b.mid,
            "g_emp_pmm3": g_emp,
            "g_theory_mean": grp["g3_theory"].mean(),
            "n": len(grp),
        })
    out_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(out_df["g4_mid"], out_df["g_emp_pmm3"], "o-", label="empirical g₃")
    ax.plot(out_df["g4_mid"], out_df["g_theory_mean"], "s--", label="theoretical g₃")
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("γ₄ bin midpoint")
    ax.set_ylabel("g₃ (variance reduction factor)")
    ax.set_title("Exp 1A: empirical vs theoretical g₃ for PMM3")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    return out_df


def fig_coverage(df: pd.DataFrame, out: Path) -> None:
    """Coverage vs SNR for OLS / PMM2 / PMM3 / AUTO."""
    rows = []
    for snr, grp in df.groupby("snr_db"):
        for m in ["OLS", "PMM2", "PMM3", "AUTO"]:
            mask = grp[f"A_{m}"].notna() & grp[f"SE_{m}"].notna()
            if not mask.any():
                continue
            a_hat = grp.loc[mask, f"A_{m}"].to_numpy()
            a_true = grp.loc[mask, "A_true"].to_numpy()
            se = grp.loc[mask, f"SE_{m}"].to_numpy()
            cov = float(np.mean((a_true >= a_hat - 1.96 * se) &
                                  (a_true <= a_hat + 1.96 * se)))
            rows.append({"snr_db": snr, "method": m, "coverage": cov})
    cov_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ["OLS", "PMM2", "PMM3", "AUTO"]:
        sub = cov_df[cov_df["method"] == m]
        ax.plot(sub["snr_db"], sub["coverage"], "o-", label=m)
    ax.axhspan(0.93, 0.97, color="green", alpha=0.15, label="Spec target")
    ax.axhline(0.95, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Coverage(95%)")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Exp 1A: coverage vs SNR")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


def fig_method_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["method_auto"].value_counts()
    ax.bar(counts.index, counts.values, color=["#3274a1", "#e1812c", "#3a923a"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v}\n({100*v/len(df):.1f}%)",
                 ha="center", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title(f"Exp 1A: pmm_dispatch method choice (n={len(df)})")
    fig.savefig(out)
    plt.close(fig)


def fig_mse_by_snr(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for snr, grp in df.groupby("snr_db"):
        for m in ["OLS", "PMM2", "PMM3", "AUTO"]:
            mse = float(((grp[f"A_{m}"] - grp["A_true"]) ** 2).mean())
            rows.append({"snr_db": snr, "method": m, "MSE": mse})
    mse_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ["OLS", "PMM2", "PMM3", "AUTO"]:
        sub = mse_df[mse_df["method"] == m]
        ax.plot(sub["snr_db"], sub["MSE"], "o-", label=m)
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE(Â)")
    ax.set_title("Exp 1A: MSE vs SNR")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


def main(csv_path: Path | None = None, fig_dir: Path | None = None) -> None:
    if csv_path is None:
        csv_path = PROJECT_ROOT / "experiments" / "exp1_pmm_estimation" / "results" / "exp1A_amplitude.csv"
    df = pd.read_csv(csv_path)
    if fig_dir is None:
        # results -> figures, results_realnoise -> figures_realnoise, etc.
        results_name = csv_path.parent.name
        fig_name = results_name.replace("results", "figures")
        fig_dir = csv_path.parent.parent / fig_name
    fig_dir.mkdir(exist_ok=True)
    fig_RE_vs_kurtosis(df, fig_dir / "fig1A_RE_vs_kurtosis.pdf")
    g_df = fig_g_emp_vs_theory(df, fig_dir / "fig1A_g_emp_vs_theory.pdf")
    fig_coverage(df, fig_dir / "fig1A_coverage.pdf")
    fig_method_distribution(df, fig_dir / "fig1A_method_distribution.pdf")
    fig_mse_by_snr(df, fig_dir / "fig1A_mse_vs_snr.pdf")
    print(f"Figures written to {fig_dir}")
    if g_df is not None:
        print("\nEmpirical vs theoretical g₃ by γ₄ bin:")
        print(g_df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--fig-dir", type=Path, default=None)
    args = p.parse_args()
    main(args.csv, args.fig_dir)
