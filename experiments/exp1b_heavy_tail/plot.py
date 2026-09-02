"""Generate exp1b figures from results CSV.

Tasks:
  8.1 Score functions ψ_α(ε) for α ∈ {0.1, 0.3, 0.5, 1.0, 2.0}
  8.2 RE(PMM_FRAC) vs α_s with theoretical curve g_frac(α_opt)
  8.3 α_opt vs γ̂₄ (monotone relationship)
  8.4 MSE vs SNR for all methods on α-stable(1.5) dataset
  8.5 RE table: rows=methods, columns=distributions
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from scipy import integrate, stats as scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    try:
        plt.style.use("seaborn-paper")
    except OSError:
        pass  # fall back to default

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------------------------------------------------------------------------
# Colour / line-style cycle (consistent across figures)
# ---------------------------------------------------------------------------
METHOD_STYLES: dict[str, dict] = {
    "OLS":      {"color": "#3274a1", "linestyle": "-",  "marker": "o"},
    "PMM2":     {"color": "#e1812c", "linestyle": "--", "marker": "s"},
    "PMM3":     {"color": "#3a923a", "linestyle": "-.", "marker": "^"},
    "PMM_FRAC": {"color": "#c03d3e", "linestyle": ":",  "marker": "D"},
    "AUTO_EXT": {"color": "#9372b2", "linestyle": "-",  "marker": "v"},
}

ALPHA_COLORS = {
    0.1: "#d62728",
    0.3: "#ff7f0e",
    0.5: "#2ca02c",
    1.0: "#1f77b4",
    2.0: "#9467bd",
}


# ---------------------------------------------------------------------------
# Helper: theoretical g_frac(α) for standard normal ε
# ---------------------------------------------------------------------------

def _normal_moment(p: float) -> float:
    """E[|Z|^p] for Z ~ N(0,1), p > -1."""
    if _SCIPY_AVAILABLE:
        # Use the half-normal formula: E[|Z|^p] = 2^(p/2) * Γ((p+1)/2) / Γ(1/2)
        from scipy.special import gamma as _gamma
        return (2.0 ** (p / 2.0)) * _gamma((p + 1.0) / 2.0) / _gamma(0.5)
    else:
        # Numerical fallback via Monte Carlo (reproducible)
        rng = np.random.default_rng(0)
        z = rng.standard_normal(200_000)
        return float(np.mean(np.abs(z) ** p))


def g_frac_normal(alpha: float) -> float:
    """
    Theoretical efficiency coefficient for PMM_FRAC vs OLS under N(0,1) noise.

    g_frac(α) = E[|ε|^(2α)] / (α² · (E[|ε|^(α-1)])²)

    g_frac < 1  ⟹  PMM_FRAC is more efficient than OLS.
    For α = 1 (OLS): g_frac = 1.
    """
    if alpha <= 0:
        return np.nan
    e_2alpha = _normal_moment(2.0 * alpha)
    e_alpha_m1 = _normal_moment(alpha - 1.0)
    denom = (alpha ** 2) * (e_alpha_m1 ** 2)
    if denom <= 0:
        return np.nan
    return e_2alpha / denom


def re_theory_normal(alpha_s: float) -> float:
    """
    Theoretical RE = 1 / g_frac(α_opt) for α-stable(α_s) noise,
    approximated by N(0,1) moments (valid for α_s = 2.0 exactly).

    For α_s = 2.0 (Gaussian): RE_theory = 1.0.
    For other α_s: α_opt = α_s - 1, RE_theory = 1 / g_frac(α_opt).
    """
    if alpha_s == 2.0:
        return 1.0
    alpha_opt = max(0.1, alpha_s - 1.0)
    g = g_frac_normal(alpha_opt)
    if g is None or np.isnan(g) or g <= 0:
        return np.nan
    return 1.0 / g


# ---------------------------------------------------------------------------
# Figure 8.1 — Score functions ψ_α(ε)
# ---------------------------------------------------------------------------

def fig_score_functions(out: Path) -> None:
    """
    Task 8.1: Plot ψ_α(ε) = sign(ε)|ε|^α for α ∈ {0.1, 0.3, 0.5, 1.0, 2.0}.

    Validates: Requirements 9.6 (score function figure)
    """
    alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
    eps = np.linspace(-3.0, 3.0, 600)

    fig, ax = plt.subplots(figsize=(7, 5))

    for alpha in alphas:
        psi = np.sign(eps) * np.abs(eps) ** alpha
        color = ALPHA_COLORS.get(alpha, None)
        ax.plot(eps, psi, color=color, linewidth=1.8,
                label=f"α = {alpha}")

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("ε")
    ax.set_ylabel(r"$\psi_\alpha(\varepsilon) = \mathrm{sign}(\varepsilon)\,|\varepsilon|^\alpha$")
    ax.set_title(r"Score functions $\psi_\alpha(\varepsilon)$ for various $\alpha$")
    ax.legend(title="α value", loc="upper left")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    fig.savefig(out)
    plt.close(fig)
    print(f"[8.1] Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 8.2 — RE(PMM_FRAC) vs α_s with theoretical curve
# ---------------------------------------------------------------------------

def fig_re_vs_alpha_s(df, out: Path) -> None:
    """
    Task 8.2: RE(PMM_FRAC) vs α_s with theoretical curve g_frac(α_opt).

    Validates: Requirements 9.2 (RE vs alpha_s figure)
    """
    alpha_s_values = [1.2, 1.5, 1.8, 2.0]

    # Theoretical curve (dense)
    alpha_s_dense = np.linspace(1.0, 2.0, 200)
    re_theory = np.array([re_theory_normal(a) for a in alpha_s_dense])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Theoretical curve
    ax.plot(alpha_s_dense, re_theory, color="#1f77b4", linewidth=2.0,
            linestyle="--", label=r"Theory: $1/g_\mathrm{frac}(\alpha_\mathrm{opt})$")

    # Empirical points from summary CSV
    if df is not None:
        # Filter PMM_FRAC rows
        frac_df = df[df["method"].str.upper() == "PMM_FRAC"].copy()

        # Map dataset names to α_s values
        dataset_to_alpha_s = {
            "alphastable_1p2": 1.2,
            "alphastable_1p5": 1.5,
            "alphastable_1p8": 1.8,
            "alphastable_2p0": 2.0,
        }

        emp_points = []
        for ds_name, a_s in dataset_to_alpha_s.items():
            sub = frac_df[frac_df["dataset"] == ds_name]
            if sub.empty:
                continue
            # Average RE over SNR levels
            if "RE" in sub.columns:
                re_mean = sub["RE"].mean()
            elif "MSE" in sub.columns and "MSE_OLS" in sub.columns:
                re_mean = (sub["MSE_OLS"] / sub["MSE"]).mean()
            else:
                continue
            emp_points.append((a_s, re_mean))

        if emp_points:
            xs, ys = zip(*sorted(emp_points))
            ax.scatter(xs, ys, color="#d62728", s=80, zorder=5,
                       label="Empirical (avg over SNR)")
            ax.plot(xs, ys, color="#d62728", linewidth=1.2, linestyle=":",
                    alpha=0.7)

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.6,
               label="RE = 1 (OLS baseline)")
    ax.set_xlabel(r"$\alpha_s$ (stability index)")
    ax.set_ylabel(r"RE = MSE(OLS) / MSE(PMM\_FRAC)")
    ax.set_title(r"RE(PMM\_FRAC) vs $\alpha_s$ with theoretical curve")
    ax.set_xticks(alpha_s_values)
    ax.legend()

    fig.savefig(out)
    plt.close(fig)
    print(f"[8.2] Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 8.3 — α_opt vs γ̂₄
# ---------------------------------------------------------------------------

def fig_alpha_opt_vs_gamma4(out: Path) -> None:
    """
    Task 8.3: α_opt vs γ̂₄ — monotone decreasing relationship.

    α_opt = max(0.1, 1 / (1 + 0.3·γ₄))

    Validates: Requirements 9.3 (alpha_opt vs gamma4 figure)
    """
    gamma4 = np.linspace(0.0, 30.0, 500)
    alpha_opt = np.maximum(0.1, 1.0 / (1.0 + 0.3 * gamma4))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(gamma4, alpha_opt, color="#1f77b4", linewidth=2.0,
            label=r"$\alpha_\mathrm{opt} = \max(0.1,\; 1/(1+0.3\hat{\gamma}_4))$")

    # Mark Student-t(3): γ₄ = 6 (theoretical excess kurtosis for t(3) is ∞,
    # but empirically with finite samples it's large; use the design doc value)
    # Design doc says: Student-t(3): γ₄=6, α_opt≈0.36
    g4_t3 = 6.0
    a_t3 = max(0.1, 1.0 / (1.0 + 0.3 * g4_t3))
    ax.scatter([g4_t3], [a_t3], color="#d62728", s=100, zorder=5,
               label=rf"Student-$t$(3): $\hat{{\gamma}}_4={g4_t3}$, $\alpha_\mathrm{{opt}}\approx{a_t3:.2f}$")
    ax.annotate(rf"$t(3)$: $\alpha_\mathrm{{opt}}\approx{a_t3:.2f}$",
                xy=(g4_t3, a_t3), xytext=(g4_t3 + 1.5, a_t3 + 0.08),
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                color="#d62728", fontsize=9)

    # Mark Student-t(5): γ₄ = 1 (theoretical excess kurtosis for t(5) is 6/(5-4)=6,
    # but design doc says γ₄=1 for t(5) — using design doc value)
    g4_t5 = 1.0
    a_t5 = max(0.1, 1.0 / (1.0 + 0.3 * g4_t5))
    ax.scatter([g4_t5], [a_t5], color="#ff7f0e", s=100, zorder=5,
               label=rf"Student-$t$(5): $\hat{{\gamma}}_4={g4_t5}$, $\alpha_\mathrm{{opt}}\approx{a_t5:.2f}$")
    ax.annotate(rf"$t(5)$: $\alpha_\mathrm{{opt}}\approx{a_t5:.2f}$",
                xy=(g4_t5, a_t5), xytext=(g4_t5 + 2.0, a_t5 + 0.10),
                arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
                color="#ff7f0e", fontsize=9)

    ax.axhline(0.1, color="grey", linewidth=0.8, linestyle=":",
               alpha=0.6, label=r"$\alpha_\mathrm{min} = 0.1$ (clipping)")
    ax.set_xlabel(r"$\hat{\gamma}_4$ (excess kurtosis)")
    ax.set_ylabel(r"$\alpha_\mathrm{opt}$")
    ax.set_title(r"Optimal $\alpha$ vs excess kurtosis $\hat{\gamma}_4$")
    ax.set_xlim(0, 30)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right")

    fig.savefig(out)
    plt.close(fig)
    print(f"[8.3] Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 8.4 — MSE vs SNR for all methods on α-stable(1.5)
# ---------------------------------------------------------------------------

def fig_mse_vs_snr_alphastable(df, out: Path,
                                dataset_name: str = "alphastable_1p5") -> None:
    """
    Task 8.4: MSE vs SNR for OLS, PMM2, PMM3, PMM_FRAC, AUTO_EXT
    on the α-stable(1.5) dataset.

    Validates: Requirements 9.4 (MSE vs SNR figure)
    """
    sub = df[df["dataset"] == dataset_name].copy()
    if sub.empty:
        warnings.warn(f"[8.4] Dataset '{dataset_name}' not found in summary CSV — skipping.")
        return

    methods = ["OLS", "PMM2", "PMM3", "PMM_FRAC", "AUTO_EXT"]
    rows = []
    for snr, grp in sub.groupby("snr_db"):
        for m in methods:
            col = f"MSE_{m}" if f"MSE_{m}" in grp.columns else None
            if col is None and "method" in grp.columns and "MSE" in grp.columns:
                mval = grp[grp["method"].str.upper() == m]["MSE"].mean()
            elif col is not None:
                mval = grp[col].mean()
            else:
                continue
            if not np.isnan(mval):
                rows.append({"snr_db": snr, "method": m, "MSE": mval})

    if not rows:
        warnings.warn(f"[8.4] No MSE data found for dataset '{dataset_name}' — skipping.")
        return

    import pandas as _pd
    mse_df = _pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        msub = mse_df[mse_df["method"] == m]
        if msub.empty:
            continue
        style = METHOD_STYLES.get(m, {})
        ax.plot(msub["snr_db"], msub["MSE"],
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker", "o"),
                linewidth=1.8, markersize=6,
                label=m)

    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE(Â)")
    ax.set_title(r"MSE vs SNR — $\alpha$-stable($\alpha_s=1.5$) noise")
    ax.legend()

    fig.savefig(out)
    plt.close(fig)
    print(f"[8.4] Saved: {out}")


# ---------------------------------------------------------------------------
# Table 8.5 — RE summary table
# ---------------------------------------------------------------------------

def table_re_summary(df, txt_out: Path, csv_out: Path) -> None:
    """
    Task 8.5: RE table — rows=methods, columns=distributions.

    Pivot: rows = methods, columns = datasets, values = mean RE (avg over SNR).

    Validates: Requirements 9.1 (RE table)
    """
    import pandas as _pd

    methods = ["OLS", "PMM2", "PMM3", "PMM_FRAC", "AUTO_EXT"]

    # Compute RE per (method, dataset, snr_db) then average over SNR
    if "RE" in df.columns:
        re_col = "RE"
    elif "MSE" in df.columns and "MSE_OLS" in df.columns:
        df = df.copy()
        df["RE"] = df["MSE_OLS"] / df["MSE"]
        re_col = "RE"
    else:
        warnings.warn("[8.5] Cannot compute RE: missing RE or MSE/MSE_OLS columns — skipping.")
        return

    # Filter to known methods
    df_filt = df[df["method"].str.upper().isin(methods)].copy()
    df_filt["method"] = df_filt["method"].str.upper()

    # Average RE over SNR levels for each (method, dataset)
    agg = (df_filt
           .groupby(["method", "dataset"])[re_col]
           .mean()
           .reset_index()
           .rename(columns={re_col: "RE_mean"}))

    # Pivot
    pivot = agg.pivot(index="method", columns="dataset", values="RE_mean")

    # Reorder rows to canonical method order
    present_methods = [m for m in methods if m in pivot.index]
    pivot = pivot.reindex(present_methods)

    # Format for display
    pivot_fmt = pivot.applymap(lambda x: f"{x:.3f}" if not np.isnan(x) else "—")

    # Print to stdout
    print("\n" + "=" * 60)
    print("RE Summary Table (mean over SNR levels)")
    print("RE = MSE(OLS) / MSE(method)  [>1 means better than OLS]")
    print("=" * 60)
    print(pivot_fmt.to_string())
    print("=" * 60 + "\n")

    # Save plain text
    txt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("RE Summary Table (mean over SNR levels)\n")
        f.write("RE = MSE(OLS) / MSE(method)  [>1 means better than OLS]\n")
        f.write("=" * 60 + "\n")
        f.write(pivot_fmt.to_string())
        f.write("\n")
    print(f"[8.5] Saved text table: {txt_out}")

    # Save CSV
    pivot.to_csv(csv_out)
    print(f"[8.5] Saved CSV table: {csv_out}")


# ---------------------------------------------------------------------------
# Auto-detect summary CSV
# ---------------------------------------------------------------------------

def _find_summary_csv(results_dir: Path) -> Path | None:
    """Search for a summary CSV in results_dir."""
    candidates = [
        results_dir / "summary.csv",
        results_dir / "exp1b_summary.csv",
    ]
    # Also look for any *summary*.csv
    candidates += sorted(results_dir.glob("*summary*.csv"))
    # And any *exp1b*.csv
    candidates += sorted(results_dir.glob("*exp1b*.csv"))
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exp1b figures and tables."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "exp1b_heavy_tail" / "results",
        help="Directory containing results CSV files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "exp1b_heavy_tail" / "figures",
        help="Output directory for figures and tables.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Path to summary CSV (auto-detected from results-dir if not given).",
    )
    args = parser.parse_args()

    fig_dir: Path = args.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figures that require NO data (always generated)
    # ------------------------------------------------------------------

    # 8.1 — Score functions
    fig_score_functions(fig_dir / "fig_score_functions.pdf")

    # 8.3 — α_opt vs γ̂₄
    fig_alpha_opt_vs_gamma4(fig_dir / "fig_alpha_opt_vs_gamma4.pdf")

    # ------------------------------------------------------------------
    # Figures that require summary CSV (graceful skip if missing)
    # ------------------------------------------------------------------

    if not _PANDAS_AVAILABLE:
        warnings.warn("pandas not available — skipping data-dependent figures (8.2, 8.4, 8.5).")
        print("Figures 8.1 and 8.3 written to", fig_dir)
        return

    # Locate summary CSV
    csv_path: Path | None = args.summary_csv
    if csv_path is None:
        csv_path = _find_summary_csv(args.results_dir)

    if csv_path is None or not csv_path.exists():
        warnings.warn(
            f"[WARNING] Summary CSV not found in {args.results_dir} — "
            "skipping data-dependent figures (8.2, 8.4, 8.5)."
        )
        print("Figures 8.1 and 8.3 written to", fig_dir)
        return

    import pandas as pd
    print(f"Loading summary CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalise column names to lower-case for robustness
    df.columns = [c.strip() for c in df.columns]

    # 8.2 — RE vs α_s
    fig_re_vs_alpha_s(df, fig_dir / "fig_re_vs_alpha_s.pdf")

    # 8.4 — MSE vs SNR on α-stable(1.5)
    fig_mse_vs_snr_alphastable(df, fig_dir / "fig_mse_vs_snr_alphastable_1p5.pdf",
                                dataset_name="alphastable_1p5")

    # 8.5 — RE table
    table_re_summary(
        df,
        txt_out=fig_dir / "table_re_summary.txt",
        csv_out=fig_dir / "table_re_summary.csv",
    )

    print(f"\nAll figures written to {fig_dir}")


if __name__ == "__main__":
    main()
