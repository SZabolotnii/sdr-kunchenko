"""Generate PATP α-transition figures from exp1b_patp results.

Figures (arXiv:2605.14610 thesis, applied to SDR amplitude estimation):
  Fig MC-1  RE(α) curves — relative efficiency vs the PATP exponent parameter α
            for each heavy-tail family and the Gaussian controls. Shows the
            optimal α moving toward the fractal edge (α→0) as tails get heavier;
            the super-linear edge (p₂→2) degenerates and is left blank (NaN).
  Fig MC-2  α* vs measured excess kurtosis γ̂₄ — the data-driven transition.

Reads:
  {results_dir}/{experiment_id}_patp_sweep.csv   (alpha, p2_alpha, RE, dataset)
  {results_dir}/{experiment_id}_{dataset}.csv    (gamma4_xi per trial)

Usage:
  python experiments/exp1b_heavy_tail/plot_patp.py --config config_patp_mc.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    pass
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 11})


def _dataset_label(name: str) -> str:
    if name.startswith("alphastable_"):
        return f"α-stable α$_s$={name.split('_')[1].replace('p', '.')}"
    if name == "studentt_gaussian":
        return "Student-t ν=∞ (Gaussian)"
    if name.startswith("studentt_"):
        return f"Student-t ν={name.split('nu')[1]}"
    return name


def fig_re_vs_alpha(sweep: pd.DataFrame, out_path: Path) -> None:
    """Fig MC-1: RE(α) transition curves, split into α-stable and Student-t panels."""
    # Each panel gets its own y-range: the α-stable curves span ~6 decades while the
    # Student-t curves span ~2, and a shared axis flattens the latter into a band.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)
    groups = [
        ("α-stable family", [d for d in sweep["dataset"].unique() if d.startswith("alphastable")]),
        ("Student-t family", [d for d in sweep["dataset"].unique() if d.startswith("studentt")]),
    ]
    for ax, (title, datasets) in zip(axes, groups):
        panel = sweep[sweep["dataset"].isin(datasets)]
        for ds in sorted(datasets):
            g = sweep[sweep["dataset"] == ds].sort_values("alpha")
            ax.plot(g["alpha"], g["RE"], marker="o", ms=4, label=_dataset_label(ds))
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
        ax.axvline(0.5, color="0.7", lw=0.8, ls=":")
        ax.set_yscale("log")
        re_pos = panel.loc[panel["RE"] > 0, "RE"]
        if len(re_pos):
            ax.set_ylim(float(re_pos.min()) / 1.6, float(re_pos.max()) * 1.6)
        ax.set_xlabel(r"PATP exponent parameter $\alpha$")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"Relative efficiency $RE=\mathrm{MSE_{OLS}}/\mathrm{MSE_{PATP}}$")
    axes[0].annotate("fractal edge\n(heavy-tail)", xy=(0.0, 1), xytext=(0.05, 1),
                     fontsize=7, color="0.35")
    fig.suptitle(r"PATP $\alpha$-transition: RE peaks at the fractal edge as tails get heavier "
                 r"(RE$>$1 $\Rightarrow$ beats OLS)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_alpha_star_vs_kurtosis(sweep: pd.DataFrame, results_dir: Path, experiment_id: str,
                               out_path: Path) -> None:
    """Fig MC-2: α* (argmax RE) vs measured mean excess kurtosis γ̂₄ per dataset."""
    rows = []
    for ds, g in sweep.groupby("dataset"):
        astar = float(g.loc[g["RE"].idxmax(), "alpha"])
        re_star = float(g["RE"].max())
        trial_csv = results_dir / f"{experiment_id}_{ds}.csv"
        gamma4 = np.nan
        if trial_csv.exists():
            t = pd.read_csv(trial_csv, usecols=lambda c: c == "gamma4_xi")
            if "gamma4_xi" in t.columns:
                gamma4 = float(np.nanmedian(t["gamma4_xi"].values))
        rows.append({"dataset": ds, "alpha_star": astar, "RE_star": re_star, "gamma4": gamma4})
    d = pd.DataFrame(rows).dropna(subset=["gamma4"]).sort_values("gamma4")

    # Deterministic label placement (offset points) with thin leader lines, so the
    # coincident Gaussian controls and the α*=0 heavy-tail cluster stay legible.
    # dx, dy in points; ha = horizontal anchor of the label text.
    placement = {
        "alphastable_2p0":   (10, 13, "left"),    # Gaussian control, label above-right
        "studentt_gaussian": (10, -17, "left"),   # coincident point, label below-right
        "studentt_nu10":     (9, 5, "left"),
        "studentt_nu5":      (0, 34, "center"),   # α*=0 cluster: alternate heights
        "studentt_nu3":      (0, 58, "center"),
        "alphastable_1p8":   (0, 34, "center"),
        "alphastable_1p5":   (0, 58, "center"),
        "alphastable_1p2":   (-4, 82, "right"),   # rightmost: lift highest, anchor right
    }
    leader = dict(arrowstyle="-", lw=0.5, color="0.55", shrinkA=0, shrinkB=3)

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    sc = ax.scatter(d["gamma4"], d["alpha_star"], c=np.log10(d["RE_star"].clip(lower=1e-3)),
                    cmap="viridis", s=70, zorder=3)
    for _, r in d.iterrows():
        dx, dy, ha = placement.get(r["dataset"], (6, 5, "left"))
        ax.annotate(_dataset_label(r["dataset"]), (r["gamma4"], r["alpha_star"]),
                    fontsize=7, xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", arrowprops=leader, zorder=4)
    ax.set_xscale("symlog")
    ax.set_xlabel(r"measured median excess kurtosis $\hat{\gamma}_4$")
    ax.set_ylabel(r"optimal PATP exponent $\alpha^{*}$")
    ax.set_ylim(-0.04, 0.62)
    ax.set_xlim(-0.5, 900)
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax.set_title(r"Data-driven transition: $\alpha^{*}\!\to\!0$ (fractal) as $\hat{\gamma}_4$ grows")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_{10}$ RE$^{*}$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PATP α-transition figures")
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config_patp_mc.yaml")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    experiment_id = cfg.get("experiment_id", "exp1b_patp_mc")
    out_cfg = cfg.get("output", {})
    results_dir = PROJECT_ROOT / out_cfg.get("results_dir", "experiments/exp1b_heavy_tail/results_patp_mc")
    figures_dir = PROJECT_ROOT / out_cfg.get("figures_dir", "experiments/exp1b_heavy_tail/figures_patp_mc")
    figures_dir.mkdir(parents=True, exist_ok=True)

    sweep_csv = results_dir / f"{experiment_id}_patp_sweep.csv"
    if not sweep_csv.exists():
        print(f"ERROR: {sweep_csv} not found. Run run.py with patp_alpha_grid first.")
        return 1
    sweep = pd.read_csv(sweep_csv)

    fig_re_vs_alpha(sweep, figures_dir / "fig_patp_re_vs_alpha.pdf")
    fig_alpha_star_vs_kurtosis(sweep, results_dir, experiment_id,
                               figures_dir / "fig_patp_alpha_star_vs_kurtosis.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
