"""Regime-map figure: the PMM/PATP family is two-sided and harmless at the
Gaussian center, whereas the standard robust baseline (Huber) is one-sided.

Pulls the robust-RE recomputation (results_robust_re/robust_re.csv) and plots,
for each noise regime, the relative efficiency (RE vs OLS) of the Huber
M-estimator versus the adaptive PMM/PATP family (AUTO_EXT, plus the
regime-appropriate member). The RE=1 line is OLS; bars below it mean the
estimator is WORSE than OLS.

RE is the MEDIAN-based functional med(e^2_OLS)/med(e^2_method) with 95% paired
block-bootstrap intervals, not the mean-based ratio used in the submitted
version: under alpha-stable noise the sample MSE of OLS is dominated by a single
extreme replicate and does not converge (68.1% of the OLS sum of squared errors
came from one replicate of 2500 at alpha_s=1.2).

Usage:
  python experiments/exp1b_heavy_tail/plot_patp_regime.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "experiments/exp1b_heavy_tail"

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    pass
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


ROBUST = BASE / "results_robust_re/robust_re.csv"


def _re(dataset: str, method: str) -> tuple[float, float, float, bool]:
    """Median-based RE with its bootstrap interval and the descriptive-only flag."""
    df = pd.read_csv(ROBUST)
    sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
    if not len(sub):
        return np.nan, np.nan, np.nan, False
    r = sub.iloc[0]
    return (float(r["RE_median"]), float(r["RE_median_lo"]),
            float(r["RE_median_hi"]), bool(r["descriptive_only"]))


def main() -> int:
    # (label, dataset, PMM-family member to show besides AUTO_EXT)
    cols = [
        ("Platykurtic\n(uniform)",       "uniform",                 "PMM3"),
        ("Platykurtic\n(Beta 2,2)",      "beta2_2",                 "PMM3"),
        ("Gaussian",                     "studentt_gaussian",       "PMM3"),
        ("Heavy tail\nStudent-t ν=3",    "studentt_nu3",            "PMM_FRAC"),
        ("Heavy tail\nα-stable 1.8",     "alphastable_1p8",         "PMM_FRAC"),
        ("Heavy tail\nα-stable 1.5",     "alphastable_1p5",         "PMM_FRAC"),
    ]
    # Real-RF sources are deliberately NOT shown here: they are descriptive only
    # (n_recordings = 1 per drone) and their median-based RE reaches ~300 on the
    # Parrot set, which would compress every controlled regime on a shared axis.
    # They are reported in Table S1 (supplementary material) instead.

    labels = [c[0] for c in cols]
    huber_ci = [_re(c[1], "HUBER") for c in cols]
    auto_ci = [_re(c[1], "AUTO_EXT") for c in cols]
    member_ci = [_re(c[1], c[2]) for c in cols]
    huber = [c[0] for c in huber_ci]
    auto = [c[0] for c in auto_ci]
    member = [c[0] for c in member_ci]

    def _err(ci: list) -> np.ndarray:
        """Asymmetric error bars from the bootstrap interval."""
        lo = np.array([max(c[0] - c[1], 0.0) for c in ci])
        hi = np.array([max(c[2] - c[0], 0.0) for c in ci])
        return np.vstack([lo, hi])

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ebar = {"capsize": 2.0, "error_kw": {"ecolor": "0.25", "elinewidth": 0.8}}
    ax.bar(x - w, huber, w, yerr=_err(huber_ci), label="Huber M (robust baseline)",
           color="#c03d3e", **ebar)
    ax.bar(x, auto, w, yerr=_err(auto_ci), label="AUTO_EXT (adaptive PMM/PATP)",
           color="#3274a1", **ebar)
    ax.bar(x + w, member, w, yerr=_err(member_ci),
           label="regime member (PMM3 / PATP-frac)", color="#3a923a", **ebar)

    ax.axhline(1.0, color="0.3", lw=1.0, ls="--", zorder=0)
    ax.text(len(labels) - 0.5, 1.02, "OLS (RE=1)", fontsize=8, color="0.3", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.62)  # headroom so the sub-OLS markers are not clipped
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median-based RE = med($e^2_{OLS}$)/med($e^2$)")
    ax.set_title("Huber is one-sided: it hurts on the platykurtic side and on the Gaussian, "
                 "where the PMM/PATP family gains or returns OLS")
    ax.text(0.005, -0.26, "Median-based RE with 95% paired block-bootstrap intervals; "
            "controlled Monte-Carlo regimes only (real RF is descriptive, see Table S1).",
            transform=ax.transAxes, fontsize=7, color="0.35", va="top")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    # annotate where Huber drops below OLS (compact single-line marker below the bar top)
    for xi, h in zip(x, huber):
        if h < 0.999:
            ax.annotate("↓ <OLS", (xi - w, h), fontsize=6.5, color="#c03d3e",
                        ha="center", va="top", xytext=(0, -3), textcoords="offset points")

    fig.tight_layout()
    out = BASE / "figures_patp_mc/fig_patp_regime_map.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {out} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
