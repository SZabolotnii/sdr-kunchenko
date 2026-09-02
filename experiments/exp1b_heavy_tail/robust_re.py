#!/usr/bin/env python3
"""Robust relative-efficiency (RE) recomputation for the exp1b PATP results.

Motivation (RADAP submission 2149, reviewer point 3): the published Tables 3–4
report RE = MSE(OLS)/MSE(method) with values such as PATP* = 5083 and
Huber = 12122 on α-stable noise with α_s = 1.2.  Under α_s < 2 the population
variance is infinite, so the *sample* MSE of OLS does not converge: it is
dominated by a handful of extreme replicates (this script measures that share
explicitly).  The ratio of two such quantities is not a well-defined efficiency.

This script recomputes, for every dataset and every estimator, four RE variants
from the *existing* per-replicate CSVs (no Monte Carlo is re-run):

  RE_mean_snr   published definition — mean over the 5 SNR cells of the per-cell
                ratio of mean squared errors (reproduces Tables 3–4 exactly);
  RE_mean_pool  ratio of mean squared errors pooled over all replicates;
  RE_median     ratio of medians of squared errors, med(e²_OLS)/med(e²_method);
  RE_trim1      ratio of 1 % upper-trimmed means of squared errors.

``RE_median`` and ``RE_trim1`` are ratios of population functionals that exist
for any error law, including infinite-variance α-stable, so they are the
defensible replacements for the ill-defined mean-based numbers.

Uncertainty: percentile bootstrap 95 % CI, resampling *noise blocks* (column
``idx``) with replacement and applying the resampled index set to numerator and
denominator simultaneously (common random numbers / paired structure).  For the
synthetic Monte Carlo families the blocks are i.i.d. draws from the generator, so
block-level resampling is replicate-level resampling and is legitimate.  For the
two real-RF sources all blocks come from a *single recording per drone*
(n_recordings = 1): those rows are flagged ``descriptive_only`` — the intervals
describe within-recording block dispersion and must not be read as generalization
statements or turned into p-values (Verification Standard 9, CLAUDE.md).

Outputs (``experiments/exp1b_heavy_tail/results_robust_re/``):
  robust_re.csv                — all variants side by side
  fig_robust_re.{png,pdf}      — mean-RE vs median-RE comparison

Usage:
  python experiments/exp1b_heavy_tail/robust_re.py
  python experiments/exp1b_heavy_tail/robust_re.py --bootstrap 4000 --seed 7
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

# regime label -> (results directory, experiment_id)
RESULT_SETS: tuple[tuple[str, str, str], ...] = (
    ("heavy_tail", "results_patp_mc", "exp1b_patp_mc"),
    ("platykurtic", "results_patp_platy", "exp1b_patp_platy"),
    ("real_rf", "results_patp_real", "exp1b_patp_real"),
)

# Method label -> per-replicate amplitude-estimate column.
METHODS: dict[str, str] = {
    "OLS": "A_ols",
    "HUBER": "A_huber",
    "PMM3": "A_pmm3",
    "PMM_FRAC": "A_pmm_frac",
    "PATP*": "A_patp_star",
    "AUTO_EXT": "A_auto_ext",
}

# Real-RF sources: every block originates from one .bin recording per drone.
SINGLE_RECORDING_DATASETS: frozenset[str] = frozenset(
    {"mavic_ocusync_heavytail", "parrot_disco_heavytail"}
)

TRIM_FRACTION: float = 0.01  # upper 1 % of squared errors removed
DEFAULT_BOOTSTRAP: int = 2000
DEFAULT_SEED: int = 20260830

with contextlib.suppress(OSError):
    plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# ---------------------------------------------------------------------------
# RE functionals
# ---------------------------------------------------------------------------


def upper_trimmed_mean(x: np.ndarray, frac: float = TRIM_FRACTION) -> float:
    """Mean of the ``floor(n·(1−frac))`` smallest values of ``x``.

    A 1 % upper-trimmed mean of the squared errors.  Unlike the raw mean it is a
    finite population functional even when the error law has infinite variance,
    and unlike a per-replicate cap it needs no arbitrary scale constant.  The
    identical rule is applied to every estimator, each trimmed at its own
    empirical (1−frac) quantile, on the same set of replicates.
    """
    n = x.size
    if n == 0:
        return float("nan")
    k = max(1, int(np.floor(n * (1.0 - frac))))
    return float(np.mean(np.sort(x)[:k]))


def _ratio(stat, num: np.ndarray, den: np.ndarray) -> float:
    """RE = stat(e²_OLS) / stat(e²_method); +inf if the denominator vanishes."""
    d = stat(den)
    return float(stat(num) / d) if d > 0 else float("inf")


def re_median(e2_ols: np.ndarray, e2_m: np.ndarray) -> float:
    """Median-based relative efficiency med(e²_OLS)/med(e²_method)."""
    return _ratio(np.median, e2_ols, e2_m)


def re_trimmed(e2_ols: np.ndarray, e2_m: np.ndarray) -> float:
    """1 % upper-trimmed-mean relative efficiency."""
    return _ratio(upper_trimmed_mean, e2_ols, e2_m)


def re_mean(e2_ols: np.ndarray, e2_m: np.ndarray) -> float:
    """Pooled mean-based relative efficiency (the ill-defined reference)."""
    return _ratio(np.mean, e2_ols, e2_m)


def re_mean_per_snr(df: pd.DataFrame, col: str) -> np.ndarray:
    """Published Tables 3–4 definition: the per-SNR-cell mean-based RE values.

    ``run.py::_compute_summary`` computes one RE per (dataset, SNR) cell and the
    manuscript averages the five cells; the mean of this array reproduces the
    published numbers exactly.  The spread of the array is itself a diagnostic:
    for synthetic noise the SNR grid is a pure replication label (the generator
    fixes the noise law), so five nominally identical cells that disagree by
    orders of magnitude show the mean-based RE has not converged.
    """
    ratios = []
    for _, grp in df.groupby("snr_db"):
        a_true = grp["A_true"].to_numpy()
        mse_ols = float(np.nanmean((grp["A_ols"].to_numpy() - a_true) ** 2))
        mse_m = float(np.nanmean((grp[col].to_numpy() - a_true) ** 2))
        ratios.append(mse_ols / mse_m if mse_m > 0 else float("inf"))
    return np.asarray(ratios, dtype=float)


# ---------------------------------------------------------------------------
# Paired cluster bootstrap over noise blocks
# ---------------------------------------------------------------------------


def _group_layout(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row order sorted by group, per-group offsets, per-group counts)."""
    codes = pd.factorize(groups)[0]
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return order, offsets, counts


def _bootstrap_rows(
    order: np.ndarray, offsets: np.ndarray, counts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One cluster-bootstrap resample: row indices of a block-level draw."""
    g = rng.integers(0, counts.size, size=counts.size)
    lens = counts[g]
    total = int(lens.sum())
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    pos = np.arange(total) - np.repeat(starts, lens)
    return order[np.repeat(offsets[g], lens) + pos]


def bootstrap_ci(
    e2_ols: np.ndarray,
    e2_m: np.ndarray,
    groups: np.ndarray,
    stat,
    *,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a paired RE functional.

    The block indices are resampled once per bootstrap iteration and the *same*
    index set is applied to numerator and denominator, so the common-random-number
    pairing between OLS and the competing estimator is preserved.
    """
    order, offsets, counts = _group_layout(groups)
    if counts.size < 2:
        return float("nan"), float("nan")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        rows = _bootstrap_rows(order, offsets, counts, rng)
        vals[b] = _ratio(stat, e2_ols[rows], e2_m[rows])
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-dataset computation
# ---------------------------------------------------------------------------


def patp_alpha_columns(df: pd.DataFrame) -> list[tuple[float, str]]:
    """Available PATP sweep columns as (alpha, column name), sorted by alpha."""
    out = []
    for c in df.columns:
        if c.startswith("A_patp_a"):
            out.append((float(c.removeprefix("A_patp_a")), c))
    return sorted(out)


def alpha_star_by(df: pd.DataFrame, stat) -> float:
    """α maximizing the ``stat``-based RE over the PATP sweep grid (oracle rule)."""
    a_true = df["A_true"].to_numpy()
    e2_ols = (df["A_ols"].to_numpy() - a_true) ** 2
    best_a, best_re = float("nan"), -np.inf
    for a, col in patp_alpha_columns(df):
        e2 = (df[col].to_numpy() - a_true) ** 2
        mask = np.isfinite(e2) & np.isfinite(e2_ols)
        if mask.sum() == 0:
            continue
        re = _ratio(stat, e2_ols[mask], e2[mask])
        if np.isfinite(re) and re > best_re:
            best_a, best_re = a, re
    return best_a


def analyse_dataset(
    df: pd.DataFrame,
    *,
    regime: str,
    dataset: str,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Compute all RE variants (+ CIs) for one dataset."""
    a_true = df["A_true"].to_numpy()
    e2_ols_all = (df["A_ols"].to_numpy() - a_true) ** 2
    groups_all = df["idx"].to_numpy()

    single_rec = dataset in SINGLE_RECORDING_DATASETS
    common = {
        "regime": regime,
        "dataset": dataset,
        "n_trials": len(df),
        "n_blocks": int(df["idx"].nunique()),
        "n_recordings": 1 if single_rec else np.nan,
        "resampling_unit": "noise block (idx)",
        "descriptive_only": bool(single_rec),
        "gamma4_median": float(np.nanmedian(df["gamma4_xi"].to_numpy())),
        "patp_alpha_star_mean": (
            float(df["patp_alpha_star"].iloc[0]) if "patp_alpha_star" in df else np.nan
        ),
        "patp_alpha_star_median": alpha_star_by(df, np.median),
        # Share of the pooled OLS sum of squared errors carried by its single
        # largest replicate — the direct measure of "MSE decided by one outlier".
        "ols_max_share": float(np.nanmax(e2_ols_all) / np.nansum(e2_ols_all)),
    }

    rows = []
    for method, col in METHODS.items():
        if col not in df.columns:
            continue
        e2_m_all = (df[col].to_numpy() - a_true) ** 2
        mask = np.isfinite(e2_m_all) & np.isfinite(e2_ols_all)
        e2_ols, e2_m, groups = e2_ols_all[mask], e2_m_all[mask], groups_all[mask]

        per_snr = re_mean_per_snr(df, col)
        row = dict(common)
        row.update(
            {
                "method": method,
                "n_used": int(mask.sum()),
                "RE_mean_snr_published": float(np.mean(per_snr)),
                "RE_mean_snr_min": float(np.min(per_snr)),
                "RE_mean_snr_max": float(np.max(per_snr)),
                "RE_mean_pooled": re_mean(e2_ols, e2_m),
                "RE_median": re_median(e2_ols, e2_m),
                "RE_trim1": re_trimmed(e2_ols, e2_m),
                # Share of replicates where the estimator returned exactly the OLS
                # value (non-convergence fallback inside run.py).  These few
                # replicates carry the whole mean-based MSE on heavy tails and
                # explain most of the mean-vs-median gap.
                "share_equals_ols": float(
                    np.mean(np.abs(df[col].to_numpy() - df["A_ols"].to_numpy()) < 1e-12)
                ),
                "med_sq_err": float(np.median(e2_m)),
                "trim1_mean_sq_err": upper_trimmed_mean(e2_m),
            }
        )
        if method == "OLS":
            row["RE_median_lo"] = row["RE_median_hi"] = 1.0
            row["RE_trim1_lo"] = row["RE_trim1_hi"] = 1.0
        else:
            row["RE_median_lo"], row["RE_median_hi"] = bootstrap_ci(
                e2_ols, e2_m, groups, np.median, n_boot=n_boot, rng=rng
            )
            row["RE_trim1_lo"], row["RE_trim1_hi"] = bootstrap_ci(
                e2_ols, e2_m, groups, upper_trimmed_mean, n_boot=n_boot, rng=rng
            )
        rows.append(row)
    return rows


def load_all(n_boot: int, seed: int) -> pd.DataFrame:
    """Run the analysis over every exp1b PATP results directory."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for regime, subdir, exp_id in RESULT_SETS:
        results_dir = EXP_DIR / subdir
        if not results_dir.is_dir():
            print(f"  [skip] {results_dir} not found")
            continue
        for csv_path in sorted(results_dir.glob(f"{exp_id}_*.csv")):
            stem = csv_path.stem.removeprefix(f"{exp_id}_")
            if stem in ("summary", "patp_sweep"):
                continue
            df = pd.read_csv(csv_path)
            print(f"  {regime:12s} {stem:26s} n={len(df)}")
            records.extend(
                analyse_dataset(df, regime=regime, dataset=stem, n_boot=n_boot, rng=rng)
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _short_label(dataset: str) -> str:
    if dataset.startswith("alphastable_"):
        return f"α$_s$={dataset.split('_')[1].replace('p', '.')}"
    if dataset == "studentt_gaussian":
        return "Gauss (ν=∞)"
    if dataset.startswith("studentt_nu"):
        return f"t, ν={dataset.split('nu')[1]}"
    if dataset == "mavic_ocusync_heavytail":
        return "Mavic RF"
    if dataset == "parrot_disco_heavytail":
        return "Parrot RF"
    return dataset


def make_figure(tab: pd.DataFrame, out_stem: Path) -> None:
    """Scatter of mean-RE vs median-RE plus a per-dataset bar comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # --- (a) mean vs median, every dataset × method ------------------------
    ax = axes[0]
    sub = tab[tab["method"] != "OLS"]
    markers = {"HUBER": "o", "PMM3": "s", "PMM_FRAC": "^", "PATP*": "D", "AUTO_EXT": "v"}
    for method, grp in sub.groupby("method"):
        ax.scatter(
            grp["RE_mean_snr_published"],
            grp["RE_median"],
            marker=markers.get(str(method), "o"),
            s=34,
            alpha=0.85,
            label=str(method),
        )
    lim = [0.5, max(2.0, float(sub["RE_mean_snr_published"].max()) * 2)]
    ax.plot(lim, lim, color="0.5", lw=0.9, ls="--", label="RE$_{median}$ = RE$_{mean}$")
    ax.axhline(1.0, color="0.75", lw=0.8, ls=":")
    ax.axvline(1.0, color="0.75", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_xlabel("published mean-based RE (Tables 3–4)")
    ax.set_ylabel("median-based RE")
    ax.set_title("(a) Mean-based RE collapses under a robust functional")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    for _, r in sub.iterrows():
        if r["RE_mean_snr_published"] > 50 or r["RE_median"] > 20:
            ax.annotate(
                f"{_short_label(str(r['dataset']))} {r['method']}",
                (r["RE_mean_snr_published"], r["RE_median"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="0.3",
            )

    # --- (b) bars per dataset for the headline estimators ------------------
    ax = axes[1]
    order = [
        "alphastable_1p2",
        "alphastable_1p5",
        "alphastable_1p8",
        "studentt_nu3",
        "studentt_nu5",
        "uniform",
        "beta2_2",
        "parrot_disco_heavytail",
        "mavic_ocusync_heavytail",
    ]
    order = [d for d in order if d in set(tab["dataset"])]
    show = ["HUBER", "PATP*", "AUTO_EXT"]
    width = 0.13
    x = np.arange(len(order))
    for j, method in enumerate(show):
        m = tab[tab["method"] == method].set_index("dataset")
        base = x + (j - 1) * 2 * width
        mean_v = [m.loc[d, "RE_mean_snr_published"] for d in order]
        med_v = [m.loc[d, "RE_median"] for d in order]
        lo = [m.loc[d, "RE_median_lo"] for d in order]
        hi = [m.loc[d, "RE_median_hi"] for d in order]
        err = np.vstack(
            [np.array(med_v) - np.array(lo), np.array(hi) - np.array(med_v)]
        ).clip(min=0)
        ax.bar(base - width / 2, mean_v, width, color=f"C{j}", alpha=0.35)
        ax.bar(base + width / 2, med_v, width, color=f"C{j}", yerr=err, capsize=2,
               label=method)
    ax.axhline(1.0, color="0.5", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(d) for d in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("relative efficiency vs OLS")
    ax.set_title("(b) Pale bar = published mean-RE, solid bar = median-RE")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Rectangle((0, 0), 1, 1, color="0.4", alpha=0.35),
        plt.Rectangle((0, 0), 1, 1, color="0.4"),
    ]
    labels += ["left bar: published mean-RE", "right bar: median-RE (95 % CI)"]
    ax.legend(handles, labels, fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path)
        print(f"Wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description="Robust RE recomputation for exp1b PATP results (no MC re-run)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                   help="Bootstrap replications for the RE confidence intervals")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Bootstrap RNG seed")
    p.add_argument("--out-dir", type=Path, default=EXP_DIR / "results_robust_re",
                   dest="out_dir", help="Output directory")
    p.add_argument("--no-figure", action="store_true", help="Skip figure generation")
    args = p.parse_args()

    print("Loading per-replicate CSVs (no Monte Carlo is re-run)...")
    tab = load_all(args.bootstrap, args.seed)
    if tab.empty:
        print("No results found.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "robust_re.csv"
    tab.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(tab)} rows)")

    cols = ["RE_mean_snr_published", "RE_mean_snr_min", "RE_mean_snr_max",
            "RE_mean_pooled", "RE_median", "RE_median_lo", "RE_median_hi", "RE_trim1"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n=== RE variants (rounded) ===")
        print(
            tab.set_index(["regime", "dataset", "method"])[cols]
            .round(3)
            .to_string()
        )
        print("\n=== OLS MSE concentration: share of Σe²_OLS in its single worst replicate ===")
        print(
            tab[tab["method"] == "OLS"]
            .set_index(["regime", "dataset"])[["ols_max_share", "n_trials", "n_blocks"]]
            .round(4)
            .to_string()
        )

    if not args.no_figure:
        make_figure(tab, args.out_dir / "fig_robust_re")

    flagged = sorted(set(tab.loc[tab["descriptive_only"], "dataset"]))
    if flagged:
        print(
            "\nNOTE: datasets " + ", ".join(flagged) + " come from a single recording each "
            "(n_recordings=1). Their intervals are within-recording descriptive dispersion "
            "only — no p-values, no generalization claims (Verification Standard 9)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
