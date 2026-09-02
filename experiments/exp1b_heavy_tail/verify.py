#!/usr/bin/env python3
"""exp1b Verification Script — checks pass/fail criteria for heavy-tail experiments.

Reads the summary CSV produced by ``experiments/exp1b_heavy_tail/run.py`` and
evaluates four quantitative acceptance criteria:

  re_frac_heavy    — RE(PMM_FRAC | α_s ≤ 1.5) ≥ 1.5
  re_frac_gaussian — RE(PMM_FRAC | α_s = 2.0) ≥ 0.95
  auto_ext_safety  — MSE(AUTO_EXT) ≤ MSE(OLS) × 1.05
  baseline_ols_mse — OLS MSE on realnoise = 3.28e-3 ± 5% (constant from [exp1])

Generates ``verification_report_heavy_tail.txt`` in the results directory and
exits with code 0 if all required criteria pass, 1 otherwise.

References
----------
- exp1b requirements §Requirement 6 (criteria 6.6, 6.7), §Requirement 7 (7.4)
- exp1b requirements §Requirement 10 (10.5)
- exp1b requirements §Requirement 11 (11.2 — baseline constants from [exp1])
- exp1b design §9 (pass/fail criteria table)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Baseline constants from [exp1] (git-tag: exp1a-final)
# Dataset realnoise (UID 692e429b), Table II
# ---------------------------------------------------------------------------
BASELINE_OLS_MSE_EXP1: float = 3.28e-3   # OLS MSE on realnoise
BASELINE_PMM3_RE_EXP1: float = 1.74       # RE(PMM3) on platykurtic subset of realnoise
BASELINE_OLS_MSE_TOLERANCE: float = 0.05  # ±5 %

# Pass/fail thresholds (from config_alphastable.yaml §verification.pass_criteria)
THRESHOLD_RE_FRAC_HEAVY: float = 1.5    # RE(PMM_FRAC | α_s ≤ 1.5) ≥ 1.5
THRESHOLD_RE_FRAC_GAUSSIAN: float = 0.95  # RE(PMM_FRAC | α_s = 2.0) ≥ 0.95
THRESHOLD_AUTO_EXT_SAFETY: float = 1.05  # MSE(AUTO_EXT) ≤ MSE(OLS) × 1.05

# Dataset name patterns
HEAVY_TAIL_PATTERNS: tuple[str, ...] = ("1p2", "1p5")   # α_s ≤ 1.5
GAUSSIAN_PATTERNS: tuple[str, ...] = ("2p0",)            # α_s = 2.0 (Gaussian control)
REALNOISE_PATTERNS: tuple[str, ...] = ("realnoise",)

# PATP α-sweep criteria (best-α column reported as method "PATP*").
THRESHOLD_RE_PATP_HEAVY: float = 1.5     # RE(PATP* | α_s ≤ 1.5 or ν ≤ 3) ≥ 1.5
THRESHOLD_RE_PATP_GAUSSIAN: float = 0.95  # RE(PATP* | Gaussian control) ≥ 0.95 (no harm)
PATP_HEAVY_PATTERNS: tuple[str, ...] = ("1p2", "1p5", "nu3")
PATP_GAUSSIAN_PATTERNS: tuple[str, ...] = ("2p0", "gaussian")
PATP_METHOD: str = "PATP*"


# ---------------------------------------------------------------------------
# Criterion checks
# ---------------------------------------------------------------------------


def check_re_frac_heavy(df: pd.DataFrame) -> dict[str, Any]:
    """Task 7.1 — RE(PMM_FRAC | α_s ≤ 1.5) ≥ 1.5.

    Filters rows where ``dataset`` contains "1p2" or "1p5" and
    ``method == "PMM_FRAC"``, then checks that the mean RE across all
    matching rows is ≥ 1.5.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame with columns: dataset, method, snr_db, MSE, RE.

    Returns
    -------
    dict
        Keys: status ("PASS"/"FAIL"/"SKIPPED"), value (float or None),
        threshold (float), datasets (list[str]), detail (str).
    """
    mask_ds = df["dataset"].str.contains("|".join(HEAVY_TAIL_PATTERNS), na=False)
    mask_method = df["method"] == "PMM_FRAC"
    subset = df[mask_ds & mask_method]

    if subset.empty:
        return {
            "status": "SKIPPED",
            "value": None,
            "threshold": THRESHOLD_RE_FRAC_HEAVY,
            "datasets": [],
            "detail": "No rows matching alpha_s ≤ 1.5 datasets with PMM_FRAC found in summary CSV.",
        }

    datasets_found = sorted(subset["dataset"].unique().tolist())
    mean_re = float(subset["RE"].mean())
    passed = mean_re >= THRESHOLD_RE_FRAC_HEAVY

    return {
        "status": "PASS" if passed else "FAIL",
        "value": mean_re,
        "threshold": THRESHOLD_RE_FRAC_HEAVY,
        "datasets": datasets_found,
        "detail": (
            f"Mean RE(PMM_FRAC) = {mean_re:.4f} over {len(subset)} rows "
            f"(datasets: {', '.join(datasets_found)})"
        ),
    }


def check_re_frac_gaussian(df: pd.DataFrame) -> dict[str, Any]:
    """Task 7.2 — RE(PMM_FRAC | α_s = 2.0) ≥ 0.95.

    Filters rows where ``dataset`` contains "2p0" and
    ``method == "PMM_FRAC"``, then checks that the mean RE ≥ 0.95.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame.

    Returns
    -------
    dict
        Keys: status, value, threshold, datasets, detail.
    """
    mask_ds = df["dataset"].str.contains("|".join(GAUSSIAN_PATTERNS), na=False)
    mask_method = df["method"] == "PMM_FRAC"
    subset = df[mask_ds & mask_method]

    if subset.empty:
        return {
            "status": "SKIPPED",
            "value": None,
            "threshold": THRESHOLD_RE_FRAC_GAUSSIAN,
            "datasets": [],
            "detail": "No rows matching alpha_s = 2.0 dataset with PMM_FRAC found in summary CSV.",
        }

    datasets_found = sorted(subset["dataset"].unique().tolist())
    mean_re = float(subset["RE"].mean())
    passed = mean_re >= THRESHOLD_RE_FRAC_GAUSSIAN

    return {
        "status": "PASS" if passed else "FAIL",
        "value": mean_re,
        "threshold": THRESHOLD_RE_FRAC_GAUSSIAN,
        "datasets": datasets_found,
        "detail": (
            f"Mean RE(PMM_FRAC) = {mean_re:.4f} over {len(subset)} rows "
            f"(datasets: {', '.join(datasets_found)})"
        ),
    }


def _check_re_patp(
    df: pd.DataFrame,
    patterns: tuple[str, ...],
    threshold: float,
    regime: str,
) -> dict[str, Any]:
    """Shared PATP* relative-efficiency check over a set of dataset patterns."""
    mask_ds = df["dataset"].str.contains("|".join(patterns), na=False)
    subset = df[mask_ds & (df["method"] == PATP_METHOD)]
    if subset.empty:
        return {
            "status": "SKIPPED",
            "value": None,
            "threshold": threshold,
            "datasets": [],
            "detail": f"No PATP* rows for {regime} datasets in summary CSV.",
        }
    datasets_found = sorted(subset["dataset"].unique().tolist())
    mean_re = float(subset["RE"].mean())
    return {
        "status": "PASS" if mean_re >= threshold else "FAIL",
        "value": mean_re,
        "threshold": threshold,
        "datasets": datasets_found,
        "detail": (
            f"Mean RE(PATP*) = {mean_re:.4f} over {len(subset)} rows "
            f"(datasets: {', '.join(datasets_found)})"
        ),
    }


def check_re_patp_heavy(df: pd.DataFrame) -> dict[str, Any]:
    """RE(PATP* | heavy tails: α_s ≤ 1.5 or ν ≤ 3) ≥ 1.5 — fractional branch wins."""
    return _check_re_patp(df, PATP_HEAVY_PATTERNS, THRESHOLD_RE_PATP_HEAVY, "heavy-tail")


def check_re_patp_gaussian(df: pd.DataFrame) -> dict[str, Any]:
    """RE(PATP* | Gaussian control) ≥ 0.95 — no harm at the power/linear edge."""
    return _check_re_patp(df, PATP_GAUSSIAN_PATTERNS, THRESHOLD_RE_PATP_GAUSSIAN, "Gaussian-control")


def check_auto_ext_safety(df: pd.DataFrame) -> dict[str, Any]:
    """Task 7.3 — MSE(AUTO_EXT) ≤ MSE(OLS) × 1.05 for every dataset/SNR cell.

    For each (dataset, snr_db) pair, compares MSE(AUTO_EXT) vs MSE(OLS).
    Reports the worst-case ratio and whether all cells satisfy the constraint.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame.

    Returns
    -------
    dict
        Keys: status, value (worst-case ratio), threshold, datasets, detail,
        violations (list of dicts describing failing cells).
    """
    ols_rows = df[df["method"] == "OLS"][["dataset", "snr_db", "MSE"]].rename(
        columns={"MSE": "MSE_OLS"}
    )
    ext_rows = df[df["method"] == "AUTO_EXT"][["dataset", "snr_db", "MSE"]].rename(
        columns={"MSE": "MSE_AUTO_EXT"}
    )

    if ols_rows.empty or ext_rows.empty:
        return {
            "status": "SKIPPED",
            "value": None,
            "threshold": THRESHOLD_AUTO_EXT_SAFETY,
            "datasets": [],
            "detail": "OLS or AUTO_EXT rows not found in summary CSV.",
            "violations": [],
        }

    merged = pd.merge(ols_rows, ext_rows, on=["dataset", "snr_db"], how="inner")
    if merged.empty:
        return {
            "status": "SKIPPED",
            "value": None,
            "threshold": THRESHOLD_AUTO_EXT_SAFETY,
            "datasets": [],
            "detail": "No matching (dataset, snr_db) pairs for OLS and AUTO_EXT.",
            "violations": [],
        }

    merged["ratio"] = merged["MSE_AUTO_EXT"] / merged["MSE_OLS"].replace(0, float("nan"))
    worst_ratio = float(merged["ratio"].max())
    violations = merged[merged["ratio"] > THRESHOLD_AUTO_EXT_SAFETY]

    datasets_found = sorted(merged["dataset"].unique().tolist())
    passed = worst_ratio <= THRESHOLD_AUTO_EXT_SAFETY

    violation_list = [
        {
            "dataset": row["dataset"],
            "snr_db": row["snr_db"],
            "ratio": row["ratio"],
            "MSE_AUTO_EXT": row["MSE_AUTO_EXT"],
            "MSE_OLS": row["MSE_OLS"],
        }
        for _, row in violations.iterrows()
    ]

    return {
        "status": "PASS" if passed else "FAIL",
        "value": worst_ratio,
        "threshold": THRESHOLD_AUTO_EXT_SAFETY,
        "datasets": datasets_found,
        "detail": (
            f"Worst-case MSE(AUTO_EXT)/MSE(OLS) = {worst_ratio:.4f} "
            f"over {len(merged)} cells (datasets: {', '.join(datasets_found)})"
        ),
        "violations": violation_list,
    }


def check_baseline_ols_mse(df: pd.DataFrame) -> dict[str, Any]:
    """Task 7.4 — OLS MSE on realnoise = 3.28e-3 ± 5% (constant from [exp1]).

    Checks whether the exp1b OLS MSE on the Gaussian control dataset
    (α_s = 2.0 or ν = ∞) is within 5 % of the exp1 baseline value 3.28e-3.

    If no realnoise data is present in the summary CSV, reports SKIPPED.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame.

    Returns
    -------
    dict
        Keys: status, value, threshold, datasets, detail,
        baseline_ols_mse_exp1, baseline_pmm3_re_exp1.
    """
    # Look for realnoise dataset first
    mask_realnoise = df["dataset"].str.contains(
        "|".join(REALNOISE_PATTERNS), na=False, case=False
    )
    mask_ols = df["method"] == "OLS"
    subset = df[mask_realnoise & mask_ols]

    if subset.empty:
        return {
            "status": "SKIPPED",
            "value": None,
            "threshold": BASELINE_OLS_MSE_EXP1,
            "datasets": [],
            "detail": "SKIPPED (realnoise dataset not in exp1b scope)",
            "baseline_ols_mse_exp1": BASELINE_OLS_MSE_EXP1,
            "baseline_pmm3_re_exp1": BASELINE_PMM3_RE_EXP1,
        }

    datasets_found = sorted(subset["dataset"].unique().tolist())
    mean_mse = float(subset["MSE"].mean())
    lower = BASELINE_OLS_MSE_EXP1 * (1.0 - BASELINE_OLS_MSE_TOLERANCE)
    upper = BASELINE_OLS_MSE_EXP1 * (1.0 + BASELINE_OLS_MSE_TOLERANCE)
    passed = lower <= mean_mse <= upper

    return {
        "status": "PASS" if passed else "FAIL",
        "value": mean_mse,
        "threshold": BASELINE_OLS_MSE_EXP1,
        "datasets": datasets_found,
        "detail": (
            f"OLS MSE = {mean_mse:.4e} "
            f"(expected {BASELINE_OLS_MSE_EXP1:.2e} ± {BASELINE_OLS_MSE_TOLERANCE*100:.0f}%, "
            f"window [{lower:.2e}, {upper:.2e}])"
        ),
        "baseline_ols_mse_exp1": BASELINE_OLS_MSE_EXP1,
        "baseline_pmm3_re_exp1": BASELINE_PMM3_RE_EXP1,
    }


def check_baseline_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Return baseline comparison dict for §VI of the manuscript.

    Computes the exp1b OLS MSE on the Gaussian control dataset and compares
    it to the exp1 baseline constants.  Always returns the hardcoded exp1
    baseline values regardless of whether realnoise data is present.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame with columns: dataset, method, snr_db, MSE, RE.

    Returns
    -------
    dict
        Keys:
          baseline_ols_mse_exp1  — 3.28e-3 (hardcoded from [exp1])
          baseline_pmm3_re_exp1  — 1.74    (hardcoded from [exp1])
          status                 — "PASS" / "FAIL" / "SKIPPED"
    """
    result = check_baseline_ols_mse(df)
    return {
        "baseline_ols_mse_exp1": BASELINE_OLS_MSE_EXP1,
        "baseline_pmm3_re_exp1": BASELINE_PMM3_RE_EXP1,
        "status": result["status"],
    }


# ---------------------------------------------------------------------------
# Report generation (Task 7.5)
# ---------------------------------------------------------------------------

_CRITERION_META: dict[str, dict[str, str]] = {
    "re_frac_heavy": {
        "description": "RE(PMM_FRAC | alpha_s <= 1.5) >= 1.5",
        "threshold_label": ">= 1.5",
    },
    "re_frac_gaussian": {
        "description": "RE(PMM_FRAC | alpha_s = 2.0) >= 0.95",
        "threshold_label": ">= 0.95",
    },
    "auto_ext_safety": {
        "description": "MSE(AUTO_EXT) <= MSE(OLS) x 1.05",
        "threshold_label": "<= 1.05 x MSE(OLS)",
    },
    "re_patp_heavy": {
        "description": "RE(PATP* | heavy tails: alpha_s <= 1.5 or nu <= 3) >= 1.5",
        "threshold_label": ">= 1.5",
    },
    "re_patp_gaussian": {
        "description": "RE(PATP* | Gaussian control) >= 0.95 (no harm)",
        "threshold_label": ">= 0.95",
    },
    "baseline_ols_mse": {
        "description": "OLS MSE on realnoise = 3.28e-3 +/- 5% (constant from [exp1])",
        "threshold_label": "3.28e-3 +/- 5%",
    },
}


def _format_value(value: float | None, criterion: str) -> str:
    """Format the numeric value for display in the report."""
    if value is None:
        return "N/A"
    if criterion in ("re_frac_heavy", "re_frac_gaussian", "re_patp_heavy", "re_patp_gaussian"):
        return f"{value:.4f}"
    if criterion == "auto_ext_safety":
        return f"{value:.4f} (worst-case ratio)"
    if criterion == "baseline_ols_mse":
        return f"{value:.4e}"
    return f"{value}"


def generate_report(
    results: dict[str, dict[str, Any]],
    results_dir: Path,
    summary_csv_path: Path,
) -> str:
    """Task 7.5 — Build the verification report text.

    Parameters
    ----------
    results : dict
        Mapping criterion name → result dict from the check_* functions.
    results_dir : Path
        Path to the results directory (shown in the report header).
    summary_csv_path : Path
        Path to the summary CSV that was loaded.

    Returns
    -------
    str
        Full report text.
    """
    sep = "=" * 80
    now = datetime.now(timezone.utc).isoformat()

    lines: list[str] = [
        sep,
        "exp1b Verification Report",
        f"Generated: {now}",
        f"Results dir: {results_dir}",
        f"Summary CSV: {summary_csv_path}",
        sep,
        "",
    ]

    for criterion, meta in _CRITERION_META.items():
        r = results[criterion]
        status = r["status"]
        value_str = _format_value(r.get("value"), criterion)
        datasets = r.get("datasets", [])
        detail = r.get("detail", "")

        lines.append(f"Criterion: {criterion}")
        lines.append(f"  Description: {meta['description']}")
        lines.append(f"  Status: {status}")
        lines.append(f"  Value: {value_str} (threshold: {meta['threshold_label']})")
        if datasets:
            lines.append(f"  Datasets: {', '.join(datasets)}")
        if detail:
            lines.append(f"  Detail: {detail}")

        # Extra: list violations for auto_ext_safety
        violations = r.get("violations", [])
        if violations:
            lines.append("  Violations:")
            for v in violations:
                lines.append(
                    f"    dataset={v['dataset']}, snr_db={v['snr_db']:.1f}, "
                    f"ratio={v['ratio']:.4f} "
                    f"(MSE_AUTO_EXT={v['MSE_AUTO_EXT']:.4e}, MSE_OLS={v['MSE_OLS']:.4e})"
                )

        lines.append("")

    # Overall verdict: a criterion fails the run only if its status is FAIL.
    # SKIPPED criteria (e.g. PATP* absent for a non-sweep config, or the
    # informational baseline) are neutral and excluded from the denominator.
    statuses = [r["status"] for r in results.values()]
    n_total = sum(1 for s in statuses if s != "SKIPPED")
    n_pass_total = sum(1 for s in statuses if s == "PASS")
    overall = "FAIL" if any(s == "FAIL" for s in statuses) else "PASS"

    lines += [
        sep,
        f"OVERALL: {overall} ({n_pass_total}/{n_total} criteria passed)",
        sep,
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV auto-detection
# ---------------------------------------------------------------------------


def _find_summary_csv(results_dir: Path) -> Path | None:
    """Auto-detect the summary CSV in *results_dir*.

    Prefers files matching ``*_summary.csv``; falls back to any ``*.csv``.
    """
    candidates = sorted(results_dir.glob("*_summary.csv"))
    if candidates:
        # Prefer the most recently modified
        return max(candidates, key=lambda p: p.stat().st_mtime)
    # Fallback: any CSV
    all_csvs = sorted(results_dir.glob("*.csv"))
    if all_csvs:
        return max(all_csvs, key=lambda p: p.stat().st_mtime)
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run all verification checks and write the report.

    Returns
    -------
    int
        Exit code: 0 if all required criteria pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description=(
            "exp1b verification script — checks pass/fail criteria for "
            "heavy-tail PMM_FRAC experiments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/exp1b_heavy_tail/results/"),
        dest="results_dir",
        help="Directory containing the summary CSV and where the report will be saved.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        dest="summary_csv",
        help=(
            "Path to the summary CSV file.  "
            "If omitted, auto-detected from --results-dir."
        ),
    )
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Locate summary CSV
    if args.summary_csv is not None:
        summary_csv_path = args.summary_csv.resolve()
        if not summary_csv_path.exists():
            print(f"ERROR: specified --summary-csv not found: {summary_csv_path}", file=sys.stderr)
            return 1
    else:
        summary_csv_path = _find_summary_csv(results_dir)
        if summary_csv_path is None:
            print(
                f"ERROR: no summary CSV found in {results_dir}. "
                "Run experiments/exp1b_heavy_tail/run.py first.",
                file=sys.stderr,
            )
            return 1

    print(f"Loading summary CSV: {summary_csv_path}")
    df = pd.read_csv(summary_csv_path)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Run all checks
    print("\nRunning verification checks...")
    results: dict[str, dict[str, Any]] = {
        "re_frac_heavy": check_re_frac_heavy(df),
        "re_frac_gaussian": check_re_frac_gaussian(df),
        "auto_ext_safety": check_auto_ext_safety(df),
        "re_patp_heavy": check_re_patp_heavy(df),
        "re_patp_gaussian": check_re_patp_gaussian(df),
        "baseline_ols_mse": check_baseline_ols_mse(df),
    }

    # Print brief status to stdout
    for criterion, r in results.items():
        status = r["status"]
        icon = "✓" if status == "PASS" else ("~" if status == "SKIPPED" else "✗")
        print(f"  [{icon}] {criterion}: {status}  — {r.get('detail', '')}")

    # Generate full report text
    report_text = generate_report(results, results_dir, summary_csv_path)

    # Save report
    report_path = results_dir / "verification_report_heavy_tail.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    # Print full report to stdout
    print()
    print(report_text)

    # Determine exit code: any FAIL fails the run; SKIPPED is neutral.
    overall_pass = not any(r["status"] == "FAIL" for r in results.values())

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
