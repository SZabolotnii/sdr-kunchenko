"""Cross-domain α-mapping tests (Phase 5, Spec §8)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdr_kunchenko.patp.cross_domain import (
    aggregate_exp4_to_snr,
    build_unified_table,
    check_monotonicity,
    compute_pareto,
    fit_alpha_mapping,
    load_exp2_alphas,
    load_exp3_alphas,
    load_exp4_blocks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Skip data-dependent tests if results files are not present yet.
EXP2_PRI = PROJECT_ROOT / "experiments" / "exp2_gsa_detection" / "results" / "exp2_patp_alpha_sweep.csv"
EXP2_REAL = PROJECT_ROOT / "experiments" / "exp2_gsa_detection" / "results_realnoise" / "exp2_patp_alpha_sweep.csv"
EXP3_CSV = PROJECT_ROOT / "experiments" / "exp3_dsge_classification" / "results" / "exp3_patp_alpha_sweep.csv"
EXP4_JSON = PROJECT_ROOT / "experiments" / "exp4_patp_denoiser" / "results" / "exp4_alpha_mapping.json"

DATA_AVAILABLE = all(p.exists() for p in [EXP2_PRI, EXP2_REAL, EXP3_CSV, EXP4_JSON])
pytestmark = pytest.mark.skipif(not DATA_AVAILABLE,
                                  reason="Phase 2/3/4 results CSV/JSON not present")


def test_load_exp4_blocks_schema():
    """Unpacking JSON yields one row per (alias, snr, block_idx)."""
    df = load_exp4_blocks(EXP4_JSON)
    expected_cols = {"dataset_alias", "dataset", "snr_db", "block_idx",
                      "training_alpha", "training_gamma4"}
    assert expected_cols.issubset(df.columns), f"missing cols: {expected_cols - set(df.columns)}"
    assert len(df) > 0
    # Two datasets × ~5 SNRs × ~300 blocks = a few thousand rows.
    assert df["dataset_alias"].nunique() == 2
    assert (df["training_alpha"].between(0.0, 1.0)).all(), \
        "training_alpha must be in [0, 1]"


def test_aggregate_exp4_alpha_in_unit_interval():
    """aggregate_exp4_to_snr produces α-mode/p25/p75 in [0, 1]."""
    blocks = load_exp4_blocks(EXP4_JSON)
    agg = aggregate_exp4_to_snr(blocks)
    for col in ("alpha_opt", "alpha_p25", "alpha_p75"):
        assert (agg[col].between(0.0, 1.0)).all(), f"{col} out of [0, 1]"
    # p25 ≤ p75 always.
    assert (agg["alpha_p25"] <= agg["alpha_p75"]).all()
    # Per-(snr, dataset) only one row.
    assert agg.groupby(["snr_db", "dataset"]).size().max() == 1


def test_unified_table_has_all_three_task_types():
    """build_unified_table exposes detection / classification / denoising."""
    e2_pri = load_exp2_alphas(EXP2_PRI,
        dataset_name="fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8")
    e2_real = load_exp2_alphas(EXP2_REAL, dataset_name="fpv_realnoise_bpsk_qpsk")
    e3 = load_exp3_alphas(EXP3_CSV)
    e4_blocks = load_exp4_blocks(EXP4_JSON)
    e4_agg = aggregate_exp4_to_snr(e4_blocks)
    unified = build_unified_table(e2_pri, e2_real, e3, e4_agg)
    tasks = set(unified["task_type"].unique())
    assert tasks == {"detection", "classification", "denoising"}, \
        f"got {tasks}"
    # gamma3, gamma4 priors attached.
    assert unified["gamma4"].notna().all()
    # Normalised metric in [0, 1].
    assert unified["metric_normalized"].between(0.0, 1.0).all()


def test_poly2_mapping_recovers_known_signal():
    """Synthetic data α = 0.5 + 0.2*γ₄ + small noise — fit should recover slope."""
    rng = np.random.default_rng(0)
    n = 60
    g4 = rng.uniform(-2, 2, n)
    g3 = rng.normal(0, 0.05, n)
    snr = rng.uniform(-5, 15, n)
    task = rng.choice(["detection", "classification", "denoising"], size=n)
    alpha_true = np.clip(0.5 + 0.2 * g4 + rng.normal(0, 0.02, n), 0, 1)
    df = pd.DataFrame({
        "experiment": "synthetic",
        "task_type": task,
        "dataset": "synthetic",
        "snr_db": snr,
        "gamma3": g3,
        "gamma4": g4,
        "alpha_opt": alpha_true,
        "metric_value": rng.uniform(0.5, 1.0, n),
        "primary_metric": "synthetic",
    })
    fit = fit_alpha_mapping(df, include_task_dummies=True, ridge_alpha=1e-4)
    assert fit["r2"] > 0.7, f"poor fit: r²={fit['r2']:.3f}"
    # Predict at γ₄=1 vs γ₄=-1 (Δ=2) — slope ~0.2 → predicted Δ ~0.4.
    a_high = fit["predict"](0.0, 1.0, 5.0, "detection")
    a_low = fit["predict"](0.0, -1.0, 5.0, "detection")
    assert a_high - a_low > 0.15, \
        f"slope direction wrong: a(γ₄=1)={a_high}, a(γ₄=-1)={a_low}"


def test_check_monotonicity_detects_monotonic_function():
    """Manually constructed strictly increasing α(γ₄) yields monotonic=True."""
    def predict(g3, g4, snr, task):
        return float(np.clip(0.5 + 0.1 * g4, 0, 1))
    out = check_monotonicity(predict, gamma4_grid=np.linspace(-2, 2, 21))
    assert out["monotonic"] is True
    assert out["n_neg_diffs"] == 0


def test_pareto_includes_at_least_one_per_task():
    """Synthetic table — Pareto frontier non-empty per task_type."""
    df = pd.DataFrame({
        "task_type": ["detection"] * 4 + ["classification"] * 4,
        "alpha_distance_to_05": [0.5, 0.3, 0.1, 0.4, 0.2, 0.6, 0.1, 0.3],
        "metric_normalized":     [0.9, 0.7, 0.5, 0.6, 0.8, 0.4, 0.95, 0.7],
        "snr_db":                [0.0] * 8,
        "gamma4":                [0.5] * 8,
    })
    out = compute_pareto(df)
    for task in df["task_type"].unique():
        sub = out[(out["task_type"] == task) & out["pareto_optimal"]]
        assert len(sub) >= 1, f"no Pareto-optimal points for task {task}"
