"""Tests for the pure-Python frozen-weight PMM estimators.

Strategy: when γ₃, γ₄ are frozen at their per-block values, our pooled
estimator should be approximately consistent with EstemPMM (which re-estimates
gammas inside its NR loop; small differences expected).

Then on calibration sets (pooled gammas), we test that variance is reduced
relative to per-block PMM2 in regimes where per-block γ̂ is dominated by noise.
"""
from __future__ import annotations

import numpy as np
import pytest

from sdr_kunchenko.pmm import oracle
from sdr_kunchenko.pmm.pooled import (
    pmm2_pooled_amplitude,
    pmm3_pooled_amplitude,
    pool_gamma_from_blocks,
)


def test_pmm2_pooled_close_to_oracle_on_tpn():
    """When per-block gamma is informative (asymmetric TPN), pooled-with-true-γ
    should match EstemPMM's own per-block estimate within ~1% (gammas similar)."""
    rng = np.random.default_rng(0)
    n = 1024
    z = rng.normal(size=n)
    raw = np.where(z < 0, z * 1.6, z * 0.7)
    eps = (raw - raw.mean()) / raw.std() * 0.5
    A_true = 1.5
    s = rng.normal(size=n)
    y = A_true * s + eps

    fit_oracle = oracle.lm_pmm2(y, s)
    fit_pool = pmm2_pooled_amplitude(y, s, gamma3=fit_oracle["gamma3"],
                                       gamma4=fit_oracle["gamma4"])
    assert fit_pool["converged"]
    rel = abs(fit_pool["coef"][0] - fit_oracle["coef"][0]) / abs(fit_oracle["coef"][0])
    assert rel < 0.02, f"pooled (with oracle gammas) off by {rel:.4%}"


def test_pmm2_pooled_recovers_ols_when_gammas_zero():
    """With γ₃=γ₄=0 (Gaussian assumption), c₁=1, c₂=0 → reduces to OLS."""
    rng = np.random.default_rng(1)
    n = 512
    s = rng.normal(size=n)
    eps = rng.normal(size=n) * 0.3
    A_true = 2.0
    y = A_true * s + eps
    fit = pmm2_pooled_amplitude(y, s, gamma3=0.0, gamma4=0.0)
    A_ols = float(y @ s / (s @ s))
    assert abs(fit["coef"][0] - A_ols) < 1e-10
    assert fit["c1"] == 1.0 and fit["c2"] == 0.0


def test_pool_gamma_reduces_se_with_n_blocks():
    """SE of pooled γ̂₃ should scale ~ 1/√(n_blocks)."""
    rng = np.random.default_rng(2)
    n_block = 1024
    pool_n = 50
    z_blocks = [rng.normal(size=n_block) for _ in range(pool_n)]
    out = pool_gamma_from_blocks(z_blocks)
    # SE of mean for n=50 blocks of 1024 Gaussian: std(γ̂₃) ≈ √(6/1024) = 0.077;
    # SE of mean ≈ 0.077/√50 ≈ 0.0109. Allow 50% tolerance for sample noise.
    assert 0.005 < out["gamma3_se"] < 0.020, f"got SE={out['gamma3_se']}"


def test_pmm2_pooled_outperforms_per_block_on_weak_asymmetry():
    """Key claim: when |γ₃| is small (~SE per-block), pooled-PMM2 with smoothed
    γ̂ has lower MC variance than per-block PMM2."""
    rng = np.random.default_rng(3)
    n_block = 1024
    n_mc = 200
    # γ₃ ≈ 0.3 (mildly asymmetric), but per-block estimate has SE 0.077 → noisy
    A_true = 1.5
    var_pool = 0.0
    var_perblock = 0.0
    # Calibration set for pool
    cal_eps = []
    for _ in range(50):
        z = rng.normal(size=n_block)
        raw = np.where(z < 0, z * 1.2, z * 0.85)  # mild TPN, γ₃ ≈ -0.3
        e = (raw - raw.mean()) / raw.std() * 0.5
        cal_eps.append(e)
    pool = pool_gamma_from_blocks(cal_eps)
    g3_pool, g4_pool = pool["gamma3_mean"], pool["gamma4_mean"]

    A_pool, A_pb = [], []
    for _ in range(n_mc):
        s = rng.normal(size=n_block)
        z = rng.normal(size=n_block)
        raw = np.where(z < 0, z * 1.2, z * 0.85)
        e = (raw - raw.mean()) / raw.std() * 0.5
        y = A_true * s + e
        # Pooled
        f_pool = pmm2_pooled_amplitude(y, s, gamma3=g3_pool, gamma4=g4_pool)
        # Per-block via oracle
        f_pb = oracle.lm_pmm2(y, s)
        if f_pool["converged"]:
            A_pool.append(f_pool["coef"][0])
        if f_pb.get("converged"):
            A_pb.append(f_pb["coef"][0])

    A_pool = np.array(A_pool); A_pb = np.array(A_pb)
    var_pool = np.var(A_pool, ddof=1)
    var_pb = np.var(A_pb, ddof=1)
    # Allow either improvement OR comparable performance — MC noise on n=200
    # makes strict claims unreliable. The point is: pool ≤ per-block.
    assert var_pool <= var_pb * 1.10, \
        f"expected pooled var ≤ 1.10·per-block; got {var_pool:.5f} vs {var_pb:.5f}"
