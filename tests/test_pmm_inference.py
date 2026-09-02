"""Tests for SE estimators (sandwich vs OLS vs bootstrap)."""
from __future__ import annotations

import numpy as np
import pytest

from sdr_kunchenko.pmm import oracle
from sdr_kunchenko.pmm.inference import (
    ols_se_amplitude,
    pmm2_bootstrap_se_amplitude,
    pmm2_sandwich_se_amplitude,
)


def test_sandwich_reduces_to_ols_for_gaussian():
    """When γ₃=γ₄=0, c₁=1 c₂=0 → ψ=ε/σ → Var(Â)=σ²/Σs². Should match OLS within FP."""
    rng = np.random.default_rng(0)
    n = 1024
    s = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    se_sw = pmm2_sandwich_se_amplitude(s, eps, gamma3=0.0, gamma4=0.0)
    se_ols = ols_se_amplitude(s, eps)
    rel = abs(se_sw - se_ols) / se_ols
    # Sandwich uses ddof=0, OLS uses ddof=1; ~0.5/n diff plus FP noise
    assert rel < 0.01, f"sandwich vs OLS diff: sandwich={se_sw}, ols={se_ols}, rel={rel}"


def test_sandwich_smaller_than_ols_on_asymmetric():
    """PMM2 sandwich SE should be smaller than OLS SE on heavily-asymmetric noise."""
    rng = np.random.default_rng(7)
    n = 2048
    s = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    raw = np.where(z < 0, z * 1.6, z * 0.7)
    eps = (raw - raw.mean()) / raw.std() * 0.5
    A_true = 1.5
    y = A_true * s + eps
    fit = oracle.lm_pmm2(y, s)
    se_sw = pmm2_sandwich_se_amplitude(s, fit["residuals"],
                                          gamma3=fit["gamma3"], gamma4=fit["gamma4"])
    se_ols = ols_se_amplitude(s, y - A_true * s)
    assert se_sw < se_ols, f"expected sandwich < OLS on TPN, got sandwich={se_sw}, ols={se_ols}"
    # Variance reduction ratio approx g₂ = 0.7
    ratio = (se_sw / se_ols) ** 2
    assert 0.5 < ratio < 0.95, f"ratio={ratio}, expected ~g₂≈0.7"


def test_hac_reduces_to_ols_for_white_residuals():
    """At lag L, HAC of white residuals ≈ OLS SE within ~5%."""
    from sdr_kunchenko.pmm.inference import hac_se_amplitude

    rng = np.random.default_rng(20)
    n = 1024
    s = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)  # white
    se_ols = ols_se_amplitude(s, eps)
    se_hac = hac_se_amplitude(s, eps)
    rel = abs(se_hac - se_ols) / se_ols
    assert rel < 0.10, f"HAC vs OLS on white residuals: {se_hac} vs {se_ols}, rel={rel:.3%}"


def test_hac_inflates_se_with_correlated_regressor_and_residuals():
    """HAC matters when BOTH regressor and residuals are autocorrelated.
    With an iid Gaussian regressor, even AR residuals give HAC ≈ OLS because
    s_t·ε_t has zero autocorrelation. We construct a smooth s (sinusoid) plus
    AR(1) ε to make the long-run variance of s·ε non-trivial."""
    from sdr_kunchenko.pmm.inference import hac_se_amplitude

    rng = np.random.default_rng(21)
    n = 2048
    t = np.arange(n)
    s = np.sin(2 * np.pi * 5 * t / n) + 0.3 * np.cos(2 * np.pi * 13 * t / n)
    # AR(1) ε with ρ=0.7
    e = np.empty(n)
    w = rng.normal(0, 0.5, n)
    e[0] = w[0]
    for k in range(1, n):
        e[k] = 0.7 * e[k - 1] + w[k]
    se_ols = ols_se_amplitude(s, e)
    se_hac = hac_se_amplitude(s, e)
    ratio = se_hac / se_ols
    # Smooth regressor × AR residuals → meaningful HAC inflation.
    assert ratio > 1.10, f"HAC/OLS on smooth-s × AR-ε: ratio={ratio:.3f}"


@pytest.mark.slow
def test_sandwich_agrees_with_bootstrap_within_15pct():
    """Bootstrap SE (slow) should agree with sandwich within 15% on Gaussian."""
    rng = np.random.default_rng(13)
    n = 1024
    s = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    A_true = 1.5
    y = A_true * s + eps
    fit = oracle.lm_pmm2(y, s)
    se_sw = pmm2_sandwich_se_amplitude(s, fit["residuals"],
                                          gamma3=fit["gamma3"], gamma4=fit["gamma4"])
    se_boot = pmm2_bootstrap_se_amplitude(y, s, B=50, seed=42)
    rel = abs(se_sw - se_boot) / se_boot
    assert rel < 0.15, f"sandwich {se_sw} vs bootstrap {se_boot}, rel={rel}"


def test_pmm3_hac_matches_sandwich_on_white_residuals():
    """With white residuals the score s·ψ has no serial correlation, so the
    HAC meat collapses to the sandwich meat → the two SEs agree within ~5%."""
    from sdr_kunchenko.pmm.inference import (
        pmm3_hac_se_amplitude,
        pmm3_sandwich_se_amplitude,
    )

    rng = np.random.default_rng(31)
    n = 2048
    s = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)  # white, symmetric
    kappa = 3.0 * float(np.var(eps, ddof=0))  # PMM3-scale κ ~ 3σ²
    se_sw = pmm3_sandwich_se_amplitude(s, eps, kappa=kappa)
    se_hac = pmm3_hac_se_amplitude(s, eps, kappa=kappa)
    rel = abs(se_hac - se_sw) / se_sw
    assert rel < 0.05, f"PMM3 HAC vs sandwich on white ε: {se_hac} vs {se_sw}, rel={rel:.3%}"


def test_pmm3_hac_inflates_se_with_correlated_regressor_and_residuals():
    """Smooth regressor × AR(1) residuals → the PMM3 score is autocorrelated,
    so HAC must exceed the naive sandwich SE (mirrors the OLS-HAC test)."""
    from sdr_kunchenko.pmm.inference import (
        pmm3_hac_se_amplitude,
        pmm3_sandwich_se_amplitude,
    )

    rng = np.random.default_rng(32)
    n = 2048
    t = np.arange(n)
    s = np.sin(2 * np.pi * 5 * t / n) + 0.3 * np.cos(2 * np.pi * 13 * t / n)
    e = np.empty(n)
    w = rng.normal(0, 0.5, n)
    e[0] = w[0]
    for k in range(1, n):
        e[k] = 0.7 * e[k - 1] + w[k]
    kappa = 3.0 * float(np.var(e, ddof=0))
    se_sw = pmm3_sandwich_se_amplitude(s, e, kappa=kappa)
    se_hac = pmm3_hac_se_amplitude(s, e, kappa=kappa)
    ratio = se_hac / se_sw
    assert ratio > 1.10, f"PMM3 HAC/sandwich on smooth-s × AR-ε: ratio={ratio:.3f}"


def test_pmm3_hac_degenerate_hessian_returns_inf():
    """κ chosen so that Σ s²(κ − 3ε²) ≈ 0 → J degenerate → inf (same guard as
    pmm3_sandwich_se_amplitude / pmm2_hac_se_amplitude)."""
    from sdr_kunchenko.pmm.inference import pmm3_hac_se_amplitude

    rng = np.random.default_rng(33)
    n = 512
    s = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    s2 = s ** 2
    kappa_deg = float(3.0 * np.sum(s2 * eps ** 2) / np.sum(s2))
    assert np.isinf(pmm3_hac_se_amplitude(s, eps, kappa=kappa_deg))


def test_pmm3_hac_lag_zero_equals_sandwich_exactly():
    """lag=0 disables the Bartlett cross-terms, so the HAC meat is exactly
    Σ (s·ψ)² — identical to the sandwich M term."""
    from sdr_kunchenko.pmm.inference import (
        pmm3_hac_se_amplitude,
        pmm3_sandwich_se_amplitude,
    )

    rng = np.random.default_rng(34)
    n = 256
    s = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    kappa = 3.0 * float(np.var(eps, ddof=0))
    assert pmm3_hac_se_amplitude(s, eps, kappa=kappa, lag=0) == pytest.approx(
        pmm3_sandwich_se_amplitude(s, eps, kappa=kappa), rel=1e-12
    )


def test_pmm3_hac_on_oracle_fit():
    """End-to-end: κ and residuals from EstemPMM lm_pmm3 give a finite, positive
    HAC SE of a plausible magnitude (same order as the OLS SE)."""
    from sdr_kunchenko.pmm.inference import pmm3_hac_se_amplitude

    rng = np.random.default_rng(35)
    n = 1024
    s = rng.normal(0, 1, n)
    # Platykurtic (uniform) noise — PMM3's target regime.
    eps = rng.uniform(-1, 1, n) * 0.5
    A_true = 1.5
    y = A_true * s + eps
    fit = oracle.lm_pmm3(y, s)
    se_hac = pmm3_hac_se_amplitude(s, fit["residuals"], kappa=fit["kappa"])
    se_ols = ols_se_amplitude(s, y - A_true * s)
    assert np.isfinite(se_hac) and se_hac > 0
    assert 0.1 * se_ols < se_hac < 10 * se_ols, f"se_hac={se_hac}, se_ols={se_ols}"
