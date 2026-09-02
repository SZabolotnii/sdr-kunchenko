"""Tests for the AUTO_EXT dispatcher routing rules.

Focus is the marginal-gain guard on the PMM2 branch: when the theoretical
variance-reduction factor g₂ = 1 − γ̂₃²/(2 + γ̂₄) is at or above ``g2_threshold``,
the expected gain over OLS is below 1 − g₂, so the block must be routed to OLS
rather than risking noisy cumulant weights for a negligible return.

The remaining tests pin the four routing levels so that the algorithm printed in
the manuscript stays in sync with the implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from sdr_kunchenko.pmm.dispatch_ext import pmm_dispatch_ext


def _skewed_block(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Asymmetric block with finite moments: γ̂₃ ≈ 0.6, γ̂₄ ≈ 0.5.

    A Gamma(k) variate has skewness 2/√k and excess kurtosis 6/k, so k = 11 keeps
    the block clear of the divergent-moment and heavy-tail guards while putting
    |γ̂₃| well above the symmetry threshold.
    """
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=11.0, size=n)


def test_pmm2_branch_activates_when_gain_is_material():
    """Strong skew with a low g₂ routes to PMM2 and reports g₂ in the rationale."""
    out = pmm_dispatch_ext(_skewed_block())
    g3, g4 = out["gamma3"], out["gamma4"]
    g2 = 1.0 - g3 ** 2 / (2.0 + g4)
    assert abs(g3) > 0.3
    assert g2 < 0.95
    assert out["method"] == "PMM2"


def test_g2_guard_reverts_marginal_pmm2_to_ols():
    """Skew above the threshold but g₂ ≥ 0.95 must fall back to OLS."""
    x = _skewed_block()
    # The guard fires on g₂ >= threshold, so a threshold above 1 disables it.
    out = pmm_dispatch_ext(x, g2_threshold=1.5)
    assert out["method"] == "PMM2", "sanity: guard disabled keeps PMM2"

    # A threshold below the block's own g2 forces the guard to fire.
    g3, g4 = out["gamma3"], out["gamma4"]
    g2 = 1.0 - g3 ** 2 / (2.0 + g4)
    guarded = pmm_dispatch_ext(x, g2_threshold=g2 - 1e-9)
    assert guarded["method"] == "OLS"
    assert "g₂" in guarded["rationale"]


def test_g2_guard_boundary_is_inclusive():
    """g₂ exactly at the threshold counts as marginal (>= comparison)."""
    x = _skewed_block()
    probe = pmm_dispatch_ext(x, g2_threshold=1.5)
    g3, g4 = probe["gamma3"], probe["gamma4"]
    g2 = 1.0 - g3 ** 2 / (2.0 + g4)
    assert pmm_dispatch_ext(x, g2_threshold=g2)["method"] == "OLS"


def test_guard_does_not_touch_symmetric_branches():
    """A symmetric platykurtic block reaches PMM3 regardless of g2_threshold."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, size=4000)  # γ̂₄ ≈ −1.2, γ̂₃ ≈ 0
    for thr in (0.0, 0.95, 1.5):
        out = pmm_dispatch_ext(x, g2_threshold=thr)
        assert out["method"] == "PMM3"


def test_gaussian_block_routes_to_ols():
    rng = np.random.default_rng(2)
    out = pmm_dispatch_ext(rng.normal(size=4000))
    assert out["method"] == "OLS"


def test_divergent_moment_guard_precedes_pmm2():
    """Heavy tails inflate the sample skewness; the Level-0 guard must win."""
    rng = np.random.default_rng(3)
    x = rng.standard_cauchy(size=4000)
    out = pmm_dispatch_ext(x, alpha_pool=0.4)
    assert out["method"] == "PMM_FRAC"
    assert out["alpha_used"] == pytest.approx(0.4)
