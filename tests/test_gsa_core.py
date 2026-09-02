"""GSA core tests: basis, coefficients, detector, threshold."""
from __future__ import annotations

import numpy as np
import pytest

from sdr_kunchenko.gsa import (
    EnergyDetector,
    GSACUSUMDetector,
    evaluate_basis,
    fit_gsa_coefficients,
    lambda_score,
    make_basis,
    patp_power,
    pe_threshold,
    simulation_threshold,
)


# ------------------------------------------------------ basis sanity


def test_patp_power_endpoints():
    """At α=0: p_i=1/i (frac); at α=1: p_i=i (poly); at α=0.5: p_i=1 (linear)."""
    for i in range(2, 6):
        assert abs(patp_power(i, 0.0) - 1.0 / i) < 1e-12
        assert abs(patp_power(i, 1.0) - i) < 1e-12
        assert abs(patp_power(i, 0.5) - 1.0) < 1e-12


def test_frac_and_patp_are_sign_preserving():
    """sign(x)·|x|^p — odd by construction. Poly includes even powers (x²)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    basis = make_basis("frac", 3)
    for fn in basis:
        assert np.allclose(fn(-x), -fn(x), atol=1e-12), "frac not odd"
    for alpha in (0.0, 0.3, 0.5, 0.8, 1.0):
        basis = make_basis("patp", 3, alpha=alpha)
        for fn in basis:
            assert np.allclose(fn(-x), -fn(x), atol=1e-12), \
                f"PATP α={alpha} not odd"


def test_poly_basis_includes_even_powers():
    """φ_2(x) = x² is even, not odd."""
    basis = make_basis("poly", 2)
    x = np.array([1.0, -1.0, 2.0, -2.0])
    assert np.allclose(basis[1](x), x ** 2)
    # Even — φ(-x) = φ(x)
    assert np.allclose(basis[1](-x), basis[1](x))


def test_augmented_patp_keeps_linear_term():
    """The augmented PATP comparison keeps φ1(x)=x as the first basis function."""
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    patp_aug = make_basis("patp_aug", 3, alpha=1.0)
    assert len(patp_aug) == 3
    assert np.allclose(patp_aug[0](x), x, atol=1e-12)
    assert np.allclose(patp_aug[1](x), np.sign(x) * np.abs(x) ** 2, atol=1e-12)


# ------------------------------------------------------ FK=Y system


def test_gsa_recovers_amplitude_under_gaussian():
    """For y = A·s + ξ, ξ~N(0,σ²), with constant s=1 and poly s=1 basis,
    GSA k_1 ≈ A/σ² (LLR slope for Gaussian mean shift)."""
    rng = np.random.default_rng(42)
    sigma = 0.5
    A = 0.5
    n = 50_000
    h0 = rng.normal(0, sigma, n)        # ξ alone
    h1 = A + rng.normal(0, sigma, n)    # A + ξ
    basis = make_basis("poly", 1)
    fit = fit_gsa_coefficients(basis, h0, h1)
    # E[x|H1] − E[x|H0] = A; Var(x|H0) = σ². So k_1 = A/σ² ≈ 2.0.
    expected = A / sigma ** 2
    assert abs(fit["k"][0] - expected) < 0.1, \
        f"k₁={fit['k'][0]}, expected ≈ {expected}"
    assert fit["J"] > 0


def test_J_monotone_with_s():
    """J(s) should be monotone non-decreasing as we extend the basis."""
    rng = np.random.default_rng(7)
    n = 20_000
    sigma = 1.0
    A = 0.4
    h0 = rng.normal(0, sigma, n)
    h1 = A + rng.normal(0, sigma, n)
    Js = []
    for s in (1, 2, 3):
        basis = make_basis("poly", s)
        fit = fit_gsa_coefficients(basis, h0, h1)
        Js.append(fit["J"])
    assert Js[1] >= Js[0] - 1e-3, f"J(2)={Js[1]} < J(1)={Js[0]}"
    assert Js[2] >= Js[1] - 1e-3, f"J(3)={Js[2]} < J(2)={Js[1]}"


# ------------------------------------------------------ detector


def test_energy_detector_fires_on_signal():
    rng = np.random.default_rng(11)
    sigma = 0.5
    n = 1024
    s_template = np.cos(2 * np.pi * np.arange(n) / n)
    A = 1.0
    h0 = rng.normal(0, sigma, n)
    h1 = A * s_template + rng.normal(0, sigma, n)

    # Calibrate energy threshold from H0 directly: 99% quantile of cumulative |x|².
    det_h0 = EnergyDetector(threshold=sigma ** 2 * 5.0, mean_under_h0=sigma ** 2)
    det_h1 = EnergyDetector(threshold=sigma ** 2 * 5.0, mean_under_h0=sigma ** 2)
    out_h0 = det_h0.feed(h0)
    out_h1 = det_h1.feed(h1)
    # H1 should fire much earlier (or at all) than H0.
    if out_h0["alarm"]:
        # Both fired: H1 must fire earlier
        assert out_h1["alarm"]
        assert out_h1["fired_at"] <= out_h0["fired_at"], \
            f"energy: H1@{out_h1['fired_at']}, H0@{out_h0['fired_at']}"
    else:
        # Only H1 fires — that's the typical happy path.
        assert out_h1["alarm"], "energy detector failed to fire on signal"


def test_gsa_detector_FAR_calibrated_via_simulation():
    """Calibrate GSA-poly-s=1 via per-block simulation threshold, verify FAR ≤ ε on fresh H0."""
    rng = np.random.default_rng(13)
    sigma = 1.0
    n_per_block = 256
    n_blocks_calib = 200
    n_blocks_test = 200
    A = 0.5
    eps = 0.05

    # Calibrate basis & coefficients
    h0_calib_flat = rng.normal(0, sigma, n_per_block * n_blocks_calib)
    h1_calib_flat = A + rng.normal(0, sigma, n_per_block * n_blocks_calib)
    basis = make_basis("poly", 1)
    fit = fit_gsa_coefficients(basis, h0_calib_flat, h1_calib_flat)
    # Use a separate H0 block-resolved set for simulation threshold
    h0_blocks_calib = rng.normal(0, sigma, (n_blocks_calib, n_per_block))
    th_sim = simulation_threshold(basis, fit["k"], h0_blocks_calib, epsilon=eps)

    n_alarms = 0
    for _ in range(n_blocks_test):
        h0_test = rng.normal(0, sigma, n_per_block)
        # Reframe: alarm when CUSUM exceeds simulated threshold (drift=0).
        det = GSACUSUMDetector(basis, fit["k"], threshold=0.0)
        out = det.feed(h0_test)
        if out["max_S"] > th_sim["h"]:
            n_alarms += 1
    far = n_alarms / n_blocks_test
    # Simulation threshold guarantees ε empirically; allow 2× slack on test stochasticity.
    assert far <= 2 * eps, f"GSA per-block FAR = {far}, expected ≤ {2*eps}"


# ------------------------------------------------------ threshold simulation


def test_simulation_threshold_returns_finite():
    rng = np.random.default_rng(17)
    M, N = 30, 256
    h0_blocks = rng.normal(0, 1, (M, N))
    basis = make_basis("poly", 1)
    fit = fit_gsa_coefficients(basis, h0_blocks.ravel(),
                                h0_blocks.ravel() + 0.5)  # mock H1
    th = simulation_threshold(basis, fit["k"], h0_blocks, epsilon=0.1)
    assert np.isfinite(th["h"]) and th["h"] > 0
    assert th["n_blocks"] == M
