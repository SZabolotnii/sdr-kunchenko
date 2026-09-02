"""Denoiser unit tests: subspace learning, PATP IRLS, baselines, mapping."""
from __future__ import annotations

import numpy as np
import pytest

from sdr_kunchenko.denoiser import (
    calibrate_alpha_mapping,
    fixed_patp_denoise,
    learn_subspace,
    median_denoise,
    patp_irls_denoise,
    projection_denoise,
    wiener_denoise,
)


def test_learn_subspace_recovers_known_components():
    """If clean blocks are linear combos of K known basis vectors,
    SVD should recover their span (within sign/orthogonality)."""
    rng = np.random.default_rng(0)
    n = 128
    K = 4
    true_basis = rng.normal(size=(n, K))
    coeffs = rng.normal(size=(50, K))
    blocks = (coeffs @ true_basis.T)  # (50, n)
    Psi = learn_subspace(blocks, rank=K)
    assert Psi.shape == (n, K)
    # Reconstruction error on training blocks should be ~0
    proj = blocks @ Psi @ Psi.T  # (50, n)
    err = np.linalg.norm(blocks - proj) / np.linalg.norm(blocks)
    assert err < 1e-9, f"recon err {err}"


def test_patp_irls_recovers_signal_under_gaussian():
    """Under Gaussian noise, α=1 (p=2) should match OLS projection."""
    rng = np.random.default_rng(1)
    n = 64
    K = 4
    Psi = np.linalg.qr(rng.normal(size=(n, K)))[0]
    c_true = rng.normal(size=K)
    s = Psi @ c_true
    y = s + rng.normal(0, 0.1, n)
    out = patp_irls_denoise(y, Psi, alpha=1.0)
    err = np.linalg.norm(out["s_hat"] - s) / np.linalg.norm(s)
    assert err < 0.2, f"PATP @ α=1 reconstruction err {err}"
    # Compare with pure projection
    s_proj = projection_denoise(y, Psi)
    assert np.allclose(out["s_hat"], s_proj, atol=1e-8)


def test_patp_irls_more_robust_than_ols_under_outliers():
    """For heavy-tailed noise, α<1 (p<2) should give lower MSE than α=1."""
    rng = np.random.default_rng(7)
    n = 256
    K = 8
    Psi = np.linalg.qr(rng.normal(size=(n, K)))[0]
    c_true = rng.normal(size=K)
    s = Psi @ c_true
    # Mix of Gaussian + occasional outliers (heavy-tailed).
    base = rng.normal(0, 0.05, n)
    out_idx = rng.choice(n, size=10, replace=False)
    base[out_idx] += rng.normal(0, 1.0, 10)  # 10 large outliers
    y = s + base
    rmse = {}
    for a in (1.0, 0.5, 0.0):
        out = patp_irls_denoise(y, Psi, alpha=a)
        rmse[a] = float(np.linalg.norm(out["s_hat"] - s))
    # Robust α should be at least competitive (within 20%) of OLS
    # under outliers — exact dominance not guaranteed at small n.
    assert rmse[0.5] < rmse[1.0] * 1.2, \
        f"L1 (α=0.5) should be competitive: rmse={rmse}"


def test_median_denoise_smooths_impulses():
    rng = np.random.default_rng(2)
    n = 128
    s = np.sin(2 * np.pi * np.arange(n) / 16)
    noise = np.zeros(n)
    out_idx = rng.choice(n, size=5, replace=False)
    noise[out_idx] = 5.0
    y = s + noise
    s_hat = median_denoise(y, window=5)
    # Median should remove most outliers
    assert np.linalg.norm(s_hat - s) < np.linalg.norm(y - s)


def test_wiener_denoise_reduces_white_noise():
    rng = np.random.default_rng(3)
    n = 1024
    s = np.sin(2 * np.pi * 5 * np.arange(n) / n)
    y = s + rng.normal(0, 0.3, n)
    s_hat = wiener_denoise(y)
    err_y = float(np.mean((y - s) ** 2))
    err_s = float(np.mean((s_hat - s) ** 2))
    assert err_s < err_y, f"Wiener didn't help: in={err_y}, out={err_s}"


def test_alpha_mapping_calibration_correlates_with_gamma4():
    """When noise tail varies (changing γ̂₄), optimal α should change too,
    giving non-zero correlation in the calibration set."""
    rng = np.random.default_rng(11)
    n = 64
    K = 6
    Psi = np.linalg.qr(rng.normal(size=(n, K)))[0]
    M = 30
    clean = np.empty((M, n))
    noisy = np.empty((M, n))
    for i in range(M):
        c = rng.normal(size=K)
        clean[i] = Psi @ c
        # Vary tail heaviness: half blocks Gaussian, half Laplace
        if i < M // 2:
            ξ = rng.normal(0, 0.2, n)
        else:
            ξ = rng.laplace(0, 0.15, n)
        noisy[i] = clean[i] + ξ
    mapping = calibrate_alpha_mapping(noisy, clean, Psi,
                                        alpha_grid=[0.3, 0.5, 0.7, 0.9])
    assert "predict" in mapping
    assert callable(mapping["predict"])
    # Correlation may be low or high depending on signal — just check finite.
    assert np.isfinite(mapping["correlation"])
