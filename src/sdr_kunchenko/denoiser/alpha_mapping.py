"""Calibration of α=f(γ̂₄) mapping for PATP denoiser.

Calibration phase (offline, on training set):
    For each block:
        compute γ̂₄ from the noise residual
        for α in α_grid:
            denoise with α, compute reconstruction MSE
        α*(block) = argmin_α MSE

    Fit α* = f(γ̂₄) — polynomial regression (degree 2) by default.

Inference phase:
    Estimate γ̂₄ from initial residual y − ŝ_init (where ŝ_init = median or
    projection denoiser output).
    α̂ = f(γ̂₄)
    ŝ = patp_irls_denoise(y, Ψ, α̂)
"""
from __future__ import annotations

import numpy as np

from sdr_kunchenko.denoiser.baselines import median_denoise, projection_denoise
from sdr_kunchenko.denoiser.patp_denoiser import patp_irls_denoise
from sdr_kunchenko.utils.cumulants import standardized_cumulants


def estimate_gamma4_from_residual(y: np.ndarray, init_estimate: np.ndarray) -> float:
    """γ̂₄ from (y − ŝ_init)."""
    return float(standardized_cumulants(y - init_estimate)["gamma4"])


def find_optimal_alpha(y: np.ndarray, clean_truth: np.ndarray, Psi: np.ndarray,
                        alpha_grid: list[float]) -> tuple[float, float]:
    """For one block, sweep α and return argmin MSE + best MSE."""
    best_alpha, best_mse = alpha_grid[0], float("inf")
    for a in alpha_grid:
        out = patp_irls_denoise(y, Psi, a)
        mse = float(np.mean((out["s_hat"] - clean_truth) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = a
    return best_alpha, best_mse


def calibrate_alpha_mapping(noisy_blocks: np.ndarray, clean_blocks: np.ndarray,
                              Psi: np.ndarray, *,
                              alpha_grid: list[float] | None = None,
                              poly_degree: int = 2,
                              init_method: str = "projection",
                              ) -> dict:
    """Fit α* = f(γ̂₄) on training pairs (y, s).

    Parameters
    ----------
    noisy_blocks, clean_blocks : (M, N) arrays
    Psi : (N, K) signal subspace
    alpha_grid : list of α values to try (default 11 points)
    poly_degree : polynomial fit order

    Returns
    -------
    dict with mapping coefficients, training γ̂₄, α* arrays, and a callable
    `predict(γ4) → α`.
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    M = noisy_blocks.shape[0]
    g4_list = np.empty(M)
    a_list = np.empty(M)
    for i in range(M):
        y = noisy_blocks[i]
        s = clean_blocks[i]
        if init_method == "projection":
            s_init = projection_denoise(y, Psi)
        else:
            s_init = median_denoise(y, window=5)
        g4_list[i] = estimate_gamma4_from_residual(y, s_init)
        a_list[i], _ = find_optimal_alpha(y, s, Psi, alpha_grid)

    # Polynomial regression α* = f(γ̂₄).
    coefs = np.polyfit(g4_list, a_list, deg=poly_degree)
    poly = np.poly1d(coefs)

    def predict(g4: float) -> float:
        a = float(np.clip(poly(g4), 0.0, 1.0))
        return a

    return {
        "alpha_grid": alpha_grid,
        "training_gamma4": g4_list,
        "training_alpha": a_list,
        "poly_coefs": coefs,
        "poly_degree": poly_degree,
        "init_method": init_method,
        "predict": predict,
        "correlation": float(np.corrcoef(g4_list, a_list)[0, 1]),
    }


def adaptive_denoise(y: np.ndarray, Psi: np.ndarray, mapping: dict,
                      init_method: str = "projection") -> dict:
    """Run PATP-adaptive denoiser using calibrated α(γ̂₄) mapping."""
    if init_method == "projection":
        s_init = projection_denoise(y, Psi)
    else:
        s_init = median_denoise(y, window=5)
    g4 = estimate_gamma4_from_residual(y, s_init)
    a_hat = mapping["predict"](g4)
    out = patp_irls_denoise(y, Psi, a_hat)
    out["gamma4_est"] = g4
    out["alpha_used"] = a_hat
    return out


__all__ = [
    "estimate_gamma4_from_residual",
    "find_optimal_alpha",
    "calibrate_alpha_mapping",
    "adaptive_denoise",
]
