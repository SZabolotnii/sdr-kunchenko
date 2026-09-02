"""GSA coefficient solver: F·k = Y system.

Spec §A.3:
    F_{ij} = E[φ_i(X)·φ_j(X) | H₀]
    Y_i   = E[φ_i(X) | H₁] − E[φ_i(X) | H₀]
    k     = F⁻¹·Y

The information functional J(s) = k^T·Y is the criterion of convergence
(Theorem 2 of GSA-CP skill: J(s) ≤ ‖z‖², monotone in s, → ‖z‖² as s→∞).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def empirical_moments(basis: Sequence[Callable], samples: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Compute E[φ_i] (length-s vector) and E[φ_i·φ_j] (s×s matrix).

    `samples` shape: (M, N) — M blocks of N samples each, OR (M·N,) flat.
    Returns (mean_phi, second_moment_matrix).
    """
    samples = np.asarray(samples, dtype=np.float64).ravel()
    s = len(basis)
    Phi = np.empty((len(samples), s))
    for i, fn in enumerate(basis):
        Phi[:, i] = fn(samples)
    mean_phi = Phi.mean(axis=0)
    M = (Phi.T @ Phi) / len(samples)
    return mean_phi, M


def solve_FK(F: np.ndarray, Y: np.ndarray, *, ridge: float = 1e-8) -> np.ndarray:
    """Solve F·k = Y with optional ridge regularization for ill-conditioned F."""
    s = F.shape[0]
    F_reg = F + ridge * np.trace(F) / s * np.eye(s)
    try:
        k = np.linalg.solve(F_reg, Y)
    except np.linalg.LinAlgError:
        k, *_ = np.linalg.lstsq(F_reg, Y, rcond=None)
    return k


def fit_gsa_coefficients(basis: Sequence[Callable],
                          h0_samples: np.ndarray,
                          h1_samples: np.ndarray,
                          *, ridge: float = 1e-8
                          ) -> dict:
    """Fit GSA coefficients k from H₀ and H₁ calibration samples.

    Returns dict with keys:
        F            — second-moment matrix under H₀ (s×s)
        Y            — mean-shift vector E_H1 − E_H0 (s,)
        k            — F⁻¹·Y coefficients (s,)
        J            — information functional k^T·Y (scalar)
        cond_F       — condition number of F (numerical health check)
        E0, E1       — H₀, H₁ mean vectors of φ
    """
    E0, M0 = empirical_moments(basis, h0_samples)
    E1, _ = empirical_moments(basis, h1_samples)
    # F = Cov_{H0} = M0 − E0·E0^T  (centered second moment).
    F = M0 - np.outer(E0, E0)
    Y = E1 - E0
    k = solve_FK(F, Y, ridge=ridge)
    J = float(k @ Y)
    return {
        "F": F, "Y": Y, "k": k, "J": J,
        "E0": E0, "E1": E1,
        "cond_F": float(np.linalg.cond(F)),
    }


def lambda_score(basis: Sequence[Callable], k: np.ndarray, x: np.ndarray
                  ) -> np.ndarray:
    """Λ̂(x) = Σ k_i · φ_i(x). Vectorized over `x` (any shape)."""
    s = len(basis)
    if k.shape[-1] != s:
        raise ValueError(f"k length {len(k)} ≠ basis length {s}")
    out = np.zeros_like(np.asarray(x, dtype=np.float64))
    for i, fn in enumerate(basis):
        out = out + k[i] * fn(x)
    return out


__all__ = [
    "empirical_moments", "solve_FK", "fit_gsa_coefficients", "lambda_score",
]
