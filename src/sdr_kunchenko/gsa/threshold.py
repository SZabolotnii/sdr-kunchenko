"""PE threshold calibration for GSA-CUSUM detector.

Spec §A.3 (Probability of Exceedance):
    h = E[Λ̂ | H₀] + √(Var(Λ̂ | H₀) / ε)

Cheap Chebyshev-style bound: P(Λ̂ − E ≥ √(Var/ε)) ≤ ε.
Guarantees FAR ≤ ε per-block (loose; simulation-based threshold sharpens).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .coefficients import lambda_score


def pe_threshold(basis: Sequence[Callable], k: np.ndarray,
                 h0_samples: np.ndarray, *, epsilon: float = 0.01,
                 scale: float = 1.0) -> dict:
    """Compute PE threshold from per-sample Λ̂ statistics under H₀.

    h = E[Λ̂|H₀] + scale · √(Var(Λ̂|H₀)/ε)

    Returns dict with mean, var, threshold, scale.
    """
    samples = np.asarray(h0_samples, dtype=np.float64).ravel()
    lam = lambda_score(basis, k, samples)
    mu = float(lam.mean())
    var = float(lam.var(ddof=1))
    h = mu + scale * np.sqrt(var / epsilon)
    return {"mu0": mu, "var0": var, "h": float(h), "epsilon": epsilon,
            "scale": scale}


def simulation_threshold(basis: Sequence[Callable], k: np.ndarray,
                          h0_samples: np.ndarray, *, epsilon: float = 0.01
                          ) -> dict:
    """Simulation-based threshold: empirical (1-ε) quantile of per-block CUSUM peak.

    Tighter than PE/Chebyshev. h0_samples should be (M, N) — M blocks of N samples.
    For each block, simulate CUSUM with current threshold guess and record max.
    """
    arr = np.asarray(h0_samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Per-sample LLR values, then per-block CUSUM peak at h=0 (no early-stop).
    M, N = arr.shape
    peaks = np.empty(M)
    for m in range(M):
        lam = lambda_score(basis, k, arr[m])
        S = 0.0
        peak = 0.0
        for v in lam:
            S = max(0.0, S + v)
            peak = max(peak, S)
        peaks[m] = peak
    # Use a slightly conservative (1−ε/2) quantile to compensate for the
    # discrete CDF inverse and small-sample bias on the tail.
    h = float(np.quantile(peaks, 1.0 - 0.5 * epsilon))
    return {"h": h, "epsilon": epsilon, "peaks_mean": float(peaks.mean()),
            "peaks_max": float(peaks.max()), "n_blocks": int(M)}


__all__ = ["pe_threshold", "simulation_threshold"]
