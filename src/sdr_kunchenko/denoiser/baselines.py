"""Classical denoiser baselines for Exp 4 comparison."""
from __future__ import annotations

import numpy as np
from scipy.signal import medfilt


def median_denoise(y: np.ndarray, *, window: int = 5) -> np.ndarray:
    """Median filter — non-linear smoothing, robust to impulse noise."""
    return medfilt(np.asarray(y, dtype=np.float64).ravel(), kernel_size=window)


def wiener_denoise(y: np.ndarray, *, sigma2_noise: float | None = None) -> np.ndarray:
    """Frequency-domain Wiener filter.

    H(f) = |S(f)|² / (|S(f)|² + σ²_noise)
    where |S(f)|² is estimated from periodogram of y minus σ².
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    Y = np.fft.rfft(y)
    P = (np.abs(Y) ** 2) / len(y)
    if sigma2_noise is None:
        # Robust noise variance from high-frequency tail (top 25% bins).
        n_tail = max(1, len(P) // 4)
        sigma2_noise = float(np.median(P[-n_tail:]))
    # Signal PSD estimate (clipped to non-negative).
    P_signal = np.clip(P - sigma2_noise, 0.0, None)
    H = P_signal / (P_signal + sigma2_noise + 1e-12)
    Y_hat = H * Y
    return np.fft.irfft(Y_hat, n=len(y))


def projection_denoise(y: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """Pure subspace projection (OLS) onto learned Ψ — α=1 PATP limit baseline."""
    y = np.asarray(y, dtype=np.float64).ravel()
    Psi = np.asarray(Psi, dtype=np.float64)
    c = np.linalg.lstsq(Psi, y, rcond=None)[0]
    return Psi @ c


__all__ = ["median_denoise", "wiener_denoise", "projection_denoise"]
