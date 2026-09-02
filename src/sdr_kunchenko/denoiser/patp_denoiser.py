"""PATP-adaptive denoiser.

Approach (Spec §7.2 reformulated for SDR baseband):
    Given y = s + ξ where s lies in a low-rank subspace Ψ learned from clean
    training data, find ŝ via PATP-weighted regression:

        ĉ = argmin_c  Σ_n |y[n] − (Ψ·c)[n]|^{p(α)},     ŝ = Ψ·ĉ

    where p(α) = patp_power(i=2, α) ∈ [0.5, 2] is the leading PATP exponent.

    α=1.0  → p=2 (OLS, optimal for Gaussian residuals)
    α=0.5  → p=1 (L1, optimal for Laplace-tail residuals)
    α=0.0  → p=0.5 (sub-linear, robust to extreme outliers)

IRLS (Iteratively Reweighted Least Squares) solver.

The α parameter is calibrated by `alpha_mapping.calibrate(γ̂₄)` on a held-out
training split: for each γ₄ bin, find α* that minimises validation MSE.
"""
from __future__ import annotations

import numpy as np

from sdr_kunchenko.gsa.basis import patp_power


def learn_subspace(clean_blocks: np.ndarray, *, rank: int = 16) -> np.ndarray:
    """Learn signal subspace via truncated SVD on stacked clean blocks.

    Parameters
    ----------
    clean_blocks : (M, N) — M clean signals of length N
    rank : K, number of leading singular vectors to keep

    Returns
    -------
    Psi : (N, K) — orthonormal basis for the signal subspace
    """
    X = np.asarray(clean_blocks, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # SVD on (M, N). We want column basis (N-dim).
    # Center per-block? Clean signals have approximate zero mean already.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Vt has shape (min(M,N), N). Top-K rows are leading right singular vectors.
    K = min(rank, Vt.shape[0])
    return Vt[:K, :].T  # (N, K)


def patp_irls_denoise(y: np.ndarray, Psi: np.ndarray, alpha: float, *,
                       max_iter: int = 30, tol: float = 1e-6,
                       eps: float = 1e-3) -> dict:
    """IRLS for Lp regression onto subspace Ψ with PATP exponent.

    Iteratively solves
        ĉ_{t+1} = argmin_c Σ w_n^{(t)} (y[n] − (Ψc)[n])²
    with weights w_n^{(t)} = |residual_n|^{p−2}, where p = p_2(α).

    For p ≤ 1, the weights are amplified for small residuals and damped for
    large ones — robust behaviour. For p = 2, weights = 1 (OLS one-shot).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    Psi = np.asarray(Psi, dtype=np.float64)
    p = patp_power(2, alpha)  # leading PATP exponent
    # OLS init
    c = np.linalg.lstsq(Psi, y, rcond=None)[0]
    history = []
    for it in range(max_iter):
        s_hat = Psi @ c
        residuals = y - s_hat
        # IRLS weight: ∂/∂c L_p = 2 (y - Ψc) |y - Ψc|^{p-2} → weighted LS.
        if abs(p - 2.0) < 1e-9:
            # OLS — converged in one step.
            return {"s_hat": s_hat, "c": c, "iterations": 0,
                     "converged": True, "p_used": p}
        w = (np.abs(residuals) + eps) ** (p - 2.0)
        # Solve weighted LS:  Psi^T diag(w) Psi · c = Psi^T diag(w) y
        WPsi = Psi * w[:, None]
        G = Psi.T @ WPsi
        rhs = Psi.T @ (w * y)
        try:
            c_new = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            c_new, *_ = np.linalg.lstsq(G, rhs, rcond=None)
        delta = float(np.linalg.norm(c_new - c) / (np.linalg.norm(c) + 1e-12))
        history.append(delta)
        c = c_new
        if delta < tol:
            return {"s_hat": Psi @ c, "c": c, "iterations": it + 1,
                     "converged": True, "p_used": p, "history": history}
    return {"s_hat": Psi @ c, "c": c, "iterations": max_iter,
            "converged": False, "p_used": p, "history": history}


def fixed_patp_denoise(y: np.ndarray, Psi: np.ndarray, alpha: float = 0.5,
                        **kwargs) -> dict:
    """Wrapper to mark a 'fixed-α' baseline (e.g. α=0.5 = L1)."""
    return patp_irls_denoise(y, Psi, alpha, **kwargs)


__all__ = ["learn_subspace", "patp_irls_denoise", "fixed_patp_denoise"]
