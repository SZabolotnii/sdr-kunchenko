"""PMM2 / PMM3 amplitude estimators with *frozen* (pre-pooled) cumulant weights.

Motivation (Phase 1 finding): on N=1024 blocks, per-block γ̂₃ has SE ≈ √(6/N) ≈ 0.077.
When the true |γ₃| is small (e.g. ~0 for symmetric polygauss), per-block γ̂₃ is
dominated by sample noise, and PMM2 weights c₁(γ̂₃, γ̂₄), c₂(γ̂₃, γ̂₄) are random
— this *amplifies* variance instead of reducing it.

Mitigation per Spec §4.5: pre-aggregate γ̂ over blocks of the same regime
(same SNR bucket) to obtain a low-variance pooled estimate, then apply PMM2 with
those frozen weights.

This module implements pure-Python frozen-weight estimators (we cannot ask
EstemPMM to use externally-supplied gammas — its NR loop re-estimates them).
The math is identical to EstemPMM's analytic form (Spec §A.1, A.2);
bit-equivalence with EstemPMM at γ̂_per_block agreement is checked in tests.
"""
from __future__ import annotations

import numpy as np


def pmm2_pooled_amplitude(y: np.ndarray, s: np.ndarray, *,
                          gamma3: float, gamma4: float,
                          max_iter: int = 25, tol: float = 1e-8) -> dict:
    """PMM2 amplitude with frozen (pooled) γ₃, γ₄.

    Newton iterations on the score:
      Σ s · ψ(ε) = 0,
      ψ(ε) = c₁·(ε/σ) + c₂·((ε/σ)² − 1)
    where (c₁, c₂) are computed once from frozen γ₃, γ₄ (independent of ε̂).
    σ² is re-estimated each iteration from current residuals.

    Returns dict {coef, residuals, sigma, iterations, converged, gamma3_used,
                   gamma4_used}.
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    denom = 2.0 + gamma4 - gamma3 ** 2
    if abs(denom) < 1e-12:
        # Degenerate — fall back to OLS.
        A = float(y @ s / (s @ s))
        return {"coef": np.array([A]), "residuals": y - A * s,
                "sigma": float(np.std(y - A * s, ddof=1)),
                "iterations": 0, "converged": False,
                "gamma3_used": gamma3, "gamma4_used": gamma4}
    c1 = (2.0 + gamma4) / denom
    c2 = -gamma3 / denom

    s2_sum = float(s @ s)
    if s2_sum <= 0:
        return {"coef": np.array([np.nan]), "residuals": y.copy(),
                "sigma": float("nan"), "iterations": 0, "converged": False,
                "gamma3_used": gamma3, "gamma4_used": gamma4}

    A = float(y @ s / s2_sum)  # OLS init
    for it in range(max_iter):
        eps = y - A * s
        m2 = float(np.var(eps, ddof=0))
        if m2 <= 0:
            break
        sigma = np.sqrt(m2)
        z = eps / sigma                     # standardized residuals
        psi = c1 * z + c2 * (z * z - 1.0)
        # ∂ψ/∂A = ψ'(ε)·(-s) where ψ'(ε) = c1/σ + 2·c2·ε/σ²
        psi_prime = c1 / sigma + 2.0 * c2 * eps / m2
        score = float(np.sum(s * psi))
        hess = float(np.sum(s * psi_prime * s))
        if abs(hess) < 1e-12:
            break
        delta = score / hess
        A_new = A + delta
        if not np.isfinite(A_new):
            break
        # Step-size cap — same spirit as EstemPMM's adaptive guard.
        if abs(delta) > 5.0:
            delta = np.sign(delta) * 5.0
            A_new = A + delta
        A = A_new
        if abs(delta) < tol:
            return {"coef": np.array([A]), "residuals": y - A * s,
                    "sigma": sigma, "iterations": it + 1, "converged": True,
                    "gamma3_used": gamma3, "gamma4_used": gamma4,
                    "c1": c1, "c2": c2}
    return {"coef": np.array([A]), "residuals": y - A * s,
            "sigma": float(np.sqrt(np.var(y - A * s, ddof=0))),
            "iterations": max_iter, "converged": False,
            "gamma3_used": gamma3, "gamma4_used": gamma4,
            "c1": c1, "c2": c2}


def kappa_from_pool(gamma4_pool: float, gamma6_pool: float, m2_block: float) -> float:
    """Derive PMM3 κ from pooled γ̂₄, γ̂₆ and per-block σ²=m₂.

    Identity (EstemPMM convention for symmetric distributions):
      m₄ = (γ₄ + 3)·m₂²,  m₆ = (γ₆ + 15γ₄ + 15)·m₂³
      κ  = (m₆ − 3·m₄·m₂) / (m₄ − 3·m₂²)
         = m₂ · (γ₆ + 12·γ₄ + 6) / γ₄

    The denominator γ₄ vanishes for Gaussian; caller should guard with |γ₄|>ε.
    """
    if abs(gamma4_pool) < 1e-6:
        return 0.0  # Gaussian fallback — PMM3 won't gain anyway.
    return m2_block * (gamma6_pool + 12.0 * gamma4_pool + 6.0) / gamma4_pool


def pmm3_pooled_amplitude(y: np.ndarray, s: np.ndarray, *,
                          kappa: float | None = None,
                          gamma4_pool: float | None = None,
                          gamma6_pool: float | None = None,
                          max_iter: int = 50, tol: float = 1e-8) -> dict:
    """PMM3 amplitude with frozen κ.

    Two ways to specify the frozen weight:
      - `kappa` directly (legacy);
      - `gamma4_pool`, `gamma6_pool` — recommended, derives κ per-block from
        the pool's standardised cumulants and the block's own σ². This is more
        stable than pooling κ across blocks (κ depends on σ² and varies wildly).

    Score: Σ s · ψ(ε) = 0, ψ(ε) = ε(κ − ε²).
    Newton: Â ← Â − (Σ s·ε(κ−ε²)) / (Σ s²·(κ−3ε²)).
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s2_sum = float(s @ s)
    if s2_sum <= 0:
        return {"coef": np.array([np.nan]), "iterations": 0, "converged": False}
    A = float(y @ s / s2_sum)
    use_gamma_path = kappa is None and gamma4_pool is not None and gamma6_pool is not None
    if not use_gamma_path and kappa is None:
        return {"coef": np.array([A]), "residuals": y - A * s, "converged": False,
                "iterations": 0, "kappa_used": None}

    # Freeze κ at the OLS-init residual variance — avoids the positive-feedback
    # loop where bad NR steps inflate m₂ and amplify κ-induced divergence.
    eps0 = y - A * s
    m2_frozen = float(np.var(eps0, ddof=0))
    if use_gamma_path:
        kappa_eff = kappa_from_pool(gamma4_pool, gamma6_pool, m2_frozen)
    else:
        kappa_eff = kappa

    for it in range(max_iter):
        eps = y - A * s
        z1 = eps * (kappa_eff - eps * eps)
        psi_prime = kappa_eff - 3.0 * eps * eps   # ψ'(ε) for ψ(ε)=ε(κ−ε²)
        score = float(np.sum(s * z1))             # Σ s · ψ(ε)
        # Newton: ∂score/∂A = -Σ s² ψ'(ε); A_new = A + score / (Σ s² ψ').
        hess = float(np.sum(s * psi_prime * s))
        if abs(hess) < 1e-12:
            break
        delta = score / hess
        if not np.isfinite(delta):
            break
        if abs(delta) > 5.0:
            delta = np.sign(delta) * 5.0
        A_new = A + delta  # see derivation in module docstring
        if not np.isfinite(A_new):
            break
        A = A_new
        if abs(delta) < tol:
            return {"coef": np.array([A]), "residuals": y - A * s,
                    "iterations": it + 1, "converged": True,
                    "kappa_used": kappa_eff}
    return {"coef": np.array([A]), "residuals": y - A * s,
            "iterations": max_iter, "converged": False,
            "kappa_used": kappa_eff if use_gamma_path else kappa}


def pool_gamma_from_blocks(noise_blocks: list[np.ndarray]) -> dict:
    """Pool γ₃, γ₄ over a calibration set of noise-only blocks.

    Returns means and SE of the pool estimates. Each block contributes one γ̂_k.
    The pooled mean has SE roughly √(SE_k_per_block / n_blocks).
    """
    from sdr_kunchenko.utils.cumulants import standardized_cumulants
    g3, g4, g6 = [], [], []
    for blk in noise_blocks:
        c = standardized_cumulants(blk)
        g3.append(c["gamma3"]); g4.append(c["gamma4"]); g6.append(c["gamma6"])
    g3a = np.array(g3); g4a = np.array(g4); g6a = np.array(g6)
    return {
        "gamma3_mean": float(g3a.mean()), "gamma3_se": float(g3a.std(ddof=1) / np.sqrt(len(g3a))),
        "gamma4_mean": float(g4a.mean()), "gamma4_se": float(g4a.std(ddof=1) / np.sqrt(len(g4a))),
        "gamma6_mean": float(g6a.mean()), "gamma6_se": float(g6a.std(ddof=1) / np.sqrt(len(g6a))),
        "n_blocks": int(len(noise_blocks)),
    }


__all__ = ["pmm2_pooled_amplitude", "pmm3_pooled_amplitude", "pool_gamma_from_blocks"]
