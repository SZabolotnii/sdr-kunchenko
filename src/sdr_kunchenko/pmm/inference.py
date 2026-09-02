"""Standard errors for PMM estimators.

Two paths:
  1. Analytic sandwich SE — closed-form, fast (no R callback per block).
  2. Bootstrap SE via EstemPMM::pmm2_inference (oracle) — slow but trusted.

Both are validated to agree within ~5% on Gaussian samples in tests/.
"""
from __future__ import annotations

import numpy as np

from sdr_kunchenko.pmm import oracle


def pmm2_sandwich_se_amplitude(s: np.ndarray, residuals: np.ndarray,
                                gamma3: float, gamma4: float) -> float:
    """Analytic sandwich SE for amplitude estimator under PMM2 weights.

    Model: y = A·s + ξ, ξ standardized cumulants (γ₃, γ₄).
    PMM2 weight: ψ(ε) = c₁·ε/σ + c₂·(ε²/σ² − 1) where (c₁, c₂) optimal:
      c₁ = (2 + γ₄) / (2 + γ₄ − γ₃²)
      c₂ = −γ₃ / (2 + γ₄ − γ₃²)

    Sandwich variance (univariate):
      Var(Â) = M / J²
      J = Σ ψ'(ε̂_i) · s_i²
      M = Σ ψ²(ε̂_i) · s_i²

    Reduces to OLS SE when γ₃=γ₄=0 (c₁=1, c₂=0, ψ=ε/σ, Var(Â)=σ²/Σs²).
    """
    s = np.asarray(s, dtype=np.float64)
    eps = np.asarray(residuals, dtype=np.float64)
    m2 = float(np.var(eps, ddof=0))
    if m2 <= 0:
        return float("inf")
    sigma = np.sqrt(m2)

    denom = 2.0 + gamma4 - gamma3 ** 2
    if abs(denom) < 1e-12:
        return float("inf")
    c1 = (2.0 + gamma4) / denom
    c2 = -gamma3 / denom

    psi = c1 * (eps / sigma) + c2 * ((eps ** 2 / m2) - 1.0)
    psi_prime = c1 / sigma + 2.0 * c2 * eps / m2

    s2 = s ** 2
    J = float(np.sum(psi_prime * s2))
    M = float(np.sum(psi ** 2 * s2))
    if abs(J) < 1e-12:
        return float("inf")
    var_A = M / (J ** 2)
    return float(np.sqrt(var_A))


def pmm3_sandwich_se_amplitude(s: np.ndarray, residuals: np.ndarray,
                                kappa: float) -> float:
    """Sandwich SE for PMM3 amplitude estimator (symmetric track, γ₃≈0).

    PMM3 weight: ψ(ε) = ε(κ − ε²) where κ is the second-order coefficient
    chosen to maximize information.

    J = Σ ψ'(ε̂_i) · s_i² = Σ (κ − 3ε̂_i²) · s_i²
    M = Σ ψ²(ε̂_i) · s_i²
    Var(Â) = M / J²
    """
    s = np.asarray(s, dtype=np.float64)
    eps = np.asarray(residuals, dtype=np.float64)
    psi = eps * (kappa - eps ** 2)
    psi_prime = kappa - 3.0 * eps ** 2
    s2 = s ** 2
    J = float(np.sum(psi_prime * s2))
    M = float(np.sum(psi ** 2 * s2))
    if abs(J) < 1e-12:
        return float("inf")
    return float(np.sqrt(M / J ** 2))


def ols_se_amplitude(s: np.ndarray, residuals: np.ndarray) -> float:
    """OLS SE for univariate amplitude: σ²/Σs²."""
    s2 = float(np.sum(s ** 2))
    if s2 <= 0:
        return float("inf")
    sigma2 = float(np.var(residuals, ddof=1))
    return float(np.sqrt(sigma2 / s2))


def pmm2_bootstrap_se_amplitude(y: np.ndarray, s: np.ndarray,
                                  *, B: int = 50, seed: int = 0) -> float:
    """Bootstrap SE via residual resampling. Slow — use for cross-check only.

    Calls oracle.lm_pmm2 B times. ~5 ms × B per block.
    """
    fit = oracle.lm_pmm2(y, s)
    A_hat = fit["coef"][0]
    eps_hat = fit["residuals"]
    s = np.asarray(s, dtype=np.float64)
    rng = np.random.default_rng(seed)
    A_boot = np.empty(B)
    n = len(eps_hat)
    for b in range(B):
        idx = rng.integers(0, n, n)
        y_b = A_hat * s + eps_hat[idx]
        f = oracle.lm_pmm2(y_b, s)
        A_boot[b] = f["coef"][0]
    return float(np.std(A_boot, ddof=1))


def newey_west_lag(n: int) -> int:
    """Default truncation lag for Bartlett kernel: L = floor(4·(N/100)^(2/9))."""
    return max(1, int(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_se_amplitude(s: np.ndarray, residuals: np.ndarray,
                     *, weight: np.ndarray | None = None,
                     lag: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) SE for univariate amplitude.

    For Â = (s'·y)/(s'·s) the variance estimator is:
      Var(Â) = (s'·s)^{-2} · Σ_{j=-L}^{L} w_j · Σ_t z_t · z_{t-j}
    where z_t = s_t · ε_t (or s_t · ψ(ε_t) for PMM-weighted score),
          w_j = 1 − |j|/(L+1) (Bartlett),
          L = floor(4·(N/100)^(2/9)) by default.

    Parameters
    ----------
    s, residuals : same length
    weight : optional ψ(ε) for PMM (use raw ε if None — that's OLS-HAC).
    lag : truncation L, default Newey-West rule of thumb.
    """
    s = np.asarray(s, dtype=np.float64)
    eps = np.asarray(residuals, dtype=np.float64)
    n = len(s)
    if weight is None:
        z = s * eps
    else:
        z = s * np.asarray(weight, dtype=np.float64)
    s2 = float(s @ s)
    if s2 <= 0:
        return float("inf")
    L = lag if lag is not None else newey_west_lag(n)
    # j=0 term
    gamma0 = float(z @ z)
    omega = gamma0
    # j>0 terms with Bartlett weight (1 − j/(L+1)).
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        gamma_j = float(np.sum(z[j:] * z[:-j]))
        omega += 2.0 * w * gamma_j
    if omega <= 0:
        return float("inf")
    # For OLS: Var(Â) = ω / s2².
    # For PMM with ψ-weighting, the proper sandwich form uses the J⁻¹·M·J⁻¹
    # structure where J = Σ ψ'·s². When weight is supplied, the caller is
    # responsible for scaling; here we return only the meat term σ²(score)/s2².
    # We expose ω directly via auxiliary so the caller can wrap it.
    return float(np.sqrt(omega / (s2 ** 2)))


def pmm2_hac_se_amplitude(s: np.ndarray, residuals: np.ndarray,
                            gamma3: float, gamma4: float,
                            lag: int | None = None) -> float:
    """HAC sandwich SE for PMM2 amplitude (combines NW meat with PMM-Hessian)."""
    s = np.asarray(s, dtype=np.float64)
    eps = np.asarray(residuals, dtype=np.float64)
    m2 = float(np.var(eps, ddof=0))
    if m2 <= 0:
        return float("inf")
    sigma = np.sqrt(m2)
    denom = 2.0 + gamma4 - gamma3 ** 2
    if abs(denom) < 1e-12:
        return float("inf")
    c1 = (2.0 + gamma4) / denom
    c2 = -gamma3 / denom
    psi = c1 * (eps / sigma) + c2 * ((eps ** 2 / m2) - 1.0)
    psi_prime = c1 / sigma + 2.0 * c2 * eps / m2

    # Hessian: J = Σ s² · ψ'.
    J = float(np.sum(s ** 2 * psi_prime))
    if abs(J) < 1e-12:
        return float("inf")
    # Meat (HAC of score s·ψ).
    z = s * psi
    n = len(s)
    L = lag if lag is not None else newey_west_lag(n)
    omega = float(z @ z)
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        omega += 2.0 * w * float(np.sum(z[j:] * z[:-j]))
    omega = max(omega, 0.0)
    return float(np.sqrt(omega) / abs(J))


def pmm3_hac_se_amplitude(s: np.ndarray, residuals: np.ndarray,
                            kappa: float,
                            lag: int | None = None) -> float:
    """HAC sandwich SE for PMM3 amplitude (combines NW meat with PMM3-Hessian).

    PMM3 weight: ψ(ε) = ε(κ − ε²), ψ'(ε) = κ − 3ε².

    Hessian J = Σ s² · ψ'(ε̂); meat ω = HAC(Bartlett) long-run variance of the
    score z_t = s_t · ψ(ε̂_t) with truncation L = floor(4·(N/100)^(2/9)).
    SE(Â) = √ω / |J| — same structure as `pmm2_hac_se_amplitude`, and reduces
    to `pmm3_sandwich_se_amplitude` when the score is serially uncorrelated.
    """
    s = np.asarray(s, dtype=np.float64)
    eps = np.asarray(residuals, dtype=np.float64)
    psi = eps * (kappa - eps ** 2)
    psi_prime = kappa - 3.0 * eps ** 2

    # Hessian: J = Σ s² · ψ'.
    J = float(np.sum(s ** 2 * psi_prime))
    if abs(J) < 1e-12:
        return float("inf")
    # Meat (HAC of score s·ψ).
    z = s * psi
    n = len(s)
    L = lag if lag is not None else newey_west_lag(n)
    omega = float(z @ z)
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        omega += 2.0 * w * float(np.sum(z[j:] * z[:-j]))
    omega = max(omega, 0.0)
    return float(np.sqrt(omega) / abs(J))


__all__ = [
    "pmm2_sandwich_se_amplitude",
    "pmm3_sandwich_se_amplitude",
    "ols_se_amplitude",
    "pmm2_bootstrap_se_amplitude",
    "hac_se_amplitude",
    "pmm2_hac_se_amplitude",
    "pmm3_hac_se_amplitude",
    "newey_west_lag",
]
