"""Sample cumulants and standardized moments via Fisher's k-statistics.

This is the *secondary* (pure-Python, no R) implementation. The *primary* path
for any PMM-related computation is `sdr_kunchenko.pmm.oracle.cumulants`, which
calls EstemPMM directly — the values from this module must agree with that
oracle on Gaussian samples (validated by tests/test_cumulants.py).

Provided here mainly for fast batch processing (no rpy2 overhead) when ground
truth is not strictly required (e.g. dataset-wide γ̂₄ histograms).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def central_moments(x: np.ndarray, max_order: int = 6) -> Mapping[int, float]:
    """Biased central moments m_k = E[(x - x̄)^k], k=2..max_order."""
    xc = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean()
    return {k: float(np.mean(xc**k)) for k in range(2, max_order + 1)}


def cumulants_from_moments(m: Mapping[int, float]) -> Mapping[int, float]:
    """Convert biased central moments to cumulants via Fisher's k-statistics.

    κ₂ = m₂
    κ₃ = m₃
    κ₄ = m₄ − 3·m₂²
    κ₅ = m₅ − 10·m₃·m₂
    κ₆ = m₆ − 15·m₄·m₂ − 10·m₃² + 30·m₂³
    """
    return {
        2: m[2],
        3: m[3],
        4: m[4] - 3 * m[2] ** 2,
        5: m[5] - 10 * m[3] * m[2],
        6: m[6] - 15 * m[4] * m[2] - 10 * m[3] ** 2 + 30 * m[2] ** 3,
    }


def standardized_cumulants(x: np.ndarray) -> dict[str, float]:
    """Standardized cumulants matching EstemPMM conventions.

    γ₃ = κ₃/σ³                                — Fisher's k-statistic.
    γ₄ = κ₄/σ⁴ = m₄/m₂² − 3                  — Fisher's k-statistic (excess).
    γ₆ = m₆/m₂³ − 15·m₄/m₂² + 30              — EstemPMM convention,
                                                 valid for symmetric distributions
                                                 (drops the −10·m₃²/m₂³ term that
                                                 full Fisher κ₆ would include).

    The γ₆ formula is intentionally aligned with EstemPMM::pmm_gamma6 because
    PMM3 only consumes γ₆ when γ₃ ≈ 0 (symmetric track). For full Fisher's
    γ₆ on asymmetric data, use `standardized_cumulants_fisher`.
    """
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    m2 = float(np.mean(xc ** 2))
    m4 = float(np.mean(xc ** 4))
    m6 = float(np.mean(xc ** 6))
    if m2 <= 0:
        return {"gamma3": 0.0, "gamma4": 0.0, "gamma6": 0.0}
    sigma = np.sqrt(m2)
    m3 = float(np.mean(xc ** 3))
    return {
        "gamma3": m3 / sigma ** 3,
        "gamma4": m4 / m2 ** 2 - 3.0,
        "gamma6": m6 / m2 ** 3 - 15.0 * (m4 / m2 ** 2) + 30.0,
    }


def standardized_cumulants_fisher(x: np.ndarray) -> dict[str, float]:
    """Full Fisher k-statistic γ_k = κ_k/σ^k for k=3..6.

    Differs from `standardized_cumulants` only in γ₆ (includes −10·m₃²/m₂³).
    Use this for asymmetric distributions where the full κ₆ matters.
    """
    m = central_moments(x, max_order=6)
    kappa = cumulants_from_moments(m)
    sigma = float(np.sqrt(m[2]))
    if sigma <= 0:
        return {f"gamma{k}": 0.0 for k in (3, 4, 5, 6)}
    return {f"gamma{k}": kappa[k] / sigma ** k for k in range(3, 7)}


def asymptotic_se(n: int) -> dict[str, float]:
    """Asymptotic SE of standardized cumulants under Gaussianity.

    SE(γ̂₃) = √(6/N), SE(γ̂₄) = √(24/N).
    For γ₅, γ₆ we use the Cramér-Rao lower bound from Fisher's table:
    SE(γ̂₅) ≈ √(120/N), SE(γ̂₆) ≈ √(720/N).
    """
    return {
        "gamma3": float(np.sqrt(6.0 / n)),
        "gamma4": float(np.sqrt(24.0 / n)),
        "gamma5": float(np.sqrt(120.0 / n)),
        "gamma6": float(np.sqrt(720.0 / n)),
    }


__all__ = [
    "central_moments",
    "cumulants_from_moments",
    "standardized_cumulants",
    "standardized_cumulants_fisher",
    "asymptotic_se",
]
