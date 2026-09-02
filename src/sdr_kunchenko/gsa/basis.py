"""Basis functions for GSA-LLR approximation.

Three families:
    - Polynomial:  φ_i(x) = x^i, i=1..s
    - Fractional:  φ_i(x) = sign(x)·|x|^{1/i}, i=1..s
    - PATP:        φ_i(x) = sign(x)·|x|^{p_i(α)}, with
                    p_i(α) = 1/i + (4−i−3/i)·α + (2i−4+2/i)·α²

All functions are sign-preserving (real-valued SDR signal).
The basis is **purely odd** — useful for symmetric H₀ where E[φ_i(ξ)] = 0.
For mean-shift H₁ (y = A·s + ξ), the conditional moments of φ_i(y) under H₁
differ from H₀ → the LLR pickup.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ------------------------------------------------------------------- helpers


def _signed_pow(x: np.ndarray, p: float) -> np.ndarray:
    """sign(x)·|x|^p — sign-preserving fractional power."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.abs(x) ** p


# ---------------------------------------------------------- basis families


def poly_basis(s: int) -> list[Callable[[np.ndarray], np.ndarray]]:
    """φ_i(x) = x^i, i = 1..s."""
    if s < 1:
        raise ValueError("s must be ≥ 1")
    return [lambda x, i=i: np.asarray(x, dtype=np.float64) ** i for i in range(1, s + 1)]


def frac_basis(s: int) -> list[Callable[[np.ndarray], np.ndarray]]:
    """φ_i(x) = sign(x)·|x|^{1/i}, i = 1..s."""
    if s < 1:
        raise ValueError("s must be ≥ 1")
    return [lambda x, i=i: _signed_pow(x, 1.0 / i) for i in range(1, s + 1)]


def patp_power(i: int, alpha: float) -> float:
    """Spec §A.4: p_i(α) = 1/i + (4 − i − 3/i)·α + (2i − 4 + 2/i)·α²

    At α=0:   p_i = 1/i             → fractional
    At α=0.5: p_i = 1                → linear (all i collapse)
    At α=1:   p_i = i                → polynomial
    """
    a = 1.0 / i
    b = 4.0 - i - 3.0 / i
    c = 2.0 * i - 4.0 + 2.0 / i
    return a + b * alpha + c * alpha ** 2


def patp_basis(s: int, alpha: float) -> list[Callable[[np.ndarray], np.ndarray]]:
    """φ_i(x) = sign(x)·|x|^{p_i(α)}, i = 2..s+1 (Spec §A.4 indexing)."""
    if s < 1:
        raise ValueError("s must be ≥ 1")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("α must be in [0, 1]")
    powers = [patp_power(i, alpha) for i in range(2, s + 2)]
    return [lambda x, p=p: _signed_pow(x, p) for p in powers]


def augmented_patp_basis(s: int, alpha: float) -> list[Callable[[np.ndarray], np.ndarray]]:
    """PATP with a fixed linear term.

    The total number of returned functions is ``s``.  This removes the missing-linear-
    term confounder from PATP comparisons while preserving the sign-preserving PATP
    convention for the parameterised terms.
    """
    if s < 1:
        raise ValueError("s must be ≥ 1")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("α must be in [0, 1]")
    if s == 1:
        return [lambda x: np.asarray(x, dtype=np.float64)]
    powers = [patp_power(i, alpha) for i in range(2, s + 1)]
    return [lambda x: np.asarray(x, dtype=np.float64)] + [
        lambda x, p=p: _signed_pow(x, p) for p in powers
    ]


def trig_basis(s: int) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Trigonometric basis (Spec §6.5): φ_i(x) = sin(i·x), cos(i·x).
    For s, alternates sin/cos and increments frequency: s=2 → [sin x, cos x],
    s=4 → [sin x, cos x, sin 2x, cos 2x], etc. We use ceil(s/2) frequencies.
    """
    if s < 1:
        raise ValueError("s must be ≥ 1")
    out: list[Callable] = []
    k = 1
    while len(out) < s:
        out.append(lambda x, k=k: np.sin(k * np.asarray(x, dtype=np.float64)))
        if len(out) < s:
            out.append(lambda x, k=k: np.cos(k * np.asarray(x, dtype=np.float64)))
        k += 1
    return out[:s]


def robust_basis(s: int, *, c: float = 1.345) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Huber-style smooth basis (Spec §6.5):
        φ_1(x) = ψ_huber(x, c) = clip(x, -c, c) / c
        φ_i(x) = sign(x) · |ψ_huber|^{i/s}
    The first function is a soft-clipped identity; higher orders are
    sign-preserving fractional powers of the clipped value.
    """
    if s < 1:
        raise ValueError("s must be ≥ 1")

    def _huber(x):
        x = np.asarray(x, dtype=np.float64)
        return np.clip(x, -c, c) / c

    out: list[Callable] = [_huber]
    for i in range(2, s + 1):
        p = 1.0 + (i - 1) / s  # 1.0 < p ≤ 2.0 — gentle rise above linear
        out.append(lambda x, p=p: np.sign(_huber(x)) * np.abs(_huber(x)) ** p)
    return out[:s]


def make_basis(kind: str, s: int, *, alpha: float = 0.5) -> list[Callable]:
    """Factory: 'poly' | 'frac' | 'patp' | 'patp_aug' | 'trig' | 'robust'."""
    if kind == "poly":
        return poly_basis(s)
    if kind == "frac":
        return frac_basis(s)
    if kind == "patp":
        return patp_basis(s, alpha)
    if kind == "patp_aug":
        return augmented_patp_basis(s, alpha)
    if kind == "trig":
        return trig_basis(s)
    if kind == "robust":
        return robust_basis(s)
    raise ValueError(f"unknown basis kind: {kind}")


def evaluate_basis(basis: list[Callable], x: np.ndarray) -> np.ndarray:
    """Apply each φ_i to x. Returns (..., s) array."""
    return np.stack([f(x) for f in basis], axis=-1)


__all__ = [
    "poly_basis", "frac_basis", "patp_basis", "augmented_patp_basis", "patp_power",
    "trig_basis", "robust_basis",
    "make_basis", "evaluate_basis",
]
