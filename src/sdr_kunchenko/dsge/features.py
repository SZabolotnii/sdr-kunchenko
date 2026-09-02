"""DSGE: Decomposition in Space with Generating Element.

Per Spec §6.2:
    For each class c, fit F_c · k_c = b_c where:
        F_c — basis matrix evaluated pointwise on pooled training blocks
        b_c — target vector of pooled raw signal samples
        k_c — class-specific reconstruction coefficients

    For new sample x, the class-c **reconstruction error** is:
        MSED_c(x) = mean((F(x) · k_c − x)²)

    The classifier then operates on the feature vector
        z(x) = [log MSED_1(x), log MSED_2(x), ..., log MSED_C(x)]
    or on raw MSED_c.

For SDR-baseband: x is a 1024-sample block, F(x) is an (N × s) matrix where
column i is φ_i(x) (Kunchenko basis evaluated pointwise). For M training
blocks of class c, F_c has shape (M·N × s), b_c has shape (M·N,), and
k_c has shape (s,). The ridge regression minimizes pooled pointwise L2.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def build_F(basis: Sequence[Callable], x: np.ndarray) -> np.ndarray:
    """Evaluate basis on a 1-D signal x. Returns (N × s) matrix of φ_i(x_n)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    cols = [fn(x) for fn in basis]
    return np.stack(cols, axis=1)  # (N, s)


def fit_class_coefficients(basis: Sequence[Callable],
                            class_blocks: list[np.ndarray],
                            *, ridge: float = 1e-4) -> np.ndarray:
    """Fit K_c by ridge regression, pooling all training blocks for class c.

    Solves:  argmin_K  Σ_b || F(x_b) · K − x_b ||²  +  λ ||K||²
    Closed form:  K = (F^T F + λI)^{-1} F^T x  with F, x stacked over blocks.
    """
    s = len(basis)
    if not class_blocks:
        raise ValueError("class_blocks is empty")
    F_full = np.vstack([build_F(basis, b) for b in class_blocks])
    x_full = np.concatenate([np.asarray(b, dtype=np.float64).ravel()
                              for b in class_blocks])
    G = F_full.T @ F_full + ridge * np.eye(s)
    rhs = F_full.T @ x_full
    K = np.linalg.solve(G, rhs)
    return K


def msed(basis: Sequence[Callable], K_c: np.ndarray, x: np.ndarray) -> float:
    """Mean-squared reconstruction error of x using class-c projection K_c."""
    F = build_F(basis, x)
    x = np.asarray(x, dtype=np.float64).ravel()
    pred = F @ K_c
    err = pred - x
    return float(np.mean(err ** 2))


def msed_features(basis: Sequence[Callable],
                   class_Ks: list[np.ndarray],
                   x: np.ndarray, *, log: bool = True) -> np.ndarray:
    """Compute MSED feature vector for a single sample x.

    Returns shape (C,) array — one MSED per class.
    """
    msed_vec = np.array([msed(basis, K_c, x) for K_c in class_Ks])
    if log:
        msed_vec = np.log(msed_vec + 1e-12)
    return msed_vec


def msed_features_batch(basis: Sequence[Callable],
                         class_Ks: list[np.ndarray],
                         X: list[np.ndarray] | np.ndarray,
                         *, log: bool = True) -> np.ndarray:
    """Batch version: for a list/array of blocks, return (M, C) feature matrix."""
    if isinstance(X, np.ndarray):
        X_iter = (X[i] for i in range(X.shape[0]))
    else:
        X_iter = iter(X)
    out = np.array([msed_features(basis, class_Ks, x, log=log) for x in X_iter])
    return out


__all__ = ["build_F", "fit_class_coefficients", "msed", "msed_features",
            "msed_features_batch"]
