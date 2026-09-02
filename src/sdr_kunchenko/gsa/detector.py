"""GSA-CUSUM detector: streaming Λ̂(y_t) → CUSUM stop rule.

Per Spec §5.2:
    S_t = max(0, S_{t-1} + Λ̂(y_t) − h)
    Detection: t̂ = inf{t : S_t > 0} (with threshold absorbed into the per-step drift,
    or equivalently t̂ = inf{t : S_t > threshold} if drift is null).

Two equivalent parametrizations:
    A. drift-form: subtract `h` per sample, alarm at S_t > 0 (Spec form).
    B. threshold-form: no per-sample subtract, alarm at S_t > h.
We implement (A) since Spec uses it; threshold-h then absorbs both the
per-step centering AND the false-alarm slack.

Energy detector baseline: identical structure with φ_1(y) = y² (no GSA fit).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .coefficients import lambda_score


class GSACUSUMDetector:
    """Sequential CUSUM detector with GSA-LLR per-sample score."""

    def __init__(self, basis: Sequence[Callable], k: np.ndarray,
                  threshold: float):
        self.basis = list(basis)
        self.k = np.asarray(k, dtype=np.float64)
        self.threshold = float(threshold)
        self.S = 0.0
        self.history: list[float] = []
        self.fired_at: int | None = None

    def reset(self) -> None:
        self.S = 0.0
        self.history = []
        self.fired_at = None

    def feed(self, x: np.ndarray) -> dict:
        """Process a stream of samples. Returns first-alarm time and CUSUM trace."""
        self.reset()
        x = np.asarray(x, dtype=np.float64).ravel()
        lam = lambda_score(self.basis, self.k, x)
        # Per-Spec drift form: per-sample increment = lam − threshold/N? No —
        # Spec uses CUSUM directly: S_t = max(0, S_{t-1} + Λ̂(y_t) − h),
        # alarm at S_t > 0. We keep h in the per-step subtraction.
        # However for SDR baseband we *already* normalized Λ̂ via GSA fit
        # (so E[Λ̂|H₀] ≈ 0 if H₀ moments correctly subtracted).
        # If E[Λ̂|H₀] ≠ 0 (calibration noise), the per-step h re-centers it.
        S = 0.0
        history = np.empty(len(lam))
        fired_at = None
        for t in range(len(lam)):
            S = max(0.0, S + lam[t] - self.threshold)
            history[t] = S
            if S > 0.0 and fired_at is None:
                fired_at = t
        self.S = S
        self.history = history.tolist()
        self.fired_at = fired_at
        return {"fired_at": fired_at, "history": history,
                "max_S": float(history.max()),
                "alarm": fired_at is not None}


class EnergyDetector:
    """Baseline: φ(y) = y², CUSUM with simple drift-form.

    threshold encodes both the noise-mean centering AND the FAR slack.
    """

    def __init__(self, threshold: float, *, mean_under_h0: float = 0.0):
        self.threshold = float(threshold)
        self.mean_h0 = float(mean_under_h0)

    def feed(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=np.float64).ravel()
        lam = x ** 2 - self.mean_h0
        S = 0.0
        history = np.empty(len(lam))
        fired_at = None
        for t in range(len(lam)):
            S = max(0.0, S + lam[t] - self.threshold)
            history[t] = S
            if S > 0.0 and fired_at is None:
                fired_at = t
        return {"fired_at": fired_at, "history": history,
                "max_S": float(history.max()),
                "alarm": fired_at is not None}


__all__ = ["GSACUSUMDetector", "EnergyDetector"]
