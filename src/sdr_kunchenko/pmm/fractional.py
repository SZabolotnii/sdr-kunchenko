"""
Fractional-power score functions for PMM amplitude estimation.

This module implements the **PMM_FRAC** estimator — a new PMM track that uses
the fractional-power score function

    ψ_α(ε) = sign(ε) · |ε|^α,   α ∈ (0, 2]

to achieve near-optimal efficiency under heavy-tailed (leptokurtic) noise
distributions such as α-stable and Student-t.

Background
----------
The existing PMM2 and PMM3 tracks (see :mod:`sdr_kunchenko.pmm.oracle` and
:mod:`sdr_kunchenko.pmm.pooled`) use integer-order score functions and are
optimised for asymmetric (γ₃ ≠ 0) and symmetric platykurtic (γ₄ < −0.7)
noise, respectively.  For heavy-tailed leptokurtic noise (γ₄ >> 1), both
tracks degrade relative to OLS.  PMM_FRAC fills this gap by adapting α to
the tail behaviour of the noise distribution.

Limiting cases of ψ_α:
  - α = 1  →  ψ₁(ε) = ε  (OLS, optimal for Gaussian)
  - α → 0  →  ψ_α(ε) → sign(ε)  (sign detector, optimal for Cauchy/Laplace)
  - α = 2  →  ψ₂(ε) = |ε|·ε  (quadratic score)

Connection to α-stable distributions:
  For α-stable(α_s) noise the optimal fractional exponent satisfies
  α_opt ≈ α_s − 1, so:
    α_s = 2.0 (Gaussian)  →  α_opt = 1.0  (OLS)
    α_s = 1.5             →  α_opt = 0.5
    α_s = 1.0 (Cauchy)    →  α_opt → 0   (sign detector)

References
----------
- exp1b requirements §Requirement 2 (Оцінювач_α)
- exp1b requirements §Requirement 3 (PMM_FRAC реалізація)
- exp1b design §2.1–§2.5 (математичний фундамент)
- exp1b design §3.1 (специфікація модуля)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

__all__ = [
    "pmm_frac_amplitude",
    "estimate_alpha_from_cumulants",
    "estimate_alpha_stable_index",
    "pmm_frac_sandwich_se",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp_abs(x: np.ndarray, eps_min: float) -> np.ndarray:
    """Return |x| clamped from below at *eps_min*.

    Avoids division-by-zero in ψ'_α(ε) = α·|ε|^(α−1) when α < 1.
    """
    return np.maximum(np.abs(x), eps_min)


# ---------------------------------------------------------------------------
# Task 1.1 — PMM_FRAC IRLS estimator
# ---------------------------------------------------------------------------


def pmm_frac_amplitude(
    y: ArrayLike,
    s: ArrayLike,
    *,
    alpha: float,
    max_iter: int = 25,
    tol: float = 1e-8,
    eps_min: float = 1e-10,
) -> dict:
    r"""PMM_FRAC amplitude estimator via Iteratively Re-weighted Least Squares.

    Solves the M-estimation equation

    .. math::

        \sum_{i} s_i \, \psi_\alpha(\varepsilon_i) = 0,
        \quad \psi_\alpha(\varepsilon) = \operatorname{sign}(\varepsilon)
                                          |\varepsilon|^\alpha

    using the IRLS algorithm with weights
    :math:`w_i = |\varepsilon_i|^{\alpha - 1}`.

    Algorithm
    ---------
    1. **Initialise** with OLS:

       .. math:: \hat{A}^{(0)} = \frac{\mathbf{y} \cdot \mathbf{s}}
                                       {\mathbf{s} \cdot \mathbf{s}}

    2. **Iterate** for ``k = 0, 1, …, max_iter − 1``:

       a. Compute residuals:
          :math:`\varepsilon_i^{(k)} = y_i - \hat{A}^{(k)} s_i`

       b. Compute IRLS weights (clamping :math:`|\varepsilon_i| < \varepsilon_{\min}`
          to :math:`\varepsilon_{\min}` to avoid division by zero when α < 1):

          .. math:: w_i^{(k)} = \max(|\varepsilon_i^{(k)}|,\,
                                     \varepsilon_{\min})^{\alpha - 1}

       c. Update estimate:

          .. math:: \hat{A}^{(k+1)} =
                    \frac{\sum_i w_i^{(k)} s_i y_i}
                         {\sum_i w_i^{(k)} s_i^2}

       d. **Convergence check**:
          if :math:`|\hat{A}^{(k+1)} - \hat{A}^{(k)}| < \text{tol}`, stop.

    3. **Fallback**: if the loop exhausts ``max_iter`` without converging,
       return the OLS estimate and log a warning.

    Limiting behaviour
    ------------------
    - ``alpha = 1``: weights are all 1 → IRLS reduces to OLS in one step.
    - ``alpha → 0``: weights ∝ 1/|ε| → sign-detector (optimal for Cauchy).
    - ``alpha = 2``: weights ∝ |ε| → quadratic score function.

    Parameters
    ----------
    y : array_like, shape (n,)
        Observed signal samples :math:`y_i = A \cdot s_i + \xi_i`.
    s : array_like, shape (n,)
        Known pilot/reference signal samples :math:`s_i`.
    alpha : float
        Fractional exponent of the score function, :math:`\alpha \in (0, 2]`.
        Use :func:`estimate_alpha_from_cumulants` or
        :func:`estimate_alpha_stable_index` to choose α adaptively.
    max_iter : int, optional
        Maximum number of IRLS iterations (default 25).
    tol : float, optional
        Convergence tolerance on :math:`|\hat{A}^{(k+1)} - \hat{A}^{(k)}|`
        (default 1e-8).
    eps_min : float, optional
        Minimum absolute residual used in weight computation to prevent
        division by zero (default 1e-10).

    Returns
    -------
    dict
        ``coef`` : np.ndarray, shape (1,)
            Estimated amplitude :math:`\hat{A}`.
        ``residuals`` : np.ndarray, shape (n,)
            Final residuals :math:`y_i - \hat{A} s_i`.
        ``sigma`` : float
            Residual standard deviation (ddof=0).
        ``iterations`` : int
            Number of IRLS iterations performed.
        ``converged`` : bool
            Whether the algorithm converged within ``max_iter``.
        ``alpha_used`` : float
            The value of α actually used (echoes the ``alpha`` argument).

    Raises
    ------
    ValueError
        If ``y`` and ``s`` have different lengths, or if ``s @ s == 0``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = np.ones(1024)
    >>> A_true = 1.5
    >>> y = A_true * s + rng.standard_normal(1024)
    >>> result = pmm_frac_amplitude(y, s, alpha=1.0)
    >>> abs(result["coef"][0] - A_true) < 0.1
    True
    >>> result["converged"]
    True

    Notes
    -----
    The interface is intentionally compatible with
    :func:`sdr_kunchenko.pmm.pooled.pmm2_pooled_amplitude` and
    :func:`sdr_kunchenko.pmm.pooled.pmm3_pooled_amplitude`:
    all return a dict with at least ``coef``, ``residuals``, ``sigma``,
    ``iterations``, and ``converged``.

    References
    ----------
    - exp1b requirements §Requirement 3, criteria 1–6
    - exp1b design §2.1 (PATP score function), §3.1 (module spec)
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if y.shape != s.shape:
        raise ValueError(
            f"y and s must have the same shape; got {y.shape} vs {s.shape}"
        )

    s2_sum = float(s @ s)
    if s2_sum <= 0.0:
        raise ValueError("s @ s must be positive (s cannot be the zero vector).")

    # OLS initialisation
    A_ols = float(y @ s / s2_sum)

    # Special case: α = 1 → IRLS is exactly OLS (weights are all 1).
    # Return immediately to avoid unnecessary iterations.
    if abs(alpha - 1.0) < 1e-14:
        eps = y - A_ols * s
        return {
            "coef": np.array([A_ols]),
            "residuals": eps,
            "sigma": float(np.sqrt(np.var(eps, ddof=0))),
            "iterations": 1,
            "converged": True,
            "alpha_used": float(alpha),
        }

    A = A_ols
    converged = False

    for it in range(max_iter):
        eps = y - A * s
        abs_eps = _clamp_abs(eps, eps_min)
        # w_i = |ε_i|^(α-1)
        w = abs_eps ** (alpha - 1.0)

        ws2 = float(np.sum(w * s * s))
        if ws2 <= 0.0:
            break
        A_new = float(np.sum(w * s * y) / ws2)

        if not np.isfinite(A_new):
            break

        if abs(A_new - A) < tol:
            A = A_new
            converged = True
            break

        A = A_new

    if not converged:
        logger.warning(
            "pmm_frac_amplitude: IRLS did not converge in %d iterations "
            "(alpha=%.4f). Falling back to OLS estimate.",
            max_iter,
            alpha,
        )
        A = A_ols

    eps_final = y - A * s
    return {
        "coef": np.array([A]),
        "residuals": eps_final,
        "sigma": float(np.sqrt(np.var(eps_final, ddof=0))),
        "iterations": it + 1 if converged else max_iter,
        "converged": converged,
        "alpha_used": float(alpha),
    }


# ---------------------------------------------------------------------------
# Task 1.2 — α estimator via method of moments
# ---------------------------------------------------------------------------


def estimate_alpha_from_cumulants(
    gamma4: float,
    gamma6: float,
    *,
    alpha_min: float = 0.1,
    alpha_max: float = 1.0,
) -> float:
    r"""Estimate the optimal fractional exponent α from standardised cumulants.

    Uses a method-of-moments approximation derived from the theoretical
    relationship between the excess kurtosis γ₄ and the optimal score-function
    exponent for the PMM_FRAC estimator.

    Mathematical background
    -----------------------
    For distributions with finite moments the optimal α minimises the
    asymptotic variance of the M-estimator.  A closed-form approximation
    (see exp1b design §2.4) is:

    .. math::

        \alpha_{\text{opt}} \approx
        \max\!\left(\alpha_{\min},\;
                    \frac{1}{1 + 0.3 \cdot \hat{\gamma}_4}\right)

    This formula is monotonically decreasing in γ₄:

    - γ₄ ≤ 0 (platykurtic): returns 1.0 → OLS / PMM3 regime.
    - γ₄ ∈ (0, 2] (mildly leptokurtic): α ∈ (0.5, 1.0).
    - γ₄ > 2 (strongly leptokurtic): α ∈ (0.1, 0.5).

    The ``gamma6`` parameter is accepted for API completeness and future
    refinement (higher-order correction), but is not used in the current
    approximation.

    Parameters
    ----------
    gamma4 : float
        Standardised excess kurtosis :math:`\gamma_4 = \mu_4/\sigma^4 - 3`.
        Positive values indicate leptokurtic (heavy-tailed) distributions.
    gamma6 : float
        Standardised sixth cumulant :math:`\gamma_6`.  Currently unused;
        reserved for future higher-order corrections.
    alpha_min : float, optional
        Lower bound for the returned α (default 0.1, for numerical stability).
    alpha_max : float, optional
        Upper bound for the returned α (default 1.0; FRAC-track is not
        beneficial above 1.0 for leptokurtic noise).

    Returns
    -------
    float
        Estimated optimal α clipped to ``[alpha_min, alpha_max]``.

    Examples
    --------
    >>> estimate_alpha_from_cumulants(-1.0, 0.0)   # platykurtic → OLS
    1.0
    >>> estimate_alpha_from_cumulants(0.0, 0.0)    # Gaussian → OLS
    1.0
    >>> round(estimate_alpha_from_cumulants(3.0, 0.0), 4)  # leptokurtic
    0.5263
    >>> round(estimate_alpha_from_cumulants(30.0, 0.0), 4)  # very heavy tail
    0.1

    Notes
    -----
    For distributions without finite moments (e.g. Cauchy, α-stable with
    α_s < 2), this estimator is unreliable.  Use
    :func:`estimate_alpha_stable_index` instead.

    References
    ----------
    - exp1b requirements §Requirement 2, criteria 1–4, 7–8
    - exp1b design §2.4 (method-of-moments formula)
    """
    if gamma4 <= 0.0:
        # Platykurtic or Gaussian regime — PMM3 / OLS is appropriate.
        return 1.0

    alpha_opt = 1.0 / (1.0 + 0.3 * gamma4)

    # Clip to [alpha_min, alpha_max] and warn if clipping occurs.
    clipped = np.clip(alpha_opt, alpha_min, alpha_max)
    if clipped != alpha_opt:
        logger.warning(
            "estimate_alpha_from_cumulants: α_opt=%.6f clipped to [%.4f, %.4f] → %.6f",
            alpha_opt,
            alpha_min,
            alpha_max,
            float(clipped),
        )

    return float(clipped)


# ---------------------------------------------------------------------------
# Task 1.3 — α_s estimator via characteristic function (MLE)
# ---------------------------------------------------------------------------


def estimate_alpha_stable_index(
    noise_block: ArrayLike,
    *,
    n_freq_points: int = 50,
    t_max: float = 2.0,
) -> float:
    r"""Estimate the stability index α_s of an α-stable distribution.

    Uses the log-characteristic-function regression method (Koutrouvelis 1980):
    for a symmetric α-stable distribution with scale σ,

    .. math::

        \log |\hat{\varphi}(t)| = -\sigma^{\alpha_s} |t|^{\alpha_s}

    Taking logarithms twice:

    .. math::

        \log\!\bigl(-\log|\hat{\varphi}(t)|\bigr)
        = \alpha_s \log|t| + \alpha_s \log \sigma

    so a linear regression of :math:`\log(-\log|\hat{\varphi}(t)|)` on
    :math:`\log|t|` yields the slope :math:`\alpha_s`.

    Algorithm
    ---------
    1. Compute the empirical characteristic function at
       :math:`t_k \in \text{linspace}(0.1, t_{\max}, n_{\text{freq}})`:

       .. math::

           \hat{\varphi}(t_k) = \frac{1}{n} \sum_{j=1}^{n}
                                 e^{i t_k x_j}

    2. Compute :math:`u_k = \log(-\log|\hat{\varphi}(t_k)|)` and
       :math:`v_k = \log t_k`.

    3. Fit a simple linear regression :math:`u_k = \alpha_s v_k + c` via
       ordinary least squares.

    4. Clip the result to :math:`(0.1, 2.0]`.

    Parameters
    ----------
    noise_block : array_like, shape (n,)
        A block of noise samples assumed to follow an α-stable distribution.
        Larger blocks (n ≥ 512) give more reliable estimates.
    n_freq_points : int, optional
        Number of frequency grid points (default 50).
    t_max : float, optional
        Maximum frequency for the regression grid (default 2.0).

    Returns
    -------
    float
        Estimated stability index :math:`\hat{\alpha}_s \in (0.1, 2.0]`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import levy_stable
    >>> rng = np.random.default_rng(42)
    >>> x = levy_stable.rvs(1.5, 0, size=4096, random_state=rng)
    >>> alpha_s = estimate_alpha_stable_index(x)
    >>> 1.2 <= alpha_s <= 1.8   # rough tolerance for finite sample
    True

    Notes
    -----
    The regression is performed on the log-log scale, which amplifies noise
    at small |φ̂(t)|.  Points where :math:`|\hat{\varphi}(t)| < 1\times10^{-6}`
    are excluded to avoid numerical instability.

    The returned α_s can be converted to an optimal PMM_FRAC exponent via
    :math:`\alpha_{\text{opt}} = \alpha_s - 1` (clipped to [0.1, 1.0]).

    References
    ----------
    - exp1b requirements §Requirement 2, criterion 5
    - exp1b design §2.5 (MLE via characteristic function)
    - Koutrouvelis, I.A. (1980). "Regression-type estimation of the
      parameters of stable laws." *JASA*, 75(372), 918–928.
    """
    x = np.asarray(noise_block, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        return 2.0  # degenerate — assume Gaussian

    t_grid = np.linspace(0.1, t_max, n_freq_points)

    # Empirical characteristic function: φ̂(t) = mean(exp(i·t·x))
    # Shape: (n_freq_points,)
    phi_hat = np.mean(np.exp(1j * t_grid[:, None] * x[None, :]), axis=1)
    abs_phi = np.abs(phi_hat)

    # Exclude points where |φ̂(t)| is too small (numerical instability).
    valid = abs_phi > 1e-6
    if valid.sum() < 3:
        logger.warning(
            "estimate_alpha_stable_index: fewer than 3 valid frequency points "
            "(|φ̂(t)| > 1e-6). Returning α_s = 2.0 (Gaussian fallback)."
        )
        return 2.0

    log_t = np.log(t_grid[valid])
    inner = -np.log(abs_phi[valid])

    # Guard against log(0) or log of negative values.
    inner = np.where(inner > 0, inner, np.nan)
    log_log_phi = np.log(inner)

    finite_mask = np.isfinite(log_log_phi)
    if finite_mask.sum() < 3:
        logger.warning(
            "estimate_alpha_stable_index: fewer than 3 finite log-log values. "
            "Returning α_s = 2.0 (Gaussian fallback)."
        )
        return 2.0

    log_t_f = log_t[finite_mask]
    log_log_phi_f = log_log_phi[finite_mask]

    # OLS slope: α_s = Cov(v, u) / Var(v)
    v_mean = log_t_f.mean()
    u_mean = log_log_phi_f.mean()
    cov = float(np.sum((log_t_f - v_mean) * (log_log_phi_f - u_mean)))
    var_v = float(np.sum((log_t_f - v_mean) ** 2))

    if var_v < 1e-12:
        return 2.0

    alpha_s_raw = cov / var_v

    # Clip to (0.1, 2.0]
    alpha_s = float(np.clip(alpha_s_raw, 0.1, 2.0))
    if alpha_s != alpha_s_raw:
        logger.warning(
            "estimate_alpha_stable_index: raw α_s=%.4f clipped to %.4f",
            alpha_s_raw,
            alpha_s,
        )

    return alpha_s


# ---------------------------------------------------------------------------
# Task 1.4 — Sandwich standard error for PMM_FRAC
# ---------------------------------------------------------------------------


def pmm_frac_sandwich_se(
    s: ArrayLike,
    residuals: ArrayLike,
    alpha: float,
    *,
    eps_min: float = 1e-10,
) -> float:
    r"""Sandwich standard error for the PMM_FRAC amplitude estimator.

    Computes the asymptotic standard error of :math:`\hat{A}` using the
    sandwich (Huber-White) variance formula for M-estimators:

    .. math::

        \hat{\sigma}^2(\hat{A}) =
        \frac{\sum_i \psi_\alpha^2(\hat{\varepsilon}_i)\, s_i^2}
             {\left(\sum_i \psi'_\alpha(\hat{\varepsilon}_i)\, s_i^2\right)^2}

    where the score function and its derivative are:

    .. math::

        \psi_\alpha(\varepsilon) &= \operatorname{sign}(\varepsilon)
                                    |\varepsilon|^\alpha \\
        \psi'_\alpha(\varepsilon) &= \alpha \, |\varepsilon|^{\alpha - 1}

    Residuals with :math:`|\hat{\varepsilon}_i| < \varepsilon_{\min}` are
    clamped to :math:`\varepsilon_{\min}` before computing
    :math:`\psi'_\alpha` to avoid division by zero when :math:`\alpha < 1`.

    Parameters
    ----------
    s : array_like, shape (n,)
        Reference signal samples :math:`s_i`.
    residuals : array_like, shape (n,)
        Fitted residuals :math:`\hat{\varepsilon}_i = y_i - \hat{A} s_i`
        from :func:`pmm_frac_amplitude`.
    alpha : float
        Fractional exponent :math:`\alpha \in (0, 2]` used in the estimator.
    eps_min : float, optional
        Minimum absolute residual for numerical stability (default 1e-10).

    Returns
    -------
    float
        Estimated standard error :math:`\hat{\sigma}(\hat{A}) \geq 0`.
        Returns ``float('inf')`` if the denominator is effectively zero.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = np.ones(1024)
    >>> A_true = 1.5
    >>> y = A_true * s + rng.standard_normal(1024)
    >>> result = pmm_frac_amplitude(y, s, alpha=1.0)
    >>> se = pmm_frac_sandwich_se(s, result["residuals"], alpha=1.0)
    >>> 0.0 < se < 0.1
    True

    Notes
    -----
    For :math:`\alpha = 1` (OLS), this reduces to the standard OLS sandwich
    SE :math:`\hat{\sigma}(\hat{A}) = \hat{\sigma}_\varepsilon / \sqrt{\sum s_i^2}`.

    The formula is consistent with the PMM2 sandwich SE in
    :func:`sdr_kunchenko.pmm.inference.pmm2_sandwich_se_amplitude` and the
    PMM3 sandwich SE in
    :func:`sdr_kunchenko.pmm.inference.pmm3_sandwich_se_amplitude`.

    References
    ----------
    - exp1b requirements §Requirement 3, criterion 5
    - exp1b design §2.1 (ψ'_α formula), §3.1 (sandwich SE spec)
    """
    s = np.asarray(s, dtype=np.float64)
    eps = np.asarray(residuals, dtype=np.float64)

    abs_eps = _clamp_abs(eps, eps_min)
    s2 = s ** 2

    # ψ_α(ε) = sign(ε) · |ε|^α
    psi = np.sign(eps) * (abs_eps ** alpha)

    # ψ'_α(ε) = α · |ε|^(α-1)
    psi_prime = alpha * (abs_eps ** (alpha - 1.0))

    # Meat: M = Σ ψ²(ε̂_i) · s_i²
    M = float(np.sum(psi ** 2 * s2))

    # Bread: J = Σ ψ'(ε̂_i) · s_i²
    J = float(np.sum(psi_prime * s2))

    if abs(J) < 1e-12:
        return float("inf")

    var_A = M / (J ** 2)
    return float(np.sqrt(max(var_A, 0.0)))
