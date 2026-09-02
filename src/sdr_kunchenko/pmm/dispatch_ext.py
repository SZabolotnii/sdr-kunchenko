"""
AUTO_EXT: 4-level dispatcher for PMM amplitude estimation.

This module implements the **AUTO_EXT** dispatcher — an extension of the
existing 2-level AUTO dispatcher (see :func:`sdr_kunchenko.pmm.oracle.pmm_dispatch`)
that adds a fourth track for heavy-tailed (leptokurtic) noise distributions.

The dispatcher routes a noise block to one of four estimation methods based on
the empirical skewness γ̂₃ and excess kurtosis γ̂₄ of the noise:

  - **PMM2**     — asymmetric noise (|γ̂₃| > 0.3)
  - **PMM3**     — symmetric platykurtic noise (γ̂₄ < −0.7)
  - **PMM_FRAC** — symmetric heavy-tailed noise (γ̂₄ > kurtosis_frac_threshold)
  - **OLS**      — intermediate / Gaussian regime (fallback)

The dispatcher returns only the *routing decision* (method name, cumulants,
alpha_used, rationale).  The actual amplitude estimation is performed
separately in ``run.py::process_block_ext()``.

Module-level constant
---------------------
``KURTOSIS_FRAC_THRESHOLD_DEFAULT = 1.0``
    Initial empirical value for the γ̂₄ threshold that activates the
    PMM_FRAC track.  This value will be refined in Phase 3 Monte Carlo
    experiments (see §Requirement 7.2 of the exp1b requirements document).
    Once the Monte Carlo results are available, this constant should be
    updated to the minimum γ̂₄ at which RE(PMM_FRAC) ≥ 1.1.

References
----------
- exp1b requirements §Requirement 7 (AUTO_EXT dispatcher)
- exp1b design §3.2 (dispatch_ext module specification)
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

logger = logging.getLogger(__name__)

__all__ = [
    "pmm_dispatch_ext",
    "KURTOSIS_FRAC_THRESHOLD_DEFAULT",
]

# ---------------------------------------------------------------------------
# Module-level constant — Phase 3 placeholder
# ---------------------------------------------------------------------------

# Initial empirical value for the γ̂₄ threshold that activates the PMM_FRAC
# track.  This will be updated after Phase 3 Monte Carlo experiments determine
# the minimum γ̂₄ at which RE(PMM_FRAC) ≥ 1.1 (see §Requirement 7.2).
KURTOSIS_FRAC_THRESHOLD_DEFAULT: float = 1.0


# ---------------------------------------------------------------------------
# Task 2.1 — AUTO_EXT 4-level dispatcher
# ---------------------------------------------------------------------------


def pmm_dispatch_ext(
    noise_block: ArrayLike,
    *,
    symmetry_threshold: float = 0.3,
    kurtosis_platykurtic: float = -0.7,
    kurtosis_frac_threshold: float = KURTOSIS_FRAC_THRESHOLD_DEFAULT,
    kurtosis_divergent: float = 10.0,
    alpha_pool: float | None = None,
    g2_threshold: float = 0.95,
) -> dict:
    r"""AUTO_EXT: 4-level dispatcher for PMM amplitude estimation.

    Estimates the empirical skewness γ̂₃ and excess kurtosis γ̂₄ from
    ``noise_block`` and routes to one of four estimation methods.

    Dispatch Logic (ASCII diagram from design §3.2)
    -------------------------------------------------------
    ::

                        ┌─────────────────┐
                        │  noise_block ξ  │
                        └────────┬────────┘
                                 │ estimate γ̂₃, γ̂₄
                                 ▼
                        ┌─────────────────┐
                        │ |γ̂₃| > 0.3?    │
                        └────────┬────────┘
                        YES ◄────┤►──── NO
                        │                │
                        ▼                ▼
                ┌───────────────┐┌─────────────────┐
                │  g₂ ≥ 0.95?   ││ γ̂₄ < -0.7?     │
                └───────┬───────┘└────────┬────────┘
                YES ◄───┤►─── NO  YES ◄───┤►──── NO
                │            │     │              │
                ▼            ▼     ▼              ▼
               OLS         PMM2   PMM3   ┌─────────────────┐
                                        │ γ̂₄ > 1.0?      │
                                        └────────┬────────┘
                                        YES ◄────┤►──── NO
                                        │                │
                                        ▼                ▼
                                   PMM_FRAC(α)          OLS

    Routing rules
    -------------
    1. **Level 1** — if ``|γ̂₃| > symmetry_threshold`` → ``"PMM2"``
       (asymmetric noise; score function of order 2), unless the theoretical
       factor ``g₂ = 1 − γ̂₃²/(2 + γ̂₄) ≥ g2_threshold``, in which case the
       expected gain is marginal and the block is routed to ``"OLS"``
    2. **Level 2** — elif ``γ̂₄ < kurtosis_platykurtic`` → ``"PMM3"``
       (symmetric platykurtic noise; score function of order 3)
    3. **Level 3** — elif ``γ̂₄ > kurtosis_frac_threshold``
       **and** ``alpha_pool is not None`` → ``"PMM_FRAC"``
       (symmetric heavy-tailed noise; fractional-power score function)
    4. **Level 4** — else → ``"OLS"``
       (intermediate / Gaussian regime; ordinary least squares)

    Note on Level 3 fallback
    ------------------------
    If ``γ̂₄ > kurtosis_frac_threshold`` but ``alpha_pool is None``, the
    dispatcher falls back to ``"OLS"`` and logs a warning.  This implements
    §Requirement 7.7: when γ̂-pooling is unavailable (fewer than N_pool
    blocks), AUTO_EXT must not activate PMM_FRAC.

    Parameters
    ----------
    noise_block : array_like, shape (n,)
        A block of noise samples used to estimate γ̂₃ and γ̂₄.
        Larger blocks (n ≥ 512) give more reliable cumulant estimates.
    symmetry_threshold : float, optional
        Threshold on |γ̂₃| for activating PMM2 (default 0.3).
    kurtosis_platykurtic : float, optional
        Upper bound on γ̂₄ for activating PMM3 (default −0.7).
    kurtosis_frac_threshold : float, optional
        Lower bound on γ̂₄ for activating PMM_FRAC (default 1.0).
        This is ``KURTOSIS_FRAC_THRESHOLD_DEFAULT`` — an initial empirical
        value to be refined in Phase 3 Monte Carlo experiments.
    alpha_pool : float or None, optional
        Pre-computed pooled fractional exponent α for PMM_FRAC, estimated
        from N_pool ≥ 100 calibration blocks via γ̂-pooling.
        If ``None``, PMM_FRAC is not activated even when γ̂₄ is large.
    g2_threshold : float, optional
        Marginal-gain guard on the PMM2 branch (default 0.95). When the
        theoretical variance-reduction factor g₂ = 1 − γ̂₃²/(2 + γ̂₄) is at or
        above this value, the expected gain over OLS is below 1 − g₂ and the
        block is routed to OLS instead, so that noisy cumulant weights are not
        risked for a negligible return.

    Returns
    -------
    dict
        ``method`` : str
            Selected estimation method: one of ``"PMM2"``, ``"PMM3"``,
            ``"PMM_FRAC"``, ``"OLS"``.
        ``gamma3`` : float
            Empirical skewness γ̂₃ (Fisher definition, bias=False).
        ``gamma4`` : float
            Empirical excess kurtosis γ̂₄ (Fisher definition, bias=False).
        ``alpha_used`` : float or None
            The value of α passed as ``alpha_pool`` when ``method == "PMM_FRAC"``;
            ``None`` for all other methods.
        ``rationale`` : str
            Human-readable explanation of the routing decision.

    Examples
    --------
    Gaussian noise → OLS:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.standard_normal(2048)
    >>> result = pmm_dispatch_ext(noise)
    >>> result["method"]
    'OLS'

    Heavy-tailed Student-t(3) noise with pooled α → PMM_FRAC:

    >>> from scipy.stats import t as student_t
    >>> noise_ht = student_t.rvs(df=3, size=2048, random_state=42)
    >>> result = pmm_dispatch_ext(noise_ht, alpha_pool=0.4)
    >>> result["method"]
    'PMM_FRAC'
    >>> result["alpha_used"]
    0.4

    Asymmetric noise → PMM2:

    >>> from scipy.stats import expon
    >>> noise_asym = expon.rvs(size=2048, random_state=0) - 1.0
    >>> result = pmm_dispatch_ext(noise_asym)
    >>> result["method"]
    'PMM2'

    References
    ----------
    - exp1b requirements §Requirement 7 (AUTO_EXT dispatcher)
    - exp1b design §3.2 (4-level dispatch logic and ASCII diagram)
    """
    x = np.asarray(noise_block, dtype=np.float64).ravel()

    if x.size < 2:
        logger.warning(
            "pmm_dispatch_ext: noise_block has fewer than 2 samples (%d). "
            "Falling back to OLS.",
            x.size,
        )
        return {
            "method": "OLS",
            "gamma3": float("nan"),
            "gamma4": float("nan"),
            "alpha_used": None,
            "rationale": "Insufficient samples for cumulant estimation; fallback to OLS.",
        }

    # Step 1: Estimate γ̂₃ and γ̂₄ using scipy.stats (Fisher definition,
    # excess kurtosis, bias-corrected).
    gamma3 = float(scipy_skew(x, bias=False))
    gamma4 = float(scipy_kurtosis(x, fisher=True, bias=False))

    # Step 2: routing logic.

    # Level 0: divergent-moment / extreme heavy-tail guard → PMM_FRAC.
    # For infinite-moment noise (e.g. α-stable with α_s < 2) the sample skewness
    # and kurtosis both explode, so the moment-based PMM2/PMM3 thresholds below
    # would misfire (a huge sample |γ̂₃| spuriously triggers PMM2). When γ̂₄ is
    # extreme or non-finite, integer-moment tracks are invalid; the only valid
    # member is the fractional branch, so route there first (if α is available).
    if not np.isfinite(gamma4) or gamma4 > kurtosis_divergent:
        if alpha_pool is not None:
            return {
                "method": "PMM_FRAC",
                "gamma3": gamma3,
                "gamma4": gamma4,
                "alpha_used": float(alpha_pool),
                "rationale": (
                    f"γ̂₄ = {gamma4:.4g} ≥ kurtosis_divergent = {kurtosis_divergent:.1f} "
                    f"(moments unreliable/divergent) → heavy-tailed; "
                    f"activating PMM_FRAC(α={alpha_pool:.4f}) before moment-based tracks."
                ),
            }
        logger.warning(
            "pmm_dispatch_ext: γ̂₄ = %.4g indicates divergent moments but "
            "alpha_pool is None; falling back to OLS.",
            gamma4,
        )
        return {
            "method": "OLS",
            "gamma3": gamma3,
            "gamma4": gamma4,
            "alpha_used": None,
            "rationale": (
                f"γ̂₄ = {gamma4:.4g} ≥ kurtosis_divergent but alpha_pool is None "
                f"→ fallback to OLS."
            ),
        }

    # Level 1: asymmetric noise → PMM2, unless the theoretical gain is marginal
    if abs(gamma3) > symmetry_threshold:
        denom = 2.0 + gamma4
        g2 = 1.0 - gamma3 ** 2 / denom if denom != 0 else float("nan")
        if np.isfinite(g2) and g2 >= g2_threshold:
            return {
                "method": "OLS",
                "gamma3": gamma3,
                "gamma4": gamma4,
                "alpha_used": None,
                "rationale": (
                    f"|γ̂₃| = {abs(gamma3):.4f} > symmetry_threshold = "
                    f"{symmetry_threshold:.4f} but g₂ = {g2:.4f} ≥ g2_threshold = "
                    f"{g2_threshold:.4f} → expected gain below {1.0 - g2_threshold:.0%}; "
                    f"reverting to OLS."
                ),
            }
        return {
            "method": "PMM2",
            "gamma3": gamma3,
            "gamma4": gamma4,
            "alpha_used": None,
            "rationale": (
                f"|γ̂₃| = {abs(gamma3):.4f} > symmetry_threshold = {symmetry_threshold:.4f} "
                f"and g₂ = {g2:.4f} < {g2_threshold:.4f} → asymmetric noise; activating PMM2."
            ),
        }

    # Level 2: symmetric platykurtic noise → PMM3
    if gamma4 < kurtosis_platykurtic:
        return {
            "method": "PMM3",
            "gamma3": gamma3,
            "gamma4": gamma4,
            "alpha_used": None,
            "rationale": (
                f"γ̂₄ = {gamma4:.4f} < kurtosis_platykurtic = {kurtosis_platykurtic:.4f} "
                f"→ platykurtic noise; activating PMM3."
            ),
        }

    # Level 3: symmetric heavy-tailed noise → PMM_FRAC (requires alpha_pool)
    if gamma4 > kurtosis_frac_threshold:
        if alpha_pool is not None:
            return {
                "method": "PMM_FRAC",
                "gamma3": gamma3,
                "gamma4": gamma4,
                "alpha_used": float(alpha_pool),
                "rationale": (
                    f"γ̂₄ = {gamma4:.4f} > kurtosis_frac_threshold = {kurtosis_frac_threshold:.4f} "
                    f"and alpha_pool = {alpha_pool:.4f} is available "
                    f"→ heavy-tailed noise; activating PMM_FRAC(α={alpha_pool:.4f})."
                ),
            }
        else:
            # §Requirement 7.7: alpha_pool unavailable → fall back to OLS
            logger.warning(
                "pmm_dispatch_ext: γ̂₄ = %.4f > kurtosis_frac_threshold = %.4f "
                "but alpha_pool is None (γ̂-pooling unavailable). "
                "Falling back to OLS.",
                gamma4,
                kurtosis_frac_threshold,
            )
            return {
                "method": "OLS",
                "gamma3": gamma3,
                "gamma4": gamma4,
                "alpha_used": None,
                "rationale": (
                    f"γ̂₄ = {gamma4:.4f} > kurtosis_frac_threshold = {kurtosis_frac_threshold:.4f} "
                    f"but alpha_pool is None (γ̂-pooling unavailable per §Req 7.7) "
                    f"→ fallback to OLS."
                ),
            }

    # Level 4: intermediate / Gaussian regime → OLS
    return {
        "method": "OLS",
        "gamma3": gamma3,
        "gamma4": gamma4,
        "alpha_used": None,
        "rationale": (
            f"|γ̂₃| = {abs(gamma3):.4f} ≤ {symmetry_threshold:.4f}, "
            f"γ̂₄ = {gamma4:.4f} ∈ [{kurtosis_platykurtic:.4f}, {kurtosis_frac_threshold:.4f}] "
            f"→ intermediate or Gaussian regime; activating OLS."
        ),
    }
