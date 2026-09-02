"""Heavy-tail dataset generators for exp1b experiments.

This module provides three dataset generators for producing and filtering
heavy-tailed noise blocks used in the PMM_FRAC evaluation pipeline:

1. :func:`generate_alpha_stable` — α-stable noise via Chambers-Mallows-Stuck
   algorithm (delegated to ``scipy.stats.levy_stable``).
2. :func:`generate_student_t` — Student-t noise with configurable degrees of
   freedom; falls back to Gaussian at ν → ∞.
3. :func:`extract_radioml_heavy_tail` — filters a RadioML2018-style array to
   retain only leptokurtic (γ̂₄ > threshold) symmetric blocks.

All generators optionally save a reproducibility manifest (JSON) containing
the UID (SHA-256[:8] of the first min(1000, n_samples) samples as float64
bytes), generation parameters, and a UTC timestamp.

References
----------
- exp1b requirements §Requirement 4 (Synthetic_HeavyTail datasets)
- exp1b requirements §Requirement 5 (RadioML_HeavyTail dataset)
- exp1b requirements §Requirement 10 (Reproducibility)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default manifest directory (relative to workspace root)
# ---------------------------------------------------------------------------
_DEFAULT_MANIFEST_DIR = Path("experiments/exp1b_heavy_tail/results/manifests")

__all__ = [
    "generate_alpha_stable",
    "generate_student_t",
    "generate_platykurtic",
    "extract_radioml_heavy_tail",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_uid(data: np.ndarray) -> str:
    """Compute SHA-256[:8] UID from the first min(1000, n_samples) float64 bytes.

    Parameters
    ----------
    data : np.ndarray
        Flat or multi-dimensional array.  The first ``min(1000, data.size)``
        elements (in C-order) are used.

    Returns
    -------
    str
        First 8 hex characters of the SHA-256 digest.
    """
    flat = data.ravel().astype(np.float64)
    n = min(1000, flat.size)
    digest = hashlib.sha256(flat[:n].tobytes()).hexdigest()
    return digest[:8]


def _save_manifest(
    manifest: dict[str, Any],
    generator: str,
    uid: str,
    manifest_dir: Path,
) -> Path:
    """Serialise *manifest* to ``manifest_{generator}_{uid}.json``.

    Creates *manifest_dir* if it does not exist.

    Parameters
    ----------
    manifest : dict
        JSON-serialisable dictionary to write.
    generator : str
        Short generator name used in the filename (e.g. ``"alpha_stable"``).
    uid : str
        8-character UID used in the filename.
    manifest_dir : Path
        Directory in which to write the file.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    filename = manifest_dir / f"manifest_{generator}_{uid}.json"
    with filename.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.debug("Manifest saved to %s", filename)
    return filename


# ---------------------------------------------------------------------------
# Task 4.1 — α-stable generator
# ---------------------------------------------------------------------------


def generate_alpha_stable(
    n_blocks: int,
    block_size: int,
    alpha_s: float,
    *,
    beta: float = 0.0,
    sigma: float = 1.0,
    mu: float = 0.0,
    seed: int = 42,
    save_manifest: bool = True,
    manifest_dir: Path | str | None = None,
) -> np.ndarray:
    r"""Generate α-stable noise blocks using the Chambers-Mallows-Stuck algorithm.

    Delegates to ``scipy.stats.levy_stable.rvs`` which implements the CMS
    algorithm internally.  The returned array has shape ``(n_blocks, block_size)``
    and dtype ``float64``.

    A reproducibility manifest is optionally saved to *manifest_dir* with the
    UID computed from the first ``min(1000, n_blocks * block_size)`` samples.

    Parameters
    ----------
    n_blocks : int
        Number of noise blocks (rows in the output array).
    block_size : int
        Number of samples per block (columns in the output array).
    alpha_s : float
        Stability index :math:`\alpha_s \in (0, 2]`.  Special cases:

        - ``alpha_s = 2.0`` → Gaussian (control case).
        - ``alpha_s = 1.0`` → Cauchy (no finite moments).
        - ``alpha_s = 1.5`` → intermediate heavy-tail.
    beta : float, optional
        Skewness parameter :math:`\beta \in [-1, 1]` (default 0.0, symmetric).
    sigma : float, optional
        Scale parameter :math:`\sigma > 0` (default 1.0).
    mu : float, optional
        Location parameter :math:`\mu` (default 0.0).
    seed : int, optional
        Random seed for reproducibility (default 42).
    save_manifest : bool, optional
        If ``True`` (default), write a ``manifest_alpha_stable_{uid}.json``
        file to *manifest_dir*.
    manifest_dir : path-like or None, optional
        Directory for the manifest file.  Defaults to
        ``experiments/exp1b_heavy_tail/results/manifests/``.

    Returns
    -------
    np.ndarray, shape (n_blocks, block_size)
        Array of α-stable noise samples, dtype ``float64``.

    Examples
    --------
    >>> blocks = generate_alpha_stable(10, 64, 1.5, save_manifest=False)
    >>> blocks.shape
    (10, 64)
    >>> blocks.dtype
    dtype('float64')

    Notes
    -----
    The optimal PMM_FRAC exponent for α-stable noise satisfies
    :math:`\alpha_{\text{opt}} \approx \alpha_s - 1` (see exp1b design §2.3).
    For ``alpha_s = 2.0`` (Gaussian) this gives ``alpha_opt = 1.0`` (OLS).

    References
    ----------
    - exp1b requirements §Requirement 4, criteria 1, 3, 5
    - exp1b design §3.3 (module spec)
    - Chambers, J.M., Mallows, C.L., Stuck, B.W. (1976). "A method for
      simulating stable random variables." *JASA*, 71(354), 340–344.
    """
    from scipy.stats import levy_stable  # lazy import — optional dependency

    total = n_blocks * block_size
    samples = levy_stable.rvs(
        alpha_s,
        beta,
        loc=mu,
        scale=sigma,
        size=total,
        random_state=seed,
    )
    blocks = np.asarray(samples, dtype=np.float64).reshape(n_blocks, block_size)

    uid = _compute_uid(blocks)

    if save_manifest:
        _mdir = Path(manifest_dir) if manifest_dir is not None else _DEFAULT_MANIFEST_DIR
        manifest: dict[str, Any] = {
            "uid": uid,
            "generator": "alpha_stable",
            "alpha_s": float(alpha_s),
            "beta": float(beta),
            "sigma": float(sigma),
            "mu": float(mu),
            "n_blocks": int(n_blocks),
            "block_size": int(block_size),
            "seed": int(seed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest, "alpha_stable", uid, _mdir)

    return blocks


# ---------------------------------------------------------------------------
# Task 4.2 — Student-t generator
# ---------------------------------------------------------------------------


def generate_student_t(
    n_blocks: int,
    block_size: int,
    nu: float,
    *,
    seed: int = 42,
    save_manifest: bool = True,
    manifest_dir: Path | str | None = None,
) -> np.ndarray:
    r"""Generate Student-t noise blocks.

    For ``nu == float('inf')`` or ``nu >= 1000`` the Gaussian limit is used
    (``numpy.random.Generator.standard_normal``).  Otherwise
    ``scipy.stats.t.rvs`` is called with ``df=nu``.

    Parameters
    ----------
    n_blocks : int
        Number of noise blocks (rows in the output array).
    block_size : int
        Number of samples per block (columns in the output array).
    nu : float
        Degrees of freedom :math:`\nu > 0`.  Special cases:

        - ``nu = float('inf')`` or ``nu >= 1000`` → Gaussian (control case).
        - ``nu = 3`` → heavy-tailed, :math:`\gamma_4 = \infty` (theoretical).
        - ``nu = 5`` → :math:`\gamma_4 = 6`.
        - ``nu = 10`` → :math:`\gamma_4 = 1`.
    seed : int, optional
        Random seed for reproducibility (default 42).
    save_manifest : bool, optional
        If ``True`` (default), write a ``manifest_student_t_{uid}.json``
        file to *manifest_dir*.
    manifest_dir : path-like or None, optional
        Directory for the manifest file.  Defaults to
        ``experiments/exp1b_heavy_tail/results/manifests/``.

    Returns
    -------
    np.ndarray, shape (n_blocks, block_size)
        Array of Student-t noise samples, dtype ``float64``.

    Examples
    --------
    >>> blocks = generate_student_t(10, 64, 3, save_manifest=False)
    >>> blocks.shape
    (10, 64)
    >>> blocks.dtype
    dtype('float64')

    >>> # Gaussian limit
    >>> blocks_gauss = generate_student_t(10, 64, float('inf'), save_manifest=False)
    >>> blocks_gauss.shape
    (10, 64)

    Notes
    -----
    The theoretical excess kurtosis for Student-t(ν) is
    :math:`\gamma_4 = 6 / (\nu - 4)` for :math:`\nu > 4`, and undefined
    (infinite) for :math:`\nu \leq 4`.

    References
    ----------
    - exp1b requirements §Requirement 4, criteria 2, 4, 5
    - exp1b design §3.3 (module spec)
    """
    total = n_blocks * block_size

    # Gaussian limit
    if math.isinf(nu) or nu >= 1000:
        rng = np.random.default_rng(seed)
        samples = rng.standard_normal(total)
    else:
        from scipy.stats import t as scipy_t  # lazy import

        samples = scipy_t.rvs(df=nu, size=total, random_state=seed)

    blocks = np.asarray(samples, dtype=np.float64).reshape(n_blocks, block_size)

    uid = _compute_uid(blocks)

    if save_manifest:
        _mdir = Path(manifest_dir) if manifest_dir is not None else _DEFAULT_MANIFEST_DIR
        # Represent infinity as null in JSON (not JSON-serialisable as float)
        nu_json: float | None = None if math.isinf(nu) else float(nu)
        manifest: dict[str, Any] = {
            "uid": uid,
            "generator": "student_t",
            "nu": nu_json,
            "n_blocks": int(n_blocks),
            "block_size": int(block_size),
            "seed": int(seed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest, "student_t", uid, _mdir)

    return blocks


def generate_platykurtic(
    n_blocks: int,
    block_size: int,
    kind: str = "uniform",
    *,
    shape: float = 4.0,
    seed: int = 42,
    save_manifest: bool = True,
    manifest_dir: Path | str | None = None,
) -> np.ndarray:
    r"""Generate symmetric platykurtic (short-tailed, :math:`\gamma_4 < 0`) noise.

    These distributions probe the *power edge* of the PATP family (the regime
    served by the integer-power cubic PMM3 score), complementing the heavy-tail
    fractal edge covered by :func:`generate_alpha_stable` / :func:`generate_student_t`.

    Parameters
    ----------
    kind : {"uniform", "beta", "gennorm"}
        - ``uniform`` → :math:`U(-1, 1)`, theoretical :math:`\gamma_4 = -1.2`.
        - ``beta``    → symmetric ``Beta(shape, shape)`` shifted to zero mean;
          excess kurtosis :math:`\gamma_4 = -6/(2\,\text{shape}+3)`.
        - ``gennorm`` → generalized Gaussian (exponential power) with shape
          ``shape``; ``shape = 2`` is Gaussian, ``shape > 2`` is platykurtic
          (``shape → ∞`` approaches uniform).
    shape : float
        Distribution shape parameter (Beta concentration / gennorm β). Default 4.

    Returns
    -------
    np.ndarray, shape (n_blocks, block_size), dtype float64.
        Samples are NOT per-block centered (the random per-block mean is the
        amplitude-estimation error under study).
    """
    total = n_blocks * block_size
    rng = np.random.default_rng(seed)

    if kind == "uniform":
        samples = rng.uniform(-1.0, 1.0, size=total)
    elif kind == "beta":
        # symmetric Beta(shape, shape) on (0,1) → shift/scale to zero-mean (-1,1)
        samples = 2.0 * (rng.beta(shape, shape, size=total) - 0.5)
    elif kind == "gennorm":
        from scipy.stats import gennorm  # lazy import

        samples = gennorm.rvs(beta=shape, size=total, random_state=seed)
    else:
        raise ValueError(f"Unknown platykurtic kind: {kind!r}. Expected 'uniform', 'beta', 'gennorm'.")

    blocks = np.asarray(samples, dtype=np.float64).reshape(n_blocks, block_size)
    uid = _compute_uid(blocks)

    if save_manifest:
        _mdir = Path(manifest_dir) if manifest_dir is not None else _DEFAULT_MANIFEST_DIR
        manifest: dict[str, Any] = {
            "uid": uid,
            "generator": "platykurtic",
            "kind": kind,
            "shape": float(shape),
            "n_blocks": int(n_blocks),
            "block_size": int(block_size),
            "seed": int(seed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest, f"platykurtic_{kind}", uid, _mdir)

    return blocks


# ---------------------------------------------------------------------------
# Task 4.3 — RadioML heavy-tail filter
# ---------------------------------------------------------------------------


def extract_radioml_heavy_tail(
    radioml_dataset: ArrayLike,
    *,
    gamma4_threshold: float = 1.0,
    gamma3_max: float = 0.5,
    min_blocks: int = 1000,
    save_manifest: bool = True,
    manifest_dir: Path | str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Filter a RadioML2018-style array to retain leptokurtic symmetric blocks.

    For each block in *radioml_dataset* the excess kurtosis :math:`\hat{\gamma}_4`
    (Fisher, i.e. :math:`\mu_4/\sigma^4 - 3`) and skewness :math:`\hat{\gamma}_3`
    are computed.  A block is retained if:

    .. math::

        \hat{\gamma}_4 > \text{gamma4\_threshold}
        \quad \text{AND} \quad
        |\hat{\gamma}_3| < \text{gamma3\_max}

    The first condition selects leptokurtic (heavy-tailed) blocks; the second
    excludes asymmetric blocks that belong to the PMM2 regime.

    Parameters
    ----------
    radioml_dataset : array_like, shape (n_total_blocks, block_size)
        Input array of noise/signal blocks.  Each row is one block.
    gamma4_threshold : float, optional
        Minimum excess kurtosis for a block to be retained (default 1.0).
    gamma3_max : float, optional
        Maximum absolute skewness for a block to be retained (default 0.5).
    min_blocks : int, optional
        Minimum number of blocks expected after filtering.  If the filtered
        count is below this value a warning is logged (default 1000).
    save_manifest : bool, optional
        If ``True`` (default), write a ``manifest_radioml_heavy_tail_{uid}.json``
        file to *manifest_dir*.
    manifest_dir : path-like or None, optional
        Directory for the manifest file.  Defaults to
        ``experiments/exp1b_heavy_tail/results/manifests/``.

    Returns
    -------
    filtered_blocks : np.ndarray, shape (n_blocks_filtered, block_size)
        Blocks that pass the filter, dtype ``float64``.
    metadata : dict
        ``uid`` : str
            SHA-256[:8] of ``filtered_blocks.ravel()[:1000]`` as float64 bytes.
        ``n_blocks_total`` : int
            Total number of input blocks.
        ``n_blocks_filtered`` : int
            Number of blocks that passed the filter.
        ``gamma4_threshold`` : float
            The threshold used.
        ``gamma3_max`` : float
            The threshold used.
        ``gamma4_distribution`` : dict
            Bin counts for the filtered blocks:
            ``"(1.0, 2.0]"``, ``"(2.0, 5.0]"``, ``"(5.0, +inf)"``.
        ``warning`` : bool
            ``True`` if ``n_blocks_filtered < min_blocks``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Create a small synthetic dataset with some heavy-tailed blocks
    >>> data = rng.standard_t(df=3, size=(200, 128))
    >>> filtered, meta = extract_radioml_heavy_tail(
    ...     data, gamma4_threshold=1.0, gamma3_max=0.5,
    ...     min_blocks=10, save_manifest=False
    ... )
    >>> filtered.ndim
    2
    >>> meta["n_blocks_total"]
    200

    Notes
    -----
    If the filtered count is below *min_blocks*, the function logs a warning
    suggesting to lower *gamma4_threshold* to 0.5.  The caller may then
    re-invoke with a lower threshold or fall back to synthetic datasets.

    References
    ----------
    - exp1b requirements §Requirement 5, criteria 1–6
    - exp1b design §3.3 (module spec)
    """
    data = np.asarray(radioml_dataset, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_total, block_size = data.shape

    # Compute per-block γ̂₃ and γ̂₄ (Fisher excess kurtosis)
    xc = data - data.mean(axis=1, keepdims=True)
    m2 = np.mean(xc ** 2, axis=1)  # shape (n_total,)

    # Guard against zero-variance blocks
    valid_var = m2 > 0
    gamma3 = np.zeros(n_total)
    gamma4 = np.full(n_total, -3.0)  # degenerate blocks get γ₄ = -3

    if valid_var.any():
        sigma = np.sqrt(m2[valid_var])
        m3 = np.mean(xc[valid_var] ** 3, axis=1)
        m4 = np.mean(xc[valid_var] ** 4, axis=1)
        gamma3[valid_var] = m3 / sigma ** 3
        gamma4[valid_var] = m4 / m2[valid_var] ** 2 - 3.0

    # Apply filter
    mask = (gamma4 > gamma4_threshold) & (np.abs(gamma3) < gamma3_max)
    filtered = data[mask]
    n_filtered = int(filtered.shape[0])

    # Warn if too few blocks pass the filter
    warn_flag = n_filtered < min_blocks
    if warn_flag:
        logger.warning(
            "extract_radioml_heavy_tail: only %d blocks passed the filter "
            "(gamma4_threshold=%.2f, gamma3_max=%.2f). "
            "Consider lowering gamma4_threshold to 0.5.",
            n_filtered,
            gamma4_threshold,
            gamma3_max,
        )

    # γ̂₄ distribution bins for filtered blocks
    g4_filtered = gamma4[mask]
    bin_lo_mid = int(np.sum((g4_filtered > 1.0) & (g4_filtered <= 2.0)))
    bin_mid_hi = int(np.sum((g4_filtered > 2.0) & (g4_filtered <= 5.0)))
    bin_hi_inf = int(np.sum(g4_filtered > 5.0))

    # UID from filtered blocks
    if n_filtered > 0:
        uid = _compute_uid(filtered)
    else:
        uid = "00000000"

    metadata: dict[str, Any] = {
        "uid": uid,
        "n_blocks_total": int(n_total),
        "n_blocks_filtered": n_filtered,
        "gamma4_threshold": float(gamma4_threshold),
        "gamma3_max": float(gamma3_max),
        "gamma4_distribution": {
            "(1.0, 2.0]": bin_lo_mid,
            "(2.0, 5.0]": bin_mid_hi,
            "(5.0, +inf)": bin_hi_inf,
        },
        "warning": warn_flag,
    }

    if save_manifest:
        _mdir = Path(manifest_dir) if manifest_dir is not None else _DEFAULT_MANIFEST_DIR
        manifest: dict[str, Any] = {
            **metadata,
            "generator": "radioml_heavy_tail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest, "radioml_heavy_tail", uid, _mdir)

    return filtered, metadata
