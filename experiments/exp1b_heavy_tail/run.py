"""Experiment 1B — PMM_FRAC amplitude estimation on heavy-tailed datasets.

Extends exp1 (experiments/exp1_pmm_estimation/run.py) by adding:
  - PMM_FRAC estimator (fractional-power score function ψ_α(ε) = sign(ε)|ε|^α)
  - AUTO_EXT dispatcher (4-level: PMM2 / PMM3 / PMM_FRAC / OLS)
  - γ̂-pooling for stable α estimation (build_alpha_pool)
  - Support for synthetic heavy-tail datasets (α-stable, Student-t)

Model: y[n] = A · s[n] + ξ[n], estimate A.

Output:
  results/{experiment_id}_{dataset_name}.csv  — one row per trial
  results/{experiment_id}_summary.csv         — aggregated MSE/RE per method/SNR/dataset

References
----------
- exp1b requirements §Requirement 6 (Monte Carlo)
- exp1b design §3.4 (run.py specification)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import kurtosis as scipy_kurtosis
from tqdm import tqdm

from sdr_kunchenko.gsa.basis import patp_power
from sdr_kunchenko.pmm.dispatch_ext import pmm_dispatch_ext
from sdr_kunchenko.pmm.fractional import (
    estimate_alpha_from_cumulants,
    pmm_frac_amplitude,
    pmm_frac_sandwich_se,
)

# Optional exp1 estimators — imported lazily to avoid hard R dependency
# when only PMM_FRAC / AUTO_EXT are needed.
try:
    from sdr_kunchenko.pmm import oracle
    from sdr_kunchenko.pmm.inference import (
        ols_se_amplitude,
        pmm2_sandwich_se_amplitude,
        pmm3_sandwich_se_amplitude,
    )
    from sdr_kunchenko.pmm.pooled import (
        pmm2_pooled_amplitude,
        pmm3_pooled_amplitude,
        pool_gamma_from_blocks,
    )
    _HAS_ORACLE = True
except Exception:  # rpy2 / R not available
    _HAS_ORACLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "config_alphastable.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 5.1 — process_block_ext
# ---------------------------------------------------------------------------


def _huber_amplitude(
    y: np.ndarray, s: np.ndarray, *, c: float = 1.345, max_iter: int = 50, tol: float = 1e-8
) -> float:
    """Huber M-estimate of amplitude A in y = A·s + ξ (robust baseline).

    Standard IRLS with Huber weights w = min(1, c/|u|), u = r/scale, where the
    scale is a robust MAD estimate. This is the natural robust competitor a
    reviewer expects on heavy-tailed noise, alongside OLS and the PMM family.
    """
    s2 = float(s @ s)
    A = float((y @ s) / s2) if s2 > 0 else 0.0
    for _ in range(max_iter):
        r = y - A * s
        mad = float(np.median(np.abs(r - np.median(r))))
        scale = 1.4826 * mad if mad > 0 else float(np.std(r))
        if scale <= 0:
            break
        u = r / scale
        w = np.ones_like(u)
        big = np.abs(u) > c
        w[big] = c / np.abs(u[big])
        denom = float((w * s * s).sum())
        if denom <= 0:
            break
        A_new = float((w * s * y).sum() / denom)
        if abs(A_new - A) < tol:
            A = A_new
            break
        A = A_new
    return A


def process_block_ext(
    *,
    idx: int,
    snr_db: float,
    clean: np.ndarray,
    noise: np.ndarray,
    A_true: float,
    pool: dict | None = None,
    alpha_pool: float | None = None,
    patp_alpha_grid: list[float] | None = None,
) -> dict:
    """One MC trial: y = A_true·s + ξ, run OLS + PMM_FRAC + AUTO_EXT.

    Extends exp1 ``process_block()`` by adding PMM_FRAC and AUTO_EXT results.

    Parameters
    ----------
    idx : int
        Block index in the dataset.
    snr_db : float
        SNR level in dB for this trial.
    clean : np.ndarray, shape (N,)
        Clean reference signal s[n].
    noise : np.ndarray, shape (N,)
        Noise block ξ[n].
    A_true : float
        True amplitude used to generate y.
    pool : dict or None
        Pre-computed calibration pool from exp1 (gamma3_mean, gamma4_mean,
        gamma6_mean, kappa_mean). Used for PMM2_POOL / PMM3_POOL if available.
    alpha_pool : float or None
        Pre-computed pooled fractional exponent α for PMM_FRAC, estimated
        from N_pool ≥ 100 calibration blocks via γ̂-pooling.
        If None, PMM_FRAC uses α=1.0 (OLS fallback) and AUTO_EXT cannot
        activate the PMM_FRAC track.

    Returns
    -------
    dict
        Keys: idx, snr_db, A_true, A_ols, SE_ols,
              A_pmm2, SE_pmm2, pmm2_converged (if oracle available),
              A_pmm3, SE_pmm3, pmm3_converged (if oracle available),
              A_auto, SE_auto, method_auto (if oracle available),
              A_pmm_frac, SE_pmm_frac, converged_frac, alpha_used,
              A_auto_ext, SE_auto_ext, method_auto_ext,
              gamma3_xi, gamma4_xi.
    """
    s = clean
    xi = noise
    y = A_true * s + xi

    # ------------------------------------------------------------------ OLS
    s2 = float(s @ s)
    A_ols = float((y @ s) / s2)
    eps_ols = y - A_ols * s
    sigma_ols = float(np.sqrt(np.var(eps_ols, ddof=0)))
    # Simple OLS SE: σ / sqrt(Σs²)
    se_ols = float(sigma_ols / np.sqrt(s2)) if s2 > 0 else float("inf")

    row: dict[str, Any] = {
        "idx": idx,
        "snr_db": snr_db,
        "A_true": A_true,
        "A_ols": A_ols,
        "SE_ols": se_ols,
    }

    # ------------------------------------------------------------------ Huber (robust baseline)
    try:
        row["A_huber"] = _huber_amplitude(y, s)
    except Exception:
        row["A_huber"] = A_ols

    # ------------------------------------------------------------------ exp1 estimators (optional)
    A_pmm2 = float("nan")
    SE_pmm2 = float("nan")
    pmm2_converged = False
    A_pmm3 = float("nan")
    SE_pmm3 = float("nan")
    pmm3_converged = False
    A_auto = float("nan")
    SE_auto = float("nan")
    method_auto = "N/A"

    if _HAS_ORACLE:
        # PMM2
        try:
            fit2 = oracle.lm_pmm2(y, s)
            pmm2_converged = bool(fit2.get("converged", False))
            if pmm2_converged:
                A_pmm2 = float(fit2["coef"][0])
                SE_pmm2 = pmm2_sandwich_se_amplitude(
                    s, fit2["residuals"],
                    gamma3=fit2["gamma3"], gamma4=fit2["gamma4"],
                )
            else:
                A_pmm2 = A_ols
                SE_pmm2 = se_ols
        except Exception:
            A_pmm2 = A_ols
            SE_pmm2 = se_ols

        # PMM3
        try:
            fit3 = oracle.lm_pmm3(y, s)
            pmm3_converged = bool(fit3.get("converged", False))
            if pmm3_converged:
                A_pmm3 = float(fit3["coef"][0])
                SE_pmm3 = pmm3_sandwich_se_amplitude(
                    s, fit3["residuals"], kappa=fit3.get("kappa", 0.0)
                )
            else:
                A_pmm3 = A_ols
                SE_pmm3 = se_ols
        except Exception:
            A_pmm3 = A_ols
            SE_pmm3 = se_ols

        # AUTO (exp1 2-level dispatcher)
        try:
            disp = oracle.pmm_dispatch(xi)
            method_auto = disp["method"]
            if method_auto == "PMM2":
                A_auto, SE_auto = A_pmm2, SE_pmm2
            elif method_auto == "PMM3":
                A_auto, SE_auto = A_pmm3, SE_pmm3
            else:
                A_auto, SE_auto = A_ols, se_ols
        except Exception:
            A_auto, SE_auto = A_ols, se_ols
            method_auto = "OLS"

    row.update({
        "A_pmm2": A_pmm2,
        "SE_pmm2": SE_pmm2,
        "pmm2_converged": pmm2_converged,
        "A_pmm3": A_pmm3,
        "SE_pmm3": SE_pmm3,
        "pmm3_converged": pmm3_converged,
        "A_auto": A_auto,
        "SE_auto": SE_auto,
        "method_auto": method_auto,
    })

    # ------------------------------------------------------------------ PMM_FRAC
    # Use alpha_pool if provided; otherwise fall back to α=1.0 (OLS).
    alpha_eff = alpha_pool if alpha_pool is not None else 1.0
    try:
        frac_result = pmm_frac_amplitude(y, s, alpha=alpha_eff, max_iter=100)
        A_pmm_frac = float(frac_result["coef"][0])
        SE_pmm_frac = pmm_frac_sandwich_se(s, frac_result["residuals"], alpha=alpha_eff)
        converged_frac = bool(frac_result["converged"])
    except Exception as exc:
        logger.warning("pmm_frac_amplitude failed (idx=%d): %s", idx, exc)
        A_pmm_frac = A_ols
        SE_pmm_frac = se_ols
        converged_frac = False

    row.update({
        "A_pmm_frac": A_pmm_frac,
        "SE_pmm_frac": SE_pmm_frac,
        "alpha_used": alpha_eff,
        "converged_frac": converged_frac,
    })

    # ------------------------------------------------------------------ AUTO_EXT (4-level)
    try:
        disp_ext = pmm_dispatch_ext(xi, alpha_pool=alpha_pool)
        method_auto_ext = disp_ext["method"]
        gamma3_xi = float(disp_ext["gamma3"])
        gamma4_xi = float(disp_ext["gamma4"])

        if method_auto_ext == "PMM_FRAC":
            A_auto_ext = A_pmm_frac
            SE_auto_ext = SE_pmm_frac
        elif method_auto_ext == "PMM2" and _HAS_ORACLE:
            A_auto_ext = A_pmm2
            SE_auto_ext = SE_pmm2
        elif method_auto_ext == "PMM3" and _HAS_ORACLE:
            A_auto_ext = A_pmm3
            SE_auto_ext = SE_pmm3
        else:
            A_auto_ext = A_ols
            SE_auto_ext = se_ols
    except Exception as exc:
        logger.warning("pmm_dispatch_ext failed (idx=%d): %s", idx, exc)
        A_auto_ext = A_ols
        SE_auto_ext = se_ols
        method_auto_ext = "OLS"
        gamma3_xi = float("nan")
        gamma4_xi = float("nan")

    row.update({
        "A_auto_ext": A_auto_ext,
        "SE_auto_ext": SE_auto_ext,
        "method_auto_ext": method_auto_ext,
        "gamma3_xi": gamma3_xi,
        "gamma4_xi": gamma4_xi,
    })

    # ------------------------------------------------------------------ PATP α-sweep
    # PATP S=2 leading fractional term is ψ(ε) = sign(ε)·|ε|^{p₂(α)} (arXiv:2605.14610).
    # We realize it via pmm_frac_amplitude with score exponent = patp_power(2, α):
    #   α=0   → p₂=1/2 (fractal / heavy-tail), α=1/2 → p₂=1 (linear≈OLS),
    #   α=1   → p₂=2   (signed-square / power regime).
    if patp_alpha_grid:
        for a in patp_alpha_grid:
            a = float(a)
            col_A = f"A_patp_a{a:.2f}"
            col_SE = f"SE_patp_a{a:.2f}"
            try:
                p = patp_power(2, a)
                pres = pmm_frac_amplitude(y, s, alpha=p, max_iter=100)
                # Record NaN (not a silent OLS fallback) on non-convergence so the
                # α-sweep curve never shows a misleading RE≈1 at the super-linear
                # edge (p₂→2), where IRLS degenerates on heavy-tailed noise.
                if pres.get("converged", False):
                    row[col_A] = float(pres["coef"][0])
                    row[col_SE] = pmm_frac_sandwich_se(s, pres["residuals"], alpha=p)
                else:
                    row[col_A] = float("nan")
                    row[col_SE] = float("nan")
            except Exception as exc:
                logger.warning("PATP α=%.2f failed (idx=%d): %s", a, idx, exc)
                row[col_A] = float("nan")
                row[col_SE] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Task 5.2 — build_alpha_pool
# ---------------------------------------------------------------------------


def build_alpha_pool(noise_blocks: np.ndarray, *, n_pool: int = 200) -> float:
    """γ̂-pooling for stable α estimation.

    Takes the first ``min(n_pool, len(noise_blocks))`` blocks, computes the
    excess kurtosis γ̂₄ (Fisher, bias-corrected) for each, averages them, and
    maps the mean γ̂₄ to an optimal fractional exponent α via
    :func:`estimate_alpha_from_cumulants`.

    Parameters
    ----------
    noise_blocks : np.ndarray, shape (n_blocks, block_size)
        Array of noise blocks.  Each row is one block.
    n_pool : int, optional
        Maximum number of blocks to use for pooling (default 200).

    Returns
    -------
    float
        Pooled α estimate in [0.1, 1.0].

    References
    ----------
    - exp1b requirements §Requirement 2, criterion 6 (γ̂-pooling)
    - exp1b design §3.4 (build_alpha_pool spec)
    """
    n_use = min(n_pool, len(noise_blocks))
    blocks = noise_blocks[:n_use]

    # Compute excess kurtosis (Fisher, bias-corrected) for each block.
    gamma4_vals = np.array([
        float(scipy_kurtosis(blocks[i], fisher=True, bias=False))
        for i in range(n_use)
    ])

    # Filter out non-finite values (can occur for very heavy-tailed blocks
    # where sample moments are unreliable).
    finite_mask = np.isfinite(gamma4_vals)
    if finite_mask.sum() == 0:
        logger.warning(
            "build_alpha_pool: all γ̂₄ values are non-finite; returning α=0.5 fallback."
        )
        return 0.5

    mean_gamma4 = float(gamma4_vals[finite_mask].mean())
    logger.debug(
        "build_alpha_pool: n_pool=%d, mean_γ̂₄=%.4f (from %d finite blocks)",
        n_use, mean_gamma4, int(finite_mask.sum()),
    )

    alpha_pool = estimate_alpha_from_cumulants(mean_gamma4, gamma6=0.0)
    return alpha_pool


# ---------------------------------------------------------------------------
# Task 5.3 — MC driver
# ---------------------------------------------------------------------------


def _generate_dataset(ds_cfg: dict, seed: int) -> np.ndarray:
    """Generate noise blocks for a single dataset config entry.

    Supports ``type: alpha_stable`` and ``type: student_t``.
    """
    from sdr_kunchenko.data_gen.heavy_tail import (
        generate_alpha_stable,
        generate_platykurtic,
        generate_student_t,
    )

    ds_type = ds_cfg.get("type", "alpha_stable")
    n_blocks = int(ds_cfg.get("n_blocks", 10000))
    block_size = int(ds_cfg.get("block_size", 1024))

    if ds_type == "alpha_stable":
        alpha_s = float(ds_cfg["alpha_s"])
        blocks = generate_alpha_stable(
            n_blocks=n_blocks,
            block_size=block_size,
            alpha_s=alpha_s,
            beta=float(ds_cfg.get("beta", 0.0)),
            sigma=float(ds_cfg.get("sigma", 1.0)),
            mu=float(ds_cfg.get("mu", 0.0)),
            seed=seed,
            save_manifest=True,
        )
    elif ds_type == "student_t":
        import math
        nu_raw = ds_cfg.get("nu", 5)
        nu = float("inf") if str(nu_raw).lower() in (".inf", "inf", "infinity") else float(nu_raw)
        blocks = generate_student_t(
            n_blocks=n_blocks,
            block_size=block_size,
            nu=nu,
            seed=seed,
            save_manifest=True,
        )
    elif ds_type == "platykurtic":
        blocks = generate_platykurtic(
            n_blocks=n_blocks,
            block_size=block_size,
            kind=str(ds_cfg.get("kind", "uniform")),
            shape=float(ds_cfg.get("shape", 4.0)),
            seed=seed,
            save_manifest=True,
        )
    elif ds_type == "radioml_heavy_tail":
        # Real RF heavy-tail samples: scan RadioML2018 over-the-air I/Q blocks and
        # retain leptokurtic, near-symmetric ones (γ̂₄ > thr, |γ̂₃| < thr) as the
        # empirical heavy-tail perturbation ξ.  Honest caveat: these are real RF
        # signal blocks used as a heavy-tail source, not literal channel residuals.
        import h5py

        from sdr_kunchenko.data_gen.heavy_tail import extract_radioml_heavy_tail

        path = ds_cfg.get("radioml_path") or os.environ.get("RADIOML_PATH")
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                "radioml_heavy_tail requires a valid HDF5 path via config "
                f"'radioml_path' or $RADIOML_PATH; got {path!r}"
            )
        max_scan = int(ds_cfg.get("max_scan_blocks", 100000))
        channel = int(ds_cfg.get("channel", 0))  # 0 = I, 1 = Q
        with h5py.File(path, "r") as f:
            X = f["X"]  # RadioML2018.01A: (M, 1024, 2) interleaved I/Q
            n_avail = X.shape[0]
            stride = max(1, n_avail // max_scan)
            arr = X[::stride][:max_scan, :, channel].astype(np.float64)
        # Scale-normalize per block (unit variance) but DO NOT center: the random
        # per-block mean is the amplitude-estimation error under study; subtracting
        # it would make OLS trivially exact. Kurtosis filtering is scale-invariant.
        std = arr.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        arr = arr / std
        blocks, manifest = extract_radioml_heavy_tail(
            arr,
            gamma4_threshold=float(ds_cfg.get("gamma4_threshold", 1.0)),
            gamma3_max=float(ds_cfg.get("gamma3_max", 0.5)),
            min_blocks=int(ds_cfg.get("min_blocks", 1000)),
            save_manifest=False,
        )
        logger.info(
            "radioml_heavy_tail: scanned %d blocks, retained %d leptokurtic "
            "(γ̂₄>%.2f, |γ̂₃|<%.2f)",
            len(arr), len(blocks),
            float(ds_cfg.get("gamma4_threshold", 1.0)),
            float(ds_cfg.get("gamma3_max", 0.5)),
        )
        return blocks
    elif ds_type == "iq_heavy_tail":
        # Real RF heavy-tail from raw interleaved-I/Q recordings (e.g. Tampere
        # DJI OcuSync .bin @ int16). Impulsive burst onsets produce genuinely
        # leptokurtic blocks; we pool I and Q channels (both symmetric) and keep
        # only γ̂₄ > thr, |γ̂₃| < thr blocks as the empirical heavy-tail source.
        from sdr_kunchenko.data_gen.heavy_tail import extract_radioml_heavy_tail

        path = ds_cfg.get("iq_path") or os.environ.get("KU_IQ_PATH")
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                "iq_heavy_tail requires a valid raw I/Q path via config "
                f"'iq_path' or $KU_IQ_PATH; got {path!r}"
            )
        dtype = np.dtype(ds_cfg.get("dtype", "int16"))
        max_complex = int(ds_cfg.get("max_complex_samples", 120_000_000))
        raw = np.fromfile(path, dtype=dtype, count=2 * max_complex)
        I = raw[0::2].astype(np.float64)
        Q = raw[1::2].astype(np.float64)
        chans = []
        for ch in (I, Q):
            m = (len(ch) // block_size) * block_size
            chans.append(ch[:m].reshape(-1, block_size))
        arr = np.vstack(chans)
        # Scale-normalize per block (unit variance) but DO NOT center: the random
        # per-block mean is exactly the amplitude-estimation error under study, so
        # subtracting it would make OLS trivially exact and the comparison moot.
        std = arr.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        arr = arr / std
        blocks, manifest = extract_radioml_heavy_tail(
            arr,
            gamma4_threshold=float(ds_cfg.get("gamma4_threshold", 1.0)),
            gamma3_max=float(ds_cfg.get("gamma3_max", 0.5)),
            min_blocks=int(ds_cfg.get("min_blocks", 1000)),
            save_manifest=False,
        )
        logger.info(
            "iq_heavy_tail[%s]: scanned %d I/Q blocks, retained %d leptokurtic "
            "(γ̂₄>%.2f, |γ̂₃|<%.2f)",
            Path(path).name, len(arr), len(blocks),
            float(ds_cfg.get("gamma4_threshold", 1.0)),
            float(ds_cfg.get("gamma3_max", 0.5)),
        )
        return blocks
    else:
        raise ValueError(
            f"Unknown dataset type: {ds_type!r}. Expected 'alpha_stable', "
            "'student_t', 'radioml_heavy_tail', or 'iq_heavy_tail'."
        )

    return blocks


def _build_mc_trials(
    noise_blocks: np.ndarray,
    snr_grid: list[float],
    n_blocks_per_cell: int,
    A_true_range: tuple[float, float],
    rng: np.random.Generator,
) -> list[dict]:
    """Build the list of MC trial descriptors.

    For each SNR level, randomly sample ``n_blocks_per_cell`` block indices
    (with replacement if needed) and assign a random A_true.

    For synthetic datasets there is no per-block SNR label, so we treat all
    blocks as eligible for every SNR level.  The SNR is used only as a label
    for stratification in the output CSV.
    """
    n_total = len(noise_blocks)
    A_lo, A_hi = A_true_range
    trials = []
    for snr_db in snr_grid:
        n_take = min(n_blocks_per_cell, n_total)
        idxs = rng.choice(n_total, size=n_take, replace=(n_take > n_total))
        for i in idxs:
            trials.append({
                "idx": int(i),
                "snr_db": float(snr_db),
                "A_true": float(rng.uniform(A_lo, A_hi)),
            })
    return trials


def _worker_ext(
    trial: dict,
    noise_blocks: np.ndarray,
    block_size: int,
    alpha_pool: float | None,
    patp_alpha_grid: list[float] | None = None,
) -> dict:
    """joblib worker for a single MC trial."""
    i = trial["idx"]
    xi = noise_blocks[i].astype(np.float64)
    s = np.ones(block_size, dtype=np.float64)  # unit pilot signal
    return process_block_ext(
        idx=i,
        snr_db=trial["snr_db"],
        clean=s,
        noise=xi,
        A_true=trial["A_true"],
        alpha_pool=alpha_pool,
        patp_alpha_grid=patp_alpha_grid,
    )


def _compute_summary(df: pd.DataFrame, experiment_id: str, dataset_name: str) -> pd.DataFrame:
    """Aggregate MSE and RE per method per SNR level.

    RE(method) = MSE(OLS) / MSE(method).  RE > 1 means method beats OLS.
    """
    methods = {
        "OLS": "A_ols",
        "HUBER": "A_huber",
        "PMM2": "A_pmm2",
        "PMM3": "A_pmm3",
        "AUTO": "A_auto",
        "PMM_FRAC": "A_pmm_frac",
        "AUTO_EXT": "A_auto_ext",
        "PATP*": "A_patp_star",  # best-α PATP column, injected when sweep is enabled
    }

    rows = []
    for snr_db, grp in df.groupby("snr_db"):
        A_true = grp["A_true"].values
        mse_ols = float(np.nanmean((grp["A_ols"].values - A_true) ** 2))
        for method_name, col in methods.items():
            if col not in grp.columns:
                continue
            A_hat = grp[col].values
            mse = float(np.nanmean((A_hat - A_true) ** 2))
            re = mse_ols / mse if mse > 0 else float("inf")
            rows.append({
                "experiment_id": experiment_id,
                "dataset": dataset_name,
                "snr_db": snr_db,
                "method": method_name,
                "n_trials": len(grp),
                "MSE": mse,
                "RE": re,
            })

    return pd.DataFrame(rows)


def _compute_patp_sweep(
    df: pd.DataFrame,
    experiment_id: str,
    dataset_name: str,
    patp_alpha_grid: list[float],
) -> pd.DataFrame:
    """Aggregate MSE/RE over all trials per PATP α (the α-transition curve).

    SNR is a replication label here (synthetic noise power is fixed by the
    generator), so the curve is pooled over all trials for statistical power.
    RE(α) = MSE(OLS) / MSE(PATP(α)); RE > 1 means PATP beats OLS at that α.
    """
    A_true = df["A_true"].values
    mse_ols = float(np.mean((df["A_ols"].values - A_true) ** 2))
    rows = []
    for a in patp_alpha_grid:
        a = float(a)
        col = f"A_patp_a{a:.2f}"
        if col not in df.columns:
            continue
        sq = (df[col].values - A_true) ** 2
        n_finite = int(np.isfinite(sq).sum())
        if n_finite == 0:
            continue  # all trials non-convergent at this α
        mse = float(np.nanmean(sq))
        re = mse_ols / mse if mse > 0 else float("inf")
        rows.append({
            "experiment_id": experiment_id,
            "dataset": dataset_name,
            "alpha": a,
            "p2_alpha": float(patp_power(2, a)),
            "n_trials": len(df),
            "n_converged": n_finite,
            "MSE": mse,
            "RE": re,
        })
    return pd.DataFrame(rows)


def run_experiment(
    config_path: str | Path,
    *,
    jobs: int = 1,
    blocks: int = 500,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Path:
    """Run the exp1b Monte Carlo experiment.

    Parameters
    ----------
    config_path : path-like
        Path to the YAML configuration file.
    jobs : int
        Number of parallel jobs (1 = serial).
    blocks : int
        Number of MC blocks per SNR cell (overrides config).
    seed : int
        Random seed.
    output_dir : path-like or None
        Output directory.  Overrides config if provided.

    Returns
    -------
    Path
        Path to the summary CSV file.
    """
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())

    experiment_id: str = cfg.get("experiment_id", "exp1b")
    cfg_seed: int = int(cfg.get("seed", seed))
    mc_cfg: dict = cfg.get("monte_carlo", {})
    snr_grid: list[float] = [float(x) for x in mc_cfg.get("snr_grid", [-5, 0, 5, 10, 15])]
    n_blocks_per_cell: int = int(mc_cfg.get("N_blocks_per_cell", blocks))
    A_true_range: tuple[float, float] = tuple(mc_cfg.get("A_true_range", [0.5, 2.0]))  # type: ignore[assignment]
    n_calib: int = int(mc_cfg.get("N_calibration_blocks", 200))
    patp_alpha_grid_cfg = mc_cfg.get("patp_alpha_grid", cfg.get("patp_alpha_grid"))
    patp_alpha_grid: list[float] | None = (
        [float(x) for x in patp_alpha_grid_cfg] if patp_alpha_grid_cfg else None
    )

    # Output directory
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = PROJECT_ROOT / cfg.get("output", {}).get("results_dir", "experiments/exp1b_heavy_tail/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_path = out_dir / f"{experiment_id}_log.txt"
    logging.basicConfig(
        filename=log_path, filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    logger.info("Experiment: %s | config: %s", experiment_id, config_path)
    logger.info("SNR grid: %s | blocks/cell: %d | seed: %d", snr_grid, n_blocks_per_cell, cfg_seed)

    all_trial_dfs: list[pd.DataFrame] = []
    all_summary_dfs: list[pd.DataFrame] = []
    all_sweep_dfs: list[pd.DataFrame] = []

    datasets_cfg: list[dict] = cfg.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("Config must contain a non-empty 'datasets' list.")

    for ds_cfg in datasets_cfg:
        ds_name: str = ds_cfg["name"]
        logger.info("--- Dataset: %s ---", ds_name)

        # Generate noise blocks
        t0 = time.time()
        noise_blocks = _generate_dataset(ds_cfg, seed=cfg_seed)
        block_size = noise_blocks.shape[1]
        logger.info(
            "Generated %d blocks × %d samples in %.2fs",
            len(noise_blocks), block_size, time.time() - t0,
        )

        # Build alpha_pool via γ̂-pooling
        t0 = time.time()
        alpha_pool = build_alpha_pool(noise_blocks, n_pool=n_calib)
        logger.info(
            "alpha_pool = %.4f (from %d calibration blocks, elapsed %.2fs)",
            alpha_pool, min(n_calib, len(noise_blocks)), time.time() - t0,
        )

        # Build MC trial list
        rng = np.random.default_rng(cfg_seed)
        trials = _build_mc_trials(
            noise_blocks,
            snr_grid=snr_grid,
            n_blocks_per_cell=n_blocks_per_cell,
            A_true_range=A_true_range,
            rng=rng,
        )
        logger.info("MC trials: %d (across %d SNR levels)", len(trials), len(snr_grid))

        # Run MC
        t0 = time.time()
        if jobs in (None, 0, 1):
            rows = [
                _worker_ext(t, noise_blocks, block_size, alpha_pool, patp_alpha_grid)
                for t in tqdm(trials, desc=f"serial MC [{ds_name}]")
            ]
        else:
            rows = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
                delayed(_worker_ext)(t, noise_blocks, block_size, alpha_pool, patp_alpha_grid)
                for t in tqdm(trials, desc=f"parallel MC [{ds_name}] (n_jobs={jobs})")
            )
        elapsed = time.time() - t0
        logger.info(
            "MC done in %.2fs (%.4f s/trial)", elapsed, elapsed / max(1, len(rows))
        )

        # Save per-trial CSV
        df = pd.DataFrame(rows)
        df["dataset"] = ds_name
        df["experiment_id"] = experiment_id

        # PATP α-sweep aggregation + α* selection (inject best-α column for summary).
        if patp_alpha_grid:
            sweep_df = _compute_patp_sweep(df, experiment_id, ds_name, patp_alpha_grid)
            all_sweep_dfs.append(sweep_df)
            if not sweep_df.empty:
                astar = float(sweep_df.loc[sweep_df["RE"].idxmax(), "alpha"])
                df["A_patp_star"] = df[f"A_patp_a{astar:.2f}"]
                df["patp_alpha_star"] = astar
                logger.info(
                    "[%s] PATP α* = %.2f (RE=%.3f, p₂=%.3f)",
                    ds_name, astar, float(sweep_df["RE"].max()), float(patp_power(2, astar)),
                )

        trial_csv = out_dir / f"{experiment_id}_{ds_name}.csv"
        df.to_csv(trial_csv, index=False)
        logger.info("Wrote %s (%d rows)", trial_csv, len(df))
        all_trial_dfs.append(df)

        # Compute summary
        summary_df = _compute_summary(df, experiment_id, ds_name)
        all_summary_dfs.append(summary_df)

    # Save combined summary CSV
    summary_all = pd.concat(all_summary_dfs, ignore_index=True)
    summary_csv = out_dir / f"{experiment_id}_summary.csv"
    summary_all.to_csv(summary_csv, index=False)
    logger.info("Wrote summary: %s (%d rows)", summary_csv, len(summary_all))

    # Save PATP α-sweep CSV (the α-transition curve) if enabled
    if all_sweep_dfs:
        sweep_all = pd.concat(all_sweep_dfs, ignore_index=True)
        sweep_csv = out_dir / f"{experiment_id}_patp_sweep.csv"
        sweep_all.to_csv(sweep_csv, index=False)
        logger.info("Wrote PATP sweep: %s (%d rows)", sweep_csv, len(sweep_all))

    # Save manifest
    manifest = _make_manifest(cfg, experiment_id, sum(len(d) for d in all_trial_dfs))
    manifest_path = out_dir / f"{experiment_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest: %s", manifest_path)

    return summary_csv


# ---------------------------------------------------------------------------
# Task 5.4 — results directory helpers
# ---------------------------------------------------------------------------


def _make_manifest(cfg: dict, experiment_id: str, n_trials: int) -> dict:
    """Build a reproducibility manifest dict."""
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        pass
    return {
        "experiment_id": experiment_id,
        "git_sha": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": cfg,
        "n_trials_total": n_trials,
        "oracle_available": _HAS_ORACLE,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for exp1b Monte Carlo runner."""
    p = argparse.ArgumentParser(
        description="exp1b: PMM_FRAC Monte Carlo experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to YAML config file",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (1 = serial, -1 = all cores)",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=500,
        help="Number of MC blocks per SNR cell (overrides config)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Output directory for results (overrides config)",
    )
    args = p.parse_args()

    summary_csv = run_experiment(
        args.config,
        jobs=args.jobs,
        blocks=args.blocks,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()
