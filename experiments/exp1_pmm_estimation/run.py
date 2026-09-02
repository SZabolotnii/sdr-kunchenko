"""Experiment 1A — PMM amplitude estimation on real SDR datasets.

Per Spec §4.2: model y[n] = A · s[n] + ξ[n], estimate A.
Run OLS, PMM2, PMM3, Auto-PMM on each block; compare to true A.
Compute analytic sandwich SE for coverage.

Output: results/exp1A_amplitude.csv (and manifest.json, log.txt).
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

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from sdr_kunchenko import __version__ as pkg_version  # noqa: F401 (informational)
from sdr_kunchenko.data_loader import SDRDataset, get_data_root
from sdr_kunchenko.pmm import oracle
from sdr_kunchenko.pmm.inference import (
    hac_se_amplitude,
    ols_se_amplitude,
    pmm2_hac_se_amplitude,
    pmm2_sandwich_se_amplitude,
    pmm3_hac_se_amplitude,
    pmm3_sandwich_se_amplitude,
)
from sdr_kunchenko.pmm.pooled import (
    pmm2_pooled_amplitude,
    pmm3_pooled_amplitude,
    pool_gamma_from_blocks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


# -------------------------------------------------------------- per-block work


def process_block(*, idx: int, snr_db: float, clean: np.ndarray,
                   noise: np.ndarray, A_true: float,
                   pool: dict | None = None) -> dict:
    """One MC trial: y = A_true·s + xi, run OLS/PMM2/PMM3/Auto + Pooled.

    `pool` (if provided): {gamma3_mean, gamma4_mean, gamma6_mean, kappa_mean}
    pre-computed from a calibration set of noise-only blocks at the same SNR.
    Used to drive PMM2/PMM3 with frozen weights ('Pool' columns).
    """
    s = clean
    xi = noise
    y = A_true * s + xi

    # OLS
    s2 = float(s @ s)
    A_ols = float((y @ s) / s2)
    eps_ols = y - A_ols * s
    se_ols = ols_se_amplitude(s, eps_ols)
    se_ols_hac = hac_se_amplitude(s, eps_ols)

    # PMM2 via oracle. Fallback to OLS on non-convergence (NR can diverge
    # on noisy γ̂₃, γ̂₄ at small N, producing huge outliers that pollute MSE).
    fit2 = oracle.lm_pmm2(y, s)
    pmm2_converged = bool(fit2.get("converged", False))
    if pmm2_converged:
        A_pmm2 = float(fit2["coef"][0])
        se_pmm2 = pmm2_sandwich_se_amplitude(
            s, fit2["residuals"],
            gamma3=fit2["gamma3"], gamma4=fit2["gamma4"],
        )
    else:
        A_pmm2 = A_ols  # safe fallback
        se_pmm2 = se_ols
    A_pmm2_raw = float(fit2["coef"][0])  # keep raw for diagnostics

    # PMM3 via oracle — same fallback strategy.
    try:
        fit3 = oracle.lm_pmm3(y, s)
        pmm3_converged = bool(fit3.get("converged", False))
        if pmm3_converged:
            A_pmm3 = float(fit3["coef"][0])
            se_pmm3 = pmm3_sandwich_se_amplitude(s, fit3["residuals"],
                                                    kappa=fit3.get("kappa", 0.0))
            se_pmm3_hac = pmm3_hac_se_amplitude(s, fit3["residuals"],
                                                   kappa=fit3.get("kappa", 0.0))
        else:
            A_pmm3 = A_ols
            se_pmm3 = se_ols
            se_pmm3_hac = se_ols_hac
        A_pmm3_raw = float(fit3["coef"][0])
        gamma6 = float(fit3.get("gamma6", float("nan")))
        g3_theory = float(fit3.get("g_coefficient", float("nan")))
        kappa = float(fit3.get("kappa", float("nan")))
    except Exception:
        A_pmm3 = A_ols
        A_pmm3_raw = float("nan")
        se_pmm3 = se_ols
        se_pmm3_hac = se_ols_hac
        gamma6 = float("nan")
        g3_theory = float("nan")
        kappa = float("nan")
        pmm3_converged = False

    # Auto dispatch on noise itself (we have it from the dataset).
    # Inherits convergence-fallback from underlying estimator.
    disp = oracle.pmm_dispatch(xi)
    method = disp["method"]
    if method == "PMM2":
        A_auto, se_auto = A_pmm2, se_pmm2
    elif method == "PMM3":
        A_auto, se_auto = A_pmm3, se_pmm3
    else:  # OLS
        A_auto, se_auto = A_ols, se_ols

    # Pooled-PMM2 / PMM3 (if calibration pool provided): frozen-γ̂ weights.
    A_pmm2_pool, A_pmm3_pool = float("nan"), float("nan")
    se_pmm2_pool, se_pmm3_pool = float("nan"), float("nan")
    if pool is not None:
        g3p, g4p = pool["gamma3_mean"], pool["gamma4_mean"]
        f2p = pmm2_pooled_amplitude(y, s, gamma3=g3p, gamma4=g4p)
        if f2p["converged"]:
            A_pmm2_pool = float(f2p["coef"][0])
            se_pmm2_pool = pmm2_sandwich_se_amplitude(
                s, f2p["residuals"], gamma3=g3p, gamma4=g4p)
        else:
            A_pmm2_pool = A_ols
            se_pmm2_pool = se_ols
        # PMM3 with frozen γ̂₄, γ̂₆ pool — κ is derived per-block from σ²(eps).
        # Avoids the κ-pooling instability (κ has dimensions of variance).
        f3p = pmm3_pooled_amplitude(
            y, s, gamma4_pool=pool["gamma4_mean"], gamma6_pool=pool["gamma6_mean"]
        )
        if f3p["converged"]:
            A_pmm3_pool = float(f3p["coef"][0])
            se_pmm3_pool = pmm3_sandwich_se_amplitude(
                s, f3p["residuals"], kappa=f3p.get("kappa_used", 0.0))
        else:
            A_pmm3_pool = A_ols
            se_pmm3_pool = se_ols

    # HAC-SE on the OLS-residuals path (cross-check coverage anomaly).
    se_pmm2_hac = pmm2_hac_se_amplitude(
        s, fit2["residuals"], gamma3=fit2["gamma3"], gamma4=fit2["gamma4"]
    ) if pmm2_converged else se_ols_hac

    return {
        "sample_id": idx,
        "snr_db": snr_db,
        "A_true": A_true,
        "gamma3_xi": float(disp["gamma3"]),
        "gamma4_xi": float(disp["gamma4"]),
        "gamma6_xi": float(disp.get("gamma6", float("nan"))),
        "g2_theory": float(disp["g2"]),
        "g3_theory": g3_theory,
        "method_auto": method,
        "A_OLS": A_ols,
        "A_PMM2": A_pmm2,            # fallback-corrected (= OLS if not converged)
        "A_PMM2_raw": A_pmm2_raw,    # raw NR output (for outlier diagnostics)
        "A_PMM3": A_pmm3,
        "A_PMM3_raw": A_pmm3_raw,
        "A_AUTO": A_auto,
        "A_PMM2_POOL": A_pmm2_pool,  # frozen-γ̂ pooled (Phase-1 cleanup)
        "A_PMM3_POOL": A_pmm3_pool,
        "SE_OLS": se_ols,
        "SE_OLS_HAC": se_ols_hac,
        "SE_PMM2": se_pmm2,
        "SE_PMM2_HAC": se_pmm2_hac,
        "SE_PMM3": se_pmm3,
        "SE_PMM3_HAC": se_pmm3_hac,
        "SE_AUTO": se_auto,
        "SE_PMM2_POOL": se_pmm2_pool,
        "SE_PMM3_POOL": se_pmm3_pool,
        "pmm2_converged": pmm2_converged,
        "pmm3_converged": pmm3_converged,
        "kappa_pmm3": kappa,
    }


# ---------------------------------------------------------------- MC driver


def build_mc_grid(cfg: dict, ds: SDRDataset, *, rng: np.random.Generator) -> list[dict]:
    """Build the list of MC trials.

    Spec §4.2: R repetitions per (SNR, K, A_true). Real datasets have block-level
    SNR (continuous in train), so we bucket blocks by nearest SNR grid point and
    sample at most n_blocks_per_cell per bucket.
    """
    snr = ds.snr
    grid = cfg["monte_carlo"]["snr_grid"]
    tol = cfg["monte_carlo"].get("snr_bucket_tol", 0.5)  # dB
    n_per_cell = cfg["monte_carlo"]["N_blocks_per_cell"]
    A_lo, A_hi = cfg["monte_carlo"]["A_true_range"]

    trials = []
    for snr_db in grid:
        mask = np.abs(snr - snr_db) <= tol
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            logging.warning("No blocks within ±%.1fdB of SNR=%d; skipping", tol, snr_db)
            continue
        n_take = min(n_per_cell, len(candidates))
        idxs = rng.choice(candidates, size=n_take, replace=False)
        for i in idxs:
            trials.append({
                "idx": int(i),
                "snr_db": float(snr_db),
                "A_true": float(rng.uniform(A_lo, A_hi)),
            })
    return trials


def build_calibration_pools(ds: SDRDataset, snr_grid: list[float],
                              snr_tol: float, n_calib: int,
                              rng: np.random.Generator) -> dict[float, dict]:
    """For each SNR bucket, pool γ̂₃, γ̂₄, γ̂₆, κ over `n_calib` noise blocks.

    Returns dict[snr] -> {gamma3_mean, gamma4_mean, gamma6_mean, kappa_mean,
                          gamma3_se, gamma4_se, n_blocks}.
    """
    snr = ds.snr
    noise_arr = ds.noise_only
    pools: dict[float, dict] = {}
    for snr_db in snr_grid:
        mask = np.abs(snr - snr_db) <= snr_tol
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            continue
        n_take = min(n_calib, len(candidates))
        idxs = rng.choice(candidates, size=n_take, replace=False)
        blocks = [np.asarray(noise_arr[i]).astype(np.float64) for i in idxs]
        p = pool_gamma_from_blocks(blocks)
        # Compute pooled κ for PMM3 from compute_moments_pmm3 averages.
        kappas = []
        for blk in blocks:
            try:
                m3 = oracle.compute_moments_pmm3(blk)
                kappas.append(m3.get("kappa", np.nan))
            except Exception:
                pass
        kappa_arr = np.array([k for k in kappas if np.isfinite(k)])
        p["kappa_mean"] = float(kappa_arr.mean()) if len(kappa_arr) else None
        p["snr_db"] = float(snr_db)
        pools[float(snr_db)] = p
    return pools


def run_trials_serial(trials: list[dict], ds: SDRDataset,
                        pools: dict[float, dict] | None = None) -> list[dict]:
    """Single-process loop. Used for sanity profile."""
    clean_arr = ds.clean
    noise_arr = ds.noise_only
    rows = []
    for t in tqdm(trials, desc="serial MC"):
        i = t["idx"]
        pool = pools.get(float(t["snr_db"])) if pools else None
        rows.append(process_block(
            idx=i,
            snr_db=t["snr_db"],
            clean=np.asarray(clean_arr[i]).astype(np.float64),
            noise=np.asarray(noise_arr[i]).astype(np.float64),
            A_true=t["A_true"],
            pool=pool,
        ))
    return rows


def _worker(trial: dict, dataset_name: str, dataset_root: str) -> dict:
    """joblib worker: each process opens its own SDRDataset (mmap) + R session."""
    ds = SDRDataset.open(dataset_name, root=Path(dataset_root))
    i = trial["idx"]
    return process_block(
        idx=i,
        snr_db=trial["snr_db"],
        clean=np.asarray(ds.clean[i]).astype(np.float64),
        noise=np.asarray(ds.noise_only[i]).astype(np.float64),
        A_true=trial["A_true"],
    )


def run_trials_parallel(trials: list[dict], dataset_name: str,
                          dataset_root: Path, n_jobs: int = -1) -> list[dict]:
    """Multi-process loop via joblib(loky). Each worker has its own R session."""
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_worker)(t, dataset_name, str(dataset_root))
        for t in tqdm(trials, desc=f"parallel MC (n_jobs={n_jobs})")
    )


# -------------------------------------------------------------- manifest


def make_manifest(cfg: dict, n_trials: int, elapsed_sec: float,
                   dataset: SDRDataset) -> dict:
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        pass
    return {
        "experiment_id": cfg["experiment_id"],
        "git_sha": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "estempmm_version": oracle.ESTEMPMM_VERSION.replace("\n", " ").strip(),
        "dataset": {
            "name": dataset.name,
            "root": str(dataset.root),
            "n_train": dataset.n_train,
            "block_size": dataset.block_size,
            "config": dataset.config,
        },
        "config": cfg,
        "n_trials_run": n_trials,
        "elapsed_sec": elapsed_sec,
        "data_root_env": str(get_data_root()),
    }


# -------------------------------------------------------------- main


def main(cfg_path: Path, *, n_jobs: int = -1, override_R: int | None = None,
          override_blocks: int | None = None,
          override_dataset: str | None = None) -> Path:
    """Run Exp 1A. Returns path to results CSV."""
    cfg = yaml.safe_load(cfg_path.read_text())
    if override_R is not None:
        cfg["monte_carlo"]["R"] = override_R
    if override_blocks is not None:
        cfg["monte_carlo"]["N_blocks_per_cell"] = override_blocks
    if override_dataset is not None:
        cfg["dataset"]["name"] = override_dataset

    out_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = PROJECT_ROOT / cfg["output"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.txt"
    logging.basicConfig(
        filename=log_path, filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    rng = np.random.default_rng(cfg["seed"])

    data_root = Path(cfg["dataset"]["root"])
    if "${KU_SDR_DATA_ROOT" in str(data_root):
        data_root = get_data_root()
    ds = SDRDataset.open(cfg["dataset"]["name"], root=data_root)
    logging.info("Dataset: %s (n=%d, block=%d)", ds.name, ds.n_train, ds.block_size)

    trials = build_mc_grid(cfg, ds, rng=rng)
    logging.info("MC trials: %d (across %d SNR bins)", len(trials),
                 len(cfg["monte_carlo"]["snr_grid"]))

    # Calibration pools (Phase-1 cleanup): pool γ̂ over disjoint noise blocks.
    n_calib = cfg["monte_carlo"].get("N_calibration_blocks", 200)
    snr_tol = cfg["monte_carlo"].get("snr_bucket_tol", 0.5)
    pools = build_calibration_pools(ds, cfg["monte_carlo"]["snr_grid"],
                                      snr_tol=snr_tol, n_calib=n_calib,
                                      rng=np.random.default_rng(cfg["seed"] + 1))
    logging.info("Calibration pools (n_calib=%d): %s", n_calib,
                 {k: f"γ₃={v['gamma3_mean']:+.3f}±{v['gamma3_se']:.3f}, "
                       f"γ₄={v['gamma4_mean']:+.3f}, κ={v.get('kappa_mean')}"
                  for k, v in pools.items()})

    t0 = time.time()
    if n_jobs in (None, 0, 1):
        rows = run_trials_serial(trials, ds, pools=pools)
    else:
        rows = run_trials_parallel(trials, ds.name, data_root, n_jobs=n_jobs)
    elapsed = time.time() - t0
    logging.info("MC done in %.2fs (%.3f s/trial)", elapsed,
                 elapsed / max(1, len(rows)))

    df = pd.DataFrame(rows)
    csv_path = out_dir / cfg["output"]["results_csv"]
    df.to_csv(csv_path, index=False)
    logging.info("Wrote %s (%d rows)", csv_path, len(df))

    manifest = make_manifest(cfg, len(rows), elapsed, ds)
    (out_dir / cfg["output"]["manifest"]).write_text(json.dumps(manifest, indent=2))

    return csv_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--jobs", type=int, default=-1, help="-1 = all cores, 1 = serial")
    p.add_argument("--R", type=int, default=None, help="override MC repetitions")
    p.add_argument("--blocks", type=int, default=None, help="override N_blocks_per_cell")
    p.add_argument("--dataset", type=str, default=None, help="override dataset name")
    args = p.parse_args()
    csv = main(args.config, n_jobs=args.jobs, override_R=args.R,
                override_blocks=args.blocks, override_dataset=args.dataset)
    print(f"\nResults: {csv}")
