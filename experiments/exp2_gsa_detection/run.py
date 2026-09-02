"""Experiment 2 — GSA-CUSUM detection on real SDR datasets.

Pipeline (Spec §5):
  1. Pick SNR bucket → split blocks into calibration (200) and test (300 H0 + 300 H1).
  2. Fit GSA coefficients k = F⁻¹·Y using calibration H0 / H1 samples.
  3. Calibrate per-block threshold via simulation (1−ε quantile of CUSUM peak under H0).
  4. Run detector on test H0 (compute FAR) and H1 (compute ADD, AUC).
  5. Sweep over detector configurations (energy, GSA-poly s=1..4, GSA-frac, GSA-PATP).
  6. PATP α-sweep at s=3 → find α* per SNR.

Output: results_dir/{exp2_gsa_results.json, exp2_per_detector.csv, exp2_patp_alpha_sweep.csv,
                       manifest.json, log.txt}.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from sdr_kunchenko import __version__ as pkg_version  # noqa: F401
from sdr_kunchenko.data_loader import SDRDataset, get_data_root
from sdr_kunchenko.gsa import (
    EnergyDetector,
    GSACUSUMDetector,
    fit_gsa_coefficients,
    lambda_score,
    make_basis,
    simulation_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


# ----------------------------------------------------------------- helpers


@dataclass
class DetectorSpec:
    name: str
    kind: str        # 'energy' | 'gsa'
    basis: str | None = None  # 'poly' | 'frac' | 'patp'
    s: int = 1
    alpha: float = 0.5


def _bucket_indices(snr: np.ndarray, target: float, tol: float) -> np.ndarray:
    return np.where(np.abs(snr - target) <= tol)[0]


def _split_indices(idxs: np.ndarray, n_calib: int, n_test: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint calibration / test splits. Returns (calib, test)."""
    rng.shuffle(idxs)
    n_calib = min(n_calib, len(idxs))
    n_test = min(n_test, len(idxs) - n_calib)
    return idxs[:n_calib], idxs[n_calib:n_calib + n_test]


def _calibration_samples(arr, idxs: np.ndarray) -> np.ndarray:
    """Concatenate idxs blocks of `arr` into one flat sample stream."""
    return np.concatenate([np.asarray(arr[i]).astype(np.float64) for i in idxs])


def _block_stack(arr, idxs: np.ndarray) -> np.ndarray:
    """Stack idxs blocks as (M, N) array for per-block runs."""
    return np.stack([np.asarray(arr[i]).astype(np.float64) for i in idxs])


# --------------------------------------------------- detector evaluation


def evaluate_detector_on_blocks(detector_obj,
                                  h0_blocks: np.ndarray,
                                  h1_blocks: np.ndarray,
                                  threshold: float) -> dict:
    """Run detector on H0 and H1 block sets. Compute FAR, ADD, AUC, MTBF.

    Detector interface: any object with `feed(x)` returning {fired_at, max_S, alarm}.
    Alarm is redefined here as `max_S > threshold`.
    """
    # H0
    M0 = len(h0_blocks)
    far_count = 0
    h0_max = np.empty(M0)
    for m in range(M0):
        out = detector_obj.feed(h0_blocks[m])
        h0_max[m] = out["max_S"]
        if out["max_S"] > threshold:
            far_count += 1
    far = far_count / M0

    # H1
    M1 = len(h1_blocks)
    add_list = []
    detected = 0
    h1_max = np.empty(M1)
    for m in range(M1):
        out = detector_obj.feed(h1_blocks[m])
        h1_max[m] = out["max_S"]
        # First sample whose CUSUM exceeds threshold = detection delay.
        history = np.asarray(out["history"])
        idx = np.where(history > threshold)[0]
        if len(idx) > 0:
            add_list.append(int(idx[0]))
            detected += 1
    add_mean = float(np.mean(add_list)) if add_list else float("inf")
    pdet = detected / M1

    # AUC (block-level binary classification: max_S as score, label = 0/1).
    scores = np.concatenate([h0_max, h1_max])
    labels = np.concatenate([np.zeros(M0), np.ones(M1)])
    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = float("nan")

    # MTBF = N / (FAR per-sample). Per-block FAR: avg run length under H0
    # = N_block / mean(false alarms per block).
    mtbf = (h0_blocks.shape[1] / max(far_count / M0, 1e-9)) if far_count > 0 else float("inf")

    return {
        "FAR": float(far),
        "ADD": float(add_mean),
        "P_detect": float(pdet),
        "AUC": auc,
        "MTBF_samples": float(mtbf),
        "h0_max_q": [float(np.quantile(h0_max, q)) for q in (0.5, 0.9, 0.99)],
        "h1_max_q": [float(np.quantile(h1_max, q)) for q in (0.1, 0.5, 0.9)],
    }


# ----------------------------------------------------------- per-SNR run


def run_snr_bucket(ds: SDRDataset, snr_db: float, *, cfg: dict,
                    rng: np.random.Generator) -> dict:
    """Full pipeline for one SNR bucket. Returns dict with all detector results."""
    tol = cfg["monte_carlo"]["snr_bucket_tol"]
    n_calib = cfg["monte_carlo"]["N_calibration_blocks"]
    n_test = cfg["monte_carlo"]["N_test_blocks"]
    eps = cfg["monte_carlo"]["epsilon"]

    snr = ds.snr
    candidates = _bucket_indices(snr, snr_db, tol)
    if len(candidates) < n_calib + n_test:
        n_test = max(0, len(candidates) - n_calib)
    if len(candidates) < n_calib + 30:
        return {"snr_db": snr_db, "skipped": True, "reason": "insufficient blocks"}
    calib_idxs, test_idxs = _split_indices(candidates.copy(), n_calib, n_test, rng)
    # Take a fresh independent split for H1 test (signals at same idxs but disjoint
    # from calib). Re-shuffle for an independent H1 set.
    h1_idxs = test_idxs
    h0_idxs = test_idxs  # we use noise-only blocks for H0 stream
    # Split calibration into FIT (for coefficients) and CAL (for threshold sim)
    # — avoids overfitting threshold to fit-set quirks, which would underestimate
    # FAR on independent test blocks.
    n_fit = len(calib_idxs) // 2
    fit_idxs = calib_idxs[:n_fit]
    cal_idxs = calib_idxs[n_fit:]
    h0_fit_flat = _calibration_samples(ds.noise_only, fit_idxs)
    h1_fit_flat = _calibration_samples(ds.noisy, fit_idxs)
    h0_calib_blocks = _block_stack(ds.noise_only, cal_idxs)
    h0_test_blocks = _block_stack(ds.noise_only, h0_idxs)
    h1_test_blocks = _block_stack(ds.noisy, h1_idxs)
    # Aliases used downstream (kept for minimal diff).
    h0_calib_flat = h0_fit_flat
    h1_calib_flat = h1_fit_flat

    out: dict = {
        "snr_db": float(snr_db),
        "n_calib": int(len(calib_idxs)),
        "n_test_h0": int(len(h0_idxs)),
        "n_test_h1": int(len(h1_idxs)),
        "detectors": {},
    }

    # ENERGY baseline.
    sigma2_h0 = float(np.var(h0_calib_flat))
    # threshold = simulated max-S for energy detector under H0.
    energy = EnergyDetector(threshold=0.0, mean_under_h0=sigma2_h0)
    e_peaks = np.array([energy.feed(b)["max_S"] for b in h0_calib_blocks])
    e_h = float(np.quantile(e_peaks, 1.0 - eps))
    res_energy = evaluate_detector_on_blocks(
        EnergyDetector(threshold=0.0, mean_under_h0=sigma2_h0),
        h0_test_blocks, h1_test_blocks, threshold=e_h,
    )
    res_energy["threshold"] = e_h
    res_energy["J"] = float("nan")
    out["detectors"]["energy"] = res_energy

    # GSA detectors.
    for det_cfg in cfg["detectors"]:
        if det_cfg["kind"] != "gsa":
            continue
        basis_kind = det_cfg["basis"]
        s = det_cfg["s"]
        basis = make_basis(basis_kind, s)
        fit = fit_gsa_coefficients(basis, h0_calib_flat, h1_calib_flat)
        sim = simulation_threshold(basis, fit["k"], h0_calib_blocks, epsilon=eps)
        h_thresh = sim["h"]
        det = GSACUSUMDetector(basis, fit["k"], threshold=0.0)
        res = evaluate_detector_on_blocks(
            det, h0_test_blocks, h1_test_blocks, threshold=h_thresh,
        )
        res["threshold"] = h_thresh
        res["J"] = fit["J"]
        res["cond_F"] = fit["cond_F"]
        res["k"] = fit["k"].tolist()
        out["detectors"][det_cfg["name"]] = res

    # PATP α-sweep at s = patp_sweep.s.
    patp_s = cfg["patp_sweep"]["s"]
    alpha_grid = cfg["patp_sweep"]["alpha_grid"]
    out["patp_sweep"] = []
    for alpha in alpha_grid:
        basis = make_basis("patp", patp_s, alpha=alpha)
        fit = fit_gsa_coefficients(basis, h0_calib_flat, h1_calib_flat)
        sim = simulation_threshold(basis, fit["k"], h0_calib_blocks, epsilon=eps)
        h_thresh = sim["h"]
        det = GSACUSUMDetector(basis, fit["k"], threshold=0.0)
        res = evaluate_detector_on_blocks(det, h0_test_blocks, h1_test_blocks,
                                            threshold=h_thresh)
        res["alpha"] = alpha
        res["J"] = fit["J"]
        res["cond_F"] = fit["cond_F"]
        res["threshold"] = h_thresh
        out["patp_sweep"].append(res)

    return out


# ------------------------------------------------------------- main driver


def make_manifest(cfg: dict, ds: SDRDataset, n_snr: int, elapsed: float,
                   dataset_alias: str) -> dict:
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        pass
    return {
        "experiment_id": cfg["experiment_id"],
        "git_sha": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dataset_alias": dataset_alias,
        "dataset": {"name": ds.name, "n_train": ds.n_train,
                     "block_size": ds.block_size, "config": ds.config},
        "config": cfg,
        "n_snr_buckets": n_snr,
        "elapsed_sec": elapsed,
    }


def main(cfg_path: Path, *, dataset_alias: str = "primary") -> Path:
    cfg = yaml.safe_load(cfg_path.read_text())
    out_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    if dataset_alias != "primary":
        out_dir = out_dir.with_name(out_dir.name + "_" + dataset_alias)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = PROJECT_ROOT / cfg["output"]["figures_dir"]
    if dataset_alias != "primary":
        fig_dir = fig_dir.with_name(fig_dir.name + "_" + dataset_alias)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.txt"
    logging.basicConfig(filename=log_path, filemode="w",
                         level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    rng = np.random.default_rng(cfg["seed"])

    data_root = get_data_root()
    ds_name = cfg["dataset"][dataset_alias]
    ds = SDRDataset.open(ds_name, root=data_root)
    logging.info("Dataset (%s): %s (n=%d)", dataset_alias, ds.name, ds.n_train)

    snr_grid = cfg["monte_carlo"]["snr_grid"]
    t0 = time.time()
    snr_results = []
    for snr_db in tqdm(snr_grid, desc=f"SNR sweep ({dataset_alias})"):
        try:
            out = run_snr_bucket(ds, snr_db, cfg=cfg, rng=rng)
            snr_results.append(out)
        except Exception as e:
            logging.exception("Failed at SNR=%s: %s", snr_db, e)
            snr_results.append({"snr_db": snr_db, "error": str(e)})
    elapsed = time.time() - t0
    logging.info("MC done in %.1fs over %d SNR buckets", elapsed, len(snr_results))

    # Save raw JSON
    out_json = {"snr_results": snr_results,
                 "manifest": make_manifest(cfg, ds, len(snr_results), elapsed, dataset_alias)}
    (out_dir / cfg["output"]["results_json"]).write_text(
        json.dumps(out_json, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    )
    (out_dir / cfg["output"]["manifest"]).write_text(
        json.dumps(out_json["manifest"], indent=2, default=str)
    )

    # Flatten per-detector results to CSV.
    rows = []
    for snr_block in snr_results:
        if snr_block.get("skipped") or snr_block.get("error"):
            continue
        snr_db = snr_block["snr_db"]
        for name, r in snr_block["detectors"].items():
            row = {"snr_db": snr_db, "detector": name}
            row.update({k: v for k, v in r.items() if not isinstance(v, list)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / cfg["output"]["per_detector_csv"], index=False)

    # PATP α-sweep CSV.
    rows = []
    for snr_block in snr_results:
        if snr_block.get("skipped") or snr_block.get("error"):
            continue
        snr_db = snr_block["snr_db"]
        for r in snr_block.get("patp_sweep", []):
            row = {"snr_db": snr_db, "alpha": r["alpha"]}
            row.update({k: v for k, v in r.items() if k != "alpha" and not isinstance(v, list)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / cfg["output"]["alpha_sweep_csv"], index=False)
    return out_dir / cfg["output"]["results_json"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--dataset", choices=["primary", "realnoise"], default="primary")
    args = p.parse_args()
    json_path = main(args.config, dataset_alias=args.dataset)
    print(f"\nResults: {json_path}")
