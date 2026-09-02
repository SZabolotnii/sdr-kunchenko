"""Experiment 4 — PATP-adaptive denoiser.

Pipeline (Spec §7.3):
    1. Train phase (per SNR bucket):
       a. Pull N_train clean+noisy paired blocks.
       b. Learn signal subspace Ψ via SVD on clean[i].
       c. Calibrate α* = f(γ̂₄) on validation pairs.
    2. Test phase:
       a. For N_test fresh blocks, run all denoising methods.
       b. Metrics: RMSE, PSNR, log-MSE, EVM (modulation-aware).
    3. Generalization probe:
       Apply mapping fitted on FPV to deep_space test blocks.

Output: results_csv with one row per (dataset, snr_db, method, sample_id).
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sdr_kunchenko.data_loader import SDRDataset, get_data_root
from sdr_kunchenko.denoiser import (
    adaptive_denoise,
    calibrate_alpha_mapping,
    fixed_patp_denoise,
    learn_subspace,
    median_denoise,
    projection_denoise,
    wiener_denoise,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


# ----------------------------------------------------------- helpers


def block_pair(ds: SDRDataset, idxs):
    """(clean, noisy) per index."""
    clean = np.stack([np.asarray(ds.clean[i]).astype(np.float64) for i in idxs])
    noisy = np.stack([np.asarray(ds.noisy[i]).astype(np.float64) for i in idxs])
    return clean, noisy


def metrics(s_hat: np.ndarray, s_true: np.ndarray) -> dict:
    err = s_hat - s_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    s_power = float(np.mean(s_true ** 2))
    if s_power > 0 and mse > 0:
        psnr = 10.0 * np.log10(np.max(s_true ** 2) / mse) if np.max(s_true ** 2) > 0 else float("nan")
        snr_out = 10.0 * np.log10(s_power / mse)
    else:
        psnr = float("nan")
        snr_out = float("nan")
    return {"mse": mse, "rmse": rmse, "psnr": psnr, "snr_out_db": snr_out}


# --------------------------------------------------- per-SNR pipeline


def run_snr(ds: SDRDataset, snr_db: float, *, cfg: dict,
             rng: np.random.Generator,
             alpha_grid: list[float],
             dataset_alias: str,
             pretrained_mapping: dict | None = None,
             pretrained_psi: np.ndarray | None = None) -> tuple[list[dict], dict | None, np.ndarray]:
    """Run all methods on one SNR bucket. Returns (rows, mapping, Ψ)."""
    snr = ds.snr
    tol = cfg["monte_carlo"]["snr_bucket_tol"]
    n_train = cfg["monte_carlo"]["N_train"]
    n_test = cfg["monte_carlo"]["N_test"]
    candidates = np.where(np.abs(snr - snr_db) <= tol)[0]
    rng.shuffle(candidates)
    if len(candidates) < n_train + n_test:
        n_train = min(n_train, len(candidates) // 2)
        n_test = min(n_test, len(candidates) - n_train)
    train_idxs = candidates[:n_train]
    test_idxs = candidates[n_train:n_train + n_test]

    clean_train, noisy_train = block_pair(ds, train_idxs)
    clean_test, noisy_test = block_pair(ds, test_idxs)

    # Subspace
    if pretrained_psi is not None:
        Psi = pretrained_psi
    else:
        Psi = learn_subspace(clean_train, rank=cfg["subspace"]["rank"])

    # Calibrate α mapping (if not pretrained)
    if pretrained_mapping is not None:
        mapping = pretrained_mapping
    else:
        # Use first half of train for mapping; reduce N for compute speed.
        n_calib = min(300, len(noisy_train))
        mapping = calibrate_alpha_mapping(
            noisy_train[:n_calib], clean_train[:n_calib], Psi,
            alpha_grid=alpha_grid,
            poly_degree=cfg["poly_degree_for_mapping"],
            init_method=cfg["init_method"],
        )

    rows = []
    for i, (s_true, y) in enumerate(zip(clean_test, noisy_test)):
        # noisy reference (no denoising)
        m = metrics(y, s_true)
        rows.append({"snr_db": snr_db, "method": "noisy", "sample_id": int(test_idxs[i]),
                     "dataset_alias": dataset_alias, **m, "alpha_used": np.nan})
        # median
        s_hat = median_denoise(y, window=5)
        m = metrics(s_hat, s_true)
        rows.append({"snr_db": snr_db, "method": "median_w5", "sample_id": int(test_idxs[i]),
                     "dataset_alias": dataset_alias, **m, "alpha_used": np.nan})
        # wiener
        s_hat = wiener_denoise(y)
        m = metrics(s_hat, s_true)
        rows.append({"snr_db": snr_db, "method": "wiener", "sample_id": int(test_idxs[i]),
                     "dataset_alias": dataset_alias, **m, "alpha_used": np.nan})
        # projection (α=1 limit)
        s_hat = projection_denoise(y, Psi)
        m = metrics(s_hat, s_true)
        rows.append({"snr_db": snr_db, "method": "projection_K",
                     "sample_id": int(test_idxs[i]),
                     "dataset_alias": dataset_alias, **m, "alpha_used": 1.0})
        # fixed PATP α=0.5
        out = fixed_patp_denoise(y, Psi, alpha=0.5)
        m = metrics(out["s_hat"], s_true)
        rows.append({"snr_db": snr_db, "method": "fixed_patp_05",
                     "sample_id": int(test_idxs[i]),
                     "dataset_alias": dataset_alias, **m, "alpha_used": 0.5})
        # adaptive PATP
        out = adaptive_denoise(y, Psi, mapping, init_method=cfg["init_method"])
        m = metrics(out["s_hat"], s_true)
        rows.append({"snr_db": snr_db, "method": "adaptive_patp",
                     "sample_id": int(test_idxs[i]),
                     "dataset_alias": dataset_alias, **m,
                     "alpha_used": float(out["alpha_used"]),
                     "gamma4_est": float(out["gamma4_est"])})

    return rows, mapping, Psi


# -------------------------------------------------------------- main


def make_manifest(cfg: dict, n_runs: int, elapsed: float) -> dict:
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                            cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        pass
    return {
        "experiment_id": cfg["experiment_id"],
        "git_sha": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": cfg,
        "n_runs": n_runs,
        "elapsed_sec": elapsed,
    }


def main(cfg_path: Path) -> Path:
    cfg = yaml.safe_load(cfg_path.read_text())
    out_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = PROJECT_ROOT / cfg["output"]["figures_dir"]
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
    snr_grid = cfg["monte_carlo"]["snr_grid"]
    alpha_grid = cfg["alpha_grid"]
    all_rows: list[dict] = []
    mappings_by_dataset: dict[str, dict[float, dict]] = {}
    psi_by_dataset: dict[str, dict[float, np.ndarray]] = {}

    t0 = time.time()
    for alias in ("primary", "realnoise"):
        ds_name = cfg["dataset"][alias]
        try:
            ds = SDRDataset.open(ds_name, root=data_root)
        except Exception as e:
            logging.exception("Cannot open %s: %s", ds_name, e)
            continue
        logging.info("Dataset %s (%s): n=%d, block=%d", alias, ds.name, ds.n_train, ds.block_size)
        mappings_by_dataset[alias] = {}
        psi_by_dataset[alias] = {}
        for snr_db in tqdm(snr_grid, desc=f"{alias} SNR sweep"):
            try:
                rows, mapping, Psi = run_snr(ds, snr_db, cfg=cfg, rng=rng,
                                                alpha_grid=alpha_grid,
                                                dataset_alias=alias)
                all_rows.extend(rows)
                mappings_by_dataset[alias][float(snr_db)] = {
                    "poly_coefs": mapping["poly_coefs"].tolist(),
                    "correlation": float(mapping["correlation"]),
                    "training_gamma4": mapping["training_gamma4"].tolist(),
                    "training_alpha": mapping["training_alpha"].tolist(),
                }
                psi_by_dataset[alias][float(snr_db)] = Psi
            except Exception as e:
                logging.exception("SNR %s failed on %s: %s", snr_db, alias, e)

    # Generalization probe: apply primary mapping to deep_space.
    if "generalization" in cfg["dataset"]:
        ds_name = cfg["dataset"]["generalization"]
        try:
            ds_gen = SDRDataset.open(ds_name, root=data_root)
            logging.info("Generalization probe on %s", ds_gen.name)
            for snr_db in tqdm(snr_grid, desc="generalization SNR sweep"):
                if snr_db not in mappings_by_dataset.get("primary", {}):
                    continue
                mapping_dict = mappings_by_dataset["primary"][float(snr_db)]
                # Reconstruct a thin mapping object with predict()
                coefs = np.asarray(mapping_dict["poly_coefs"])
                poly = np.poly1d(coefs)
                mapping = {
                    "predict": lambda g4, p=poly: float(np.clip(p(g4), 0.0, 1.0)),
                    "training_gamma4": np.asarray(mapping_dict["training_gamma4"]),
                    "training_alpha": np.asarray(mapping_dict["training_alpha"]),
                    "poly_coefs": coefs,
                    "correlation": mapping_dict["correlation"],
                }
                Psi = psi_by_dataset["primary"][float(snr_db)]
                try:
                    rows, _, _ = run_snr(ds_gen, snr_db, cfg=cfg, rng=rng,
                                            alpha_grid=alpha_grid,
                                            dataset_alias="generalization",
                                            pretrained_mapping=mapping,
                                            pretrained_psi=Psi)
                    all_rows.extend(rows)
                except Exception as e:
                    logging.exception("Gen probe SNR %s failed: %s", snr_db, e)
        except Exception as e:
            logging.exception("Cannot load generalization dataset: %s", e)

    elapsed = time.time() - t0
    logging.info("Total time %.1fs", elapsed)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / cfg["output"]["results_csv"], index=False)
    logging.info("Wrote %s (%d rows)", cfg["output"]["results_csv"], len(df))

    (out_dir / cfg["output"]["alpha_mapping_json"]).write_text(
        json.dumps(mappings_by_dataset, indent=2, default=str))
    (out_dir / cfg["output"]["manifest"]).write_text(
        json.dumps(make_manifest(cfg, len(all_rows), elapsed), indent=2, default=str))

    return out_dir / cfg["output"]["results_csv"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    csv_path = main(args.config)
    print(f"\nResults: {csv_path}")
