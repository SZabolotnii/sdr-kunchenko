"""Phase 5 — Cross-cutting α-mapping driver (Spec §8)."""
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

from sdr_kunchenko.patp.cross_domain import (
    aggregate_exp4_to_snr,
    build_unified_table,
    check_monotonicity,
    compute_pareto,
    fit_alpha_mapping,
    load_exp2_alphas,
    load_exp3_alphas,
    load_exp4_blocks,
    transferability_test,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def make_manifest(cfg: dict, elapsed: float, n_unified_rows: int) -> dict:
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
        "n_unified_rows": n_unified_rows,
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

    t0 = time.time()

    # ---- Ingest layer ----
    inputs = cfg["inputs"]
    aliases = cfg["dataset_aliases"]
    e2_pri = load_exp2_alphas(PROJECT_ROOT / inputs["exp2_primary_csv"],
                                dataset_name=aliases["exp2_primary"])
    e2_real = load_exp2_alphas(PROJECT_ROOT / inputs["exp2_realnoise_csv"],
                                 dataset_name=aliases["exp2_realnoise"])
    e3 = load_exp3_alphas(PROJECT_ROOT / inputs["exp3_csv"])
    e4_blocks = load_exp4_blocks(PROJECT_ROOT / inputs["exp4_json"])
    e4_agg = aggregate_exp4_to_snr(e4_blocks)
    unified = build_unified_table(e2_pri, e2_real, e3, e4_agg)
    unified_csv = out_dir / cfg["output"]["unified_csv"]
    unified.to_csv(unified_csv, index=False)
    logging.info("Unified table: %d rows", len(unified))
    logging.info("By task: %s", unified["task_type"].value_counts().to_dict())

    # ---- Regression ----
    fit = fit_alpha_mapping(
        unified,
        include_task_dummies=cfg["regression"]["include_task_dummies"],
        ridge_alpha=cfg["regression"]["ridge_alpha"],
    )
    logging.info("Mapping R² = %.3f", fit["r2"])
    mapping_json = out_dir / cfg["output"]["mapping_json"]
    coefs = fit["model"].coef_.tolist()
    mapping_json.write_text(json.dumps({
        "feature_names": fit["feature_names"],
        "coefficients": coefs,
        "intercept": float(fit["model"].intercept_),
        "r2": fit["r2"],
        "residuals_per_task": fit["residuals_per_task"],
        "include_task_dummies": fit["include_task_dummies"],
    }, indent=2, default=str))

    # ---- Monotonicity ----
    mono = check_monotonicity(
        fit["predict"],
        gamma4_grid=np.array(cfg["monotonicity"]["gamma4_grid"]),
        gamma3=cfg["monotonicity"]["fixed_gamma3"],
        snr_db=cfg["monotonicity"]["fixed_snr_db"],
        task_type=cfg["monotonicity"]["task_type"],
    )
    logging.info("Monotonicity check: monotonic=%s, sign_changes=%d",
                 mono["monotonic"], mono["n_sign_changes"])

    # ---- Transferability ----
    trans = transferability_test(
        unified,
        exp2_csv=PROJECT_ROOT / inputs["exp2_realnoise_csv"]
            if cfg["transferability"]["exp2_dataset_for_test"] == aliases["exp2_realnoise"]
            else PROJECT_ROOT / inputs["exp2_primary_csv"],
        exp2_dataset_name=cfg["transferability"]["exp2_dataset_for_test"],
        train_on=cfg["transferability"]["train_on"],
        test_on=cfg["transferability"]["test_on"],
        metric=cfg["transferability"]["metric"],
    )
    if "per_snr" in trans:
        logging.info("Transferability mean ratio = %.3f, passes ≥0.90: %s",
                     trans["mean_ratio"], trans["passes_90pct"])

    # ---- Pareto ----
    pareto_df = compute_pareto(
        unified,
        axes=tuple(cfg["pareto"]["axes"]),
    )
    pareto_csv = out_dir / "phase5_pareto.csv"
    pareto_df.to_csv(pareto_csv, index=False)
    n_pareto_per_task = (pareto_df[pareto_df["pareto_optimal"]]
                          .groupby("task_type").size().to_dict())
    logging.info("Pareto-optimal points per task: %s", n_pareto_per_task)

    # ---- Verification JSON ----
    elapsed = time.time() - t0
    verification = {
        "monotonicity": mono,
        "transferability": ({k: v for k, v in trans.items() if k != "per_snr"}
                              | {"per_snr_records":
                                  trans["per_snr"].to_dict("records")
                                  if "per_snr" in trans else []}),
        "pareto": {
            "n_optimal_per_task": n_pareto_per_task,
            "non_trivial": all(v >= 1 for v in n_pareto_per_task.values()),
        },
        "regression": {
            "r2": fit["r2"],
            "n_features": len(fit["feature_names"]),
            "feature_names": fit["feature_names"],
            "residuals_per_task": fit["residuals_per_task"],
        },
    }
    verif_json = out_dir / cfg["output"]["verification_json"]
    verif_json.write_text(json.dumps(verification, indent=2, default=str))

    (out_dir / cfg["output"]["manifest"]).write_text(
        json.dumps(make_manifest(cfg, elapsed, len(unified)), indent=2, default=str))

    logging.info("Phase 5 done in %.1fs", elapsed)
    return unified_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    out = main(args.config)
    print(f"\nUnified table: {out}")
