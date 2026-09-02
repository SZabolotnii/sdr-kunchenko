"""Reviewer-response analyses for Exp 3 DSGE noise-type classification.

This script complements the original historical run.py with checks requested by
the reviewer: block-level paired bootstrap, s/lambda sensitivity, augmented PATP
with a linear term, reduced block sizes, and feature-extraction timing.
"""
from __future__ import annotations

import argparse
import json
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
from run import collect_class_blocks, make_clf, spectral_features
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sdr_kunchenko.data_loader import get_data_root
from sdr_kunchenko.dsge import fit_class_coefficients, msed_features_batch
from sdr_kunchenko.gsa import make_basis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"
DEFAULT_OUT = Path(__file__).parent / "results_review_response"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                y_test: np.ndarray, *, kind: str) -> dict[str, Any]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    clf = make_clf(kind)
    clf.fit(x_train_scaled, y_train)
    pred = clf.predict(x_test_scaled)
    proba = clf.predict_proba(x_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
    out: dict[str, Any] = {
        "pred": pred,
        "proba": proba,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }
    if proba is not None:
        try:
            out["auc"] = float(roc_auc_score(y_test, proba))
            out["log_loss"] = float(log_loss(y_test, proba))
        except Exception:
            out["auc"] = float("nan")
    return out


def load_bucket(cfg: dict, snr_db: float, rng: np.random.Generator,
                *, block_size: int | None = None) -> dict[str, Any]:
    root = get_data_root()
    n_train = cfg["monte_carlo"]["N_train_per_class"]
    n_test = cfg["monte_carlo"]["N_test_per_class"]
    snr_tol = cfg["monte_carlo"]["snr_bucket_tol"]
    train_blocks_per_class = []
    test_blocks_per_class = []
    for class_cfg in cfg["dataset"]["classes"]:
        train, test = collect_class_blocks(class_cfg, root, snr_db, snr_tol, n_train, n_test, rng)
        if block_size is not None:
            train = [np.asarray(x[:block_size], dtype=np.float64) for x in train]
            test = [np.asarray(x[:block_size], dtype=np.float64) for x in test]
        train_blocks_per_class.append(train)
        test_blocks_per_class.append(test)

    x_test_blocks = []
    y_test = []
    for cls_id, test in enumerate(test_blocks_per_class):
        x_test_blocks.extend(test)
        y_test.extend([cls_id] * len(test))
    y_train = np.concatenate([
        np.full(len(train), cls_id) for cls_id, train in enumerate(train_blocks_per_class)
    ])
    return {
        "train_blocks_per_class": train_blocks_per_class,
        "test_blocks": x_test_blocks,
        "y_train": y_train,
        "y_test": np.asarray(y_test),
        "spectral_train": np.vstack([spectral_features(train) for train in train_blocks_per_class]),
        "spectral_test": spectral_features(x_test_blocks),
    }


def dsge_features_for_basis(
    bucket: dict[str, Any], *, basis_kind: str, s: int,
    ridge: float, alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = {"alpha": alpha} if alpha is not None else {}
    basis = make_basis(basis_kind, s, **kwargs)
    coeffs = [
        fit_class_coefficients(basis, train, ridge=ridge)
        for train in bucket["train_blocks_per_class"]
    ]
    train_parts = [
        msed_features_batch(basis, coeffs, train)
        for train in bucket["train_blocks_per_class"]
    ]
    return np.vstack(train_parts), msed_features_batch(basis, coeffs, bucket["test_blocks"])


def paired_block_bootstrap(deltas: np.ndarray, *, seed: int, n_boot: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(deltas)
    for i in range(n_boot):
        boot[i] = rng.choice(deltas, size=n, replace=True).mean()
    return {
        "delta": float(deltas.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_value_one_sided": float((boot <= 0).mean()),
        "n_blocks": int(n),
        "n_boot": int(n_boot),
    }


def run_block_bootstrap(cfg: dict, out_dir: Path, *, n_boot: int) -> dict[str, float]:
    rng = np.random.default_rng(cfg["seed"])
    ridge = float(cfg.get("regularization", {}).get("ridge", 1e-4))
    rows = []
    for snr_db in cfg["monte_carlo"]["snr_grid"]:
        bucket = load_bucket(cfg, snr_db, rng)
        spec = fit_predict(
            bucket["spectral_train"], bucket["y_train"],
            bucket["spectral_test"], bucket["y_test"], kind="random_forest"
        )
        xd_train, xd_test = dsge_features_for_basis(
            bucket, basis_kind="poly", s=4, ridge=ridge
        )
        hybrid = fit_predict(
            np.hstack([xd_train, bucket["spectral_train"]]), bucket["y_train"],
            np.hstack([xd_test, bucket["spectral_test"]]), bucket["y_test"],
            kind="random_forest",
        )
        for idx, y in enumerate(bucket["y_test"]):
            rows.append({
                "snr_db": snr_db,
                "block_index": idx,
                "y_true": int(y),
                "spectral_pred": int(spec["pred"][idx]),
                "hybrid_pred": int(hybrid["pred"][idx]),
                "spectral_correct": int(spec["pred"][idx] == y),
                "hybrid_correct": int(hybrid["pred"][idx] == y),
            })
    pred_df = pd.DataFrame(rows)
    pred_df["paired_delta"] = pred_df["hybrid_correct"] - pred_df["spectral_correct"]
    pred_df.to_csv(out_dir / "block_predictions_hybrid_vs_spectral.csv", index=False)
    return paired_block_bootstrap(
        pred_df["paired_delta"].to_numpy(dtype=float), seed=0, n_boot=n_boot
    )


def run_sensitivity(cfg: dict, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for ridge in (1e-6, 1e-4, 1e-2):
        for s in (2, 3, 4, 5):
            for snr_db in cfg["monte_carlo"]["snr_grid"]:
                bucket = load_bucket(cfg, snr_db, rng)
                xd_train, xd_test = dsge_features_for_basis(
                    bucket, basis_kind="poly", s=s, ridge=ridge
                )
                dsge = fit_predict(xd_train, bucket["y_train"], xd_test, bucket["y_test"],
                                   kind="random_forest")
                hybrid = fit_predict(
                    np.hstack([xd_train, bucket["spectral_train"]]), bucket["y_train"],
                    np.hstack([xd_test, bucket["spectral_test"]]), bucket["y_test"],
                    kind="random_forest",
                )
                rows.append({
                    "snr_db": snr_db,
                    "basis": "poly",
                    "s": s,
                    "ridge": ridge,
                    "dsge_accuracy": dsge["accuracy"],
                    "hybrid_accuracy": hybrid["accuracy"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "s_lambda_sensitivity.csv", index=False)
    return df


def run_augmented_patp(cfg: dict, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    ridge = float(cfg.get("regularization", {}).get("ridge", 1e-4))
    alpha_grid = cfg["patp_sweep"]["alpha_grid"]
    rows = []
    for snr_db in cfg["monte_carlo"]["snr_grid"]:
        bucket = load_bucket(cfg, snr_db, rng)
        for alpha in alpha_grid:
            xd_train, xd_test = dsge_features_for_basis(
                bucket, basis_kind="patp_aug", s=3, alpha=float(alpha), ridge=ridge
            )
            res = fit_predict(xd_train, bucket["y_train"], xd_test, bucket["y_test"],
                              kind="logistic")
            rows.append({
                "snr_db": snr_db,
                "alpha": float(alpha),
                "accuracy": res["accuracy"],
                "auc": res.get("auc", float("nan")),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "augmented_patp_alpha_sweep.csv", index=False)
    return df


def run_block_sizes(cfg: dict, out_dir: Path) -> pd.DataFrame:
    ridge = float(cfg.get("regularization", {}).get("ridge", 1e-4))
    rows = []
    for block_size in (128, 256, 512, 1024):
        rng = np.random.default_rng(cfg["seed"])
        for snr_db in cfg["monte_carlo"]["snr_grid"]:
            bucket = load_bucket(cfg, snr_db, rng, block_size=block_size)
            spec = fit_predict(
                bucket["spectral_train"], bucket["y_train"],
                bucket["spectral_test"], bucket["y_test"], kind="random_forest"
            )
            xd_train, xd_test = dsge_features_for_basis(
                bucket, basis_kind="poly", s=4, ridge=ridge
            )
            dsge = fit_predict(xd_train, bucket["y_train"], xd_test, bucket["y_test"],
                               kind="random_forest")
            hybrid = fit_predict(
                np.hstack([xd_train, bucket["spectral_train"]]), bucket["y_train"],
                np.hstack([xd_test, bucket["spectral_test"]]), bucket["y_test"],
                kind="random_forest",
            )
            rows.append({
                "block_size": block_size,
                "snr_db": snr_db,
                "spectral_accuracy": spec["accuracy"],
                "dsge_accuracy": dsge["accuracy"],
                "hybrid_accuracy": hybrid["accuracy"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "block_size_sensitivity.csv", index=False)
    return df


def run_timing(cfg: dict, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    ridge = float(cfg.get("regularization", {}).get("ridge", 1e-4))
    snr_db = cfg["monte_carlo"]["snr_grid"][0]
    bucket = load_bucket(cfg, snr_db, rng)
    basis = make_basis("poly", 4)

    t0 = time.perf_counter()
    _ = spectral_features(bucket["test_blocks"])
    spectral_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    coeffs = [
        fit_class_coefficients(basis, train, ridge=ridge)
        for train in bucket["train_blocks_per_class"]
    ]
    fit_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = msed_features_batch(basis, coeffs, bucket["test_blocks"])
    dsge_transform_sec = time.perf_counter() - t0

    n_test = len(bucket["test_blocks"])
    df = pd.DataFrame([
        {
            "snr_db": snr_db,
            "n_test_blocks": n_test,
            "spectral_sec_per_block": spectral_sec / n_test,
            "dsge_fit_sec": fit_sec,
            "dsge_transform_sec_per_block": dsge_transform_sec / n_test,
        }
    ])
    df.to_csv(out_dir / "runtime_summary.csv", index=False)
    return df


def write_markdown_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Exp 3 Reviewer-Response Analyses",
        "",
        f"Generated: {summary['manifest']['timestamp_utc']}",
        f"Git SHA: `{summary['manifest']['git_sha']}`",
        "",
        "## Block-level paired bootstrap",
        "",
        json.dumps(summary["block_bootstrap"], indent=2),
        "",
        "## Generated tables",
        "",
        "- `block_predictions_hybrid_vs_spectral.csv`",
        "- `s_lambda_sensitivity.csv`",
        "- `augmented_patp_alpha_sweep.csv`",
        "- `block_size_sensitivity.csv`",
        "- `runtime_summary.csv`",
    ]
    (out_dir / "review_response_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    block_bootstrap = run_block_bootstrap(cfg, out_dir, n_boot=args.n_boot)
    sensitivity = run_sensitivity(cfg, out_dir)
    augmented_patp = run_augmented_patp(cfg, out_dir)
    block_sizes = run_block_sizes(cfg, out_dir)
    timing = run_timing(cfg, out_dir)
    elapsed = time.time() - t0

    summary: dict[str, Any] = {
        "manifest": {
            "experiment_id": f"{cfg['experiment_id']}_review_response",
            "git_sha": git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "elapsed_sec": elapsed,
            "config": cfg,
        },
        "block_bootstrap": block_bootstrap,
        "s_lambda_mean": sensitivity.groupby(["s", "ridge"])[
            ["dsge_accuracy", "hybrid_accuracy"]
        ].mean().reset_index().to_dict(orient="records"),
        "augmented_patp_best": augmented_patp.loc[
            augmented_patp.groupby("snr_db")["accuracy"].idxmax()
        ].to_dict(orient="records"),
        "block_size_mean": block_sizes.groupby("block_size")[
            ["spectral_accuracy", "dsge_accuracy", "hybrid_accuracy"]
        ].mean().reset_index().to_dict(orient="records"),
        "runtime": timing.to_dict(orient="records"),
    }
    (out_dir / "review_response_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_summary(out_dir, summary)
    print(out_dir / "review_response_report.md")


if __name__ == "__main__":
    main()
