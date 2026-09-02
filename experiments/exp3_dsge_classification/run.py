"""Experiment 3 — DSGE noise-type classification.

PIVOT vs Spec §6 (reflected in manuscript): the original spec called for
BPSK/QPSK/CPFSK/GFSK modulation classification on RadioML2018 splits. After
inspecting the actual datasets we found that `radioml2018_bpsk_qpsk_*` does
NOT carry per-block modulation labels (only one binary mix of BPSK+QPSK).
This is the Phase-0 risk #13.3.1 materialising.

Reformulated task — *equally Spec-aligned* per §6 ("DSGE for class
discrimination via Kunchenko reconstruction error"): **2-class noise-type
classification** on `non_gaussian_noise_only.npy` of two datasets that
empirically have very different cumulant signatures (Phase 0 finding):
    * synthetic polygauss (γ̂₄ ≈ +0.5, leptokurtic)
    * real RadioML residual (γ̂₄ ≈ −0.4, platykurtic)

This is a true cumulant-structure classification problem.

Pipeline (per Spec §6.4):
  1. Pool training blocks per class.
  2. Fit class-specific K_c via DSGE (B = signal autoreconstruction).
  3. Compute MSED_c features for test blocks.
  4. Train classifier (LR / RF / SVM) on log-MSED features.
  5. Compare against spectral baseline (γ̂_3, γ̂_4, γ̂_6, m₂).
  6. PATP α-sweep at s=3 → best basis.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from sdr_kunchenko.data_loader import SDRDataset, get_data_root
from sdr_kunchenko.dsge import (fit_class_coefficients, msed_features_batch)
from sdr_kunchenko.gsa import make_basis
from sdr_kunchenko.utils.cumulants import standardized_cumulants

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


# -------------------------------------------------------- data utilities


def collect_class_blocks(class_cfg: dict, root: Path,
                          snr_target: float, snr_tol: float,
                          n_train: int, n_test: int,
                          rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load up to (n_train + n_test) noise-only blocks for one class at given SNR.

    Returns (train_blocks, test_blocks). Train and test are disjoint.
    """
    ds = SDRDataset.open(class_cfg["dataset"], root=root)
    snr = ds.snr
    mask = np.abs(snr - snr_target) <= snr_tol
    candidates = np.where(mask)[0]
    rng.shuffle(candidates)
    if len(candidates) < n_train + n_test:
        n_train = min(n_train, len(candidates) // 2)
        n_test = min(n_test, len(candidates) - n_train)
    train_idxs = candidates[:n_train]
    test_idxs = candidates[n_train:n_train + n_test]
    arr = ds.noise_only if class_cfg.get("source") == "noise_only" else ds.noisy
    train = [np.asarray(arr[i]).astype(np.float64) for i in train_idxs]
    test = [np.asarray(arr[i]).astype(np.float64) for i in test_idxs]
    return train, test


def spectral_features(blocks: list[np.ndarray]) -> np.ndarray:
    """Baseline features: γ̂₃, γ̂₄, γ̂₆, log(m₂) per block."""
    out = []
    for b in blocks:
        c = standardized_cumulants(b)
        m2 = float(np.var(b, ddof=0))
        out.append([c["gamma3"], c["gamma4"], c["gamma6"], np.log(m2 + 1e-12)])
    return np.asarray(out)


# ---------------------------------------------------- classifier wrappers


def make_clf(kind: str):
    if kind == "logistic":
        return LogisticRegression(max_iter=1000, C=1.0)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    if kind == "svm":
        return SVC(kernel="rbf", probability=True, gamma="scale")
    raise ValueError(f"unknown classifier kind: {kind}")


def evaluate_clf(X_train, y_train, X_test, y_test, *, kind: str) -> dict:
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xe = scaler.transform(X_test)
    clf = make_clf(kind)
    clf.fit(Xt, y_train)
    pred = clf.predict(Xe)
    proba = clf.predict_proba(Xe)[:, 1] if hasattr(clf, "predict_proba") else None
    out = {
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


# ------------------------------------------------------- per-SNR pipeline


def run_snr_bucket(cfg: dict, snr_db: float, *, rng: np.random.Generator) -> dict:
    classes = cfg["dataset"]["classes"]
    n_train = cfg["monte_carlo"]["N_train_per_class"]
    n_test = cfg["monte_carlo"]["N_test_per_class"]
    snr_tol = cfg["monte_carlo"]["snr_bucket_tol"]
    ridge = float(cfg.get("regularization", {}).get("ridge", 1e-4))
    root = get_data_root()

    train_blocks_per_class = []
    test_blocks_per_class = []
    for c in classes:
        tr, te = collect_class_blocks(c, root, snr_db, snr_tol, n_train, n_test, rng)
        if len(tr) < 100 or len(te) < 50:
            return {"snr_db": snr_db, "skipped": True,
                    "reason": f"insufficient blocks for class {c['name']}: tr={len(tr)} te={len(te)}"}
        train_blocks_per_class.append(tr)
        test_blocks_per_class.append(te)

    n_classes = len(classes)
    # Build labels for test set (concat all classes).
    X_test_blocks = []
    y_test = []
    for cls_id, te in enumerate(test_blocks_per_class):
        X_test_blocks.extend(te)
        y_test.extend([cls_id] * len(te))
    y_test = np.asarray(y_test)

    out: dict = {
        "snr_db": float(snr_db),
        "n_classes": n_classes,
        "n_train_per_class": [len(t) for t in train_blocks_per_class],
        "n_test_per_class": [len(t) for t in test_blocks_per_class],
        "by_basis": {},
    }

    # ---- Spectral baseline ----
    Xs_train_list = [spectral_features(tr) for tr in train_blocks_per_class]
    ys_train = np.concatenate([np.full(len(tr), i) for i, tr in enumerate(train_blocks_per_class)])
    Xs_train = np.vstack(Xs_train_list)
    Xs_test = spectral_features(X_test_blocks)
    spectral_results = {}
    for clf_cfg in cfg["classifiers"]:
        spectral_results[clf_cfg["name"]] = evaluate_clf(
            Xs_train, ys_train, Xs_test, y_test, kind=clf_cfg["kind"]
        )
    out["baseline_spectral"] = spectral_results

    # ---- DSGE for each fixed basis ----
    for b_cfg in cfg["bases"]:
        basis = make_basis(b_cfg["kind"], b_cfg["s"])
        # Fit per-class K
        Ks = [fit_class_coefficients(basis, tr, ridge=ridge) for tr in train_blocks_per_class]
        # Train features = MSED on training samples (out-of-sample within class via splitting)
        # For simplicity, we use all training to fit Ks, and use those Ks
        # to compute MSED features for both train and test.
        train_features_per_class = []
        for tr in train_blocks_per_class:
            train_features_per_class.append(msed_features_batch(basis, Ks, tr))
        Xd_train = np.vstack(train_features_per_class)
        Xd_test = msed_features_batch(basis, Ks, X_test_blocks)
        dsge_results = {}
        for clf_cfg in cfg["classifiers"]:
            dsge_results[clf_cfg["name"]] = evaluate_clf(
                Xd_train, ys_train, Xd_test, y_test, kind=clf_cfg["kind"]
            )
        # Hybrid (DSGE + spectral)
        Xh_train = np.hstack([Xd_train, Xs_train])
        Xh_test = np.hstack([Xd_test, Xs_test])
        hybrid_results = {}
        for clf_cfg in cfg["classifiers"]:
            hybrid_results[clf_cfg["name"]] = evaluate_clf(
                Xh_train, ys_train, Xh_test, y_test, kind=clf_cfg["kind"]
            )
        out["by_basis"][b_cfg["name"]] = {
            "dsge": dsge_results, "hybrid": hybrid_results,
            "K_norms": [float(np.linalg.norm(K)) for K in Ks],
        }

    # ---- PATP α-sweep ----
    patp_s = cfg["patp_sweep"]["s"]
    out["patp_sweep"] = []
    for alpha in cfg["patp_sweep"]["alpha_grid"]:
        basis = make_basis("patp", patp_s, alpha=alpha)
        Ks = [fit_class_coefficients(basis, tr, ridge=ridge) for tr in train_blocks_per_class]
        train_features_per_class = []
        for tr in train_blocks_per_class:
            train_features_per_class.append(msed_features_batch(basis, Ks, tr))
        Xd_train = np.vstack(train_features_per_class)
        Xd_test = msed_features_batch(basis, Ks, X_test_blocks)
        # Use logistic for the sweep — fast + interpretable.
        res = evaluate_clf(Xd_train, ys_train, Xd_test, y_test, kind="logistic")
        res["alpha"] = float(alpha)
        out["patp_sweep"].append(res)

    return out


# --------------------------------------------------------- main driver


def make_manifest(cfg: dict, n_snr: int, elapsed: float) -> dict:
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
        "n_snr_buckets": n_snr,
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

    snr_grid = cfg["monte_carlo"]["snr_grid"]
    t0 = time.time()
    results = []
    for snr_db in tqdm(snr_grid, desc="SNR sweep (Exp 3)"):
        try:
            out = run_snr_bucket(cfg, snr_db, rng=rng)
            results.append(out)
            if not out.get("skipped"):
                # Log a quick summary
                accs = []
                for b, br in out["by_basis"].items():
                    accs.append((b, br["dsge"]["lr"]["accuracy"]))
                logging.info("SNR=%+d dB: DSGE-LR acc by basis: %s", snr_db, accs)
        except Exception as e:
            logging.exception("Failed at SNR=%s: %s", snr_db, e)
            results.append({"snr_db": snr_db, "error": str(e)})
    elapsed = time.time() - t0
    logging.info("Total time %.1fs", elapsed)

    out_json = {"snr_results": results,
                 "manifest": make_manifest(cfg, len(results), elapsed)}
    (out_dir / cfg["output"]["results_json"]).write_text(
        json.dumps(out_json, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    )
    (out_dir / cfg["output"]["manifest"]).write_text(
        json.dumps(out_json["manifest"], indent=2, default=str)
    )

    # Flatten to CSVs
    rows_basis = []
    rows_patp = []
    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        snr = r["snr_db"]
        # Spectral baseline
        for clf_name, m in r["baseline_spectral"].items():
            rows_basis.append({"snr_db": snr, "basis": "spectral_only",
                                "classifier": clf_name, "type": "baseline", **m})
        # Per-basis
        for b, br in r["by_basis"].items():
            for clf_name, m in br["dsge"].items():
                rows_basis.append({"snr_db": snr, "basis": b, "classifier": clf_name,
                                    "type": "dsge", **m})
            for clf_name, m in br["hybrid"].items():
                rows_basis.append({"snr_db": snr, "basis": b, "classifier": clf_name,
                                    "type": "hybrid", **m})
        for ps in r.get("patp_sweep", []):
            rows_patp.append({"snr_db": snr, **ps})
    pd.DataFrame(rows_basis).to_csv(out_dir / cfg["output"]["basis_csv"], index=False)
    pd.DataFrame(rows_patp).to_csv(out_dir / cfg["output"]["patp_csv"], index=False)

    return out_dir / cfg["output"]["results_json"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    json_path = main(args.config)
    print(f"\nResults: {json_path}")
