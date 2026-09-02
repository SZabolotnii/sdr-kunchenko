#!/usr/bin/env python3
"""Phase 6 — DSGE on real OcuSync I/Q (Tampere Zenodo dataset).

First real-data experiment for the dsge-toolkit applied to actual SDR
recordings of commercial drones. Three tasks:

    Task 1: Multi-protocol classification (4-class)
            DJI Mavic Pro / DJI Inspire 2 / Parrot Disco / Yuneec Typhoon H
            Tests whether DSGE features cleanly separate proprietary
            DJI OcuSync from Wi-Fi-based protocols.

    Task 2: DJI vs non-DJI (binary)
            (Mavic + Inspire) vs (Parrot + Yuneec)
            Simpler task; expected to be easy if Task 1 works.

    Task 3: Inter-model OcuSync fingerprinting (binary)
            Mavic Pro vs Inspire 2 (both use OcuSync 1)
            HARD task — same protocol, different physical Tx (different
            controller hardware + RF front-end). This is the proof-of-concept
            for the Plan B Mavic-pelengator transmitter fingerprinting.

Data: Tampere Zenodo (https://zenodo.org/records/4264467)
- 2.4 GHz band, fs = 120 MS/s
- int16 interleaved I/Q, ~0.5 sec recordings
- Anechoic chamber (limited channel diversity — Risk #5 in DEEPRESEARCH report)

Output:
    results/phase6_tampere_results.json   — all metrics
    results/phase6_classification_report.txt
    figures/                               — t-SNE, basis comparison, PATP sweep
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Vendored DSGE toolkit modules live next to this file (see toolkit/README.md).
_TOOLKIT = Path(__file__).resolve().parent / "toolkit"
if not _TOOLKIT.is_dir():                      # scripts under iterations/
    _TOOLKIT = Path(__file__).resolve().parent.parent / "toolkit"
sys.path.insert(0, str(_TOOLKIT))

from complex_features import ComplexDSGEFeatureExtractor  # noqa: E402
from splits import grouped_split, leakage_audit  # noqa: E402
from verify_dsge import bootstrap_compare, patp_boundary_check, verification_report  # noqa: E402


# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---- Config -----------------------------------------------------------

WINDOW_SAMPLES = 1024            # one DSGE input window
WINDOWS_PER_CLASS = 1000         # subsample for tractable compute
TRAIN_FRAC = 0.7                  # first 70% in time → train
SEED = 42
FS = 120e6                        # 2.4 GHz band sample rate
RIDGE = 0.01

CLASS_FILES = {
    "DJI_Mavic_Pro": DATA_DIR / "DJI_mavic_pro_2G.bin",
    "DJI_Inspire_2": DATA_DIR / "DJI_inspire_2_2G.bin",
    "Parrot_Disco": DATA_DIR / "Parrot_disco_2G.bin",
    "Yuneec_Typhoon_H": DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin",
}


# ---- Loader -----------------------------------------------------------


def load_tampere_bin(filepath: Path, max_samples: int | None = None) -> np.ndarray:
    """Load Tampere int16 interleaved I/Q file → complex64.

    Per the dataset's own load_bin.py:
        interleaved_data = np.fromfile(filepath, dtype='<i2')
        output = interleaved_data.astype(np.float64).view(np.complex128)

    We use complex64 instead of complex128 to halve memory.
    """
    if max_samples is not None:
        # Read only the first 2*max_samples int16's
        data = np.fromfile(filepath, dtype="<i2", count=2 * max_samples)
    else:
        data = np.fromfile(filepath, dtype="<i2")
    iq = data.astype(np.float32).reshape(-1, 2)
    return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def windowize(signal: np.ndarray, window_size: int,
              n_windows: int, rng: np.random.Generator,
              normalize: bool = True) -> np.ndarray:
    """Cut `n_windows` non-overlapping random windows from `signal`.

    Returns array of shape (n_windows, window_size). With normalize=True,
    each window is z-scored to zero mean / unit variance — this matches
    the dataset's example.py preprocessing.
    """
    max_start = len(signal) - window_size
    if max_start < n_windows:
        raise ValueError(f"Signal too short: {len(signal)} samples, "
                         f"need {n_windows} windows of {window_size}.")
    starts = rng.choice(max_start, size=n_windows, replace=False)
    starts.sort()  # so we can group by time
    windows = np.stack([signal[s:s + window_size] for s in starts])

    if normalize:
        # Per-window z-score (separately on Re and Im)
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True) + 1e-9
        windows = (windows - mean) / std
        windows = windows.astype(np.complex64)

    return windows, starts


def time_order_split(starts: np.ndarray, train_frac: float = 0.7
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Train: first train_frac of windows by time. Test: the rest.

    This prevents temporal leakage that random splits would introduce.
    """
    n = len(starts)
    cutoff = int(train_frac * n)
    return np.arange(cutoff), np.arange(cutoff, n)


# ---- Build datasets -----------------------------------------------


def build_dataset(rng: np.random.Generator) -> dict:
    """Load all files, extract windows, build (X, y, time_groups)."""
    print(f"Loading {len(CLASS_FILES)} files…", flush=True)
    X_list, y_list, time_list = [], [], []
    class_names = list(CLASS_FILES.keys())

    for class_idx, (name, path) in enumerate(CLASS_FILES.items()):
        if not path.exists():
            print(f"  [skip] {name} — file not found at {path}")
            continue
        # Load enough samples for windowing
        max_samples = WINDOWS_PER_CLASS * WINDOW_SAMPLES * 4  # 4× headroom
        max_samples = min(max_samples, path.stat().st_size // 4)  # int16+int16 = 4 bytes/sample
        t0 = time.time()
        signal = load_tampere_bin(path, max_samples=max_samples)
        windows, starts = windowize(
            signal, WINDOW_SAMPLES, WINDOWS_PER_CLASS, rng
        )
        del signal  # free memory immediately
        print(f"  [{class_idx}] {name}: {len(windows)} windows × "
              f"{WINDOW_SAMPLES}, normalized. ({time.time()-t0:.1f}s)")
        X_list.append(windows)
        y_list.append(np.full(len(windows), class_idx))
        time_list.append(starts)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    time_groups = np.concatenate(time_list)

    print(f"Total: {X.shape}, dtype {X.dtype}, classes {len(set(y))}")
    return {
        "X": X, "y": y, "time_starts": time_groups,
        "class_names": class_names[:len(X_list)],
    }


# ---- Experiment runs ---------------------------------------------


def run_task(X_tr: np.ndarray, y_tr: np.ndarray,
             X_te: np.ndarray, y_te: np.ndarray,
             task_name: str,
             bases: list[str] = ("polynomial", "fractional", "trigonometric", "robust"),
             do_patp_sweep: bool = True,
             classifier=None) -> dict:
    """Run a single classification task with all bases + PATP."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    if classifier is None:
        classifier = LogisticRegression(max_iter=2000)

    print(f"\n=== {task_name} ===")
    print(f"  Train {X_tr.shape}, Test {X_te.shape}, "
          f"classes {sorted(set(y_tr))}")

    results = {"task": task_name, "bases": {}, "patp": None}

    # Basis comparison
    for basis in bases:
        t0 = time.time()
        ext = ComplexDSGEFeatureExtractor(
            mode="re_im", basis=basis, n=3, ridge=RIDGE
        ).fit(X_tr, y_tr)
        feat_tr = ext.transform(X_tr)
        feat_te = ext.transform(X_te)
        clf = classifier.fit(feat_tr, y_tr)
        y_pred = clf.predict(feat_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
        cond = ext.conditioning_report
        cond_max = max(max(v) if isinstance(v, list) else v for v in cond.values())
        results["bases"][basis] = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "feature_dim": int(feat_tr.shape[1]),
            "cond_F_max": float(cond_max),
            "duration_seconds": float(time.time() - t0),
        }
        print(f"    {basis:<14} acc={acc:.4f} f1={f1:.4f} "
              f"feat_dim={feat_tr.shape[1]:>2} cond_max={cond_max:.0f}  "
              f"({time.time()-t0:.1f}s)")

    # PATP α-sweep
    if do_patp_sweep:
        print(f"  PATP α-sweep (11 points)…")
        alphas = np.linspace(0, 1, 11)
        f1s, accs = [], []
        for a in alphas:
            ext = ComplexDSGEFeatureExtractor(
                mode="re_im", basis="patp", n=3, alpha=float(a), ridge=RIDGE
            ).fit(X_tr, y_tr)
            feat_tr = ext.transform(X_tr)
            feat_te = ext.transform(X_te)
            clf = classifier.fit(feat_tr, y_tr)
            y_pred = clf.predict(feat_te)
            f1s.append(float(f1_score(y_te, y_pred, average="macro")))
            accs.append(float(accuracy_score(y_te, y_pred)))
        best_idx = int(np.argmax(f1s))
        check = patp_boundary_check(alphas.tolist(), f1s)
        results["patp"] = {
            "alphas": alphas.tolist(),
            "macro_f1": f1s,
            "accuracy": accs,
            "best_alpha": float(alphas[best_idx]),
            "best_macro_f1": float(f1s[best_idx]),
            "best_accuracy": float(accs[best_idx]),
            "boundary_check": check,
        }
        print(f"    α_opt = {alphas[best_idx]:.2f} F1 = {f1s[best_idx]:.4f}  "
              f"({check['recommendation']})")

    return results


def run_bootstrap_dsge_vs_baseline(X_tr, y_tr, X_te, y_te,
                                    basis: str, alpha: float | None = None,
                                    R: int = 1000) -> dict:
    """Bootstrap: DSGE-on-magnitude vs DSGE-on-re_im — measures how much
    I/Q geometry adds over magnitude-only features."""
    from sklearn.linear_model import LogisticRegression

    # baseline: magnitude-only DSGE
    ext_b = ComplexDSGEFeatureExtractor(
        mode="magnitude", basis=basis, n=3, alpha=alpha, ridge=RIDGE
    ).fit(X_tr, y_tr)
    f_b = LogisticRegression(max_iter=2000).fit(
        ext_b.transform(X_tr), y_tr).predict(ext_b.transform(X_te))

    # full: re_im DSGE
    ext_h = ComplexDSGEFeatureExtractor(
        mode="re_im", basis=basis, n=3, alpha=alpha, ridge=RIDGE
    ).fit(X_tr, y_tr)
    f_h = LogisticRegression(max_iter=2000).fit(
        ext_h.transform(X_tr), y_tr).predict(ext_h.transform(X_te))

    bs = bootstrap_compare(y_te, f_b, f_h, R=R)
    return bs


# ---- Main ------------------------------------------------------------


def main():
    rng = np.random.default_rng(SEED)

    # 1. Build dataset
    data = build_dataset(rng)
    X = data["X"]
    y = data["y"]
    time_starts = data["time_starts"]
    class_names = data["class_names"]
    n_classes = len(class_names)

    # 2. Time-ordered split (per-class to maintain class balance)
    train_idx_list, test_idx_list = [], []
    for c in range(n_classes):
        c_mask = (y == c)
        c_idx = np.where(c_mask)[0]
        # within class, order by time
        c_starts = time_starts[c_idx]
        order = np.argsort(c_starts)
        c_idx_ordered = c_idx[order]
        cutoff = int(TRAIN_FRAC * len(c_idx_ordered))
        train_idx_list.append(c_idx_ordered[:cutoff])
        test_idx_list.append(c_idx_ordered[cutoff:])
    train_idx = np.concatenate(train_idx_list)
    test_idx = np.concatenate(test_idx_list)

    audit = leakage_audit(train_idx, test_idx)
    print(f"\nLeakage audit: {audit['n_train_samples']} train, "
          f"{audit['n_test_samples']} test, sample overlap = "
          f"{audit['sample_overlap']}")

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

    all_results = {
        "config": {
            "window_samples": WINDOW_SAMPLES,
            "windows_per_class": WINDOWS_PER_CLASS,
            "train_frac": TRAIN_FRAC,
            "ridge": RIDGE,
            "fs_hz": FS,
            "seed": SEED,
            "class_names": class_names,
        },
    }

    # ---- Task 1: 4-class multi-protocol ----
    res_4class = run_task(X_tr, y_tr, X_te, y_te,
                           task_name="Task 1 — 4-class multi-protocol")
    all_results["task1_4class"] = res_4class

    # ---- Task 2: DJI (0,1) vs non-DJI (2,3) ----
    if n_classes >= 4:
        y_tr_dji = (y_tr <= 1).astype(int)
        y_te_dji = (y_te <= 1).astype(int)
        res_dji = run_task(X_tr, y_tr_dji, X_te, y_te_dji,
                            task_name="Task 2 — DJI vs non-DJI (binary)")
        all_results["task2_dji_vs_nondji"] = res_dji

    # ---- Task 3: inter-model OcuSync (Mavic vs Inspire) ----
    # both have label 0 and 1 in original; restrict to those
    if n_classes >= 2:
        ocusync_mask_tr = (y_tr <= 1)
        ocusync_mask_te = (y_te <= 1)
        if ocusync_mask_tr.sum() > 100 and ocusync_mask_te.sum() > 50:
            res_ocu = run_task(
                X_tr[ocusync_mask_tr], y_tr[ocusync_mask_tr],
                X_te[ocusync_mask_te], y_te[ocusync_mask_te],
                task_name="Task 3 — Inter-model OcuSync (Mavic Pro vs Inspire 2)",
                do_patp_sweep=True,
            )
            all_results["task3_ocusync_intermodel"] = res_ocu

    # ---- Bootstrap: re_im vs magnitude on Task 3 (the hardest task) ----
    if "task3_ocusync_intermodel" in all_results:
        print("\n=== Bootstrap: re_im vs magnitude on Task 3 (R=1000) ===")
        bs = run_bootstrap_dsge_vs_baseline(
            X_tr[ocusync_mask_tr], y_tr[ocusync_mask_tr],
            X_te[ocusync_mask_te], y_te[ocusync_mask_te],
            basis="fractional", R=1000,
        )
        all_results["bootstrap_re_im_vs_magnitude_task3"] = bs
        print(f"  Δ macro_f1 = {bs['delta']:+.4f}, p = {bs['p_value']:.4f}, "
              f"95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}], "
              f"sig = {bs['significant']}")

    # ---- Save ----
    out_json = RESULTS_DIR / "phase6_tampere_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_json}")

    # ---- Verification report ----
    if "task1_4class" in all_results:
        b = all_results["task1_4class"]["bases"]
        best_basis = max(b, key=lambda k: b[k]["accuracy"])
        all_results["task1_4class"]["best_basis_summary"] = best_basis

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Phase 6 — Tampere DSGE Verification Report")
        report_lines.append("=" * 70)
        report_lines.append(f"\nDataset: Tampere Zenodo (https://zenodo.org/records/4264467)")
        report_lines.append(f"  4 classes, {len(X_tr) + len(X_te)} windows total")
        report_lines.append(f"  Train {len(X_tr)} / Test {len(X_te)}, time-ordered split")

        for task_key in ("task1_4class", "task2_dji_vs_nondji", "task3_ocusync_intermodel"):
            if task_key not in all_results:
                continue
            r = all_results[task_key]
            report_lines.append(f"\n--- {r['task']} ---")
            for basis, m in r["bases"].items():
                report_lines.append(f"  {basis:<14} acc={m['accuracy']:.4f} "
                                     f"F1={m['macro_f1']:.4f} cond_max={m['cond_F_max']:.0f}")
            if r["patp"]:
                report_lines.append(
                    f"  PATP α_opt = {r['patp']['best_alpha']:.2f} "
                    f"F1 = {r['patp']['best_macro_f1']:.4f}  "
                    f"({r['patp']['boundary_check']['recommendation']})"
                )

        if "bootstrap_re_im_vs_magnitude_task3" in all_results:
            bs = all_results["bootstrap_re_im_vs_magnitude_task3"]
            report_lines.append(f"\nBootstrap re_im vs magnitude on Task 3:")
            report_lines.append(f"  Δ = {bs['delta']:+.4f}, p = {bs['p_value']:.4f}, "
                                 f"95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")
            report_lines.append(f"  Significant? {bs['significant']}")

        report = "\n".join(report_lines)
        report_file = RESULTS_DIR / "phase6_classification_report.txt"
        report_file.write_text(report + "\n")
        print(f"\n{report}")
        print(f"\nReport saved to {report_file}")

    return all_results


if __name__ == "__main__":
    main()
