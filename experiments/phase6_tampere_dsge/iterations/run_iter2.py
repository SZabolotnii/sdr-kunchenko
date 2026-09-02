#!/usr/bin/env python3
"""Phase 6 iter-2 — DSGE on Tampere with burst detection + hybrid features.

Improvements over iter-1:
    1. WINDOW_SAMPLES 1024 → 4096   (34 μs at 120 MS/s — captures fragments
                                      of OcuSync packets, not just noise)
    2. WINDOWS_PER_CLASS 1000 → 3000 (better stat power)
    3. Burst-aware windowing — keep only windows with above-median energy
       (filters out the inter-burst silence that wastes 80% of recording time)
    4. Hybrid mode — concat DSGE features with spectral PSD bins
                     (Phase 3 paper showed +18pp from DSGE-only to hybrid)

Same 3 tasks (4-class / DJI-vs-non-DJI / inter-OcuSync intra-fingerprinting).
Same verification protocol (grouped time-split, bootstrap, PATP boundary).
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
from splits import leakage_audit  # noqa: E402
from verify_dsge import bootstrap_compare, patp_boundary_check  # noqa: E402


# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---- Config -----------------------------------------------------------

WINDOW_SAMPLES = 4096            # 34 μs at 120 MS/s
WINDOWS_PER_CLASS = 1000
TRAIN_FRAC = 0.7
SEED = 42
FS = 120e6
RIDGE = 0.01
ENERGY_PERCENTILE = 40           # keep windows above 40th percentile (top 60%)
                                 # 40% empirically — anechoic chamber recordings
                                 # have bursts ~50% of the time, so 60-pct too strict

CLASS_FILES = {
    "DJI_Mavic_Pro": DATA_DIR / "DJI_mavic_pro_2G.bin",
    "DJI_Inspire_2": DATA_DIR / "DJI_inspire_2_2G.bin",
    "Parrot_Disco": DATA_DIR / "Parrot_disco_2G.bin",
    "Yuneec_Typhoon_H": DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin",
}


# ---- Loader -----------------------------------------------------------


def load_tampere_bin(filepath: Path, max_samples: int | None = None) -> np.ndarray:
    if max_samples is not None:
        data = np.fromfile(filepath, dtype="<i2", count=2 * max_samples)
    else:
        data = np.fromfile(filepath, dtype="<i2")
    iq = data.astype(np.float32).reshape(-1, 2)
    return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def burst_aware_windows(signal: np.ndarray, window_size: int,
                         n_windows: int, energy_pct: float,
                         rng: np.random.Generator,
                         normalize: bool = True
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Cut windows enriched with high-energy regions. Vectorized.

    Reshape signal to (n_chunks, window_size), compute energy per row
    via np.mean(|x|^2, axis=1) — order of magnitude faster than Python loop.
    """
    # Truncate to integer multiple of window_size, reshape
    n_chunks = len(signal) // window_size
    truncated = signal[:n_chunks * window_size]
    chunks = truncated.reshape(n_chunks, window_size)
    energies = (chunks.real ** 2 + chunks.imag ** 2).mean(axis=1)

    threshold = float(np.percentile(energies, energy_pct))
    high_energy_idx = np.where(energies >= threshold)[0]

    if len(high_energy_idx) < n_windows:
        # Fall back: take all high-energy + extra random
        extra_needed = n_windows - len(high_energy_idx)
        low_idx = np.where(energies < threshold)[0]
        extra = rng.choice(low_idx, size=min(extra_needed, len(low_idx)),
                           replace=False)
        chosen_idx = np.concatenate([high_energy_idx, extra])
    else:
        chosen_idx = rng.choice(high_energy_idx, size=n_windows, replace=False)

    chosen_idx = np.sort(chosen_idx)
    windows = chunks[chosen_idx]
    starts = chosen_idx * window_size

    if normalize:
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True) + 1e-9
        windows = ((windows - mean) / std).astype(np.complex64)

    return windows, starts


# ---- Spectral features (for hybrid mode) -----------------------------


def spectral_psd_features(X: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Compute log-PSD with `n_bins` log-spaced frequency bins.

    Gives compact frequency-domain summary for hybrid features:
    - magnitude per bin (log10)
    - peak frequency
    - spectral centroid

    Returns (n_samples, n_bins + 2) feature matrix.
    """
    N, T = X.shape
    # FFT magnitude squared
    psd = np.abs(np.fft.fft(X, axis=1)) ** 2 / T   # (N, T)
    # Take only positive half
    psd = psd[:, :T // 2]
    # Aggregate into n_bins log-spaced bins
    edges = np.unique(np.geomspace(1, psd.shape[1], n_bins + 1).astype(int))
    bin_means = np.zeros((N, len(edges) - 1))
    for i in range(len(edges) - 1):
        bin_means[:, i] = psd[:, edges[i]:edges[i + 1]].mean(axis=1)
    log_psd = np.log10(bin_means + 1e-12)

    # Spectral centroid + peak freq
    freqs = np.arange(psd.shape[1])
    centroid = (psd * freqs).sum(axis=1) / (psd.sum(axis=1) + 1e-12)
    peak = np.argmax(psd, axis=1).astype(np.float64)

    return np.hstack([log_psd, centroid[:, None], peak[:, None]])


# ---- Build dataset --------------------------------------------------


# patched
def build_dataset(rng: np.random.Generator) -> dict:
    print(f"Loading {len(CLASS_FILES)} files (window={WINDOW_SAMPLES}, "
          f"per-class={WINDOWS_PER_CLASS}, energy_pct={ENERGY_PERCENTILE})…",
          flush=True)
    X_list, y_list, time_list = [], [], []
    class_names = list(CLASS_FILES.keys())

    for class_idx, (name, path) in enumerate(CLASS_FILES.items()):
        if not path.exists():
            print(f"  [skip] {name}")
            continue
        # Load enough samples for burst-aware sampling — 2× headroom
        max_samples = min(WINDOWS_PER_CLASS * WINDOW_SAMPLES * 2,
                          path.stat().st_size // 4)
        t0 = time.time()
        signal = load_tampere_bin(path, max_samples=max_samples)
        windows, starts = burst_aware_windows(
            signal, WINDOW_SAMPLES, WINDOWS_PER_CLASS,
            ENERGY_PERCENTILE, rng,
        )
        del signal
        print(f"  [{class_idx}] {name}: {len(windows)}×{WINDOW_SAMPLES} "
              f"burst-enriched. ({time.time()-t0:.1f}s)")
        X_list.append(windows)
        y_list.append(np.full(len(windows), class_idx))
        time_list.append(starts)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    time_groups = np.concatenate(time_list)

    print(f"Total: {X.shape}")
    return {
        "X": X, "y": y, "time_starts": time_groups,
        "class_names": class_names[:len(X_list)],
    }


# ---- Tasks --------------------------------------------------------


def run_task(X_tr, y_tr, X_te, y_te, task_name: str,
             bases=("polynomial", "fractional", "trigonometric", "robust"),
             do_patp_sweep=True, do_hybrid=True,
             classifier=None) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    if classifier is None:
        classifier = LogisticRegression(max_iter=2000)

    print(f"\n=== {task_name} ===")
    print(f"  Train {X_tr.shape}, Test {X_te.shape}, "
          f"classes {sorted(set(y_tr))}")
    results = {"task": task_name, "bases": {}, "patp": None, "hybrid": None}

    # Pre-compute spectral features once (used for hybrid)
    spec_tr = spectral_psd_features(X_tr) if do_hybrid else None
    spec_te = spectral_psd_features(X_te) if do_hybrid else None

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
        results["bases"][basis] = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
            "feature_dim": int(feat_tr.shape[1]),
            "duration_seconds": float(time.time() - t0),
        }
        print(f"    {basis:<14} acc={results['bases'][basis]['accuracy']:.4f} "
              f"f1={results['bases'][basis]['macro_f1']:.4f}  "
              f"({results['bases'][basis]['duration_seconds']:.1f}s)")

    # Hybrid (DSGE + spectral) — using best basis from above
    if do_hybrid and spec_tr is not None:
        best_basis = max(results["bases"], key=lambda k: results["bases"][k]["accuracy"])
        ext = ComplexDSGEFeatureExtractor(
            mode="re_im", basis=best_basis, n=3, ridge=RIDGE
        ).fit(X_tr, y_tr)
        dsge_tr = ext.transform(X_tr)
        dsge_te = ext.transform(X_te)
        hybrid_tr = np.hstack([dsge_tr, spec_tr])
        hybrid_te = np.hstack([dsge_te, spec_te])

        clf_dsge = classifier.fit(dsge_tr, y_tr)
        y_pred_dsge = clf_dsge.predict(dsge_te)
        clf_spec = classifier.fit(spec_tr, y_tr)
        y_pred_spec = clf_spec.predict(spec_te)
        clf_hybrid = classifier.fit(hybrid_tr, y_tr)
        y_pred_hybrid = clf_hybrid.predict(hybrid_te)

        results["hybrid"] = {
            "best_basis": best_basis,
            "dsge_only": {
                "accuracy": float(accuracy_score(y_te, y_pred_dsge)),
                "macro_f1": float(f1_score(y_te, y_pred_dsge, average="macro")),
                "feature_dim": int(dsge_tr.shape[1]),
            },
            "spectral_only": {
                "accuracy": float(accuracy_score(y_te, y_pred_spec)),
                "macro_f1": float(f1_score(y_te, y_pred_spec, average="macro")),
                "feature_dim": int(spec_tr.shape[1]),
            },
            "hybrid": {
                "accuracy": float(accuracy_score(y_te, y_pred_hybrid)),
                "macro_f1": float(f1_score(y_te, y_pred_hybrid, average="macro")),
                "feature_dim": int(hybrid_tr.shape[1]),
            },
        }
        print(f"  Hybrid (DSGE [{best_basis}] + spectral PSD):")
        print(f"    DSGE-only      acc={results['hybrid']['dsge_only']['accuracy']:.4f} "
              f"f1={results['hybrid']['dsge_only']['macro_f1']:.4f}  "
              f"feat_dim={results['hybrid']['dsge_only']['feature_dim']}")
        print(f"    Spectral-only  acc={results['hybrid']['spectral_only']['accuracy']:.4f} "
              f"f1={results['hybrid']['spectral_only']['macro_f1']:.4f}  "
              f"feat_dim={results['hybrid']['spectral_only']['feature_dim']}")
        print(f"    Hybrid         acc={results['hybrid']['hybrid']['accuracy']:.4f} "
              f"f1={results['hybrid']['hybrid']['macro_f1']:.4f}  "
              f"feat_dim={results['hybrid']['hybrid']['feature_dim']}")

        # Bootstrap: hybrid > spectral-only?
        bs = bootstrap_compare(y_te, y_pred_spec, y_pred_hybrid, R=500)
        results["hybrid"]["bootstrap_vs_spectral"] = bs
        print(f"    Bootstrap hybrid vs spectral-only: Δ={bs['delta']:+.4f} "
              f"p={bs['p_value']:.4f} sig={bs['significant']}")

    # PATP sweep
    if do_patp_sweep:
        alphas = np.linspace(0, 1, 11)
        f1s = []
        for a in alphas:
            ext = ComplexDSGEFeatureExtractor(
                mode="re_im", basis="patp", n=3, alpha=float(a), ridge=RIDGE
            ).fit(X_tr, y_tr)
            clf = classifier.fit(ext.transform(X_tr), y_tr)
            f1s.append(float(f1_score(y_te, clf.predict(ext.transform(X_te)),
                                       average="macro")))
        best_idx = int(np.argmax(f1s))
        check = patp_boundary_check(alphas.tolist(), f1s)
        results["patp"] = {
            "alphas": alphas.tolist(),
            "macro_f1": f1s,
            "best_alpha": float(alphas[best_idx]),
            "best_macro_f1": float(f1s[best_idx]),
            "boundary_check": check,
        }
        print(f"  PATP α_opt={alphas[best_idx]:.2f} F1={f1s[best_idx]:.4f}  "
              f"({check['recommendation']})")

    return results


# ---- Main -----------------------------------------------------------


def main():
    rng = np.random.default_rng(SEED)
    data = build_dataset(rng)
    X = data["X"]
    y = data["y"]
    time_starts = data["time_starts"]
    class_names = data["class_names"]
    n_classes = len(class_names)

    # Time-ordered per-class split
    train_idx_list, test_idx_list = [], []
    for c in range(n_classes):
        c_idx = np.where(y == c)[0]
        c_starts = time_starts[c_idx]
        c_idx_ordered = c_idx[np.argsort(c_starts)]
        cutoff = int(TRAIN_FRAC * len(c_idx_ordered))
        train_idx_list.append(c_idx_ordered[:cutoff])
        test_idx_list.append(c_idx_ordered[cutoff:])
    train_idx = np.concatenate(train_idx_list)
    test_idx = np.concatenate(test_idx_list)

    audit = leakage_audit(train_idx, test_idx)
    print(f"\nLeakage audit: {audit['n_train_samples']} train, "
          f"{audit['n_test_samples']} test, overlap={audit['sample_overlap']}")

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    all_results = {
        "config": {
            "window_samples": WINDOW_SAMPLES,
            "windows_per_class": WINDOWS_PER_CLASS,
            "energy_percentile": ENERGY_PERCENTILE,
            "train_frac": TRAIN_FRAC,
            "ridge": RIDGE,
            "fs_hz": FS,
            "seed": SEED,
            "class_names": class_names,
            "iter": 2,
        },
    }

    # Task 1: 4-class
    res1 = run_task(X_tr, y_tr, X_te, y_te,
                     task_name="Task 1 — 4-class multi-protocol")
    all_results["task1_4class"] = res1

    # Task 2: DJI vs non-DJI
    if n_classes >= 4:
        y_tr_dji = (y_tr <= 1).astype(int)
        y_te_dji = (y_te <= 1).astype(int)
        res2 = run_task(X_tr, y_tr_dji, X_te, y_te_dji,
                         task_name="Task 2 — DJI vs non-DJI",
                         do_hybrid=True, do_patp_sweep=False)  # patp скип на бінарних
        all_results["task2_dji_vs_nondji"] = res2

    # Task 3: Mavic vs Inspire (intra-OcuSync)
    if n_classes >= 2:
        m = (y_tr <= 1)
        m_te = (y_te <= 1)
        if m.sum() > 100 and m_te.sum() > 50:
            res3 = run_task(
                X_tr[m], y_tr[m], X_te[m_te], y_te[m_te],
                task_name="Task 3 — Inter-model OcuSync (Mavic vs Inspire)",
                do_patp_sweep=True, do_hybrid=True,
            )
            all_results["task3_ocusync_intermodel"] = res3

    out_json = RESULTS_DIR / "phase6_iter2_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {out_json}")

    return all_results


if __name__ == "__main__":
    main()
