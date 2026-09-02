#!/usr/bin/env python3
"""Hybrid variants — Сергій's hypothesis check.

Question: чи polynomial DSGE дублює spectral, а fractional DSGE
несе complementary information?

Theoretical motivation:
    - Spectral PSD = magnitude of FFT bins → captures 2nd-moment
      energy distribution in frequency domain.
    - DSGE-polynomial on time-domain → captures higher-order moments
      (γ₃, γ₄) but FFT-energy heavy-tail correlates with γ₄.
    - DSGE-fractional sign(x)|x|^p, p≤½ → soft-amplifier for small
      signals, captures heavy-tail structure that PSD averages out.
    - DSGE-trig sin(kx) → periodicity in waveform = same as PSD.

Test 4 hybrid variants per task with bootstrap:
    1. poly + spec       (current default)
    2. frac + spec       ← Сергій's prediction
    3. poly + frac + spec
    4. all 4 bases + spec (ensemble like HAR paper)
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
from complex_features import ComplexDSGEFeatureExtractor
from verify_dsge import bootstrap_compare

WIN = 4096
N_PER_CLASS = 1000
TRAIN_FRAC = 0.7
# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
OUT = Path(__file__).parent.parent / "results" / "phase6_iter2_hybrid_variants.json"
OUT.parent.mkdir(exist_ok=True)

CLASSES = [
    ("DJI_Mavic_Pro",   DATA_DIR / "DJI_mavic_pro_2G.bin"),
    ("DJI_Inspire_2",   DATA_DIR / "DJI_inspire_2_2G.bin"),
    ("Parrot_Disco",    DATA_DIR / "Parrot_disco_2G.bin"),
    ("Yuneec_Typhoon_H",DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin"),
]


def load_burst(path, n_windows, win, rng):
    data = np.fromfile(path, dtype="<i2", count=2 * n_windows * win * 2)
    iq = data.astype(np.float32).reshape(-1, 2)
    sig = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    nc = len(sig) // win
    chunks = sig[:nc * win].reshape(nc, win)
    energies = (chunks.real ** 2 + chunks.imag ** 2).mean(axis=1)
    threshold = float(np.percentile(energies, 40))
    high_idx = np.where(energies >= threshold)[0]
    if len(high_idx) >= n_windows:
        chosen = rng.choice(high_idx, size=n_windows, replace=False)
    else:
        low_idx = np.where(energies < threshold)[0]
        extra = rng.choice(low_idx, size=n_windows - len(high_idx), replace=False)
        chosen = np.concatenate([high_idx, extra])
    chosen = np.sort(chosen)
    windows = chunks[chosen]
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-9
    return ((windows - mean) / std).astype(np.complex64), chosen * win


def spectral_features(X, n_bins=16):
    N, T = X.shape
    psd = np.abs(np.fft.fft(X, axis=1)) ** 2 / T
    psd = psd[:, :T // 2]
    edges = np.unique(np.geomspace(1, psd.shape[1], n_bins + 1).astype(int))
    bins = np.zeros((N, len(edges) - 1))
    for i in range(len(edges) - 1):
        bins[:, i] = psd[:, edges[i]:edges[i + 1]].mean(axis=1)
    return np.log10(bins + 1e-12)


def dsge_features(X_tr, y_tr, X_te, basis):
    ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=basis, n=3, ridge=0.01).fit(X_tr, y_tr)
    return ext.transform(X_tr), ext.transform(X_te)


print("Loading…", flush=True)
rng = np.random.default_rng(42)
X_list, y_list, t_list = [], [], []
for c, (name, path) in enumerate(CLASSES):
    w, ts = load_burst(path, N_PER_CLASS, WIN, rng)
    X_list.append(w); y_list.append(np.full(N_PER_CLASS, c)); t_list.append(ts)

X = np.vstack(X_list); y = np.concatenate(y_list); t = np.concatenate(t_list)

tr_idx, te_idx = [], []
for c in range(len(CLASSES)):
    ci = np.where(y == c)[0]
    co = ci[np.argsort(t[ci])]
    cut = int(TRAIN_FRAC * len(co))
    tr_idx.append(co[:cut]); te_idx.append(co[cut:])
tr = np.concatenate(tr_idx); te = np.concatenate(te_idx)
X_tr, X_te = X[tr], X[te]; y_tr, y_te = y[tr], y[te]
print(f"Train {X_tr.shape}, Test {X_te.shape}", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate_features(feat_tr, feat_te, y_tr, y_te):
    clf = LogisticRegression(max_iter=2000).fit(feat_tr, y_tr)
    yp = clf.predict(feat_te)
    return {
        "accuracy": float(accuracy_score(y_te, yp)),
        "macro_f1": float(f1_score(y_te, yp, average="macro")),
        "feature_dim": int(feat_tr.shape[1]),
        "predictions": yp,
    }


def run_hybrid_battery(X_tr, y_tr, X_te, y_te, label):
    print(f"\n=== {label} ===", flush=True)
    print(f"  Train {X_tr.shape}, Test {X_te.shape}", flush=True)

    # Baseline: spectral-only
    spec_tr, spec_te = spectral_features(X_tr), spectral_features(X_te)
    base = evaluate_features(spec_tr, spec_te, y_tr, y_te)
    print(f"  spectral-only           acc={base['accuracy']:.4f} dim={base['feature_dim']}", flush=True)

    # Compute DSGE features for each basis once
    dsge = {}
    for basis in ("polynomial", "fractional", "trigonometric", "robust"):
        t0 = time.time()
        dsge[basis] = dsge_features(X_tr, y_tr, X_te, basis)
        print(f"    [{basis}] DSGE features computed ({time.time()-t0:.1f}s)", flush=True)

    # Hybrid variants
    variants = {
        "poly + spec":              [dsge["polynomial"][0], spec_tr],
        "frac + spec":              [dsge["fractional"][0], spec_tr],
        "trig + spec":              [dsge["trigonometric"][0], spec_tr],
        "poly + frac + spec":       [dsge["polynomial"][0], dsge["fractional"][0], spec_tr],
        "all 4 bases + spec":       [dsge["polynomial"][0], dsge["fractional"][0],
                                      dsge["trigonometric"][0], dsge["robust"][0], spec_tr],
        "frac alone (no spec)":     [dsge["fractional"][0]],
        "poly + frac (no spec)":    [dsge["polynomial"][0], dsge["fractional"][0]],
    }

    results = {"spectral_only": base, "variants": {}}
    for name, train_parts in variants.items():
        train_concat = np.hstack(train_parts)
        # Build matching test set
        if "spec" in name and len(train_parts) > 1:
            test_parts_d = []
            for basis in ("polynomial", "fractional", "trigonometric", "robust"):
                if dsge[basis][0] is train_parts[0]:  # match by first ref — fragile
                    pass
            # Reconstruct test parts in the same order
            test_parts = []
            for tp in train_parts:
                if tp is spec_tr:
                    test_parts.append(spec_te)
                else:
                    for basis, (tr_arr, te_arr) in dsge.items():
                        if tp is tr_arr:
                            test_parts.append(te_arr)
                            break
        else:
            test_parts = []
            for tp in train_parts:
                if tp is spec_tr:
                    test_parts.append(spec_te)
                else:
                    for basis, (tr_arr, te_arr) in dsge.items():
                        if tp is tr_arr:
                            test_parts.append(te_arr)
                            break
        test_concat = np.hstack(test_parts)

        r = evaluate_features(train_concat, test_concat, y_tr, y_te)
        bs = bootstrap_compare(y_te, base["predictions"], r["predictions"], R=500)
        r.pop("predictions")
        r["bootstrap_vs_spectral"] = bs
        results["variants"][name] = r
        sig_marker = " ★" if bs["significant"] else ""
        print(f"  {name:<28} acc={r['accuracy']:.4f} dim={r['feature_dim']:>2} "
              f"Δ={bs['delta']:+.4f} p={bs['p_value']:.4f}{sig_marker}", flush=True)

    return results


all_results = {}

# Task 2: DJI vs non-DJI
y_tr_dji = (y_tr <= 1).astype(int)
y_te_dji = (y_te <= 1).astype(int)
all_results["task2"] = run_hybrid_battery(X_tr, y_tr_dji, X_te, y_te_dji,
                                           label="Task 2 — DJI vs non-DJI")

# Task 3: intra-OcuSync
m_tr = (y_tr <= 1); m_te = (y_te <= 1)
all_results["task3"] = run_hybrid_battery(
    X_tr[m_tr], y_tr[m_tr], X_te[m_te], y_te[m_te],
    label="Task 3 — Mavic Pro vs Inspire 2 (intra-OcuSync)"
)

with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved {OUT}")
