#!/usr/bin/env python3
"""PATP α-sweep with complementarity-aware setup.

For each α ∈ [0, 1]:
    features = [PATP_DSGE(α), spectral_PSD]
    classifier = LogReg
    metric = bootstrap Δ vs spectral-only (R=500)

Hypothesis (from corrected analysis): α=0 (frac) wins clearly,
α=1 (poly) gives ~0 boost (duplicates spectral).

This is the empirical PATP α-sweep that *should* show interior or
boundary-at-0 optimum — different from previous task-3 sweep that
hit α=1 boundary.
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
from verify_dsge import bootstrap_compare, patp_boundary_check

WIN = 4096
N_PER_CLASS = 1000
TRAIN_FRAC = 0.7
# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
OUT = Path(__file__).parent.parent / "results" / "phase6_patp_complementarity.json"
OUT.parent.mkdir(exist_ok=True)

CLASSES = [
    ("DJI_Mavic_Pro",   DATA_DIR / "DJI_mavic_pro_2G.bin"),
    ("DJI_Inspire_2",   DATA_DIR / "DJI_inspire_2_2G.bin"),
    ("Parrot_Disco",    DATA_DIR / "Parrot_disco_2G.bin"),
    ("Yuneec_Typhoon_H",DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin"),
]


def load_burst(path, n, w, rng):
    data = np.fromfile(path, dtype="<i2", count=2 * n * w * 2)
    iq = data.astype(np.float32).reshape(-1, 2)
    sig = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    nc = len(sig) // w
    chunks = sig[:nc * w].reshape(nc, w)
    energies = (chunks.real ** 2 + chunks.imag ** 2).mean(axis=1)
    threshold = float(np.percentile(energies, 40))
    high = np.where(energies >= threshold)[0]
    if len(high) >= n:
        chosen = rng.choice(high, n, replace=False)
    else:
        low = np.where(energies < threshold)[0]
        extra = rng.choice(low, n - len(high), replace=False)
        chosen = np.concatenate([high, extra])
    chosen = np.sort(chosen)
    windows = chunks[chosen]
    m, s = windows.mean(axis=1, keepdims=True), windows.std(axis=1, keepdims=True) + 1e-9
    return ((windows - m) / s).astype(np.complex64), chosen * w


def spectral_features(X, n_bins=16):
    N, T = X.shape
    psd = np.abs(np.fft.fft(X, axis=1)) ** 2 / T
    psd = psd[:, :T // 2]
    edges = np.unique(np.geomspace(1, psd.shape[1], n_bins + 1).astype(int))
    bins = np.zeros((N, len(edges) - 1))
    for i in range(len(edges) - 1):
        bins[:, i] = psd[:, edges[i]:edges[i + 1]].mean(axis=1)
    return np.log10(bins + 1e-12)


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

# Task 2: DJI vs non-DJI
y_tr2 = (y_tr <= 1).astype(int)
y_te2 = (y_te <= 1).astype(int)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

print("Computing spectral baseline…", flush=True)
spec_tr, spec_te = spectral_features(X_tr), spectral_features(X_te)
clf = LogisticRegression(max_iter=2000).fit(spec_tr, y_tr2)
y_spec = clf.predict(spec_te)
spec_acc = accuracy_score(y_te2, y_spec)
spec_f1 = f1_score(y_te2, y_spec, average="macro")
print(f"  spectral-only: acc={spec_acc:.4f} f1={spec_f1:.4f}", flush=True)

print("\nPATP α-sweep with hybrid (PATP_DSGE + spectral)…", flush=True)
alphas = np.linspace(0, 1, 11)
results_per_alpha = []
for a in alphas:
    t0 = time.time()
    ext = ComplexDSGEFeatureExtractor(mode="re_im", basis="patp", n=3,
                                       alpha=float(a), ridge=0.01).fit(X_tr, y_tr2)
    dsge_tr = ext.transform(X_tr)
    dsge_te = ext.transform(X_te)
    hyb_tr = np.hstack([dsge_tr, spec_tr])
    hyb_te = np.hstack([dsge_te, spec_te])
    clf = LogisticRegression(max_iter=2000).fit(hyb_tr, y_tr2)
    y_hyb = clf.predict(hyb_te)
    acc = accuracy_score(y_te2, y_hyb)
    f1 = f1_score(y_te2, y_hyb, average="macro")
    bs = bootstrap_compare(y_te2, y_spec, y_hyb, R=500)
    sig_marker = " ★" if bs["significant"] else ""
    print(f"  α={a:.2f}  acc={acc:.4f}  Δ={bs['delta']:+.4f}  "
          f"p={bs['p_value']:.4f}{sig_marker}  ({time.time()-t0:.1f}s)", flush=True)
    results_per_alpha.append({
        "alpha": float(a),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "delta_vs_spectral": float(bs["delta"]),
        "p_value": float(bs["p_value"]),
        "ci_low": float(bs["ci_low"]),
        "ci_high": float(bs["ci_high"]),
        "significant": bool(bs["significant"]),
    })

# Find best α (by accuracy, then by Δ for tie)
best = max(results_per_alpha, key=lambda r: (r["accuracy"], r["delta_vs_spectral"]))
print(f"\nBest α = {best['alpha']:.2f}, accuracy = {best['accuracy']:.4f}, "
      f"Δ = {best['delta_vs_spectral']:+.4f}", flush=True)

# Boundary check on accuracy curve
accs = [r["accuracy"] for r in results_per_alpha]
boundary = patp_boundary_check(alphas.tolist(), accs)
print(f"Boundary check: {boundary['recommendation']}", flush=True)

results = {
    "task": "Task 2 — DJI vs non-DJI, complementarity-aware PATP α-sweep",
    "spectral_baseline": {"accuracy": float(spec_acc), "macro_f1": float(spec_f1)},
    "alphas": alphas.tolist(),
    "per_alpha": results_per_alpha,
    "best_alpha": best["alpha"],
    "best_accuracy": best["accuracy"],
    "boundary_check": boundary,
}
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}")
