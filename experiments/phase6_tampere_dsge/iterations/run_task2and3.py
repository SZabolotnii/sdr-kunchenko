#!/usr/bin/env python3
"""Task 2 (DJI vs non-DJI binary) + Task 3 (Mavic vs Inspire intra-OcuSync) +
hybrid mode + PATP α-sweep + bootstrap. Reuses load+windowing from task1."""
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
OUT = Path(__file__).parent.parent / "results" / "phase6_iter2_task23.json"
OUT.parent.mkdir(exist_ok=True)

CLASSES = [
    ("DJI_Mavic_Pro",   DATA_DIR / "DJI_mavic_pro_2G.bin"),
    ("DJI_Inspire_2",   DATA_DIR / "DJI_inspire_2_2G.bin"),
    ("Parrot_Disco",    DATA_DIR / "Parrot_disco_2G.bin"),
    ("Yuneec_Typhoon_H",DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin"),
]


def load_burst(path: Path, n_windows: int, win: int, rng):
    max_samples = n_windows * win * 2
    data = np.fromfile(path, dtype="<i2", count=2 * max_samples)
    iq = data.astype(np.float32).reshape(-1, 2)
    sig = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    n_chunks = len(sig) // win
    chunks = sig[:n_chunks * win].reshape(n_chunks, win)
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


def evaluate(X_tr, y_tr, X_te, y_te, basis, alpha=None):
    ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=basis, n=3,
                                       alpha=alpha, ridge=0.01).fit(X_tr, y_tr)
    feat_tr, feat_te = ext.transform(X_tr), ext.transform(X_te)
    clf = LogisticRegression(max_iter=2000).fit(feat_tr, y_tr)
    yp = clf.predict(feat_te)
    return {
        "accuracy": float(accuracy_score(y_te, yp)),
        "macro_f1": float(f1_score(y_te, yp, average="macro")),
        "predictions": yp,
        "feat_train": feat_tr,
        "feat_test": feat_te,
    }


results = {}

# ---- Task 2: DJI vs non-DJI ---
print("\n=== Task 2: DJI vs non-DJI binary ===", flush=True)
y_tr_dji = (y_tr <= 1).astype(int)
y_te_dji = (y_te <= 1).astype(int)

t2 = {"bases": {}}
for basis in ("polynomial", "fractional", "trigonometric", "robust"):
    t0 = time.time()
    r = evaluate(X_tr, y_tr_dji, X_te, y_te_dji, basis)
    t2["bases"][basis] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
    print(f"  {basis:<14} acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} ({time.time()-t0:.1f}s)", flush=True)

# Hybrid for task 2
best_basis = max(t2["bases"], key=lambda k: t2["bases"][k]["accuracy"])
spec_tr = spectral_features(X_tr); spec_te = spectral_features(X_te)
ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=best_basis, n=3, ridge=0.01).fit(X_tr, y_tr_dji)
dsge_tr, dsge_te = ext.transform(X_tr), ext.transform(X_te)

# Spectral-only
clf_s = LogisticRegression(max_iter=2000).fit(spec_tr, y_tr_dji)
y_spec = clf_s.predict(spec_te)
# Hybrid
hyb_tr = np.hstack([dsge_tr, spec_tr]); hyb_te = np.hstack([dsge_te, spec_te])
clf_h = LogisticRegression(max_iter=2000).fit(hyb_tr, y_tr_dji)
y_hyb = clf_h.predict(hyb_te)
t2["spectral_only"] = {"accuracy": float(accuracy_score(y_te_dji, y_spec)),
                       "macro_f1": float(f1_score(y_te_dji, y_spec, average="macro"))}
t2["hybrid"] = {"best_basis": best_basis, "feature_dim": int(hyb_tr.shape[1]),
                "accuracy": float(accuracy_score(y_te_dji, y_hyb)),
                "macro_f1": float(f1_score(y_te_dji, y_hyb, average="macro"))}
bs = bootstrap_compare(y_te_dji, y_spec, y_hyb, R=500)
t2["bootstrap_hybrid_vs_spectral"] = bs
print(f"  Spectral-only:  acc={t2['spectral_only']['accuracy']:.4f}", flush=True)
print(f"  Hybrid (DSGE[{best_basis}]+spec):  acc={t2['hybrid']['accuracy']:.4f}", flush=True)
print(f"  Bootstrap Δ={bs['delta']:+.4f} p={bs['p_value']:.4f} sig={bs['significant']}", flush=True)
results["task2"] = t2

# ---- Task 3: Mavic vs Inspire (intra-OcuSync) ---
print("\n=== Task 3: Mavic Pro vs Inspire 2 intra-OcuSync ===", flush=True)
m_tr = (y_tr <= 1); m_te = (y_te <= 1)
X_tr3 = X_tr[m_tr]; y_tr3 = y_tr[m_tr]
X_te3 = X_te[m_te]; y_te3 = y_te[m_te]

t3 = {"bases": {}}
for basis in ("polynomial", "fractional", "trigonometric", "robust"):
    t0 = time.time()
    r = evaluate(X_tr3, y_tr3, X_te3, y_te3, basis)
    t3["bases"][basis] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
    print(f"  {basis:<14} acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} ({time.time()-t0:.1f}s)", flush=True)

# PATP sweep on Task 3
print("  PATP α-sweep…", flush=True)
alphas = np.linspace(0, 1, 11); f1s = []
for a in alphas:
    r = evaluate(X_tr3, y_tr3, X_te3, y_te3, "patp", alpha=float(a))
    f1s.append(r["macro_f1"])
best_idx = int(np.argmax(f1s))
check = patp_boundary_check(alphas.tolist(), f1s)
t3["patp"] = {"alphas": alphas.tolist(), "macro_f1": f1s,
              "best_alpha": float(alphas[best_idx]), "best_f1": float(f1s[best_idx]),
              "boundary_check": check}
print(f"  PATP α_opt={alphas[best_idx]:.2f} F1={f1s[best_idx]:.4f}", flush=True)
print(f"    {check['recommendation']}", flush=True)

# Hybrid for Task 3
best_basis3 = max(t3["bases"], key=lambda k: t3["bases"][k]["accuracy"])
spec_tr3 = spectral_features(X_tr3); spec_te3 = spectral_features(X_te3)
ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=best_basis3, n=3, ridge=0.01).fit(X_tr3, y_tr3)
dsge_tr3, dsge_te3 = ext.transform(X_tr3), ext.transform(X_te3)
hyb_tr3 = np.hstack([dsge_tr3, spec_tr3]); hyb_te3 = np.hstack([dsge_te3, spec_te3])
clf_s = LogisticRegression(max_iter=2000).fit(spec_tr3, y_tr3)
y_spec3 = clf_s.predict(spec_te3)
clf_h = LogisticRegression(max_iter=2000).fit(hyb_tr3, y_tr3)
y_hyb3 = clf_h.predict(hyb_te3)
t3["spectral_only"] = {"accuracy": float(accuracy_score(y_te3, y_spec3)),
                        "macro_f1": float(f1_score(y_te3, y_spec3, average="macro"))}
t3["hybrid"] = {"best_basis": best_basis3, "feature_dim": int(hyb_tr3.shape[1]),
                "accuracy": float(accuracy_score(y_te3, y_hyb3)),
                "macro_f1": float(f1_score(y_te3, y_hyb3, average="macro"))}
bs3 = bootstrap_compare(y_te3, y_spec3, y_hyb3, R=500)
t3["bootstrap_hybrid_vs_spectral"] = bs3
print(f"  Spectral-only:  acc={t3['spectral_only']['accuracy']:.4f}", flush=True)
print(f"  Hybrid (DSGE[{best_basis3}]+spec):  acc={t3['hybrid']['accuracy']:.4f}", flush=True)
print(f"  Bootstrap Δ={bs3['delta']:+.4f} p={bs3['p_value']:.4f} sig={bs3['significant']}", flush=True)
results["task3"] = t3

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}")
