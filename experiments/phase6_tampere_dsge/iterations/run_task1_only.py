#!/usr/bin/env python3
"""Standalone runner for Task 1 only — fast feedback."""
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

WIN = 4096
N_PER_CLASS = 1000
TRAIN_FRAC = 0.7
# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
OUT = Path(__file__).parent.parent / "results" / "phase6_iter2_task1.json"
OUT.parent.mkdir(exist_ok=True)

CLASSES = [
    ("DJI_Mavic_Pro",   DATA_DIR / "DJI_mavic_pro_2G.bin"),
    ("DJI_Inspire_2",   DATA_DIR / "DJI_inspire_2_2G.bin"),
    ("Parrot_Disco",    DATA_DIR / "Parrot_disco_2G.bin"),
    ("Yuneec_Typhoon_H",DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin"),
]


def load_burst(path: Path, n_windows: int, win: int, rng) -> np.ndarray:
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


print("Loading…", flush=True)
rng = np.random.default_rng(42)
X_list, y_list, t_list = [], [], []
for c, (name, path) in enumerate(CLASSES):
    t0 = time.time()
    w, ts = load_burst(path, N_PER_CLASS, WIN, rng)
    print(f"  [{c}] {name}: {w.shape} ({time.time()-t0:.1f}s)", flush=True)
    X_list.append(w); y_list.append(np.full(N_PER_CLASS, c)); t_list.append(ts)

X = np.vstack(X_list); y = np.concatenate(y_list); t = np.concatenate(t_list)

# Time-ordered per-class split
tr_idx, te_idx = [], []
for c in range(len(CLASSES)):
    cm = y == c
    ci = np.where(cm)[0]
    co = ci[np.argsort(t[ci])]
    cut = int(TRAIN_FRAC * len(co))
    tr_idx.append(co[:cut]); te_idx.append(co[cut:])
tr = np.concatenate(tr_idx); te = np.concatenate(te_idx)
X_tr, X_te = X[tr], X[te]; y_tr, y_te = y[tr], y[te]
print(f"Train {X_tr.shape}, Test {X_te.shape}", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

results = {"task": "Task 1 — 4-class multi-protocol", "bases": {}}
for basis in ("polynomial", "fractional", "trigonometric", "robust"):
    t0 = time.time()
    ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=basis, n=3, ridge=0.01).fit(X_tr, y_tr)
    feat_tr = ext.transform(X_tr)
    feat_te = ext.transform(X_te)
    clf = LogisticRegression(max_iter=2000).fit(feat_tr, y_tr)
    yp = clf.predict(feat_te)
    acc = accuracy_score(y_te, yp)
    f1 = f1_score(y_te, yp, average="macro")
    cond = ext.conditioning_report
    cond_max = max(max(v) if isinstance(v, list) else v for v in cond.values())
    results["bases"][basis] = {"accuracy": float(acc), "macro_f1": float(f1),
                                "cond_F_max": float(cond_max),
                                "duration_sec": float(time.time() - t0)}
    print(f"  {basis:<14} acc={acc:.4f} f1={f1:.4f} cond_max={cond_max:.0f} ({time.time()-t0:.1f}s)", flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT}")
