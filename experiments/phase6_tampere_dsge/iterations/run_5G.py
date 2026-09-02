#!/usr/bin/env python3
"""Phase 6 — 5.8 GHz band experiment.

Different drones, different sample rate (200 MS/s instead of 120),
different physical band. Test whether the complementarity finding from
2.4 GHz reproduces, or whether the higher sample rate / different
protocol mix (Phantom 4 Pro+, Inspire 2, Matrice 210, Yuneec)
changes the picture.

Files (each 400 MB = 50M complex samples = 250 ms at 200 MS/s):
    DJI_phantom_4_pro_plus_5G_1of2.bin (OcuSync 2)
    DJI_inspire_2_5G_1of2.bin           (OcuSync 1)
    DJI_matrice_210_5G_1of2.bin         (OcuSync 2)
    Yuneec_typhoon_h_5G.bin             (Wi-Fi based)

Three tasks:
    Task 1: 4-class multi-protocol
    Task 2: DJI vs non-DJI (3 DJI vs 1 Yuneec)
    Task 3: OcuSync 1 vs OcuSync 2 (Inspire vs Phantom+Matrice)
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
FS = 200e6
# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
OUT = Path(__file__).parent.parent / "results" / "phase6_5G_results.json"

CLASSES = [
    ("DJI_Phantom_4_Pro+",  DATA_DIR / "DJI_phantom_4_pro_plus_5G_1of2.bin", "OcuSync_2"),
    ("DJI_Inspire_2",        DATA_DIR / "DJI_inspire_2_5G_1of2.bin",         "OcuSync_1"),
    ("DJI_Matrice_210",      DATA_DIR / "DJI_matrice_210_5G_1of2.bin",       "OcuSync_2"),
    ("Yuneec_Typhoon_H",     DATA_DIR / "Yuneec_typhoon_h_5G.bin",           "WiFi_based"),
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


def dsge_features(X_tr, y_tr, X_te, basis, alpha=None):
    ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=basis, n=3,
                                       alpha=alpha, ridge=0.01).fit(X_tr, y_tr)
    return ext.transform(X_tr), ext.transform(X_te)


print(f"Loading {len(CLASSES)} files (5.8 GHz, fs={FS/1e6:.0f} MS/s)…", flush=True)
rng = np.random.default_rng(42)
X_list, y_list, t_list = [], [], []
for c, (name, path, proto) in enumerate(CLASSES):
    t0 = time.time()
    w, ts = load_burst(path, N_PER_CLASS, WIN, rng)
    print(f"  [{c}] {name} ({proto}): {w.shape} ({time.time()-t0:.1f}s)", flush=True)
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
print(f"Train {X_tr.shape}, Test {X_te.shape}\n", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate(feat_tr, feat_te, y_tr, y_te):
    clf = LogisticRegression(max_iter=2000).fit(feat_tr, y_tr)
    yp = clf.predict(feat_te)
    return {
        "accuracy": float(accuracy_score(y_te, yp)),
        "macro_f1": float(f1_score(y_te, yp, average="macro")),
        "predictions": yp,
    }


def run_battery(X_tr, y_tr, X_te, y_te, label):
    print(f"=== {label} ===", flush=True)
    print(f"  Train {X_tr.shape}, Test {X_te.shape}, classes {sorted(set(y_tr))}", flush=True)

    spec_tr, spec_te = spectral_features(X_tr), spectral_features(X_te)
    base = evaluate(spec_tr, spec_te, y_tr, y_te)
    print(f"  spectral-only         acc={base['accuracy']:.4f}", flush=True)

    results = {"spectral": {k: v for k, v in base.items() if k != "predictions"}, "bases": {},
               "hybrid_variants": {}}
    dsge_cache = {}

    for basis in ("polynomial", "fractional", "trigonometric", "robust"):
        t0 = time.time()
        d_tr, d_te = dsge_features(X_tr, y_tr, X_te, basis)
        dsge_cache[basis] = (d_tr, d_te)
        r = evaluate(d_tr, d_te, y_tr, y_te)
        results["bases"][basis] = {k: v for k, v in r.items() if k != "predictions"}
        print(f"  {basis:<14} DSGE-only acc={r['accuracy']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    # Hybrid variants
    hybrid_specs = {
        "poly + spec":         [dsge_cache["polynomial"], spec_tr, spec_te],
        "frac + spec":         [dsge_cache["fractional"], spec_tr, spec_te],
        "trig + spec":         [dsge_cache["trigonometric"], spec_tr, spec_te],
        "all 4 bases + spec":  None,  # special
    }
    for name, parts in hybrid_specs.items():
        if name == "all 4 bases + spec":
            tr_concat = np.hstack([dsge_cache[b][0] for b in
                                    ("polynomial", "fractional", "trigonometric", "robust")] + [spec_tr])
            te_concat = np.hstack([dsge_cache[b][1] for b in
                                    ("polynomial", "fractional", "trigonometric", "robust")] + [spec_te])
        else:
            (d_tr, d_te), s_tr, s_te = parts
            tr_concat = np.hstack([d_tr, s_tr])
            te_concat = np.hstack([d_te, s_te])
        r = evaluate(tr_concat, te_concat, y_tr, y_te)
        bs = bootstrap_compare(y_te, base["predictions"], r["predictions"], R=500)
        sig = " ★" if bs["significant"] else ""
        results["hybrid_variants"][name] = {
            "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
            "feature_dim": int(tr_concat.shape[1]),
            "delta": float(bs["delta"]), "p_value": float(bs["p_value"]),
            "significant": bool(bs["significant"]),
        }
        print(f"  {name:<22} acc={r['accuracy']:.4f}  Δ={bs['delta']:+.4f}  p={bs['p_value']:.4f}{sig}", flush=True)

    return results


all_results = {"config": {"window_samples": WIN, "windows_per_class": N_PER_CLASS,
                           "fs_hz": FS, "band": "5.8_GHz", "seed": 42}}

# Task 1: 4-class
all_results["task1_4class"] = run_battery(X_tr, y_tr, X_te, y_te,
                                           label="Task 1 — 4-class multi-protocol (5.8 GHz)")
print()

# Task 2: DJI (0,1,2) vs non-DJI (3)
y_tr_dji = (y_tr <= 2).astype(int)
y_te_dji = (y_te <= 2).astype(int)
all_results["task2_dji_vs_nondji"] = run_battery(
    X_tr, y_tr_dji, X_te, y_te_dji,
    label="Task 2 — DJI (3) vs non-DJI (1)"
)
print()

# Task 3: OcuSync 1 (Inspire) vs OcuSync 2 (Phantom+Matrice)
# Re-map: y=1 (Inspire OcuSync 1) -> 0, y=0 or y=2 (OcuSync 2) -> 1; drop Yuneec
ocu_mask_tr = y_tr <= 2
ocu_mask_te = y_te <= 2
y_tr_ocu = (y_tr[ocu_mask_tr] != 1).astype(int)  # 0=OcuSync1 (Inspire), 1=OcuSync2 (Phantom+Matrice)
y_te_ocu = (y_te[ocu_mask_te] != 1).astype(int)
all_results["task3_ocusync1_vs_2"] = run_battery(
    X_tr[ocu_mask_tr], y_tr_ocu, X_te[ocu_mask_te], y_te_ocu,
    label="Task 3 — OcuSync 1 (Inspire) vs OcuSync 2 (Phantom+Matrice)"
)

with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved {OUT}")
