#!/usr/bin/env python3
"""Sanity-check 5.8 GHz findings.

Concerns to investigate:
    1. Class balance — Task 2 on 5.8 GHz is 3 DJI vs 1 Yuneec = 3:1.
       Spectral 0.9725 may be 0.75 (majority baseline) + small real gain.
    2. Task 3 — Inspire (OcuSync 1) vs Phantom+Matrice (OcuSync 2) =
       1:2 imbalance. Spectral 0.6689 ≈ 0.667 majority baseline → spectral
       basically does nothing.
    3. p=2.000 on frac+spec at α=0.0-0.5: predictions BIT-IDENTICAL to
       spectral-only. Why? Conditioning issue? Numeric collapse?
    4. Replication — re-run with seed=43, 44 to see if poly-vs-frac
       finding survives or is a single-seed artifact.

Output: console report + JSON with replicated metrics.
"""
from __future__ import annotations
import json
import os
import sys
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
OUT = Path(__file__).parent.parent / "results" / "phase6_5G_verification.json"


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


def majority_baseline_acc(y):
    """Accuracy of always predicting most frequent class."""
    classes, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


CLASSES_5G = [
    DATA_DIR / "DJI_phantom_4_pro_plus_5G_1of2.bin",  # 0 = OcuSync 2
    DATA_DIR / "DJI_inspire_2_5G_1of2.bin",            # 1 = OcuSync 1
    DATA_DIR / "DJI_matrice_210_5G_1of2.bin",          # 2 = OcuSync 2
    DATA_DIR / "Yuneec_typhoon_h_5G.bin",              # 3 = WiFi
]


def build_dataset(seed, paths):
    rng = np.random.default_rng(seed)
    X_list, y_list, t_list = [], [], []
    for c, path in enumerate(paths):
        w, ts = load_burst(path, N_PER_CLASS, WIN, rng)
        X_list.append(w); y_list.append(np.full(N_PER_CLASS, c)); t_list.append(ts)
    X = np.vstack(X_list); y = np.concatenate(y_list); t = np.concatenate(t_list)
    tr_idx, te_idx = [], []
    for c in range(len(paths)):
        ci = np.where(y == c)[0]
        co = ci[np.argsort(t[ci])]
        cut = int(TRAIN_FRAC * len(co))
        tr_idx.append(co[:cut]); te_idx.append(co[cut:])
    tr = np.concatenate(tr_idx); te = np.concatenate(te_idx)
    return X[tr], y[tr], X[te], y[te]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter


def evaluate(feat_tr, y_tr, feat_te, y_te):
    clf = LogisticRegression(max_iter=2000).fit(feat_tr, y_tr)
    yp = clf.predict(feat_te)
    return {
        "accuracy": float(accuracy_score(y_te, yp)),
        "macro_f1": float(f1_score(y_te, yp, average="macro")),
        "predictions": yp,
        "class_counts_predicted": dict(Counter(yp.tolist())),
    }


# ============== CHECK 1: Class balance on 5.8 GHz ==============
print("=" * 70)
print("CHECK 1: Class balance on 5.8 GHz tasks")
print("=" * 70, flush=True)

X_tr, y_tr, X_te, y_te = build_dataset(seed=42, paths=CLASSES_5G)
print(f"Test set 4-class counts: {dict(Counter(y_te.tolist()))}")

# Task 2: DJI vs non-DJI
y_te2 = (y_te <= 2).astype(int)
print(f"\nTask 2 (DJI vs Yuneec) test counts: {dict(Counter(y_te2.tolist()))}")
print(f"  Majority baseline accuracy: {majority_baseline_acc(y_te2):.4f}")
print(f"  Spectral reported acc:      0.9725")
print(f"  Boost over majority baseline: {0.9725 - majority_baseline_acc(y_te2):+.4f}")

# Task 3: Inspire vs (Phantom+Matrice)
m = y_te <= 2
y_te3 = (y_te[m] != 1).astype(int)
print(f"\nTask 3 (Inspire vs Phantom+Matrice) test counts: {dict(Counter(y_te3.tolist()))}")
print(f"  Majority baseline accuracy: {majority_baseline_acc(y_te3):.4f}")
print(f"  Spectral reported acc:      0.6689")
print(f"  Boost over majority baseline: {0.6689 - majority_baseline_acc(y_te3):+.4f}")

# ============== CHECK 2: Conditioning of fractional DSGE on 5.8 GHz ==============
print("\n" + "=" * 70)
print("CHECK 2: Conditioning of fractional vs polynomial DSGE on 5.8 GHz")
print("=" * 70, flush=True)

for basis in ("polynomial", "fractional", "trigonometric"):
    ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=basis, n=3, ridge=0.01).fit(X_tr, y_tr)
    cond = ext.conditioning_report
    flat_cond = []
    for v in cond.values():
        if isinstance(v, list):
            flat_cond.extend(v)
        else:
            flat_cond.append(v)
    feat = ext.transform(X_te)
    print(f"  {basis:<14}: cond max={max(flat_cond):.0f}  feat std (per col): "
          f"min={feat.std(axis=0).min():.4e} max={feat.std(axis=0).max():.4e}")

# ============== CHECK 3: Identical predictions investigation ==============
print("\n" + "=" * 70)
print("CHECK 3: Why frac+spec gives bit-identical predictions to spec-only?")
print("=" * 70, flush=True)

spec_tr, spec_te = spectral_features(X_tr), spectral_features(X_te)
y_tr2 = (y_tr <= 2).astype(int)

# Spectral alone
clf_s = LogisticRegression(max_iter=2000).fit(spec_tr, y_tr2)
y_spec = clf_s.predict(spec_te)
print(f"  spectral-only:         acc={accuracy_score(y_te2, y_spec):.4f}")
print(f"  predicted class counts: {dict(Counter(y_spec.tolist()))}")
print(f"  → if predicting majority everywhere: would be {majority_baseline_acc(y_te2):.4f}")

# Frac DSGE alone (NOT hybrid)
ext_frac = ComplexDSGEFeatureExtractor(mode="re_im", basis="fractional", n=3,
                                        ridge=0.01).fit(X_tr, y_tr2)
frac_tr, frac_te = ext_frac.transform(X_tr), ext_frac.transform(X_te)
clf_f = LogisticRegression(max_iter=2000).fit(frac_tr, y_tr2)
y_frac = clf_f.predict(frac_te)
print(f"\n  fractional-only DSGE:  acc={accuracy_score(y_te2, y_frac):.4f}")
print(f"  predicted class counts: {dict(Counter(y_frac.tolist()))}")

# Hybrid frac+spec
hyb_tr = np.hstack([frac_tr, spec_tr])
hyb_te = np.hstack([frac_te, spec_te])
clf_h = LogisticRegression(max_iter=2000).fit(hyb_tr, y_tr2)
y_hyb = clf_h.predict(hyb_te)
print(f"\n  hybrid (frac+spec):    acc={accuracy_score(y_te2, y_hyb):.4f}")
print(f"  predicted class counts: {dict(Counter(y_hyb.tolist()))}")
print(f"  predictions identical to spectral? {np.array_equal(y_spec, y_hyb)}")

# Examine logreg weights — does frac get effectively zero weight?
print(f"\n  LogReg coef magnitudes (hybrid):")
print(f"    frac DSGE features (4 dims): {np.abs(clf_h.coef_[0, :4]).round(3)}")
print(f"    spectral features (16 dims): max={np.abs(clf_h.coef_[0, 4:]).max():.3f}")

# ============== CHECK 4: Replication with seed 43, 44 ==============
print("\n" + "=" * 70)
print("CHECK 4: Replication of poly-vs-frac complementarity on 5.8 GHz")
print("=" * 70, flush=True)

verification_results = {
    "task_2_balance": {
        "test_counts": dict(Counter(y_te2.tolist())),
        "majority_baseline_acc": majority_baseline_acc(y_te2),
    },
    "task_3_balance": {
        "test_counts": dict(Counter(y_te3.tolist())),
        "majority_baseline_acc": majority_baseline_acc(y_te3),
    },
    "replications": [],
}

for seed in (42, 43, 44):
    print(f"\n--- seed {seed} ---")
    X_tr, y_tr, X_te, y_te = build_dataset(seed=seed, paths=CLASSES_5G)
    y_tr2 = (y_tr <= 2).astype(int)
    y_te2 = (y_te <= 2).astype(int)
    spec_tr, spec_te = spectral_features(X_tr), spectral_features(X_te)
    clf_s = LogisticRegression(max_iter=2000).fit(spec_tr, y_tr2)
    y_spec = clf_s.predict(spec_te)
    spec_acc = accuracy_score(y_te2, y_spec)

    rep = {"seed": seed, "spectral_acc": float(spec_acc), "variants": {}}
    for basis in ("polynomial", "fractional"):
        ext = ComplexDSGEFeatureExtractor(mode="re_im", basis=basis, n=3,
                                           ridge=0.01).fit(X_tr, y_tr2)
        d_tr, d_te = ext.transform(X_tr), ext.transform(X_te)
        hyb_tr = np.hstack([d_tr, spec_tr])
        hyb_te = np.hstack([d_te, spec_te])
        clf = LogisticRegression(max_iter=2000).fit(hyb_tr, y_tr2)
        y_h = clf.predict(hyb_te)
        bs = bootstrap_compare(y_te2, y_spec, y_h, R=500)
        sig = " ★" if bs["significant"] else ""
        rep["variants"][f"{basis}+spec"] = {
            "accuracy": float(accuracy_score(y_te2, y_h)),
            "delta": float(bs["delta"]),
            "p_value": float(bs["p_value"]),
            "significant": bool(bs["significant"]),
        }
        print(f"    {basis}+spec  acc={accuracy_score(y_te2, y_h):.4f}  "
              f"Δ={bs['delta']:+.4f}  p={bs['p_value']:.4f}{sig}")
    verification_results["replications"].append(rep)

# ============== Summary ==============
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Across seeds, count: how often poly+spec significant, frac+spec significant?
poly_sig_count = sum(1 for r in verification_results["replications"]
                     if r["variants"]["polynomial+spec"]["significant"])
frac_sig_count = sum(1 for r in verification_results["replications"]
                     if r["variants"]["fractional+spec"]["significant"])
print(f"Across 3 seeds:")
print(f"  poly+spec significant: {poly_sig_count}/3")
print(f"  frac+spec significant: {frac_sig_count}/3")

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump(verification_results, f, indent=2)
print(f"\nSaved verification results: {OUT}")
