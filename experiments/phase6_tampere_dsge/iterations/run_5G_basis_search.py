#!/usr/bin/env python3
"""Extended basis search for 5.8 GHz Task 2 (DJI vs non-DJI).

Standard 4 bases (poly/frac/trig/robust) gave 1/3 seeds significant
with <1pp effect — not reproducible. Try wider basis families with
3-seed replication built-in:

    1. Hermite polynomials — H_2(x)=x²-1, H_3(x)=x³-3x, H_4(x)=x⁴-6x²+3.
       Orthogonal to Gaussian → captures deviation from Gaussianity.
    2. Mixed-exponent fractional — sign(x)|x|^p for p ∈ {0.25, 0.5, 0.75,
       1.5, 2.5} — exponents NOT in standard PATP grid.
    3. Explicit cumulant features — γ₃, γ₄, γ₅, γ₆ computed directly per
       window (not via DSGE reconstruction). Direct moment-based.
    4. Envelope features — statistics of |y(t)| time series:
       std, kurtosis, peak-to-mean, autocorrelation lag-1.
    5. Phase-derivative features — instantaneous frequency variation.

For each candidate, hybrid (candidate + spectral) is bootstrap-tested
against spectral-only on seeds 42, 43, 44.
A finding is "reproducible" only if 2+/3 seeds significant.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis

# Vendored DSGE toolkit modules live next to this file (see toolkit/README.md).
_TOOLKIT = Path(__file__).resolve().parent / "toolkit"
if not _TOOLKIT.is_dir():                      # scripts under iterations/
    _TOOLKIT = Path(__file__).resolve().parent.parent / "toolkit"
sys.path.insert(0, str(_TOOLKIT))
from complex_features import ComplexDSGEFeatureExtractor
from kunchenko_features import DSGEFeatureExtractor
from verify_dsge import bootstrap_compare

WIN = 4096
N_PER_CLASS = 1000
TRAIN_FRAC = 0.7
# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
OUT = Path(__file__).parent.parent / "results" / "phase6_5G_basis_search.json"

CLASSES = [
    DATA_DIR / "DJI_phantom_4_pro_plus_5G_1of2.bin",
    DATA_DIR / "DJI_inspire_2_5G_1of2.bin",
    DATA_DIR / "DJI_matrice_210_5G_1of2.bin",
    DATA_DIR / "Yuneec_typhoon_h_5G.bin",
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


# ---------- Custom feature extractors ----------------------------

def cumulant_features(X: np.ndarray) -> np.ndarray:
    """Direct moment-based features per window — no DSGE reconstruction.

    Computes γ₃ (skew), γ₄ (excess kurtosis) for Re, Im, |·|. 6 features.
    """
    re, im, mag = X.real, X.imag, np.abs(X)
    out = np.column_stack([
        skew(re, axis=1, bias=False),
        kurtosis(re, axis=1, bias=False),
        skew(im, axis=1, bias=False),
        kurtosis(im, axis=1, bias=False),
        skew(mag, axis=1, bias=False),
        kurtosis(mag, axis=1, bias=False),
    ])
    # Replace any NaN/inf with 0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def envelope_features(X: np.ndarray) -> np.ndarray:
    """Statistics of |y(t)| time-series — temporal envelope shape.

    Per window: std(|y|), peak/mean(|y|), autocorr(|y|, lag=1),
                kurtosis(|y|), entropy of |y| histogram.
    """
    mag = np.abs(X)
    std_mag = mag.std(axis=1)
    mean_mag = mag.mean(axis=1) + 1e-9
    peak_mag = mag.max(axis=1)
    p2m = peak_mag / mean_mag
    # Lag-1 autocorrelation
    mag_centered = mag - mag.mean(axis=1, keepdims=True)
    ac1 = np.sum(mag_centered[:, :-1] * mag_centered[:, 1:], axis=1) / (
        np.sum(mag_centered ** 2, axis=1) + 1e-12)
    kurt_mag = kurtosis(mag, axis=1, bias=False)
    # Energy entropy via 16-bin histogram
    h_entropy = []
    for row in mag:
        hist, _ = np.histogram(row, bins=16, density=False)
        hist = hist[hist > 0] / hist.sum()
        h_entropy.append(-np.sum(hist * np.log(hist + 1e-12)))
    h_entropy = np.array(h_entropy)
    out = np.column_stack([std_mag, p2m, ac1, kurt_mag, h_entropy])
    return np.nan_to_num(out)


def phase_derivative_features(X: np.ndarray) -> np.ndarray:
    """Instantaneous frequency variation via d(arg(y))/dn.

    Per window: mean, std, skew, kurt of unwrapped phase derivative.
    """
    phase = np.unwrap(np.angle(X), axis=1)
    dphi = np.diff(phase, axis=1)
    out = np.column_stack([
        dphi.mean(axis=1),
        dphi.std(axis=1),
        skew(dphi, axis=1, bias=False),
        kurtosis(dphi, axis=1, bias=False),
    ])
    return np.nan_to_num(out)


# ---------- Custom DSGE bases via dsge-toolkit ----------------------

# Using a custom basis function in DSGEFeatureExtractor by registering it.
# Workaround: construct features by hand using KunchenkoReconstructor on
# magnitude/phase branches with custom phi.

# But simpler: compute features directly via numpy lambdas, then run
# per-class log-MSED a la the toolkit. Use DSGEFeatureExtractor on
# real arrays |X| where we apply custom phi via monkey-patch.

# Easier approach: project complex X to a real-valued vector V via
# basis-aware aggregation, then feed V to the toolkit's DSGEFeatureExtractor
# (real-valued branch). This is a separate experiment from the I/Q-aware
# ComplexDSGEFeatureExtractor.

def hermite_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """DSGE features with Hermite-inspired basis.

    Direct moment-style features per window:
        f_1 = mean(H_2(x)) where H_2(x)=x²-1
        f_2 = mean(H_3(x)) where H_3(x)=x³-3x
        f_3 = mean(H_4(x)) where H_4(x)=x⁴-6x²+3
    Done separately on Re and Im branches: 6 features total.
    """
    re, im = X.real, X.imag
    h2 = lambda x: x ** 2 - 1
    h3 = lambda x: x ** 3 - 3 * x
    h4 = lambda x: x ** 4 - 6 * x ** 2 + 3
    out = np.column_stack([
        h2(re).mean(axis=1),
        h3(re).mean(axis=1),
        h4(re).mean(axis=1),
        h2(im).mean(axis=1),
        h3(im).mean(axis=1),
        h4(im).mean(axis=1),
    ])
    return np.nan_to_num(out)


def mixed_exp_features(X: np.ndarray, exponents=(0.25, 0.75, 1.5, 2.5)) -> np.ndarray:
    """Mean of sign(x)·|x|^p for various exponents on Re, Im.

    Captures non-Gaussian moments at exponents NOT covered by PATP standard grid.
    """
    re, im = X.real, X.imag
    out = []
    for p in exponents:
        out.append(np.mean(np.sign(re) * np.abs(re) ** p, axis=1))
        out.append(np.mean(np.sign(im) * np.abs(im) ** p, axis=1))
    return np.column_stack(out)


# ---------- Main loop ----------------------------------------------

def build_dataset(seed):
    rng = np.random.default_rng(seed)
    X_list, y_list, t_list = [], [], []
    for c, path in enumerate(CLASSES):
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
    return X[tr], y[tr], X[te], y[te]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


CANDIDATES = {
    "cumulant_explicit": (cumulant_features, "γ₃, γ₄ on Re/Im/|·| — 6 features"),
    "envelope_stats":    (envelope_features, "Statistics of |y(t)| — 5 features"),
    "phase_derivative":  (phase_derivative_features, "d(arg)/dn statistics — 4 features"),
    "hermite_orth":      (lambda X: hermite_features(X, None), "H_2, H_3, H_4 means on Re/Im — 6 features"),
    "mixed_exponents":   (mixed_exp_features, "sign(x)|x|^p, p∈{0.25,0.75,1.5,2.5} — 8 features"),
}

print(f"Search space: {len(CANDIDATES)} candidate feature sets, 3 seeds × bootstrap R=500")
print(f"  Reproducibility threshold: 2/3 seeds significant\n", flush=True)

results = {"candidates": {name: {"description": desc, "per_seed": []}
                         for name, (_, desc) in CANDIDATES.items()},
           "spectral_baselines": {}}

for seed in (42, 43, 44):
    print(f"=== seed {seed} ===", flush=True)
    X_tr, y_tr, X_te, y_te = build_dataset(seed)
    y_tr2 = (y_tr <= 2).astype(int)
    y_te2 = (y_te <= 2).astype(int)

    # spectral baseline
    spec_tr = spectral_features(X_tr)
    spec_te = spectral_features(X_te)
    clf_s = LogisticRegression(max_iter=2000).fit(spec_tr, y_tr2)
    y_spec = clf_s.predict(spec_te)
    spec_acc = accuracy_score(y_te2, y_spec)
    results["spectral_baselines"][seed] = float(spec_acc)
    print(f"  spectral-only acc={spec_acc:.4f}", flush=True)

    for name, (fn, _) in CANDIDATES.items():
        t0 = time.time()
        # Compute new features
        f_tr = fn(X_tr)
        f_te = fn(X_te)
        # Standardize
        scaler = StandardScaler().fit(f_tr)
        f_tr_s, f_te_s = scaler.transform(f_tr), scaler.transform(f_te)
        # Hybrid with spectral
        hyb_tr = np.hstack([f_tr_s, spec_tr])
        hyb_te = np.hstack([f_te_s, spec_te])
        clf = LogisticRegression(max_iter=2000).fit(hyb_tr, y_tr2)
        y_hyb = clf.predict(hyb_te)
        acc = accuracy_score(y_te2, y_hyb)
        bs = bootstrap_compare(y_te2, y_spec, y_hyb, R=500)
        sig_marker = "★" if bs["significant"] else " "
        results["candidates"][name]["per_seed"].append({
            "seed": seed,
            "accuracy": float(acc),
            "delta": float(bs["delta"]),
            "p_value": float(bs["p_value"]),
            "ci_low": float(bs["ci_low"]),
            "ci_high": float(bs["ci_high"]),
            "significant": bool(bs["significant"]),
        })
        print(f"  {name:<22} acc={acc:.4f} Δ={bs['delta']:+.4f} "
              f"p={bs['p_value']:.4f} {sig_marker}  ({time.time()-t0:.1f}s)", flush=True)
    print(flush=True)

# ---------- Summary across seeds ----------
print("=" * 70)
print("REPRODUCIBILITY SUMMARY")
print("=" * 70)
for name, info in results["candidates"].items():
    n_sig = sum(1 for r in info["per_seed"] if r["significant"])
    deltas = [r["delta"] for r in info["per_seed"]]
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    accs = [r["accuracy"] for r in info["per_seed"]]
    print(f"  {name:<22} {n_sig}/3 sig — Δ mean={mean_delta:+.4f} std={std_delta:.4f}  "
          f"acc range=[{min(accs):.4f}, {max(accs):.4f}]")
    info["summary"] = {
        "n_seeds_significant": n_sig,
        "mean_delta": float(mean_delta),
        "std_delta": float(std_delta),
        "accuracy_range": [float(min(accs)), float(max(accs))],
        "reproducible_2of3": n_sig >= 2,
    }

# Best reproducible candidate
reproducible = {n: info for n, info in results["candidates"].items()
                if info["summary"]["reproducible_2of3"]}
print(f"\nReproducible candidates (2+/3 seeds significant): {len(reproducible)}")
for name in reproducible:
    print(f"  • {name}")

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}")
