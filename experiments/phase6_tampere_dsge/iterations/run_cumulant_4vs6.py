#!/usr/bin/env python3
"""Cumulant explicit features — order 4 vs order 6, 2.4 GHz vs 5.8 GHz.

Two questions:
    Q1. Чи cumulant_explicit працює також на 2.4 GHz?
        (раніше тестували тільки на 5.8 GHz — там 3/3 reproducible)
    Q2. Чи γ₅, γ₆ додають інформацію над γ₃, γ₄?

Two feature configurations:
    cumulant_order4: γ₃, γ₄ на Re/Im/|y|        — 6 features
    cumulant_order6: γ₃, γ₄, γ₅, γ₆ на Re/Im/|y| — 12 features

Tested on:
    Task 2 — DJI vs non-DJI, on each band
    3 seeds (42, 43, 44), bootstrap R=500, p<0.05 threshold

Reproducible = 2+/3 seeds significant.
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
from verify_dsge import bootstrap_compare

WIN = 4096
N_PER_CLASS = 1000
TRAIN_FRAC = 0.7
# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
OUT = Path(__file__).parent.parent / "results" / "phase6_cumulant_4vs6.json"

CLASSES_24 = [
    DATA_DIR / "DJI_mavic_pro_2G.bin",
    DATA_DIR / "DJI_inspire_2_2G.bin",
    DATA_DIR / "Parrot_disco_2G.bin",
    DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin",
]
CLASSES_58 = [
    DATA_DIR / "DJI_phantom_4_pro_plus_5G_1of2.bin",
    DATA_DIR / "DJI_inspire_2_5G_1of2.bin",
    DATA_DIR / "DJI_matrice_210_5G_1of2.bin",
    DATA_DIR / "Yuneec_typhoon_h_5G.bin",
]

BAND_CONFIG = {
    "2.4_GHz": {"paths": CLASSES_24, "fs_msps": 120, "task2_threshold": 1},  # 0,1=DJI Mavic+Inspire, 2,3=non-DJI
    "5.8_GHz": {"paths": CLASSES_58, "fs_msps": 200, "task2_threshold": 2},  # 0,1,2=DJI, 3=non-DJI
}


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


def standardized_cumulants(x_2d, max_order=6):
    """Standardized cumulants γ_k = κ_k / σ^k for k=3..max_order, vectorized.

    x_2d: (N, T) — N windows, T samples each.
    Returns: (N, max_order-2) array — γ₃, γ₄, [γ₅, γ₆]

    Cumulant formulas (Fisher k-statistics, central moments):
        κ₂ = m₂
        κ₃ = m₃
        κ₄ = m₄ - 3·m₂²
        κ₅ = m₅ - 10·m₂·m₃
        κ₆ = m₆ - 15·m₂·m₄ - 10·m₃² + 30·m₂³
    """
    x = x_2d.astype(np.float64)
    x_c = x - x.mean(axis=1, keepdims=True)
    m2 = np.mean(x_c ** 2, axis=1)
    m3 = np.mean(x_c ** 3, axis=1)
    m4 = np.mean(x_c ** 4, axis=1)
    sigma = np.sqrt(m2) + 1e-12

    cols = []
    cols.append(m3 / sigma ** 3)                          # γ₃
    cols.append((m4 - 3 * m2 ** 2) / sigma ** 4)          # γ₄
    if max_order >= 5:
        m5 = np.mean(x_c ** 5, axis=1)
        cols.append((m5 - 10 * m2 * m3) / sigma ** 5)     # γ₅
    if max_order >= 6:
        m6 = np.mean(x_c ** 6, axis=1)
        cols.append((m6 - 15 * m2 * m4 - 10 * m3 ** 2 + 30 * m2 ** 3) / sigma ** 6)  # γ₆
    out = np.stack(cols, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def cumulant_features(X, max_order):
    """Compute cumulants on Re, Im, |y| separately, concatenate."""
    re = X.real
    im = X.imag
    mag = np.abs(X)
    cols = [standardized_cumulants(re, max_order=max_order),
            standardized_cumulants(im, max_order=max_order),
            standardized_cumulants(mag, max_order=max_order)]
    return np.hstack(cols)


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
    return X[np.concatenate(tr_idx)], y[np.concatenate(tr_idx)], \
           X[np.concatenate(te_idx)], y[np.concatenate(te_idx)]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ---- Run experiment ----
results = {}
for band_name, cfg in BAND_CONFIG.items():
    print(f"\n=== {band_name} (fs={cfg['fs_msps']} MS/s) ===", flush=True)
    band_results = {"per_seed": []}
    for seed in (42, 43, 44):
        X_tr, y_tr, X_te, y_te = build_dataset(seed, cfg["paths"])
        thr = cfg["task2_threshold"]
        y_tr2 = (y_tr <= thr).astype(int)
        y_te2 = (y_te <= thr).astype(int)

        # Spectral baseline
        spec_tr, spec_te = spectral_features(X_tr), spectral_features(X_te)
        clf_s = LogisticRegression(max_iter=2000).fit(spec_tr, y_tr2)
        y_spec = clf_s.predict(spec_te)
        spec_acc = accuracy_score(y_te2, y_spec)

        seed_data = {"seed": seed, "spectral_acc": float(spec_acc), "configurations": {}}
        print(f"  --- seed {seed} (spec={spec_acc:.4f}) ---", flush=True)

        for cfg_name, max_order in (("cumulant_order4", 4), ("cumulant_order6", 6)):
            t0 = time.time()
            f_tr = cumulant_features(X_tr, max_order=max_order)
            f_te = cumulant_features(X_te, max_order=max_order)
            sc = StandardScaler().fit(f_tr)
            f_tr_s = sc.transform(f_tr)
            f_te_s = sc.transform(f_te)

            # Hybrid with spectral
            hyb_tr = np.hstack([f_tr_s, spec_tr])
            hyb_te = np.hstack([f_te_s, spec_te])
            clf = LogisticRegression(max_iter=2000).fit(hyb_tr, y_tr2)
            y_hyb = clf.predict(hyb_te)
            acc = accuracy_score(y_te2, y_hyb)
            bs = bootstrap_compare(y_te2, y_spec, y_hyb, R=500)
            sig = " ★" if bs["significant"] else ""
            seed_data["configurations"][cfg_name] = {
                "feature_dim_dsge": int(f_tr.shape[1]),
                "feature_dim_total": int(hyb_tr.shape[1]),
                "accuracy": float(acc),
                "delta": float(bs["delta"]),
                "p_value": float(bs["p_value"]),
                "ci_low": float(bs["ci_low"]),
                "ci_high": float(bs["ci_high"]),
                "significant": bool(bs["significant"]),
            }
            print(f"    {cfg_name}  ({f_tr.shape[1]} dims) "
                  f" acc={acc:.4f}  Δ={bs['delta']:+.4f}  p={bs['p_value']:.4f}{sig}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

            # Also test cumulants alone (without spectral) as sanity check
            clf_alone = LogisticRegression(max_iter=2000).fit(f_tr_s, y_tr2)
            y_alone = clf_alone.predict(f_te_s)
            seed_data["configurations"][cfg_name]["cumulants_only_accuracy"] = float(
                accuracy_score(y_te2, y_alone))

        band_results["per_seed"].append(seed_data)

    # Summary
    for cfg_name in ("cumulant_order4", "cumulant_order6"):
        deltas = [s["configurations"][cfg_name]["delta"] for s in band_results["per_seed"]]
        sig_count = sum(1 for s in band_results["per_seed"]
                        if s["configurations"][cfg_name]["significant"])
        accs_alone = [s["configurations"][cfg_name]["cumulants_only_accuracy"]
                      for s in band_results["per_seed"]]
        band_results.setdefault("summary", {})[cfg_name] = {
            "n_seeds_significant": sig_count,
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "reproducible_2of3": sig_count >= 2,
            "cumulants_only_acc_mean": float(np.mean(accs_alone)),
        }

    results[band_name] = band_results

# ---- Cross-config comparison: does γ₅, γ₆ add over γ₃, γ₄? ----
print("\n" + "=" * 70)
print("REPRODUCIBILITY ACROSS BANDS AND ORDERS")
print("=" * 70)
print(f"{'Band':<12}{'Config':<20}{'sig/3':>8}{'mean Δ':>10}{'std Δ':>10}{'cum-only acc':>15}")
print("-" * 75)
for band_name, band_data in results.items():
    for cfg_name in ("cumulant_order4", "cumulant_order6"):
        s = band_data["summary"][cfg_name]
        repro = "★" if s["reproducible_2of3"] else " "
        print(f"{band_name:<12}{cfg_name:<20}{s['n_seeds_significant']}/3 {repro:>4}"
              f"{s['mean_delta']:+10.4f}{s['std_delta']:>10.4f}"
              f"{s['cumulants_only_acc_mean']:>15.4f}")

# Order-4 vs order-6 head-to-head
print("\n" + "=" * 70)
print("DOES γ₅, γ₆ ADD OVER γ₃, γ₄?")
print("=" * 70)
for band_name, band_data in results.items():
    s4 = band_data["summary"]["cumulant_order4"]["mean_delta"]
    s6 = band_data["summary"]["cumulant_order6"]["mean_delta"]
    delta_increase = s6 - s4
    print(f"{band_name}: order4 Δ={s4:+.4f}  order6 Δ={s6:+.4f}  "
          f"increment from γ₅,γ₆: {delta_increase:+.4f}")

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}")
