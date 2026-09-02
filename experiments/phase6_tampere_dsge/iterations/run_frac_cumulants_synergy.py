#!/usr/bin/env python3
"""Final synergy test: frac DSGE + cumulant_order6 hybrid.

Hypothesis (from manuscript §3.5.2 і §3.6.5):
    На 2.4 GHz frac DSGE-reconstruction acts as smoothed cumulant estimator
    (averaging benefit на noisy per-window cumulants). Тому combined
    frac+cumulants+spec має бути ≈ cumulants+spec (redundant), not synergy.

Falsifiable test:
    1. spec only baseline
    2. frac DSGE + spec        — captures cumulants implicitly via reconstruction
    3. cumulant_order6 + spec  — captures cumulants explicitly via formulas
    4. frac + cumulant + spec  — synergy test: чи додавання обох дає більше?

Predictions:
    На 2.4 GHz: 4 ≈ 2 ≈ 3 (~99.83%) — equivalence (mechanism hypothesis CORRECT)
    На 5.8 GHz: 4 ≈ 3 (~98.4%), 2 ≈ baseline (frac degenerate) — also expected

Якщо 4 значимо > max(2, 3): equivalence hypothesis FALSIFIED — frac contributes
unique information beyond cumulants. Це change manuscript naратив.

3 seeds × bootstrap R=500.
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
OUT = Path(__file__).parent.parent / "results" / "phase6_frac_cumulants_synergy.json"

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
    "2.4_GHz": {"paths": CLASSES_24, "task2_threshold": 1},
    "5.8_GHz": {"paths": CLASSES_58, "task2_threshold": 2},
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
    x = x_2d.astype(np.float64)
    x_c = x - x.mean(axis=1, keepdims=True)
    m2 = np.mean(x_c ** 2, axis=1)
    m3 = np.mean(x_c ** 3, axis=1)
    m4 = np.mean(x_c ** 4, axis=1)
    sigma = np.sqrt(m2) + 1e-12
    cols = [m3 / sigma ** 3,
            (m4 - 3 * m2 ** 2) / sigma ** 4]
    if max_order >= 5:
        m5 = np.mean(x_c ** 5, axis=1)
        cols.append((m5 - 10 * m2 * m3) / sigma ** 5)
    if max_order >= 6:
        m6 = np.mean(x_c ** 6, axis=1)
        cols.append((m6 - 15 * m2 * m4 - 10 * m3 ** 2 + 30 * m2 ** 3) / sigma ** 6)
    return np.nan_to_num(np.stack(cols, axis=1))


def cumulant_features(X, max_order=6):
    return np.hstack([
        standardized_cumulants(X.real, max_order=max_order),
        standardized_cumulants(X.imag, max_order=max_order),
        standardized_cumulants(np.abs(X), max_order=max_order),
    ])


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


results = {"hypothesis": "frac DSGE + cumulant_order6 ≈ cumulants only (mechanism redundancy)"}

for band_name, cfg in BAND_CONFIG.items():
    print(f"\n=== {band_name} ===", flush=True)
    band_data = {"per_seed": []}

    for seed in (42, 43, 44):
        print(f"  --- seed {seed} ---", flush=True)
        X_tr, y_tr, X_te, y_te = build_dataset(seed, cfg["paths"])
        thr = cfg["task2_threshold"]
        y_tr2 = (y_tr <= thr).astype(int)
        y_te2 = (y_te <= thr).astype(int)

        # Compute features once per seed
        spec_tr, spec_te = spectral_features(X_tr), spectral_features(X_te)

        # Frac DSGE
        ext_frac = ComplexDSGEFeatureExtractor(mode="re_im", basis="fractional",
                                                n=3, ridge=0.01).fit(X_tr, y_tr2)
        frac_tr = ext_frac.transform(X_tr); frac_te = ext_frac.transform(X_te)

        # Cumulants
        cum_tr = cumulant_features(X_tr, max_order=6)
        cum_te = cumulant_features(X_te, max_order=6)
        sc = StandardScaler().fit(cum_tr)
        cum_tr_s = sc.transform(cum_tr); cum_te_s = sc.transform(cum_te)

        # Configurations
        configs = {
            "spec_only":          (spec_tr, spec_te),
            "frac+spec":          (np.hstack([frac_tr, spec_tr]),
                                    np.hstack([frac_te, spec_te])),
            "cum6+spec":          (np.hstack([cum_tr_s, spec_tr]),
                                    np.hstack([cum_te_s, spec_te])),
            "frac+cum6+spec":     (np.hstack([frac_tr, cum_tr_s, spec_tr]),
                                    np.hstack([frac_te, cum_te_s, spec_te])),
        }

        # Train classifiers
        preds = {}
        accs = {}
        for cfg_name, (f_tr, f_te) in configs.items():
            clf = LogisticRegression(max_iter=2000).fit(f_tr, y_tr2)
            yp = clf.predict(f_te)
            preds[cfg_name] = yp
            accs[cfg_name] = float(accuracy_score(y_te2, yp))

        # Compute deltas vs spec, and synergy delta vs each component
        seed_data = {"seed": seed, "accuracies": accs, "feature_dims": {
            k: int(v[0].shape[1]) for k, v in configs.items()
        }, "deltas_vs_spec": {}, "synergy_check": {}}

        # vs spec baseline
        y_spec = preds["spec_only"]
        for cfg_name in ("frac+spec", "cum6+spec", "frac+cum6+spec"):
            bs = bootstrap_compare(y_te2, y_spec, preds[cfg_name], R=500)
            seed_data["deltas_vs_spec"][cfg_name] = {
                "delta": float(bs["delta"]), "p_value": float(bs["p_value"]),
                "ci_low": float(bs["ci_low"]), "ci_high": float(bs["ci_high"]),
                "significant": bool(bs["significant"]),
            }
            sig = " ★" if bs["significant"] else ""
            print(f"    {cfg_name:<22} acc={accs[cfg_name]:.4f}  vs spec: "
                  f"Δ={bs['delta']:+.4f} p={bs['p_value']:.4f}{sig}", flush=True)

        # SYNERGY CHECK — combined vs each component
        # Test 1: combined vs frac alone
        bs_vs_frac = bootstrap_compare(y_te2, preds["frac+spec"], preds["frac+cum6+spec"], R=500)
        # Test 2: combined vs cumulants alone
        bs_vs_cum = bootstrap_compare(y_te2, preds["cum6+spec"], preds["frac+cum6+spec"], R=500)
        seed_data["synergy_check"] = {
            "combined_vs_frac+spec": {
                "delta": float(bs_vs_frac["delta"]), "p_value": float(bs_vs_frac["p_value"]),
                "significant": bool(bs_vs_frac["significant"]),
            },
            "combined_vs_cum6+spec": {
                "delta": float(bs_vs_cum["delta"]), "p_value": float(bs_vs_cum["p_value"]),
                "significant": bool(bs_vs_cum["significant"]),
            },
        }
        sig_f = " ★" if bs_vs_frac["significant"] else ""
        sig_c = " ★" if bs_vs_cum["significant"] else ""
        print(f"    SYNERGY: combined vs frac+spec  Δ={bs_vs_frac['delta']:+.4f} "
              f"p={bs_vs_frac['p_value']:.4f}{sig_f}", flush=True)
        print(f"             combined vs cum6+spec  Δ={bs_vs_cum['delta']:+.4f} "
              f"p={bs_vs_cum['p_value']:.4f}{sig_c}", flush=True)

        band_data["per_seed"].append(seed_data)

    # Cross-seed summary
    band_data["summary"] = {}
    for cfg_name in ("frac+spec", "cum6+spec", "frac+cum6+spec"):
        deltas = [s["deltas_vs_spec"][cfg_name]["delta"] for s in band_data["per_seed"]]
        sig_count = sum(1 for s in band_data["per_seed"]
                        if s["deltas_vs_spec"][cfg_name]["significant"])
        accs = [s["accuracies"][cfg_name] for s in band_data["per_seed"]]
        band_data["summary"][cfg_name] = {
            "n_seeds_significant": sig_count,
            "mean_accuracy": float(np.mean(accs)),
            "mean_delta_vs_spec": float(np.mean(deltas)),
        }

    # Synergy summary
    syn_vs_frac_sig = sum(1 for s in band_data["per_seed"]
                          if s["synergy_check"]["combined_vs_frac+spec"]["significant"])
    syn_vs_cum_sig = sum(1 for s in band_data["per_seed"]
                         if s["synergy_check"]["combined_vs_cum6+spec"]["significant"])
    syn_vs_frac_deltas = [s["synergy_check"]["combined_vs_frac+spec"]["delta"]
                           for s in band_data["per_seed"]]
    syn_vs_cum_deltas = [s["synergy_check"]["combined_vs_cum6+spec"]["delta"]
                          for s in band_data["per_seed"]]
    band_data["summary"]["synergy"] = {
        "vs_frac+spec_seeds_significant": syn_vs_frac_sig,
        "vs_frac+spec_mean_delta": float(np.mean(syn_vs_frac_deltas)),
        "vs_cum6+spec_seeds_significant": syn_vs_cum_sig,
        "vs_cum6+spec_mean_delta": float(np.mean(syn_vs_cum_deltas)),
    }

    results[band_name] = band_data

# ---- Final verdict ----
print("\n" + "=" * 75)
print("FINAL VERDICT — does combined add over each component?")
print("=" * 75)
for band_name in ("2.4_GHz", "5.8_GHz"):
    s = results[band_name]["summary"]
    print(f"\n{band_name}:")
    print(f"  spec only:           acc={results[band_name]['per_seed'][0]['accuracies']['spec_only']:.4f}")
    print(f"  frac+spec:           acc≈{s['frac+spec']['mean_accuracy']:.4f}  Δ={s['frac+spec']['mean_delta_vs_spec']:+.4f}  ({s['frac+spec']['n_seeds_significant']}/3 sig)")
    print(f"  cum6+spec:           acc≈{s['cum6+spec']['mean_accuracy']:.4f}  Δ={s['cum6+spec']['mean_delta_vs_spec']:+.4f}  ({s['cum6+spec']['n_seeds_significant']}/3 sig)")
    print(f"  frac+cum6+spec:      acc≈{s['frac+cum6+spec']['mean_accuracy']:.4f}  Δ={s['frac+cum6+spec']['mean_delta_vs_spec']:+.4f}  ({s['frac+cum6+spec']['n_seeds_significant']}/3 sig)")
    print(f"  SYNERGY check:")
    print(f"    combined vs frac+spec:  Δ_mean={s['synergy']['vs_frac+spec_mean_delta']:+.4f}  ({s['synergy']['vs_frac+spec_seeds_significant']}/3 sig)")
    print(f"    combined vs cum6+spec:  Δ_mean={s['synergy']['vs_cum6+spec_mean_delta']:+.4f}  ({s['synergy']['vs_cum6+spec_seeds_significant']}/3 sig)")

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}")
