#!/usr/bin/env python3
"""Distribution diagnostics v2 — block bootstrap + corrected sampling.

Fixes from Codex adversarial review:

    [F1] Per-window cumulants now computed on a SEEDED RANDOM SUBSAMPLE
         of selected windows, not on the temporal prefix windows[:200].
         Subsample size is recorded in JSON.

    [F2] Bootstrap CIs now WINDOW-LEVEL: each replicate resamples whole
         windows (with replacement), pools their samples, and computes
         cumulants on the resulting pool. This honours within-window
         correlation that sample-level bootstrap ignored.

    [F3] JSON reporting now matches actual compute (no n_windows=500
         while internally using 200).

Compare to v1 to see how much CIs widen — especially for γ₅, γ₆ which
are tail-sensitive.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Vendored DSGE toolkit modules live next to this file (see toolkit/README.md).
_TOOLKIT = Path(__file__).resolve().parent / "toolkit"
if not _TOOLKIT.is_dir():                      # scripts under iterations/
    _TOOLKIT = Path(__file__).resolve().parent.parent / "toolkit"
sys.path.insert(0, str(_TOOLKIT))

WIN = 4096
N_PER_CLASS = 500
PER_WINDOW_SUBSAMPLE = 200       # random subsample for per-window stats
PER_WINDOW_SEED = 0
BOOTSTRAP_R = 100                 # window-level bootstrap replicates
BOOTSTRAP_SEED = 42

# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
RESULTS = Path(__file__).parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figures"

CLASSES_24 = [
    ("Mavic Pro",   DATA_DIR / "DJI_mavic_pro_2G.bin",         "DJI_OcuSync_1"),
    ("Inspire 2",   DATA_DIR / "DJI_inspire_2_2G.bin",         "DJI_OcuSync_1"),
    ("Parrot Disco",DATA_DIR / "Parrot_disco_2G.bin",          "WiFi"),
    ("Yuneec H",    DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin", "WiFi"),
]
CLASSES_58 = [
    ("Phantom 4 Pro+", DATA_DIR / "DJI_phantom_4_pro_plus_5G_1of2.bin", "DJI_OcuSync_2"),
    ("Inspire 2",      DATA_DIR / "DJI_inspire_2_5G_1of2.bin",          "DJI_OcuSync_1"),
    ("Matrice 210",    DATA_DIR / "DJI_matrice_210_5G_1of2.bin",        "DJI_OcuSync_2"),
    ("Yuneec H",       DATA_DIR / "Yuneec_typhoon_h_5G.bin",            "WiFi"),
]

BANDS = {
    "2.4 GHz": (CLASSES_24, 120e6),
    "5.8 GHz": (CLASSES_58, 200e6),
}


def load_burst(path, n_windows, win, rng):
    """Note: returns windows in chunk-index order (sorted by chosen index).
    For per-window cumulant stats we'll random-subsample INSIDE the analysis.
    """
    data = np.fromfile(path, dtype="<i2", count=2 * n_windows * win * 2)
    iq = data.astype(np.float32).reshape(-1, 2)
    sig = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    nc = len(sig) // win
    chunks = sig[:nc * win].reshape(nc, win)
    energies = (chunks.real ** 2 + chunks.imag ** 2).mean(axis=1)
    threshold = float(np.percentile(energies, 40))
    high = np.where(energies >= threshold)[0]
    if len(high) >= n_windows:
        chosen = rng.choice(high, n_windows, replace=False)
    else:
        low = np.where(energies < threshold)[0]
        extra = rng.choice(low, n_windows - len(high), replace=False)
        chosen = np.concatenate([high, extra])
    chosen = np.sort(chosen)  # NOTE: still sorted by time within return
    windows = chunks[chosen]
    m, s = windows.mean(axis=1, keepdims=True), windows.std(axis=1, keepdims=True) + 1e-9
    return ((windows - m) / s).astype(np.complex64), chosen * win


def cumulants_from_pool(x):
    """Standardized γ₃-γ₆ from a 1D sample (pooled)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x_c = x - x.mean()
    m2 = np.mean(x_c ** 2)
    m3 = np.mean(x_c ** 3)
    m4 = np.mean(x_c ** 4)
    m5 = np.mean(x_c ** 5)
    m6 = np.mean(x_c ** 6)
    sigma = np.sqrt(m2) + 1e-12
    return {
        "gamma3": float(m3 / sigma ** 3),
        "gamma4": float((m4 - 3 * m2 ** 2) / sigma ** 4),
        "gamma5": float((m5 - 10 * m2 * m3) / sigma ** 5),
        "gamma6": float((m6 - 15 * m2 * m4 - 10 * m3 ** 2 + 30 * m2 ** 3) / sigma ** 6),
    }


def window_level_bootstrap(windows: np.ndarray, R: int, seed: int,
                            component: str = "Re") -> dict:
    """[F2 fix] Window-level block bootstrap for pooled-cumulant CIs.

    Each replicate resamples whole windows (with replacement), pools
    their samples, and computes cumulants on the pool. This respects
    within-window correlation that sample-level bootstrap ignored.

    Parameters
    ----------
    windows : (N, T) complex array
    R : number of bootstrap replicates
    seed : RNG seed
    component : 'Re' | 'Im' | '|y|'

    Returns
    -------
    dict with 95% CI for each cumulant.
    """
    rng = np.random.default_rng(seed)
    N = len(windows)
    g3 = np.empty(R); g4 = np.empty(R); g5 = np.empty(R); g6 = np.empty(R)
    for r in range(R):
        idx = rng.integers(0, N, size=N)
        sample = windows[idx]
        if component == "Re":
            pool = sample.real.ravel()
        elif component == "Im":
            pool = sample.imag.ravel()
        else:
            pool = np.abs(sample).ravel()
        c = cumulants_from_pool(pool)
        g3[r] = c["gamma3"]; g4[r] = c["gamma4"]
        g5[r] = c["gamma5"]; g6[r] = c["gamma6"]
    return {
        "gamma3": (float(np.percentile(g3, 2.5)), float(np.percentile(g3, 97.5))),
        "gamma4": (float(np.percentile(g4, 2.5)), float(np.percentile(g4, 97.5))),
        "gamma5": (float(np.percentile(g5, 2.5)), float(np.percentile(g5, 97.5))),
        "gamma6": (float(np.percentile(g6, 2.5)), float(np.percentile(g6, 97.5))),
        "n_replicates": R,
        "n_windows_resampled_per_replicate": N,
        "bootstrap_seed": seed,
        "component": component,
    }


def per_window_cumulants(windows: np.ndarray, n_subsample: int, seed: int,
                          component: str = "Re") -> dict:
    """[F1 fix] Random subsample (with seed) of windows, then per-window cumulants.

    Replaces the previous broken `windows[:n]` (temporal prefix).
    """
    rng = np.random.default_rng(seed)
    N = len(windows)
    n = min(n_subsample, N)
    sub_idx = np.sort(rng.choice(N, size=n, replace=False))
    sub = windows[sub_idx]
    if component == "Re":
        per_w = np.array([cumulants_from_pool(w.real) for w in sub])
    elif component == "Im":
        per_w = np.array([cumulants_from_pool(w.imag) for w in sub])
    else:
        per_w = np.array([cumulants_from_pool(np.abs(w)) for w in sub])

    # Convert array of dicts to arrays
    arrs = {k: np.array([d[k] for d in per_w]) for k in ("gamma3", "gamma4", "gamma5", "gamma6")}
    return {
        "n_subsampled_windows": int(n),
        "subsample_seed": seed,
        "subsample_indices_used": sub_idx.tolist()[:10] + ["…"] if n > 10 else sub_idx.tolist(),
        "subsample_indices_total_count": int(n),
        "stats": {
            k: {
                "mean": float(arrs[k].mean()),
                "std": float(arrs[k].std()),
                "p2.5": float(np.percentile(arrs[k], 2.5)),
                "p25": float(np.percentile(arrs[k], 25)),
                "p50": float(np.percentile(arrs[k], 50)),
                "p75": float(np.percentile(arrs[k], 75)),
                "p97.5": float(np.percentile(arrs[k], 97.5)),
                "min": float(arrs[k].min()),
                "max": float(arrs[k].max()),
            }
            for k in arrs
        },
    }


print(f"Diagnostics v2 — block bootstrap (R={BOOTSTRAP_R}, window-level)\n", flush=True)
diagnostics = {"config": {
    "n_windows_loaded_per_class": N_PER_CLASS,
    "window_samples": WIN,
    "per_window_subsample": PER_WINDOW_SUBSAMPLE,
    "per_window_seed": PER_WINDOW_SEED,
    "bootstrap_R": BOOTSTRAP_R,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_type": "window-level (block) — resample whole windows then pool",
}}

for band_name, (classes, fs) in BANDS.items():
    print(f"=== {band_name} (fs={fs/1e6:.0f} MS/s) ===", flush=True)
    rng = np.random.default_rng(42)
    band_data = {"sample_rate_MSps": fs / 1e6, "classes": {}}

    for cls_name, path, proto in classes:
        if not path.exists():
            print(f"  [skip] {cls_name}")
            continue
        t0 = time.time()
        windows, _starts = load_burst(path, N_PER_CLASS, WIN, rng)

        # ---- Pooled point estimates (unchanged: stable on 2M samples) ----
        c_re = cumulants_from_pool(windows.real.ravel())
        c_mag = cumulants_from_pool(np.abs(windows).ravel())

        # ---- v2 fix: window-level bootstrap CI ----
        ci_re_block = window_level_bootstrap(
            windows, R=BOOTSTRAP_R, seed=BOOTSTRAP_SEED, component="Re"
        )

        # ---- v2 fix: per-window cumulants on RANDOM SUBSAMPLE ----
        per_window = per_window_cumulants(
            windows, n_subsample=PER_WINDOW_SUBSAMPLE,
            seed=PER_WINDOW_SEED, component="Re"
        )

        band_data["classes"][cls_name] = {
            "protocol": proto,
            "n_windows_loaded": int(len(windows)),
            "n_samples_pooled": int(windows.size),
            "pooled_cumulants_Re": c_re,
            "pooled_cumulants_|y|": c_mag,
            "Re_bootstrap_95CI_window_level": ci_re_block,
            "per_window_cumulants_Re": per_window,
        }
        elapsed = time.time() - t0
        print(f"  {cls_name:<20} (n={len(windows)}, {elapsed:.1f}s)  γ₃(Re)={c_re['gamma3']:+.3f}  "
              f"γ₄(Re)={c_re['gamma4']:+.3f}  γ₆(Re)={c_re['gamma6']:+.3f}", flush=True)
        ci3, ci4, ci6 = (ci_re_block["gamma3"], ci_re_block["gamma4"], ci_re_block["gamma6"])
        print(f"      window-block 95%CI:  γ₃[{ci3[0]:+.3f}, {ci3[1]:+.3f}]  "
              f"γ₄[{ci4[0]:+.3f}, {ci4[1]:+.3f}]  γ₆[{ci6[0]:+.3f}, {ci6[1]:+.3f}]", flush=True)

    diagnostics[band_name] = band_data
    print(flush=True)

# ---- Compare new vs old CIs ----
print("=" * 78)
print("COMPARING WINDOW-BLOCK vs SAMPLE-LEVEL BOOTSTRAP CI WIDTHS (γ₆ on Re)")
print("=" * 78)
old_path = RESULTS / "phase6_distribution_diagnostics.json"
if old_path.exists():
    old = json.loads(old_path.read_text())
    print(f"{'Band':<10}{'Class':<18}{'γ₆ point':>10}"
          f"{'sample-CI half':>18}{'block-CI half':>18}{'widening':>12}")
    print("-" * 78)
    for band_name, band in diagnostics.items():
        if band_name == "config" or "classes" not in band:
            continue
        for cls_name, info in band["classes"].items():
            point = info["pooled_cumulants_Re"]["gamma6"]
            ci_new = info["Re_bootstrap_95CI_window_level"]["gamma6"]
            half_new = (ci_new[1] - ci_new[0]) / 2
            try:
                old_band = old.get(band_name, {})
                old_class = old_band.get("classes", {}).get(cls_name, {})
                ci_old = old_class.get("Re_bootstrap_95CI", {}).get("gamma6", None)
                if ci_old:
                    half_old = (ci_old[1] - ci_old[0]) / 2
                    ratio = half_new / max(half_old, 1e-9)
                    print(f"{band_name:<10}{cls_name:<18}{point:>+10.3f}"
                          f"{half_old:>18.4f}{half_new:>18.4f}{ratio:>10.1f}×")
            except Exception:
                pass

# Save corrected JSON
RESULTS.mkdir(exist_ok=True)
with open(RESULTS / "phase6_distribution_diagnostics_v2.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print(f"\nSaved {RESULTS / 'phase6_distribution_diagnostics_v2.json'}")

# ---- Mechanism check: do γ₆ CIs across protocols still NOT overlap? ----
print("\n" + "=" * 78)
print("MECHANISM CHECK: do γ₆(Re) CIs across classes still separate?")
print("=" * 78)
for band_name, band in diagnostics.items():
    if band_name == "config" or "classes" not in band:
        continue
    print(f"\n{band_name}:")
    intervals = []
    for cls_name, info in band["classes"].items():
        ci = info["Re_bootstrap_95CI_window_level"]["gamma6"]
        intervals.append((cls_name, ci[0], ci[1]))
        print(f"  {cls_name:<18} γ₆ ∈ [{ci[0]:+.3f}, {ci[1]:+.3f}]  width={ci[1]-ci[0]:.3f}")
    # Range of point estimates
    points = [band["classes"][c]["pooled_cumulants_Re"]["gamma6"]
              for c in band["classes"]]
    spread = max(points) - min(points)
    widths = [hi - lo for _, lo, hi in intervals]
    avg_ci_width = np.mean(widths)
    print(f"  Point-estimate spread = {spread:.3f}, avg CI width = {avg_ci_width:.3f}, "
          f"ratio = {spread/avg_ci_width:.1f}×")
    # Pairwise non-overlap check
    print("  Pairwise CI overlaps:")
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            n_i, lo_i, hi_i = intervals[i]
            n_j, lo_j, hi_j = intervals[j]
            non_overlap = (hi_i < lo_j) or (hi_j < lo_i)
            print(f"    {n_i:<14} vs {n_j:<14}  "
                  f"{'NON-OVERLAP (discriminative)' if non_overlap else 'overlap'}")
