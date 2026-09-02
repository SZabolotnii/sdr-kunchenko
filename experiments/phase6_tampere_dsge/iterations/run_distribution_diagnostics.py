#!/usr/bin/env python3
"""Distribution diagnostics — why does frac win on 2.4 GHz and cumulants on 5.8?

Computes per-class per-band:
    - Histograms of Re(y), Im(y), |y|
    - Cumulants γ₃, γ₄, γ₅, γ₆ (skewness, excess kurtosis, 5th, 6th)
    - Bootstrap 95% CI for each cumulant
    - Within-class variance of per-window cumulants — captures information
      that DSGE/cumulant features actually use

Builds two layered figures:
    fig7: histograms 4-class × 2-band, |y| dimension (cleanest diagnostic)
    fig8: cumulant comparison table — γ₃-γ₆ across classes/bands with CI

Output:
    results/phase6_distribution_diagnostics.json
    figures/fig7_histograms.pdf
    figures/fig8_cumulants.pdf
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
N_PER_CLASS = 500           # smaller pool for fast cumulant estimation
SUBSAMPLE_FOR_BOOTSTRAP = 100_000  # cap pooled sample for bootstrap
# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
RESULTS = Path(__file__).parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figures"
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

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
    """Identical loader to other experiments."""
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
    chosen = np.sort(chosen)
    windows = chunks[chosen]
    m, s = windows.mean(axis=1, keepdims=True), windows.std(axis=1, keepdims=True) + 1e-9
    return ((windows - m) / s).astype(np.complex64)


def central_moment(x, k):
    """k-th central moment of 1D array x."""
    return np.mean((x - x.mean()) ** k)


def cumulants(x):
    """Standardized cumulants γ₃, γ₄, γ₅, γ₆ from a 1D sample.

    γ_n = κ_n / σ^n  where κ_n is n-th cumulant (Fisher).

    Formulas:
        κ₂ = m₂                    (variance)
        κ₃ = m₃                    (skewness numerator)
        κ₄ = m₄ - 3·m₂²            (excess kurtosis)
        κ₅ = m₅ - 10·m₂·m₃
        κ₆ = m₆ - 15·m₂·m₄ - 10·m₃² + 30·m₂³
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x_c = x - x.mean()
    m2 = np.mean(x_c ** 2)
    m3 = np.mean(x_c ** 3)
    m4 = np.mean(x_c ** 4)
    m5 = np.mean(x_c ** 5)
    m6 = np.mean(x_c ** 6)
    sigma = np.sqrt(m2) + 1e-12
    kappa3 = m3
    kappa4 = m4 - 3 * m2 ** 2
    kappa5 = m5 - 10 * m2 * m3
    kappa6 = m6 - 15 * m2 * m4 - 10 * m3 ** 2 + 30 * m2 ** 3
    return {
        "gamma3": kappa3 / sigma ** 3,
        "gamma4": kappa4 / sigma ** 4,
        "gamma5": kappa5 / sigma ** 5,
        "gamma6": kappa6 / sigma ** 6,
    }


def bootstrap_cumulants(x, R=200, seed=42):
    """Bootstrap 95% CI for γ₃-γ₆ on a 1D sample."""
    rng = np.random.default_rng(seed)
    n = len(x)
    g3, g4, g5, g6 = [], [], [], []
    for _ in range(R):
        idx = rng.integers(0, n, size=n)
        c = cumulants(x[idx])
        g3.append(c["gamma3"]); g4.append(c["gamma4"])
        g5.append(c["gamma5"]); g6.append(c["gamma6"])
    return {
        "gamma3": (float(np.percentile(g3, 2.5)), float(np.percentile(g3, 97.5))),
        "gamma4": (float(np.percentile(g4, 2.5)), float(np.percentile(g4, 97.5))),
        "gamma5": (float(np.percentile(g5, 2.5)), float(np.percentile(g5, 97.5))),
        "gamma6": (float(np.percentile(g6, 2.5)), float(np.percentile(g6, 97.5))),
    }


print("Computing distribution diagnostics for both bands…\n", flush=True)
diagnostics = {}

for band_name, (classes, fs) in BANDS.items():
    print(f"=== {band_name} (fs={fs/1e6:.0f} MS/s) ===", flush=True)
    rng = np.random.default_rng(42)
    band_data = {"sample_rate_MSps": fs / 1e6, "classes": {}}

    for cls_name, path, proto in classes:
        if not path.exists():
            continue
        t0 = time.time()
        windows = load_burst(path, N_PER_CLASS, WIN, rng)
        # Pool all windows: pool of N×T samples
        pool_re = windows.real.ravel()
        pool_im = windows.imag.ravel()
        pool_mag = np.abs(windows).ravel()

        c_re = cumulants(pool_re)
        c_im = cumulants(pool_im)
        c_mag = cumulants(pool_mag)

        # Bootstrap CIs on subsample (large pools blow up bootstrap time)
        rng_b = np.random.default_rng(0)
        sub_idx = rng_b.choice(len(pool_re), size=min(SUBSAMPLE_FOR_BOOTSTRAP, len(pool_re)),
                                replace=False)
        ci_re = bootstrap_cumulants(pool_re[sub_idx], R=80, seed=42)

        # Per-window cumulants (γ₄ shape across windows — captures DSGE-level info)
        # Subsample of windows for speed
        sub_w = windows[:200] if len(windows) > 200 else windows
        per_window_g3 = np.array([cumulants(w.real)["gamma3"] for w in sub_w])
        per_window_g4 = np.array([cumulants(w.real)["gamma4"] for w in sub_w])

        band_data["classes"][cls_name] = {
            "protocol": proto,
            "n_windows": len(windows),
            "n_samples_pooled": int(pool_re.size),
            "pooled_cumulants": {
                "Re": c_re, "Im": c_im, "|y|": c_mag,
            },
            "Re_bootstrap_95CI": ci_re,
            "per_window_gamma3": {
                "mean": float(per_window_g3.mean()),
                "std": float(per_window_g3.std()),
                "min": float(per_window_g3.min()),
                "max": float(per_window_g3.max()),
            },
            "per_window_gamma4": {
                "mean": float(per_window_g4.mean()),
                "std": float(per_window_g4.std()),
                "min": float(per_window_g4.min()),
                "max": float(per_window_g4.max()),
            },
        }

        print(f"  {cls_name:<20} γ₃(Re)={c_re['gamma3']:+.3f}±[{ci_re['gamma3'][0]:+.3f},{ci_re['gamma3'][1]:+.3f}]"
              f"  γ₄(Re)={c_re['gamma4']:+.3f}  γ₆(Re)={c_re['gamma6']:+.3f}"
              f"  γ₄(|y|)={c_mag['gamma4']:+.3f}", flush=True)

    diagnostics[band_name] = band_data
    print(flush=True)

# ============== Figure 7: Histograms ==============

print("Generating Fig 7 — histograms…", flush=True)
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row")

for i, (band_name, (classes, fs)) in enumerate(BANDS.items()):
    rng = np.random.default_rng(42)
    for j, (cls_name, path, proto) in enumerate(classes):
        if not path.exists():
            continue
        windows = load_burst(path, 500, WIN, rng)  # smaller for histogram
        pool_re = windows.real.ravel()
        # Sample 200K points to keep histogram fast
        rng_s = np.random.default_rng(0)
        sub = rng_s.choice(len(pool_re), size=min(200_000, len(pool_re)), replace=False)
        pool_re = pool_re[sub]

        ax = axes[i, j]
        ax.hist(pool_re, bins=80, density=True, alpha=0.6, color=f"C{j}",
                edgecolor="black", linewidth=0.3)

        # Overlay standard normal for reference
        x_grid = np.linspace(pool_re.min(), pool_re.max(), 200)
        gauss = np.exp(-x_grid ** 2 / 2) / np.sqrt(2 * np.pi)
        ax.plot(x_grid, gauss, "k--", lw=1.2, label="N(0,1)")

        # Annotate cumulants
        c = cumulants(pool_re)
        text = f"γ₃={c['gamma3']:+.2f}\nγ₄={c['gamma4']:+.2f}\nγ₆={c['gamma6']:+.2f}"
        ax.text(0.04, 0.96, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_title(f"{cls_name} ({proto})", fontsize=10)
        if i == 1:
            ax.set_xlabel("Re(y), z-normalized")
        if j == 0:
            ax.set_ylabel(f"{band_name}\nDensity")
        if i == 0 and j == 3:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-5, 5)

fig.suptitle("Distribution of Re(y) for 4 protocols × 2 bands (z-normalized)\n"
             "Dashed = standard normal reference; γ values from pooled samples",
             fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(FIGS / "fig7_histograms.pdf", bbox_inches="tight")
plt.close()
print(f"  saved {FIGS / 'fig7_histograms.pdf'}")

# ============== Figure 8: Cumulants comparison ==============

print("Generating Fig 8 — cumulants…", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)

cumulant_names = ["gamma3", "gamma4", "gamma5", "gamma6"]
cumulant_labels = [r"$\gamma_3$ (skewness)", r"$\gamma_4$ (excess kurt.)",
                   r"$\gamma_5$", r"$\gamma_6$"]
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for c_idx, (cum_name, cum_label) in enumerate(zip(cumulant_names, cumulant_labels)):
    ax = axes[positions[c_idx]]
    bar_pos = []
    bar_vals = []
    bar_lo = []
    bar_hi = []
    bar_labels = []
    bar_colors = []

    for b_idx, (band_name, band) in enumerate(diagnostics.items()):
        for cls_idx, (cls_name, info) in enumerate(band["classes"].items()):
            x_pos = b_idx * 5 + cls_idx
            val = info["pooled_cumulants"]["Re"][cum_name]
            ci = info["Re_bootstrap_95CI"][cum_name]
            bar_pos.append(x_pos)
            bar_vals.append(val)
            bar_lo.append(val - ci[0])
            bar_hi.append(ci[1] - val)
            bar_labels.append(f"{cls_name}\n({band_name})")
            color = "tab:blue" if "DJI" in info["protocol"] else "tab:orange"
            bar_colors.append(color)

    err = [bar_lo, bar_hi]
    bars = ax.bar(bar_pos, bar_vals, yerr=err, color=bar_colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5, capsize=3)
    ax.axhline(0, ls="-", color="#888", lw=0.5)
    if cum_name == "gamma4":
        # Reference: 0 = Gaussian
        pass
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(cum_label)
    ax.set_title(f"Pooled-sample {cum_label} on Re(y), 95% bootstrap CI")
    ax.grid(axis="y", alpha=0.3)

    # Vertical separator between bands
    ax.axvline(3.5, ls=":", color="black", alpha=0.4, lw=1)

# Custom legend
from matplotlib.patches import Patch
legend = [Patch(facecolor="tab:blue", label="DJI (OcuSync)"),
           Patch(facecolor="tab:orange", label="WiFi-based (Parrot/Yuneec)")]
fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Higher-order cumulants γ₃–γ₆ across protocols and bands",
             fontsize=13, y=1.05)
plt.tight_layout(rect=[0, 0, 1, 1.0])
plt.savefig(FIGS / "fig8_cumulants.pdf", bbox_inches="tight")
plt.close()
print(f"  saved {FIGS / 'fig8_cumulants.pdf'}")

# ============== Save JSON ==============
with open(RESULTS / "phase6_distribution_diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print(f"\nSaved {RESULTS / 'phase6_distribution_diagnostics.json'}")

# ============== Summary print ==============
print("\n" + "=" * 70)
print("CROSS-BAND CUMULANT SUMMARY (Re(y), pooled)")
print("=" * 70)
print(f"{'Band':<10}{'Class':<18}{'Protocol':<18}{'γ₃':>9}{'γ₄':>9}{'γ₅':>10}{'γ₆':>10}")
print("-" * 84)
for band_name, band in diagnostics.items():
    for cls_name, info in band["classes"].items():
        c = info["pooled_cumulants"]["Re"]
        print(f"{band_name:<10}{cls_name:<18}{info['protocol']:<18}"
              f"{c['gamma3']:>+9.3f}{c['gamma4']:>+9.3f}"
              f"{c['gamma5']:>+10.3f}{c['gamma6']:>+10.3f}")

print("\n" + "=" * 70)
print("PER-WINDOW γ₄ DISPERSION (DSGE/cumulant_features capture this)")
print("=" * 70)
print(f"{'Band':<10}{'Class':<18}{'γ₄ mean':>10}{'γ₄ std':>10}{'γ₄ range':>20}")
print("-" * 68)
for band_name, band in diagnostics.items():
    for cls_name, info in band["classes"].items():
        pw = info["per_window_gamma4"]
        print(f"{band_name:<10}{cls_name:<18}"
              f"{pw['mean']:>+10.3f}{pw['std']:>10.3f}"
              f"  [{pw['min']:+.2f}, {pw['max']:+.2f}]")
