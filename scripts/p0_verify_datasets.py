"""
Phase 0 dataset reality check.

Verifies for each dataset under KU_SDR_DATA_ROOT:
- expected files present
- shapes consistent
- invariant non_gaussian_noise_only == non_gaussian_signals - clean_signals (within float32 tolerance)
- per-block cumulants γ₃..γ₆ on noise

Outputs docs/PHASE0_REPORT.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sdr_kunchenko.data_loader import get_data_root

DATA_ROOT = get_data_root()
INVARIANT_TOL = 1e-5
N_SAMPLE = 200  # blocks to sanity-check invariant


def standardized_cumulants(x: np.ndarray) -> dict[str, float]:
    """Compute γ₃..γ₆ on a 1-D array via Fisher's k-statistics."""
    xc = x - x.mean()
    m = {k: float(np.mean(xc**k)) for k in range(2, 7)}
    sigma = float(np.sqrt(m[2]))
    kappa = {
        2: m[2],
        3: m[3],
        4: m[4] - 3 * m[2] ** 2,
        5: m[5] - 10 * m[3] * m[2],
        6: m[6] - 15 * m[4] * m[2] - 10 * m[3] ** 2 + 30 * m[2] ** 3,
    }
    return {f"gamma{k}": kappa[k] / sigma**k for k in range(3, 7)}


def check_dataset(ds: Path) -> dict:
    train = ds / "train"
    if not train.exists():
        return {"name": ds.name, "status": "skipped", "reason": "no train/"}

    expected = ["clean_signals.npy", "gaussian_signals.npy", "non_gaussian_signals.npy",
                "non_gaussian_noise_only.npy", "snr_values.npy"]
    missing = [f for f in expected if not (train / f).exists()]
    if missing:
        return {"name": ds.name, "status": "FAIL", "missing": missing}

    clean = np.load(train / "clean_signals.npy", mmap_mode="r")
    noisy = np.load(train / "non_gaussian_signals.npy", mmap_mode="r")
    nonly = np.load(train / "non_gaussian_noise_only.npy", mmap_mode="r")
    snr = np.load(train / "snr_values.npy")

    n = min(N_SAMPLE, clean.shape[0])
    diff = np.asarray(noisy[:n]) - np.asarray(clean[:n]) - np.asarray(nonly[:n])
    invariant_err = float(np.abs(diff).max())

    cum_per_block = [standardized_cumulants(np.asarray(nonly[i])) for i in range(n)]
    g3 = np.array([c["gamma3"] for c in cum_per_block])
    g4 = np.array([c["gamma4"] for c in cum_per_block])

    cfg = {}
    if (ds / "dataset_config.json").exists():
        cfg = json.loads((ds / "dataset_config.json").read_text())

    return {
        "name": ds.name,
        "status": "OK" if invariant_err < INVARIANT_TOL else "WARN",
        "n_train": int(clean.shape[0]),
        "block_size": int(clean.shape[1]),
        "invariant_max_err": invariant_err,
        "snr_min": float(snr.min()),
        "snr_max": float(snr.max()),
        "gamma3_mean": float(g3.mean()),
        "gamma3_std": float(g3.std()),
        "gamma4_mean": float(g4.mean()),
        "gamma4_std": float(g4.std()),
        "noise_types": cfg.get("noise_types", []),
        "modulation": cfg.get("modulation_type", "?"),
    }


def main() -> None:
    if not DATA_ROOT.exists():
        raise SystemExit(f"DATA_ROOT does not exist: {DATA_ROOT}")

    reports = []
    for ds in sorted(DATA_ROOT.iterdir()):
        if ds.is_dir():
            print(f"checking {ds.name}...")
            reports.append(check_dataset(ds))

    md = ["# Phase 0 Dataset Verification Report", "",
          f"**Root**: `{DATA_ROOT}`", ""]
    md.append("| Dataset | Status | n_train | block | inv_err | SNR | γ₃(μ±σ) | γ₄(μ±σ) | Noise | Mod |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        if r["status"] == "skipped":
            md.append(f"| {r['name']} | SKIP | — | — | — | — | — | — | — | — |")
            continue
        if r["status"] == "FAIL":
            md.append(f"| {r['name']} | **FAIL** missing {r['missing']} | — | — | — | — | — | — | — | — |")
            continue
        md.append(
            f"| {r['name']} | {r['status']} | {r['n_train']:,} | {r['block_size']} "
            f"| {r['invariant_max_err']:.2e} | [{r['snr_min']:.0f},{r['snr_max']:.0f}] "
            f"| {r['gamma3_mean']:+.2f}±{r['gamma3_std']:.2f} "
            f"| {r['gamma4_mean']:+.2f}±{r['gamma4_std']:.2f} "
            f"| {','.join(r['noise_types'])} | {r['modulation']} |"
        )

    out = Path(__file__).parent.parent / "docs" / "PHASE0_REPORT.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
