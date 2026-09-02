#!/usr/bin/env python3
"""Multi-segment spectrogram analysis (Codex D fix).

Original run_spectrogram_verify.py used only first 100 ms prefix —
generalized to "all recordings bimodal" without evidence на повний 1 sec.

This script analyses 3 predeclared offsets (start/mid/late) і reports
per-segment proportions з mean ± std для honest temporal characterization.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
FS = 120e6
SEG_DURATION = 0.1
OFFSETS = [0.0, 0.4, 0.8]   # 3 segments at 0-100ms, 400-500ms, 800-900ms

TARGETS = [
    ("Mavic Pro",   DATA_DIR / "DJI_mavic_pro_2G.bin"),
    ("Inspire 2",   DATA_DIR / "DJI_inspire_2_2G.bin"),
    ("Parrot Disco",DATA_DIR / "Parrot_disco_2G.bin"),
    ("Yuneec H",    DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin"),
]


def load_segment(path: Path, offset_sec: float, duration_sec: float) -> np.ndarray:
    n_offset = int(offset_sec * FS)
    n_samples = int(duration_sec * FS)
    # int16 interleaved: 2 ints per sample
    byte_offset = 2 * n_offset * 2  # 2 bytes/int × 2 ints/sample
    data = np.fromfile(path, dtype="<i2", count=2 * n_samples, offset=byte_offset)
    iq = data.astype(np.float32).reshape(-1, 2)
    return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def per_time_bw_distribution(sig: np.ndarray, fs: float,
                              nfft: int = 1024, threshold_db: float = -10
                              ) -> np.ndarray:
    """Per-time-bin occupied bandwidth (MHz)."""
    n = len(sig)
    n_windows = (n - nfft) // (nfft // 2) + 1
    widths_mhz = np.zeros(n_windows)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    for i in range(n_windows):
        s = i * (nfft // 2)
        chunk = sig[s:s + nfft]
        if len(chunk) < nfft:
            break
        psd = np.abs(np.fft.fftshift(np.fft.fft(chunk))) ** 2 / nfft
        psd_db = 10 * np.log10(psd + 1e-30)
        peak = psd_db.max()
        active = psd_db >= peak + threshold_db
        if active.any():
            idx = np.where(active)[0]
            widths_mhz[i] = abs(freqs[idx.max()] - freqs[idx.min()]) / 1e6
    return widths_mhz


print("Multi-segment spectrogram analysis\n", flush=True)
print(f"  Segments: {len(OFFSETS)} × {SEG_DURATION*1000:.0f} ms")
print(f"  Offsets: {[f'{o*1000:.0f}ms' for o in OFFSETS]}\n", flush=True)

results = {"config": {
    "segment_duration_sec": SEG_DURATION,
    "offsets_sec": OFFSETS,
    "fs_hz": FS,
    "rc_threshold_mhz": 5,
    "video_threshold_mhz": 15,
}}

for cls_name, path in TARGETS:
    if not path.exists():
        continue
    print(f"=== {cls_name} ===", flush=True)
    per_segment = []
    for offset in OFFSETS:
        t0 = time.time()
        seg = load_segment(path, offset, SEG_DURATION)
        bw = per_time_bw_distribution(seg, FS)
        n_total = len(bw)
        n_rc = (bw < 5).sum()
        n_video = (bw > 15).sum()
        n_inter = ((bw >= 5) & (bw <= 15)).sum()
        per_segment.append({
            "offset_sec": float(offset),
            "n_time_bins": int(n_total),
            "median_bw_mhz": float(np.median(bw)),
            "p25_bw_mhz": float(np.percentile(bw, 25)),
            "p75_bw_mhz": float(np.percentile(bw, 75)),
            "n_rc_like": int(n_rc),
            "n_video_like": int(n_video),
            "n_intermediate": int(n_inter),
            "rc_fraction": float(n_rc / n_total),
            "video_fraction": float(n_video / n_total),
        })
        print(f"  offset {offset*1000:.0f}ms: median={np.median(bw):.1f} MHz, "
              f"RC={n_rc/n_total:.1%}, video={n_video/n_total:.1%}, "
              f"inter={n_inter/n_total:.1%}  ({time.time()-t0:.1f}s)", flush=True)
        del seg

    # Aggregate across segments
    rcs = np.array([s["rc_fraction"] for s in per_segment])
    videos = np.array([s["video_fraction"] for s in per_segment])
    medians = np.array([s["median_bw_mhz"] for s in per_segment])
    aggregate = {
        "rc_fraction_mean": float(rcs.mean()),
        "rc_fraction_std": float(rcs.std()),
        "rc_fraction_min": float(rcs.min()),
        "rc_fraction_max": float(rcs.max()),
        "video_fraction_mean": float(videos.mean()),
        "video_fraction_std": float(videos.std()),
        "median_bw_mean": float(medians.mean()),
        "median_bw_std": float(medians.std()),
    }
    print(f"  → cross-segment: RC frac = {aggregate['rc_fraction_mean']:.1%} ± "
          f"{aggregate['rc_fraction_std']:.1%}  "
          f"(range [{aggregate['rc_fraction_min']:.1%}, {aggregate['rc_fraction_max']:.1%}])",
          flush=True)
    print(f"  → cross-segment: video frac = {aggregate['video_fraction_mean']:.1%} ± "
          f"{aggregate['video_fraction_std']:.1%}", flush=True)
    print(flush=True)

    results[cls_name] = {
        "per_segment": per_segment,
        "aggregate": aggregate,
    }

out_path = RESULTS / "phase6_spectrogram_multi_segment.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {out_path}")

# Verdict per file
print("\n" + "=" * 70)
print("VERDICT — bimodality consistent across recording?")
print("=" * 70)
print(f"{'Class':<14}{'RC frac':>14}{'Video frac':>16}{'σ across 3 segs':>20}")
for cls_name, info in results.items():
    if cls_name == "config":
        continue
    a = info["aggregate"]
    consistency = ("CONSISTENT" if a["rc_fraction_std"] < 0.10 and
                                    a["video_fraction_std"] < 0.10
                   else "VARIES")
    print(f"  {cls_name:<14}{a['rc_fraction_mean']*100:>10.1f}%  "
          f"{a['video_fraction_mean']*100:>14.1f}%   "
          f"σ_RC={a['rc_fraction_std']*100:.1f}%  "
          f"σ_video={a['video_fraction_std']*100:.1f}%  → {consistency}")
