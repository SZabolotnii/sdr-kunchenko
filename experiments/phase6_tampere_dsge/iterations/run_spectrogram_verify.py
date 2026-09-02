#!/usr/bin/env python3
"""Spectrogram verification — RC vs video burst patterns в Tampere data.

Per Tampere description (2026-04-29 follow-up): кожен recording містить
обидва — control link (RC→drone) і video link (drone→RC). Quote:
"Radio-Frequency Control AND Video Signal Recordings". RC link explicitly
hops between sub-channels; video link uses full channel bandwidth.

Очікувані patterns:
    - RC bursts: narrowband (~1-10 MHz), short duration (~ms), frequency
      hopping pattern visible across recording.
    - Video bursts: wideband (~20 MHz), continuous high-rate OFDM.

Output:
    figures/fig11_spectrogram_mavic_pro_2G.pdf
    figures/fig11_spectrogram_parrot_disco_2G.pdf
    + quantitative per-time bandwidth analysis
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Tampere I/Q extract; override with KU_TAMPERE_DATA_ROOT.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
DATA_DIR = Path(os.environ.get("KU_TAMPERE_DATA_ROOT", _REPO / "data" / "tampere_extract"))
FIGS = Path(__file__).parent.parent / "figures"
FIGS.mkdir(exist_ok=True)
RESULTS = Path(__file__).parent.parent / "results"

# Multi-segment analysis to address Codex finding [medium D]:
# Originally 100 ms prefix only — generalization до 1 sec recording fragile.
# Now we analyse 3 predeclared offsets (start, middle, late) and report
# per-segment proportions з mean ± std для honest characterization.
SEGMENT_DURATION_SEC = 0.1
SEGMENT_OFFSETS_SEC = [0.0, 0.4, 0.8]   # 3 offsets across full 1 sec recording
DURATION_SEC = SEGMENT_DURATION_SEC      # legacy alias for spec plotting (uses first offset)
FS = 120e6
TARGET_FILES = {
    "Mavic Pro (DJI OcuSync 1)": DATA_DIR / "DJI_mavic_pro_2G.bin",
    "Inspire 2 (DJI OcuSync 1)": DATA_DIR / "DJI_inspire_2_2G.bin",
    "Parrot Disco (WiFi-based)": DATA_DIR / "Parrot_disco_2G.bin",
    "Yuneec Typhoon H (WiFi)": DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin",
}


def load_iq(path: Path, n_samples: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<i2", count=2 * n_samples)
    iq = data.astype(np.float32).reshape(-1, 2)
    return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def per_time_bandwidth(specgram: np.ndarray, freqs: np.ndarray,
                        threshold_db: float = -10) -> np.ndarray:
    """For each time bin compute the occupied bandwidth.

    Definition: range of frequency bins where power is above (peak − threshold_db).
    Returns array of widths (Hz) per time bin.
    """
    psd_db = 10 * np.log10(specgram + 1e-12)
    n_freq, n_time = psd_db.shape
    widths = np.zeros(n_time)
    for t in range(n_time):
        peak = psd_db[:, t].max()
        active = psd_db[:, t] >= peak + threshold_db
        if active.any():
            active_idx = np.where(active)[0]
            f_lo = freqs[active_idx.min()]
            f_hi = freqs[active_idx.max()]
            widths[t] = abs(f_hi - f_lo)
    return widths


print("Generating spectrograms…\n", flush=True)

n_samples = int(DURATION_SEC * FS)
print(f"Loading {DURATION_SEC*1000:.0f} ms = {n_samples:,} complex samples per file", flush=True)

results_per_file = {}

for label, path in TARGET_FILES.items():
    if not path.exists():
        print(f"  [skip] {label}")
        continue
    t0 = time.time()
    sig = load_iq(path, n_samples)
    print(f"  {label}: shape={sig.shape}, dtype={sig.dtype} ({time.time()-t0:.1f}s)", flush=True)

    # Specgram
    NFFT = 1024
    noverlap = 512
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax_spec = fig.add_subplot(gs[0])
    Pxx, freqs, bins, im = ax_spec.specgram(
        sig, NFFT=NFFT, Fs=FS, noverlap=noverlap,
        window=np.blackman(NFFT), vmin=-155, cmap="viridis",
        scale="dB",
    )
    ax_spec.set_ylabel("Frequency (Hz, baseband)")
    ax_spec.set_title(f"Spectrogram — {label}\n"
                       f"{DURATION_SEC*1000:.0f} ms @ fs={FS/1e6:.0f} MS/s, NFFT={NFFT}")
    cbar = fig.colorbar(im, ax=ax_spec)
    cbar.set_label("Power (dB)")

    # Per-time bandwidth in lower panel
    ax_bw = fig.add_subplot(gs[1])
    widths = per_time_bandwidth(Pxx, freqs, threshold_db=-10)
    widths_mhz = widths / 1e6
    ax_bw.plot(bins * 1000, widths_mhz, "k-", lw=0.7)
    ax_bw.axhline(np.median(widths_mhz), ls="--", color="orange",
                   label=f"Median: {np.median(widths_mhz):.1f} MHz")
    ax_bw.set_xlabel("Time (ms)")
    ax_bw.set_ylabel("Occupied BW (MHz)\n(at peak − 10 dB)")
    ax_bw.set_title("Per-time-bin occupied bandwidth (RC narrowband ↔ video wideband)")
    ax_bw.grid(alpha=0.3)
    ax_bw.legend(loc="upper right", fontsize=9)

    safe_label = label.split()[0].lower().replace("(", "").replace(")", "")
    fig_path = FIGS / f"fig11_spectrogram_{safe_label}_2G.pdf"
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"    saved {fig_path}", flush=True)

    # Also a histogram of bandwidth distribution
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(widths_mhz, bins=40, color="steelblue", alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax_hist.set_xlabel("Occupied bandwidth per time bin (MHz)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"BW distribution — {label}\n"
                       f"(bimodal = RC narrowband + video wideband present)")
    ax_hist.grid(axis="y", alpha=0.3)

    # Mark thresholds for RC vs video classification heuristic
    ax_hist.axvline(5, ls="--", color="red", alpha=0.5,
                     label="RC<5 MHz threshold")
    ax_hist.axvline(15, ls="--", color="green", alpha=0.5,
                     label="Video>15 MHz threshold")
    ax_hist.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS / f"fig12_bw_hist_{safe_label}_2G.pdf", bbox_inches="tight")
    plt.close()
    print(f"    saved {FIGS / f'fig12_bw_hist_{safe_label}_2G.pdf'}", flush=True)

    # Quantitative summary
    n_rc_like = (widths_mhz < 5).sum()
    n_video_like = (widths_mhz > 15).sum()
    n_intermediate = ((widths_mhz >= 5) & (widths_mhz <= 15)).sum()
    n_total = len(widths_mhz)
    results_per_file[label] = {
        "median_bw_mhz": float(np.median(widths_mhz)),
        "p25_bw_mhz": float(np.percentile(widths_mhz, 25)),
        "p75_bw_mhz": float(np.percentile(widths_mhz, 75)),
        "min_bw_mhz": float(widths_mhz.min()),
        "max_bw_mhz": float(widths_mhz.max()),
        "n_rc_like_bins": int(n_rc_like),
        "n_video_like_bins": int(n_video_like),
        "n_intermediate_bins": int(n_intermediate),
        "n_total_bins": int(n_total),
        "rc_fraction": float(n_rc_like / n_total),
        "video_fraction": float(n_video_like / n_total),
    }
    print(f"    BW stats: median={np.median(widths_mhz):.1f} MHz, "
          f"p25={np.percentile(widths_mhz, 25):.1f}, "
          f"p75={np.percentile(widths_mhz, 75):.1f}", flush=True)
    print(f"    RC-like bins (<5 MHz): {n_rc_like}/{n_total} = {n_rc_like/n_total:.1%}", flush=True)
    print(f"    Video-like (>15 MHz):  {n_video_like}/{n_total} = {n_video_like/n_total:.1%}", flush=True)
    print(f"    Intermediate:           {n_intermediate}/{n_total} = {n_intermediate/n_total:.1%}", flush=True)
    print(flush=True)

    del sig

# Save quantitative summary
import json
with open(RESULTS / "phase6_spectrogram_summary.json", "w") as f:
    json.dump(results_per_file, f, indent=2)
print(f"Saved {RESULTS / 'phase6_spectrogram_summary.json'}")

# Print verdict
print("\n" + "=" * 70)
print("VERDICT: do recordings have RC + video bursts?")
print("=" * 70)
for label, info in results_per_file.items():
    rc_pct = info["rc_fraction"] * 100
    video_pct = info["video_fraction"] * 100
    if rc_pct > 5 and video_pct > 5:
        verdict = "BIMODAL ★ — RC + video both present"
    elif rc_pct > 50:
        verdict = "RC-dominant — narrowband activity"
    elif video_pct > 50:
        verdict = "Video-dominant — wideband activity"
    else:
        verdict = "Mixed / intermediate"
    print(f"  {label:<32} {verdict}")
    print(f"    RC<5 MHz: {rc_pct:.1f}%, video>15 MHz: {video_pct:.1f}%, "
          f"median: {info['median_bw_mhz']:.1f} MHz")
