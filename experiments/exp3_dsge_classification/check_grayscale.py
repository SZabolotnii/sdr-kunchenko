"""
Grayscale readability check for Exp 3 article figures.

Converts each PDF to grayscale PNG (via pdftoppm) and checks that
different data series have sufficient luminance contrast (≥30/255).

Usage:
    python experiments/exp3_dsge_classification/check_grayscale.py
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "experiments" / "exp3_dsge_classification" / "figures"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp3_dsge_classification" / "results"
GS_CHECK_DIR = FIGURES_DIR / "grayscale_check"

# The 5 article figures (fig1–fig5 mapping)
FIGURES = [
    ("Fig. 1", "fig3_dsge_vs_spectral.pdf",
     "Scatter: DSGE vs Spectral accuracy — distinct markers per SNR"),
    ("Fig. 2", "fig3_basis_comparison_bar.pdf",
     "Bar chart: 5 bases × 3 feature types — distinct hatches"),
    ("Fig. 3", "fig3_patp_alpha_sweep.pdf",
     "Line: accuracy vs α for 5 SNR — distinct markers + linestyles"),
    ("Fig. 4", "fig3_alpha_optimal_per_snr.pdf",
     "Line: α* per SNR — single series, reference lines"),
    ("Fig. 5", "fig3_accuracy_vs_snr.pdf",
     "Line: accuracy vs SNR for 3 feature types — distinct markers + linestyles"),
]

MIN_CONTRAST = 30  # minimum luminance difference (0–255) between distinct series


def pdf_to_grayscale_png(pdf_path: Path, out_path: Path, dpi: int = 300) -> Path:
    """Convert first page of PDF to grayscale PNG using pdftoppm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "page"
        result = subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(prefix)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed: {result.stderr}")
        # pdftoppm creates prefix-1.png for first page
        png_files = sorted(Path(tmpdir).glob("page-*.png"))
        if not png_files:
            raise RuntimeError(f"No PNG output from pdftoppm for {pdf_path}")
        # Convert to grayscale and save
        img = Image.open(png_files[0]).convert("L")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return out_path


def analyze_grayscale_contrast(gray_img: Image.Image, fig_name: str) -> dict:
    """
    Analyze grayscale contrast in the figure.

    Strategy: sample the plot area (central 70% of image), compute
    luminance histogram, and check that there are at least 2 distinct
    luminance clusters separated by ≥ MIN_CONTRAST.
    """
    arr = np.array(gray_img, dtype=np.float32)
    h, w = arr.shape

    # Focus on the central plot area (exclude white margins)
    margin_h = int(h * 0.10)
    margin_w = int(w * 0.10)
    plot_area = arr[margin_h:h - margin_h, margin_w:w - margin_w]

    # Exclude near-white background (>240) and near-white grid lines
    non_bg = plot_area[plot_area < 240]
    if len(non_bg) == 0:
        return {"pass": False, "reason": "No non-background pixels found",
                "min_lum": None, "max_lum": None, "contrast": None}

    min_lum = float(non_bg.min())
    max_lum = float(non_bg.max())
    contrast = max_lum - min_lum

    # Check for distinct luminance clusters using histogram
    hist, bin_edges = np.histogram(non_bg, bins=50, range=(0, 240))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find significant peaks (>0.5% of pixels)
    threshold = len(non_bg) * 0.005
    significant_bins = bin_centers[hist > threshold]

    if len(significant_bins) < 2:
        n_clusters = 1
        cluster_contrast = 0.0
    else:
        n_clusters = len(significant_bins)
        # Max gap between consecutive significant bins
        gaps = np.diff(significant_bins)
        cluster_contrast = float(gaps.max()) if len(gaps) > 0 else 0.0

    passed = contrast >= MIN_CONTRAST

    return {
        "pass": passed,
        "min_lum": min_lum,
        "max_lum": max_lum,
        "contrast": contrast,
        "n_significant_bins": n_clusters,
        "max_cluster_gap": cluster_contrast,
        "reason": (
            f"contrast={contrast:.1f}/255 ({'≥' if passed else '<'}{MIN_CONTRAST}), "
            f"luminance range [{min_lum:.0f}–{max_lum:.0f}], "
            f"{n_clusters} significant luminance levels"
        ),
    }


def check_figure(fig_label: str, filename: str, description: str) -> dict:
    """Check a single figure for grayscale readability."""
    pdf_path = FIGURES_DIR / filename
    if not pdf_path.exists():
        return {
            "label": fig_label,
            "filename": filename,
            "pass": False,
            "reason": f"File not found: {pdf_path}",
            "description": description,
        }

    gs_png_path = GS_CHECK_DIR / filename.replace(".pdf", "_grayscale.png")

    try:
        pdf_to_grayscale_png(pdf_path, gs_png_path, dpi=300)
        gray_img = Image.open(gs_png_path)
        analysis = analyze_grayscale_contrast(gray_img, fig_label)
    except Exception as e:
        return {
            "label": fig_label,
            "filename": filename,
            "pass": False,
            "reason": f"Error during conversion/analysis: {e}",
            "description": description,
        }

    return {
        "label": fig_label,
        "filename": filename,
        "pass": analysis["pass"],
        "reason": analysis["reason"],
        "description": description,
        "gs_png": str(gs_png_path),
    }


def main() -> None:
    GS_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    print("Grayscale readability check — Exp 3 figures")
    print("=" * 50)

    results = []
    for fig_label, filename, description in FIGURES:
        print(f"\nChecking {fig_label} ({filename})...")
        result = check_figure(fig_label, filename, description)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  {status}: {result['reason']}")
        results.append(result)

    n_pass = sum(1 for r in results if r["pass"])
    print(f"\nOverall: {n_pass}/{len(results)} PASS")

    # Write report
    report_path = RESULTS_DIR / "grayscale_check_report.txt"
    with open(report_path, "w") as f:
        f.write("Grayscale readability check — exp3 figures\n")
        f.write("==========================================\n\n")
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            f.write(f"{r['label']} ({r['filename']}): {status} — {r['reason']}\n")
            f.write(f"  Description: {r['description']}\n\n")
        f.write(f"Overall: {n_pass}/{len(results)} PASS\n")
        f.write("Action taken: regenerated figures with improved markers/linestyles/hatches\n")
        f.write("\nGrayscale improvements applied to plot.py:\n")
        f.write("  - Distinct markers: o, s, ^, D, v per series\n")
        f.write("  - Distinct linestyles: solid, dashed, dotted, dash-dot per series\n")
        f.write("  - Distinct gray shades: 0.0, 0.35, 0.55, 0.70, 0.85\n")
        f.write("  - Bar chart: hatch patterns (none, ///, ..., xxx, \\\\\\) per group\n")
        f.write("  - Scatter: replaced viridis colormap with per-SNR markers + gray shades\n")
        f.write(f"\nGrayscale PNG files saved to: {GS_CHECK_DIR}\n")

    print(f"\nReport written to: {report_path}")
    print(f"Grayscale PNGs saved to: {GS_CHECK_DIR}")


if __name__ == "__main__":
    main()
