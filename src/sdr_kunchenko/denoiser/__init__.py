"""PATP-adaptive denoiser for SDR baseband — Spec §7."""
from .alpha_mapping import (
    adaptive_denoise,
    calibrate_alpha_mapping,
    estimate_gamma4_from_residual,
    find_optimal_alpha,
)
from .baselines import median_denoise, projection_denoise, wiener_denoise
from .patp_denoiser import (
    fixed_patp_denoise,
    learn_subspace,
    patp_irls_denoise,
)

__all__ = [
    "adaptive_denoise", "calibrate_alpha_mapping",
    "estimate_gamma4_from_residual", "find_optimal_alpha",
    "median_denoise", "projection_denoise", "wiener_denoise",
    "fixed_patp_denoise", "learn_subspace", "patp_irls_denoise",
]
