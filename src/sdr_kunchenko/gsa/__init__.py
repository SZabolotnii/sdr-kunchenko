"""GSA-LLR change-point / signal detection module.

Pipeline:
    basis (poly/frac/PATP) -> moments (E_H0, E_H1, F) -> coefficients k = F⁻¹·Y
    -> per-sample Λ̂(y) -> CUSUM streaming -> threshold-controlled alarm.

Spec §5; theorem-code mapping in .claude/skills/gsa-change-point/.
"""
from .basis import (
    augmented_patp_basis,
    evaluate_basis,
    frac_basis,
    make_basis,
    patp_basis,
    patp_power,
    poly_basis,
)
from .coefficients import empirical_moments, fit_gsa_coefficients, lambda_score, solve_FK
from .detector import EnergyDetector, GSACUSUMDetector
from .threshold import pe_threshold, simulation_threshold

__all__ = [
    "augmented_patp_basis", "evaluate_basis", "frac_basis", "make_basis", "patp_basis",
    "patp_power", "poly_basis",
    "empirical_moments", "fit_gsa_coefficients", "lambda_score", "solve_FK",
    "EnergyDetector", "GSACUSUMDetector",
    "pe_threshold", "simulation_threshold",
]
