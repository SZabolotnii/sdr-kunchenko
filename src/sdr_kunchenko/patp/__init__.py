"""PATP cross-domain α-mapping utilities (Spec §8 / Phase 5)."""
from .cross_domain import (
    DATASET_GAMMA_PRIOR,
    aggregate_exp4_to_snr,
    build_unified_table,
    check_monotonicity,
    compute_pareto,
    fit_alpha_mapping,
    load_exp2_alphas,
    load_exp3_alphas,
    load_exp4_blocks,
    transferability_test,
)

__all__ = [
    "DATASET_GAMMA_PRIOR",
    "load_exp2_alphas", "load_exp3_alphas",
    "load_exp4_blocks", "aggregate_exp4_to_snr",
    "build_unified_table",
    "fit_alpha_mapping", "check_monotonicity",
    "transferability_test", "compute_pareto",
]
