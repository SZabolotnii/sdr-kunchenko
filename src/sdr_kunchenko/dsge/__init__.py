"""DSGE — Decomposition in Space with Generating Element.

Per-class reconstruction error in Kunchenko basis as a feature for classification.
Spec §6.
"""
from .features import (
    build_F,
    fit_class_coefficients,
    msed,
    msed_features,
    msed_features_batch,
)

__all__ = ["build_F", "fit_class_coefficients", "msed",
            "msed_features", "msed_features_batch"]
