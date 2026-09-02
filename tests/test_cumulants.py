"""Pure-Python cumulants must match EstemPMM oracle on Gaussian samples.

Tests in this file do not require the dataset; they only need rpy2 + EstemPMM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdr_kunchenko.pmm import oracle
from sdr_kunchenko.utils.cumulants import (
    asymptotic_se,
    central_moments,
    cumulants_from_moments,
    standardized_cumulants,
)


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def fixture_arrays():
    df = pd.read_csv(FIXTURES / "pmm_reference_arrays.csv")
    return {key: grp["eps"].to_numpy(dtype=np.float64)
            for key, grp in df.groupby("dgp")}


def test_central_moments_recover_known_gaussian():
    """For Gaussian σ²=0.25: m₂ ≈ 0.25, m₄ ≈ 3·m₂² ≈ 0.187."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.5, 100_000)
    m = central_moments(x, max_order=4)
    assert abs(m[2] - 0.25) < 0.005
    assert abs(m[4] - 3 * 0.25 ** 2) < 0.02


def test_cumulants_from_moments_zero_third_for_gaussian():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.0, 100_000)
    kappa = cumulants_from_moments(central_moments(x, max_order=6))
    assert abs(kappa[3]) < 0.05
    assert abs(kappa[4]) < 0.1
    assert abs(kappa[6]) < 0.5


@pytest.mark.parametrize("dgp_key", ["dgp1", "dgp2", "dgp3"])
def test_python_cumulants_match_oracle_gamma(fixture_arrays, dgp_key):
    """Python γ_k via Fisher k-statistics must match EstemPMM's pmm_skewness/kurtosis/gamma6."""
    eps = fixture_arrays[dgp_key]
    py = standardized_cumulants(eps)
    rr = oracle.cumulants(eps)
    assert abs(py["gamma3"] - rr["gamma3"]) < 1e-10, \
        f"{dgp_key}.gamma3: py={py['gamma3']}, R={rr['gamma3']}"
    assert abs(py["gamma4"] - rr["gamma4"]) < 1e-10, \
        f"{dgp_key}.gamma4: py={py['gamma4']}, R={rr['gamma4']}"
    assert abs(py["gamma6"] - rr["gamma6"]) < 1e-10, \
        f"{dgp_key}.gamma6: py={py['gamma6']}, R={rr['gamma6']}"


def test_asymptotic_se_formulas_at_n_1024():
    se = asymptotic_se(1024)
    assert abs(se["gamma3"] - np.sqrt(6 / 1024)) < 1e-12
    assert abs(se["gamma4"] - np.sqrt(24 / 1024)) < 1e-12
    # Spec quotes SE(γ₃) ≈ 0.08, SE(γ₄) ≈ 0.15 at N=1024.
    assert abs(se["gamma3"] - 0.0765) < 0.005
    assert abs(se["gamma4"] - 0.1531) < 0.005
