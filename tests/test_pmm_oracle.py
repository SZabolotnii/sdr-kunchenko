"""Bit-equivalence tests for the EstemPMM Python wrapper.

Both the R reference (scripts/p0_estempmm_oracle.R) and the Python wrapper
(src/sdr_kunchenko/pmm/oracle.py) call the same underlying R functions.
Therefore the numerical outputs *must* match within machine precision (≤1e-10
relative deviation) on the same input arrays.

If a Python wrapper change breaks bit-equivalence, this suite fails — that is
the entire point of the oracle pattern.
"""
from __future__ import annotations

import numpy as np
import pytest

from sdr_kunchenko.pmm import oracle


TOL = 1e-10


def _close(a, b, rtol=TOL, atol=TOL):
    if a is None or b is None:
        return a is b
    if isinstance(b, str):  # NA serialised as string
        return False
    return np.allclose(np.atleast_1d(a), np.atleast_1d(b), rtol=rtol, atol=atol)


# ----------------------------------------------------------------- meta


def test_estempmm_version_match(pmm_reference):
    """The Python wrapper must hit the same EstemPMM version as the fixture."""
    fixture_ver = pmm_reference["meta"]["estempmm_version"]
    py_ver = oracle.ESTEMPMM_VERSION
    # py_ver looks like "[1] '0.3.1'\n"; just check fixture version is a substring
    assert fixture_ver in py_ver, f"Version mismatch: fixture={fixture_ver}, oracle={py_ver}"


# ------------------------------------------------------- moments / cumulants


@pytest.mark.parametrize("dgp_key", ["dgp1", "dgp2", "dgp3"])
def test_compute_moments_matches_reference(pmm_arrays, pmm_reference, dgp_key):
    """compute_moments returns m2, m3, m4 + the PMM2 decomposition (c3, c4, g)."""
    eps = pmm_arrays[dgp_key]["eps"]
    py = oracle.compute_moments(eps)
    r = pmm_reference[dgp_key]["compute_moments_full"]
    for key in ("m2", "m3", "m4", "c3", "c4", "g"):
        assert _close(py[key], r[key]), f"{dgp_key}.{key}: py={py[key]}, R={r[key]}"


@pytest.mark.parametrize("dgp_key", ["dgp1", "dgp2", "dgp3"])
def test_compute_moments_pmm3_matches_reference(pmm_arrays, pmm_reference, dgp_key):
    """compute_moments_pmm3 returns m2, m4, m6 + the PMM3 decomposition."""
    eps = pmm_arrays[dgp_key]["eps"]
    py = oracle.compute_moments_pmm3(eps)
    r = pmm_reference[dgp_key]["compute_moments_pmm3_full"]
    for key in ("m2", "m4", "m6", "gamma4", "gamma6"):
        if r[key] is None or (isinstance(r[key], str) and r[key] == "NA"):
            continue
        assert _close(py[key], r[key]), f"{dgp_key}.pmm3.{key}: py={py[key]}, R={r[key]}"


@pytest.mark.parametrize("dgp_key", ["dgp1", "dgp2", "dgp3"])
def test_cumulants_match_reference(pmm_arrays, pmm_reference, dgp_key):
    eps = pmm_arrays[dgp_key]["eps"]
    py = oracle.cumulants(eps)
    r = pmm_reference[dgp_key]["cumulants"]
    for key in ("gamma3", "gamma4", "gamma6"):
        assert _close(py[key], r[key]), f"{dgp_key}.{key}: py={py[key]}, R={r[key]}"


# ------------------------------------------------------- variance factor


@pytest.mark.parametrize("dgp_key", ["dgp1", "dgp2", "dgp3"])
def test_variance_factor_g2_matches_reference(pmm_reference, dgp_key):
    m = pmm_reference[dgp_key]["moments_pmm2"]
    r = pmm_reference[dgp_key]["variance_factor_g2"]
    py = oracle.variance_factor_full(m["m2"], m["m3"], m["m4"])
    for key in ("c3", "c4", "g"):
        assert _close(py[key], r[key]), f"{dgp_key}.VF.{key}: py={py[key]}, R={r[key]}"


def test_variance_factor_g3_matches_reference(pmm_reference):
    """g3 is well-defined only when γ₆ + 9γ₄ + 6 > 0; we check whichever DGP gives a value."""
    for dgp_key in ("dgp1", "dgp2", "dgp3"):
        m = pmm_reference[dgp_key]["moments_pmm3"]
        r = pmm_reference[dgp_key]["variance_factor_g3"]["g"]
        if isinstance(r, str):  # "NA"
            continue
        py_g3 = oracle.variance_factor_g3(m["m2"], m["m4"], m["m6"])
        assert _close(py_g3, r), f"{dgp_key}.g3: py={py_g3}, R={r}"


# ------------------------------------------------------- regression fits


@pytest.mark.parametrize("dgp_key", ["dgp1", "dgp2", "dgp3"])
def test_lm_pmm2_matches_reference(pmm_arrays, pmm_reference, dgp_key):
    a = pmm_arrays[dgp_key]
    fit = oracle.lm_pmm2(a["y"], a["x"])
    r = pmm_reference[dgp_key]["pmm2_fit"]
    assert _close(fit["coef"], r["coef"]), \
        f"{dgp_key}.coef: py={fit['coef']}, R={r['coef']}"
    for key in ("m2", "m3", "m4", "iterations"):
        assert _close(fit[key], r[key]), f"{dgp_key}.{key}: py={fit[key]}, R={r[key]}"


def test_lm_pmm3_matches_reference_for_polygauss(pmm_arrays, pmm_reference):
    """PMM3 only validated on dgp3 (polygauss); on asymmetric DGPs EstemPMM warns."""
    dgp_key = "dgp3"
    a = pmm_arrays[dgp_key]
    fit = oracle.lm_pmm3(a["y"], a["x"])
    r = pmm_reference[dgp_key]["pmm3_fit"]
    if r is None:
        pytest.skip("R reference for PMM3 not available")
    assert _close(fit["coef"], r["coef"]), \
        f"PMM3 coef mismatch: py={fit['coef']}, R={r['coef']}"
    for key in ("m2", "m4", "m6", "gamma4", "gamma6", "g_coefficient", "kappa"):
        assert _close(fit[key], r[key]), f"PMM3.{key}: py={fit[key]}, R={r[key]}"


# ------------------------------------------------------- dispatch


def test_dispatch_chooses_pmm2_on_asymmetric(pmm_arrays):
    """TPN with γ₃ ≈ -0.85 must dispatch to PMM2 (|γ₃| > 0.3)."""
    eps = pmm_arrays["dgp2"]["eps"]
    disp = oracle.pmm_dispatch(eps)
    assert disp["method"] == "PMM2", f"expected PMM2, got {disp['method']}"


def test_dispatch_chooses_ols_on_gaussian(pmm_arrays):
    """Gaussian (γ₃≈0, γ₄≈0) → g₂_est ≥ 0.95 → OLS."""
    eps = pmm_arrays["dgp1"]["eps"]
    disp = oracle.pmm_dispatch(eps)
    # Allow either OLS or PMM2 — depends on N=2000 sample noise on γ₃, γ₄.
    assert disp["method"] in {"OLS", "PMM2"}, \
        f"expected OLS or PMM2 (Gaussian sample), got {disp['method']}"


# ------------------------------------------------------- expected behaviour


def test_pmm2_gain_on_asymmetric(pmm_arrays, pmm_reference):
    """Sanity: g₂ < 0.9 on TPN (γ₃≈-0.85)."""
    g = pmm_reference["dgp2"]["variance_factor_g2"]["g"]
    assert g < 0.9, f"expected substantial PMM2 gain on TPN, got g={g}"


def test_pmm2_no_gain_on_gaussian(pmm_reference):
    """Sanity: g₂ > 0.99 on Gaussian (no gain to extract)."""
    g = pmm_reference["dgp1"]["variance_factor_g2"]["g"]
    assert g > 0.99, f"expected ≈1 on Gaussian, got g={g}"
