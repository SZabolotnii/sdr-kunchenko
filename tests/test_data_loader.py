"""Smoke tests for the dataset loader. Skipped if data root not present."""
from __future__ import annotations

import os

import numpy as np
import pytest

from sdr_kunchenko.data_loader import (
    SDRDataset,
    data_root_is_configured,
    get_data_root,
    list_available_datasets,
)

pytestmark = pytest.mark.skipif(not data_root_is_configured(),
                                  reason="KU_SDR_DATA_ROOT not configured")


def test_list_available_datasets_finds_at_least_one():
    names = list_available_datasets()
    assert len(names) >= 1, f"no datasets at {get_data_root()}"


def test_open_primary_polygauss_dataset():
    name = "fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8"
    if name not in list_available_datasets():
        pytest.skip(f"primary dataset {name} not available")
    ds = SDRDataset.open(name)
    assert ds.block_size == 1024
    assert ds.n_train == 100_000
    assert ds.config["scenario"] == "fpv_telemetry"
    assert ds.config["modulation_type"] == "qpsk"


def test_invariant_holds_on_first_blocks():
    name = "fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8"
    if name not in list_available_datasets():
        pytest.skip(f"primary dataset {name} not available")
    ds = SDRDataset.open(name)
    n = 50
    clean = np.asarray(ds.clean[:n])
    noisy = np.asarray(ds.noisy[:n])
    nonly = np.asarray(ds.noise_only[:n])
    diff = noisy - clean - nonly
    assert np.abs(diff).max() < 1e-5, \
        f"invariant broken: max|err|={np.abs(diff).max():.2e}"


def test_iter_blocks_yields_correct_shapes():
    name = "fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8"
    if name not in list_available_datasets():
        pytest.skip(f"primary dataset {name} not available")
    ds = SDRDataset.open(name)
    block = next(ds.iter_blocks(n=1))
    assert block["clean"].shape == (1024,)
    assert block["noise"].shape == (1024,)
    assert isinstance(block["snr_db"], float)
    assert -10 <= block["snr_db"] <= 25
