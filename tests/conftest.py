"""pytest config / shared fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def pmm_reference() -> dict:
    """Load reference scalar values from R-generated fixture."""
    with (FIXTURES / "pmm_reference.json").open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def pmm_arrays() -> dict[str, dict[str, np.ndarray]]:
    """Load (x, eps, y) arrays for the 3 reference DGPs."""
    df = pd.read_csv(FIXTURES / "pmm_reference_arrays.csv")
    return {
        key: {
            "x": grp["x"].to_numpy(dtype=np.float64),
            "eps": grp["eps"].to_numpy(dtype=np.float64),
            "y": grp["y"].to_numpy(dtype=np.float64),
        }
        for key, grp in df.groupby("dgp")
    }
