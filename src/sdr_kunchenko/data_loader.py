"""SDR dataset loader with KU_SDR_DATA_ROOT env-var support.

Datasets follow the schema produced by `signal-denoising/data_generation/`:

    <root>/<dataset_name>/
        dataset_config.json
        train/
            clean_signals.npy            (M, block_size)  float32
            gaussian_signals.npy         (M, block_size)  float32
            non_gaussian_signals.npy     (M, block_size)  float32
            non_gaussian_noise_only.npy  (M, block_size)  float32
            snr_values.npy               (M,)             float32
        test/
            test_<snr_label>_<type>.npy  (10 SNR × 4 type = 40 files)

Invariant: `non_gaussian_noise_only[i] == non_gaussian_signals[i] - clean_signals[i]`
within float32 noise (~1e-6 relative).

Configuration: KU_SDR_DATA_ROOT env-var must point at the directory containing
the dataset folders. There is no silent fallback — see .env.example.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

ENV_VAR = "KU_SDR_DATA_ROOT"


def get_data_root() -> Path:
    """Resolve dataset root from the KU_SDR_DATA_ROOT env-var.

    Raises:
        EnvironmentError: if KU_SDR_DATA_ROOT is unset. Set it to the directory
            containing dataset folders (see .env.example).
    """
    value = os.environ.get(ENV_VAR)
    if not value:
        raise EnvironmentError(
            f"{ENV_VAR} is not set. Point it at the directory containing the "
            f"dataset folders (e.g. signal-denoising/data_generation/datasets/). "
            f"See .env.example at the repo root."
        )
    return Path(value)


def data_root_is_configured() -> bool:
    """Return True if KU_SDR_DATA_ROOT is set and points at an existing directory.

    Use this in test gates and CLI scripts that need a graceful skip when data
    is not provisioned on the host.
    """
    value = os.environ.get(ENV_VAR)
    return bool(value) and Path(value).is_dir()


@dataclass
class SDRDataset:
    """Memory-mapped accessor for a single dataset folder."""

    name: str
    root: Path
    config: dict = field(repr=False)
    block_size: int
    n_train: int

    @classmethod
    def open(cls, name: str, root: Path | None = None) -> "SDRDataset":
        root = Path(root) if root else get_data_root()
        ds_path = root / name
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found at {ds_path}")
        cfg_path = ds_path / "dataset_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        clean = np.load(ds_path / "train" / "clean_signals.npy", mmap_mode="r")
        return cls(
            name=name,
            root=ds_path,
            config=cfg,
            block_size=int(clean.shape[1]),
            n_train=int(clean.shape[0]),
        )

    # ---- train arrays (memory-mapped) ----------------------------------------
    @property
    def clean(self) -> np.ndarray:
        return np.load(self.root / "train" / "clean_signals.npy", mmap_mode="r")

    @property
    def noisy(self) -> np.ndarray:
        return np.load(self.root / "train" / "non_gaussian_signals.npy", mmap_mode="r")

    @property
    def noise_only(self) -> np.ndarray:
        return np.load(self.root / "train" / "non_gaussian_noise_only.npy", mmap_mode="r")

    @property
    def gaussian(self) -> np.ndarray:
        return np.load(self.root / "train" / "gaussian_signals.npy", mmap_mode="r")

    @property
    def snr(self) -> np.ndarray:
        return np.load(self.root / "train" / "snr_values.npy")

    # ---- test files ----------------------------------------------------------
    def list_test_files(self) -> dict[str, list[str]]:
        """Group test files by SNR label."""
        out: dict[str, list[str]] = {}
        test_dir = self.root / "test"
        if not test_dir.exists():
            return out
        for f in sorted(test_dir.glob("test_*.npy")):
            # filenames: test_m5dB_clean.npy, test_p10dB_non_gaussian.npy, ...
            parts = f.stem.split("_", 2)  # ["test", "m5dB", "clean"]
            if len(parts) == 3:
                snr_label, type_label = parts[1], parts[2]
                out.setdefault(snr_label, []).append(type_label)
        return out

    def load_test_block(self, snr_label: str, type_label: str) -> np.ndarray:
        """Load a single test array, e.g. ('p10dB', 'non_gaussian_noise_only')."""
        path = self.root / "test" / f"test_{snr_label}_{type_label}.npy"
        return np.load(path)

    # ---- batch iterator for MC ----------------------------------------------
    def iter_blocks(self, *, n: int | None = None,
                     rng: np.random.Generator | None = None) -> Iterator[dict]:
        """Yield up to `n` (clean, noise, snr) tuples.

        If `rng` is given, blocks are sampled without replacement. Otherwise
        blocks are taken in storage order.
        """
        total = self.n_train
        if n is None or n >= total:
            indices = np.arange(total)
        elif rng is not None:
            indices = rng.choice(total, size=n, replace=False)
        else:
            indices = np.arange(n)

        clean = self.clean
        noise = self.noise_only
        snr = self.snr
        for i in indices:
            yield {
                "idx": int(i),
                "clean": np.asarray(clean[i]).astype(np.float64),
                "noise": np.asarray(noise[i]).astype(np.float64),
                "snr_db": float(snr[i]),
            }


def list_available_datasets(root: Path | None = None) -> list[str]:
    """Return names of all dataset folders under root."""
    r = Path(root) if root else get_data_root()
    if not r.exists():
        return []
    return sorted(d.name for d in r.iterdir() if d.is_dir() and (d / "train").exists())


__all__ = [
    "SDRDataset",
    "ENV_VAR",
    "get_data_root",
    "data_root_is_configured",
    "list_available_datasets",
]
