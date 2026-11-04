import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only, also on machines with a GPU

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def y_grid():
    """A stretched 201-point wall-normal grid on [0, 2], like the real one (tanh clustering)."""
    s = np.linspace(-1, 1, 201)
    return (1 + np.tanh(1.8 * s) / np.tanh(1.8)).astype(np.float64)


@pytest.fixture
def small_fields(rng):
    """A tiny synthetic ensemble (N=12, n_y=9, n_x=16) with a mean profile and random fluctuations."""
    n, n_y, n_x = 12, 9, 16
    profile = np.linspace(0.0, 1.0, n_y)
    return (profile[None, :, None] + 0.1 * rng.standard_normal((n, n_y, n_x))).astype(np.float32)
