"""Loading, padding, normalising and splitting the planes (Stage 1 cells 9 and 11; Stage 3 cell 5)."""

from pathlib import Path

import numpy as np

FILE_GLOB = "P125_83pi.*.h5"


def pad_amount(n_y_raw, multiple=16):
    """Rows to append so that n_y is a multiple of `multiple` (201 -> 7, giving 208)."""
    return multiple - (n_y_raw % multiple) if (n_y_raw % multiple) != 0 else 0


def pad_field(u, pad):
    """Edge-pad a (n_y, n_x) plane by `pad` rows at the top (y[-1] side), as the notebooks do."""
    return np.pad(u, ((0, pad), (0, 0)), mode="edge") if pad > 0 else u


def normalise(fields):
    """Global mean/std normalisation; returns (fields_norm, data_mean, data_std)."""
    data_mean = fields.mean()
    data_std = fields.std() + 1e-8
    return (fields - data_mean) / data_std, data_mean, data_std


def split_indices(n_fields, val_fraction=0.1, seed=0):
    """Stage 1's split: a default_rng(seed) permutation, the first 10 % held out.

    Returns (val_idx, trn_idx). For 5,777 fields this gives 577 held-out, 5,200 training.
    """
    n_val = max(1, int(val_fraction * n_fields))
    perm = np.random.default_rng(seed).permutation(n_fields)
    return perm[:n_val], perm[n_val:]


def snapshot_and_plane(path):
    """Snapshot and plane number from a file name `P125_83pi.<snapshot>.<plane>.h5` (Stage 3)."""
    parts = Path(path).name.split(".")
    return int(parts[1]), int(parts[2])


def list_planes(data_dir, file_glob=FILE_GLOB):
    """Sorted list of plane files in `data_dir`."""
    return sorted(Path(data_dir).glob(file_glob))


def load_planes(files, multiple=16, verbose_every=None):
    """Read `u` from every HDF5 file into a (N, n_y_padded, n_x, 1) float32 array.

    Returns (fields, y_grid, n_y_raw, n_x). Requires h5py (imported here so the rest of
    the package works without it).
    """
    import h5py

    files = list(files)
    with h5py.File(files[0], "r") as f:
        n_y_raw, n_x = f["u"].shape
        y_grid = f["y"][:]
    pad = pad_amount(n_y_raw, multiple)
    fields = np.zeros((len(files), n_y_raw + pad, n_x, 1), dtype=np.float32)
    for i, file_path in enumerate(files):
        with h5py.File(file_path, "r") as f:
            u = f["u"][:]
        fields[i, :, :, 0] = pad_field(u, pad)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"Loaded {i + 1} / {len(files)} files")
    return fields, y_grid, n_y_raw, n_x


def crop_physical(s_norm, n_y_raw, data_std, data_mean):
    """Normalised padded samples (N, n_y_padded, n_x, 1) -> physical, cropped (N, n_y_raw, n_x).

    Stage 1's `ex_phys`, with the closed-over constants as arguments.
    """
    return np.asarray(s_norm)[:, :n_y_raw, :, 0] * data_std + data_mean


def gaussian_baseline(mean_profile, std_profile, n, n_x, rng):
    """Independent-pixel Gaussian baseline with per-row mean and std (Stage 1 Experiment 1, Stage 2)."""
    n_y = len(mean_profile)
    return (mean_profile[None, :, None]
            + std_profile[None, :, None] * rng.standard_normal((n, n_y, n_x))).astype(np.float32)
