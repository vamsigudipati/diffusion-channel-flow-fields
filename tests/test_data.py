import numpy as np
import pytest

from diffusion_flow import data


def test_pad_amount_matches_the_notebooks():
    assert data.pad_amount(201) == 7       # 201 -> 208
    assert data.pad_amount(208) == 0
    assert data.pad_amount(384) == 0       # n_x is already a multiple of 16


def test_pad_field_repeats_last_row_at_the_top():
    u = np.arange(12, dtype=np.float32).reshape(3, 4)
    p = data.pad_field(u, 2)
    assert p.shape == (5, 4)
    np.testing.assert_array_equal(p[:3], u)
    np.testing.assert_array_equal(p[3], u[-1])
    np.testing.assert_array_equal(p[4], u[-1])
    assert data.pad_field(u, 0) is u


def test_normalise_round_trip(rng):
    f = rng.standard_normal((5, 4, 4, 1)).astype(np.float32) * 3 + 2
    fn, mu, sd = data.normalise(f)
    assert fn.mean() == pytest.approx(0.0, abs=1e-5)
    assert fn.std() == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(fn * sd + mu, f, rtol=1e-5)


def test_split_indices_is_the_notebook_split():
    val_idx, trn_idx = data.split_indices(5777)
    assert len(val_idx) == 577 and len(trn_idx) == 5200
    assert len(np.intersect1d(val_idx, trn_idx)) == 0
    assert sorted(np.concatenate([val_idx, trn_idx])) == list(range(5777))
    # the first held-out index of default_rng(0).permutation(5777), as every notebook computes it
    assert val_idx[0] == np.random.default_rng(0).permutation(5777)[0]


def test_snapshot_and_plane_from_file_name():
    assert data.snapshot_and_plane("P125_83pi.0042.17.h5") == (42, 17)
    assert data.snapshot_and_plane("/some/dir/P125_83pi.7.3.h5") == (7, 3)


def test_crop_physical(rng):
    s = rng.standard_normal((2, 16, 8, 1)).astype(np.float32)
    phys = data.crop_physical(s, n_y_raw=13, data_std=2.0, data_mean=0.5)
    assert phys.shape == (2, 13, 8)
    np.testing.assert_allclose(phys, s[:, :13, :, 0] * 2.0 + 0.5)


def test_gaussian_baseline_matches_profiles_by_construction(rng):
    mean_p = np.linspace(0, 1, 9); std_p = np.full(9, 0.2)
    g = data.gaussian_baseline(mean_p, std_p, n=400, n_x=32, rng=rng)
    assert g.shape == (400, 9, 32) and g.dtype == np.float32
    np.testing.assert_allclose(g.mean(axis=(0, 2)), mean_p, atol=0.01)
    np.testing.assert_allclose(g.std(axis=(0, 2)), std_p, atol=0.01)


def test_load_planes_from_tiny_hdf5_files(tmp_path):
    h5py = pytest.importorskip("h5py")
    n_y, n_x = 5, 16
    y = np.linspace(0, 2, n_y)
    for snap, plane in [(1, 0), (1, 1), (2, 0)]:
        with h5py.File(tmp_path / f"P125_83pi.{snap}.{plane}.h5", "w") as f:
            f["u"] = np.full((n_y, n_x), float(snap * 10 + plane), np.float32)
            f["y"] = y
    files = data.list_planes(tmp_path)
    assert [p.name for p in files] == ["P125_83pi.1.0.h5", "P125_83pi.1.1.h5", "P125_83pi.2.0.h5"]
    fields, y_grid, n_y_raw, nx = data.load_planes(files)
    assert (n_y_raw, nx) == (n_y, n_x) and fields.shape == (3, 16, n_x, 1) and fields.dtype == np.float32
    np.testing.assert_array_equal(y_grid, y)
    assert fields[2, 0, 0, 0] == 20.0 and fields[2, -1, 0, 0] == 20.0   # padded rows copy the last row
