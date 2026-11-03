import numpy as np
import pytest

from diffusion_flow import stats


def test_yplus_and_rows(y_grid):
    yp_b = stats.yplus_bottom(y_grid); yp_n = stats.yplus_nearest(y_grid)
    assert yp_b[0] == 0 and yp_b[-1] == pytest.approx(250.0)
    assert yp_n.max() == pytest.approx(125.0) and yp_n[0] == 0 and yp_n[-1] == 0
    j15 = stats.row_at_yplus(y_grid, 15.0)
    assert 0 < j15 < 100 and abs(yp_b[j15] - 15) == np.abs(yp_b - 15).min()
    rows = stats.band_rows(y_grid)
    assert len(rows) == 4 and sum(len(r) for r in rows) == 201      # bands cover every row once
    assert len(np.unique(np.concatenate(rows))) == 201


def test_spectrum_of_a_single_sine_puts_energy_in_one_mode():
    n_x, k = 64, 5
    x = np.arange(n_x)
    u = np.stack([np.stack([2.0 * np.cos(2 * np.pi * k * x / n_x)] * 3)] * 4)   # (4, 3, 64), constant over samples
    E = stats.spectrum(u)
    assert E.shape == (3, n_x // 2 + 1)
    # |FFT|^2 / n_x^2 of A cos(k x): A^2 / 4 at mode k, nothing elsewhere
    assert E[0, k] == pytest.approx(1.0, rel=1e-9)
    assert np.abs(np.delete(E[0], k)).max() < 1e-12


def test_spectrum_removes_the_row_mean_and_is_chunk_independent(small_fields):
    E64 = stats.spectrum(small_fields, chunk=64); E5 = stats.spectrum(small_fields, chunk=5)
    np.testing.assert_allclose(E64, E5, rtol=1e-6)     # float32 inputs: summation order only
    # u' removes the ensemble-and-x mean per row, so mode 0 is the variance of the per-sample row means
    # (which is why the notebooks' spectral distances use modes >= 1 only)
    row_means = small_fields.mean(axis=2)                              # (N, n_y)
    np.testing.assert_allclose(E64[:, 0], ((row_means - row_means.mean(0)) ** 2).mean(0), rtol=1e-5)
    assert np.all(E64 >= 0)


def test_corr_x_known_cases():
    n_x = 64; x = np.arange(n_x)
    u = np.cos(2 * np.pi * 4 * x / n_x)[None, None, :] * np.ones((2, 1, 1))
    R = stats.corr_x(u, 0)
    assert R.shape == (n_x // 2 + 1,) and R[0] == pytest.approx(1.0)
    np.testing.assert_allclose(R, np.cos(2 * np.pi * 4 * np.arange(n_x // 2 + 1) / n_x), atol=1e-10)
    assert stats.first_below(R) == 3                                  # cos(2 pi 4 r / 64) < 0.5 first at r = 3
    assert stats.correlation_length(R, dx=0.5) == 1.5
    white = np.random.default_rng(1).standard_normal((2000, 1, n_x))
    Rw = stats.corr_x(white, 0)
    assert Rw[0] == pytest.approx(1.0) and np.abs(Rw[1:]).max() < 0.1
    assert stats.first_below(Rw) == 1                                 # independent pixels: one grid step


def test_first_below_never_crossing():
    assert stats.first_below(np.ones(10)) == 10


def test_corr_y_is_one_at_the_reference_row(small_fields):
    Ry = stats.corr_y(small_fields, 3)
    assert Ry.shape == (small_fields.shape[1],)
    assert Ry[3] == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.abs(Ry) <= 1 + 1e-6)


def test_tau_bottom_is_exact_for_a_quadratic():
    y = np.array([0.0, 0.1, 0.3, 0.6])
    a, b, c = 0.7, 1.3, -2.0
    u = (a + b * y + c * y ** 2)[None, :, None] * np.ones((2, 1, 5))
    tau = stats.tau_bottom(u, y)
    assert tau.shape == (2, 5)
    np.testing.assert_allclose(tau, b, rtol=1e-12)                    # du/dy at y = 0 of a quadratic
    tau_ns = stats.tau_bottom(u, y, noslip=True)
    assert not np.allclose(tau_ns, b)                                 # imposing u(0) = 0 changes the answer


def test_profile_mae_and_log_spectral_distance(small_fields):
    mp, sp = small_fields.mean(axis=(0, 2)), small_fields.std(axis=(0, 2))
    assert stats.profile_mae(small_fields, mp, sp) == (0.0, 0.0)
    Eb = np.ones((4, 9))
    assert stats.log_spectral_distance(Eb, Eb) == 0.0
    assert stats.log_spectral_distance(10 * Eb, Eb) == pytest.approx(1.0)


def test_pdf_and_wasserstein(small_fields):
    bins = stats.pdf_bins(small_fields, small_fields, 2)
    assert bins.shape == (61,)
    p = stats.pdf(small_fields, 2, bins)
    assert p.shape == (60,) and np.sum(p) * np.diff(bins)[0] == pytest.approx(1.0)
    assert stats.wasserstein_1d(p, p, bins) == 0.0
    q = np.roll(p, 1)
    assert stats.wasserstein_1d(p, q, bins) > 0


def test_stage2_stats_against_itself_is_zero(small_fields):
    j15, jc = 2, 4; dx = 0.1
    bins = stats.pdf_bins(small_fields, small_fields, j15)
    ref = stats.reference(small_fields, j15, jc, bins)
    scalars, curves = stats.stage2_stats(small_fields, ref, j15, jc, bins, dx, kmax=3)
    expected_keys = {"mean_profile_mae", "rms_profile_mae", "spectral_distance_y15", "spectral_distance_centre",
                     "spectral_distance_all_rows", "small_scale_energy_ratio_y15", "corr_length_y15",
                     "corr_length_centre", "corr_mae_y15", "corr_mae_centre", "pdf_wasserstein_y15"}
    assert set(scalars) == expected_keys                              # the columns of a_statistics_runs.csv
    for k in expected_keys - {"small_scale_energy_ratio_y15", "corr_length_y15", "corr_length_centre"}:
        assert scalars[k] == pytest.approx(0.0, abs=1e-12), k
    assert scalars["small_scale_energy_ratio_y15"] == pytest.approx(1.0)
    assert curves["mean"].shape == (small_fields.shape[1],)


def test_nn_dist_zero_for_a_training_field_and_exclusion(rng):
    train = rng.standard_normal((20, 6, 8)).astype(np.float32)
    TR = train.reshape(20, -1)
    d, j = stats.nn_dist(train[[3, 7]], TR)
    np.testing.assert_allclose(d, 0.0, atol=1e-3)
    assert list(j) == [3, 7]
    # a query 0.5 away from field 3 in every point has RMS distance 0.5 (sqrt of mean square)
    q = train[[3]] + 0.5
    d1, j1 = stats.nn_dist(q, TR)
    assert j1[0] == 3 and d1[0] == pytest.approx(0.5, rel=1e-3)
    # exclude the snapshot that field 3 belongs to: the answer moves to another field, farther away
    snap_train = np.arange(20) // 5
    d2, j2 = stats.nn_dist(q, TR, exclude_snap=np.array([snap_train[3]]), snap_train=snap_train)
    assert snap_train[j2[0]] != snap_train[3] and d2[0] > d1[0]
    # chunking does not change the result
    d3, j3 = stats.nn_dist(train[:7], TR, chunk=2)
    np.testing.assert_allclose(d3, 0.0, atol=1e-3); assert list(j3) == list(range(7))
