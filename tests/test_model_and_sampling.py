import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from diffusion_flow import sampling, schedules  # noqa: E402
from diffusion_flow.model import create_denoiser, sinusoidal_embedding  # noqa: E402

SMALL = (16, 16, 1)   # three max-poolings need a multiple of 8


def test_sinusoidal_embedding_shape_and_range():
    e = sinusoidal_embedding(tf.constant([0.0, 7.0]), dim=32).numpy()
    assert e.shape == (2, 32)
    np.testing.assert_allclose(e[0, :16], 0.0, atol=1e-7)             # sin(0) = 0
    np.testing.assert_allclose(e[0, 16:], 1.0, atol=1e-7)             # cos(0) = 1
    assert np.abs(e).max() <= 1.0 + 1e-6


def test_parameter_counts_match_the_notebooks():
    # Stage 1 summary / README: 508,785 weights; device_timing_*.csv: conditional denoiser 508,929
    assert create_denoiser((208, 384, 1), base_filters=16).count_params() == 508785
    assert create_denoiser((208, 384, 2), base_filters=16).count_params() == 508929
    # the count does not depend on the spatial size, so the small test model has the same weights
    assert create_denoiser(SMALL, base_filters=16).count_params() == 508785


def test_forward_pass_two_sample_batch_on_cpu():
    m = create_denoiser(SMALL, base_filters=16)
    x = tf.random.stateless_normal((2, *SMALL), seed=(1, 2))
    t = tf.constant([0.0, 199.0])
    with tf.device("/CPU:0"):
        y = m([x, t], training=False)
    assert y.shape == (2, *SMALL)
    assert np.all(np.isfinite(y.numpy()))
    assert len(m.inputs) == 2 and m.inputs[0].name.startswith("noisy_field") and m.inputs[1].name.startswith("timestep")


def test_timestep_changes_the_output():
    m = create_denoiser(SMALL, base_filters=4)
    x = tf.random.stateless_normal((2, *SMALL), seed=(3, 4))
    y0 = m([x, tf.constant([0.0, 0.0])], training=False).numpy()
    y1 = m([x, tf.constant([150.0, 150.0])], training=False).numpy()
    assert not np.allclose(y0, y1)


def test_ddim_timesteps():
    ts = sampling.ddim_timesteps(200, 10)
    assert ts[0] == 199 and ts[-1] == 0 and np.all(np.diff(ts) < 0) and len(ts) == 10
    assert len(sampling.ddim_timesteps(200, 20)) == 20
    assert list(sampling.ddim_timesteps(200, 200)) == list(range(199, -1, -1))


def test_ddpm_mean_plain_equals_clipped_form_when_no_clipping_binds():
    b = schedules.linear_beta(200).astype(np.float64); a = 1 - b; ab = np.cumprod(a)
    x = np.random.default_rng(0).standard_normal((2, 4, 4, 1)); eps = 0.1 * x
    for t in (1, 100, 199):
        plain = sampling.ddpm_mean(x, eps, t, b, a, ab)
        wide = sampling.ddpm_mean(x, eps, t, b, a, ab, x0_clip=(-1e9, 1e9))
        np.testing.assert_allclose(plain, wide, rtol=1e-9, atol=1e-9)   # the two forms are the same mean


def test_ddim_step_with_perfect_noise_prediction_recovers_x0():
    b = schedules.linear_beta(200).astype(np.float64); ab = np.cumprod(1 - b)
    x0 = np.random.default_rng(1).standard_normal((1, 4, 4, 1)); eps = np.random.default_rng(2).standard_normal(x0.shape)
    t = 150; xt = np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps
    x_final = sampling.ddim_step(xt, eps, t, 1.0, ab)                 # jump straight to alpha_bar = 1
    np.testing.assert_allclose(x_final, x0, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("method,kw", [("ddpm", {}), ("ddpm", {"x0_clip": (-3.0, 3.0)}), ("ddim", {"steps": 4})])
def test_sample_shapes_and_determinism(method, kw):
    m = create_denoiser(SMALL, base_filters=4)
    beta = schedules.linear_beta(8)
    s1 = sampling.sample(m, beta, 3, seed=5, shape=SMALL, method=method, chunk=2, **kw)
    s2 = sampling.sample(m, beta, 3, seed=5, shape=SMALL, method=method, chunk=2, **kw)
    assert s1.shape == (3, *SMALL) and s1.dtype == np.float32
    assert np.all(np.isfinite(s1))
    np.testing.assert_array_equal(s1, s2)                              # same seed, same samples
    s3 = sampling.sample(m, beta, 3, seed=6, shape=SMALL, method=method, chunk=2, **kw)
    assert not np.allclose(s1, s3)
