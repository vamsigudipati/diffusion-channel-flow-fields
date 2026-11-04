import numpy as np
import pytest

from diffusion_flow import schedules


def test_linear_beta_endpoints_and_dtype():
    b = schedules.linear_beta(200)
    assert b.shape == (200,) and b.dtype == np.float32
    assert b[0] == pytest.approx(1e-4) and b[-1] == pytest.approx(0.02)
    assert np.all(np.diff(b) > 0)


def test_linear_alpha_bar_last_value_matches_readme():
    # README / Stage 1: "The linear schedule leaves 36 % of the signal at the last noise level (alpha_bar_T = 0.132)"
    ab = schedules.alpha_bar(schedules.linear_beta(200))
    assert ab[-1] == pytest.approx(0.132, abs=5e-4)
    assert np.sqrt(ab[-1]) == pytest.approx(0.36, abs=5e-3)          # 36 % of the signal amplitude
    assert np.all(np.diff(ab) < 0) and 0 < ab[-1] < ab[0] < 1


def test_cosine_beta_reaches_pure_noise_and_is_clipped():
    # Stage 3: "with the cosine schedule the last noise level is practically pure noise (alpha_bar_T ≈ 6e-8)"
    b = schedules.cosine_beta(200)
    assert b.shape == (200,) and b.dtype == np.float32
    assert b.max() <= 0.999 and b.min() >= 0.0
    ab_T = np.cumprod(1 - b.astype(np.float64))[-1]
    assert ab_T == pytest.approx(6.07e-8, rel=0.05)
    assert ab_T < 1e-6


def test_cosine_alpha_bar_follows_formula_before_clipping():
    T, s = 500, 0.008
    b = schedules.cosine_beta(T, s)
    ab = np.cumprod(1 - b.astype(np.float64))
    tt = np.arange(1, T + 1) / T
    f = np.cos((tt + s) / (1 + s) * np.pi / 2) ** 2 / np.cos(s / (1 + s) * np.pi / 2) ** 2
    unclipped = b < 0.999
    np.testing.assert_allclose(ab[unclipped], f[unclipped], rtol=1e-4)


def test_diffusion_constants_consistency():
    alpha, ab, sab, somab = schedules.diffusion_constants(schedules.linear_beta(50))
    np.testing.assert_allclose(sab ** 2 + somab ** 2, 1.0, atol=1e-6)
    np.testing.assert_allclose(ab, np.cumprod(alpha), rtol=1e-6)


def test_q_sample_shapes_and_limits():
    _, _, sab, somab = schedules.diffusion_constants(schedules.linear_beta(200))
    x0 = np.ones((3, 8, 8, 1), np.float32)
    noise = np.full((3, 8, 8, 1), 2.0, np.float32)
    t_idx = np.array([0, 100, 199])
    xt = schedules.q_sample(x0, t_idx, noise, sab, somab)
    assert xt.shape == x0.shape
    # t = 0: almost all signal; t = T-1: alpha_bar = 0.132 -> 0.3636 * 1 + 0.9317 * 2
    assert xt[0, 0, 0, 0] == pytest.approx(sab[0] * 1 + somab[0] * 2, rel=1e-6)
    assert xt[2, 0, 0, 0] == pytest.approx(np.sqrt(0.13218) + np.sqrt(1 - 0.13218) * 2, abs=1e-3)
