"""Noise schedules and the forward diffusion process (Stage 1, cells 13 and 40)."""

import numpy as np

T_DEFAULT = 200


def linear_beta(T=T_DEFAULT):
    """Stage 1's linear schedule: beta from 1e-4 to 0.02 over T steps, float32."""
    return np.linspace(1e-4, 0.02, T).astype(np.float32)


def cosine_beta(T_, s=0.008):
    """Cosine schedule of Nichol and Dhariwal, as written in Stage 1's schedule experiment.

    alpha_bar(t) = cos^2((t/T + s) / (1 + s) * pi/2) / alpha_bar(0); beta clipped at 0.999.
    """
    tt = np.arange(T_ + 1) / T_
    f = np.cos((tt + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return np.clip(1.0 - ab[1:] / ab[:-1], 0.0, 0.999).astype(np.float32)


def alpha_bar(beta):
    """Cumulative product of (1 - beta), in the dtype of beta (float32 in the notebooks)."""
    beta = np.asarray(beta)
    return np.cumprod(1.0 - beta).astype(beta.dtype)


def diffusion_constants(beta):
    """(alpha, alpha_bar, sqrt(alpha_bar), sqrt(1 - alpha_bar)) exactly as Stage 1 precomputes them."""
    beta = np.asarray(beta, np.float32)
    alpha = 1.0 - beta
    ab = np.cumprod(alpha).astype(np.float32)
    return alpha, ab, np.sqrt(ab), np.sqrt(1.0 - ab)


def q_sample(x0, t_idx, noise, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
    """Forward process: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps.

    The notebook's `q_sample` closes over the schedule arrays; here they are arguments.
    t_idx: int array, shape (batch,). x0 and noise: (batch, H, W, C).
    """
    sab = sqrt_alpha_bar[t_idx].reshape(-1, 1, 1, 1)
    somab = sqrt_one_minus_alpha_bar[t_idx].reshape(-1, 1, 1, 1)
    return sab * x0 + somab * noise
