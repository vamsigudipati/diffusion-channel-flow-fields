"""DDPM and DDIM samplers for a trained noise-predicting denoiser (Stage 1 cell 32, Stage 3 cell 9).

`sample` is Stage 3's `sample` (= Stage 1's `ex_sample` without conditioning) with the data
constants as arguments. It needs TensorFlow only when called.
"""

import numpy as np


def ddim_timesteps(T_, steps):
    """The retained timesteps of the deterministic DDIM sampler, descending (np.unique(...)[::-1])."""
    return np.unique(np.linspace(0, T_ - 1, steps).round().astype(int))[::-1]


def ddpm_mean(x, eps, t, beta, alpha, ab, x0_clip=None, clip_fn=np.clip):
    """One DDPM posterior mean, in Stage 1's plain form or written through a clipped x0 estimate.

    Works on numpy arrays (clip_fn=np.clip) or tensors (clip_fn=tf.clip_by_value).
    """
    if x0_clip is None:      # exactly the original DDPM update
        return (x - float(beta[t] / np.sqrt(1 - ab[t])) * eps) / float(np.sqrt(alpha[t]))
    x0 = clip_fn((x - float(np.sqrt(1 - ab[t])) * eps) / float(np.sqrt(ab[t])), *x0_clip)
    abp = ab[t - 1] if t > 0 else 1.0
    return (float(np.sqrt(abp) * beta[t] / (1 - ab[t])) * x0
            + float(np.sqrt(alpha[t]) * (1 - abp) / (1 - ab[t])) * x)


def ddim_step(x, eps, t, t_next_ab, ab, x0_clip=None, clip_fn=np.clip):
    """One deterministic DDIM (eta = 0) jump from t to the step whose alpha_bar is `t_next_ab`."""
    x0 = (x - float(np.sqrt(1 - ab[t])) * eps) / float(np.sqrt(ab[t]))
    if x0_clip is not None:
        x0 = clip_fn(x0, *x0_clip)
    return float(np.sqrt(t_next_ab)) * x0 + float(np.sqrt(1 - t_next_ab)) * eps


def make_caller(model):
    """A compiled `model([x, t], training=False)` call, as the notebooks' `ex_call`/`call_model`."""
    import tensorflow as tf

    return tf.function(lambda a, b: model([a, b], training=False), reduce_retracing=True)


def sample(model, beta_, n, seed, shape, method="ddpm", steps=10, x0_clip=None, chunk=64, caller=None):
    """Draw n normalised, padded samples of `shape` (n_y_padded, n_x, 1) from a denoiser.

    DDPM runs all T steps (plain update, or the clipped form when x0_clip is given); DDIM
    runs `steps` deterministic steps. Randomness comes from tf.random.Generator.from_seed(seed),
    exactly as in the notebooks, so the same seed gives the same samples as Stage 3's `sample`
    for the same model. Returns a float32 numpy array (n, *shape).
    """
    import tensorflow as tf

    f = caller or make_caller(model)
    b = np.asarray(beta_, np.float64); a = 1.0 - b; ab = np.cumprod(a); T_ = len(b)
    gen = tf.random.Generator.from_seed(seed); out = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s); x = gen.normal((m, *shape))
        if method == "ddpm":
            for t in reversed(range(T_)):
                eps = f(x, tf.fill((m,), float(t)))
                mean = ddpm_mean(x, eps, t, b, a, ab, x0_clip, tf.clip_by_value)
                x = mean + float(np.sqrt(b[t])) * gen.normal(tf.shape(x)) if t > 0 else mean
        else:
            ts = ddim_timesteps(T_, steps)
            for i, t in enumerate(ts):
                eps = f(x, tf.fill((m,), float(t)))
                abp = ab[ts[i + 1]] if i + 1 < len(ts) else 1.0
                x = ddim_step(x, eps, t, abp, ab, x0_clip, tf.clip_by_value)
        out.append(x.numpy())
    return np.concatenate(out)
