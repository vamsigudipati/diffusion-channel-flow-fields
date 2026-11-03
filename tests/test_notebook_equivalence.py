"""The package must compute exactly what the notebooks' inline definitions compute.

The function definitions are read from the .ipynb JSON (never executed as notebooks) the
same way Stage 2 imports Stage 1's denoiser: parse the cell, keep only def/class/import
statements, exec them in a namespace that provides the constants the notebook closures use.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from diffusion_flow import data, sampling, schedules, stats

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def defs_from_notebook(name, markers, namespace):
    """exec the def/class/import statements of every code cell containing one of `markers`."""
    nb = json.loads((NB / name).read_text())
    found = 0
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if c["cell_type"] != "code" or not any(m in src for m in markers):
            continue
        tree = ast.parse(src)
        # defs only; the namespace supplies np / tf, so the cells' library imports are not needed here
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, f"{name} cell {i}", "exec"), namespace)
        found += 1
    assert found, f"no cell of {name} matched {markers}"
    return namespace


@pytest.fixture(scope="module")
def ens():
    rng = np.random.default_rng(7)
    n, n_y, n_x = 10, 21, 32
    prof = np.linspace(0, 1, n_y)
    u = (prof[None, :, None] + 0.2 * rng.standard_normal((n, n_y, n_x))).astype(np.float32)
    y = np.linspace(0, 2, n_y) ** 1.3 / 2 ** 0.3
    return dict(u=u, n_x=n_x, n_y=n_y, y_grid=y, rng=rng)


def test_stage1_schedule_and_stats_helpers(ens):
    u, n_x, y_grid = ens["u"], ens["n_x"], ens["y_grid"]
    ns = dict(np=np, n_x=n_x, y_grid=y_grid, ex_band_rows=stats.band_rows(y_grid))
    ns["ex_band_rows"] = [r for r in ns["ex_band_rows"] if len(r)]      # the tiny grid leaves a band empty
    defs_from_notebook("01_diffusion_channel_flow.ipynb",
                       ("def ex_spec", "def ex_cosine_beta"), ns)
    np.testing.assert_array_equal(ns["ex_cosine_beta"](200), schedules.cosine_beta(200))
    np.testing.assert_array_equal(ns["ex_cosine_beta"](500), schedules.cosine_beta(500))
    np.testing.assert_allclose(ns["ex_spec"](u), stats.spectrum(u, chunk=128), rtol=1e-12)
    E = stats.spectrum(u)
    np.testing.assert_array_equal(ns["ex_band_spec"](E), stats.band_spectrum(E, ns["ex_band_rows"]))
    for j in (0, 5, 10):
        np.testing.assert_allclose(ns["ex_corr_x"](u, j), stats.corr_x(u, j), rtol=1e-12)
        np.testing.assert_allclose(ns["ex_corr_y"](u, j), stats.corr_y(u, j), rtol=1e-12)
    R = stats.corr_x(u, 5)
    assert ns["ex_first_below"](R) == stats.first_below(R)
    for noslip in (False, True):
        np.testing.assert_allclose(ns["ex_tau_bottom"](u, noslip), stats.tau_bottom(u, y_grid, noslip), rtol=1e-12)
    # ex_phys closes over n_y_raw, data_std, data_mean
    ns.update(n_y_raw=15, data_std=0.34, data_mean=0.78)
    s = ens["rng"].standard_normal((3, 16, n_x, 1)).astype(np.float32)
    np.testing.assert_array_equal(ns["ex_phys"](s), data.crop_physical(s, 15, 0.34, 0.78))


def test_stage1_q_sample_and_linear_schedule(ens):
    beta = schedules.linear_beta(200)
    alpha, ab, sab, somab = schedules.diffusion_constants(beta)
    # the notebook computes the same constants in the same order
    nb_beta = np.linspace(1e-4, 0.02, 200).astype(np.float32)
    nb_ab = np.cumprod(1.0 - nb_beta).astype(np.float32)
    np.testing.assert_array_equal(beta, nb_beta); np.testing.assert_array_equal(ab, nb_ab)
    ns = dict(np=np, sqrt_alpha_bar=sab, sqrt_one_minus_alpha_bar=somab)
    defs_from_notebook("01_diffusion_channel_flow.ipynb", ("def q_sample",), ns)
    x0 = ens["rng"].standard_normal((4, 8, 8, 1)).astype(np.float32); noise = ens["rng"].standard_normal(x0.shape).astype(np.float32)
    t = np.array([0, 50, 120, 199])
    np.testing.assert_array_equal(ns["q_sample"](x0, t, noise), schedules.q_sample(x0, t, noise, sab, somab))


def test_stage2_statistics_and_nn_dist(ens):
    u, n_x, y_grid = ens["u"], ens["n_x"], ens["y_grid"]
    held, train = u[:4], u[4:]
    j15, jc = 3, ens["n_y"] // 2; dx = stats.LX / n_x
    bins = stats.pdf_bins(held, train, j15)
    ns = dict(np=np, n_x=n_x, n_y_raw=ens["n_y"], j15=j15, jc=jc, dx=dx, BINS=bins, KMAX=2, held=held, train=train)
    defs_from_notebook("02_field_statistics_and_memorisation.ipynb", ("def spectrum", "def nn_dist"), ns)
    # the notebook's BINS/REF are module-level expressions, not defs: rebuild them with its own functions
    nb_bins = np.linspace(min(held[:, j15].min(), train[:, j15].min()) - 0.05, max(held[:, j15].max(), train[:, j15].max()) + 0.05, 61)
    np.testing.assert_array_equal(nb_bins, bins)
    ns["REF"] = dict(mean=held.mean(axis=(0, 2)), rms=held.std(axis=(0, 2)), E=ns["spectrum"](held),
                     R15=ns["corr_x"](held, j15), Rc=ns["corr_x"](held, jc), pdf=ns["pdf"](held))
    ref = stats.reference(held, j15, jc, bins)
    for k in ref:
        np.testing.assert_allclose(ns["REF"][k], ref[k], rtol=1e-12, err_msg=k)
    nb_scalars, nb_curves = ns["stats"](train)
    scalars, curves = stats.stage2_stats(train, ref, j15, jc, bins, dx, kmax=2)
    assert set(nb_scalars) == set(scalars)
    for k in scalars:
        assert nb_scalars[k] == pytest.approx(scalars[k], rel=1e-10, abs=1e-14), k
    for k in nb_curves:
        np.testing.assert_allclose(nb_curves[k], curves[k], rtol=1e-12, err_msg=k)
    # copying test
    TR = train.reshape(len(train), -1).astype(np.float32)
    ns.update(TR=TR, tr_sq=(TR ** 2).sum(1))
    d_nb, j_nb = ns["nn_dist"](held)
    d, j = stats.nn_dist(held, TR)
    np.testing.assert_allclose(d_nb, d, rtol=1e-6); np.testing.assert_array_equal(j_nb, j)


def test_stage3_nn_dist_with_snapshot_exclusion(ens):
    u = ens["u"]; held, train = u[:4], u[4:]
    TR = train.reshape(len(train), -1).astype(np.float32)
    snap_tr = np.array([0, 0, 1, 1, 2, 2]); snap_held = np.array([0, 1, 2, 3])
    ns = dict(np=np, TR=TR, tr_sq=(TR ** 2).sum(1), snap_tr=snap_tr)
    defs_from_notebook("03_cosine_schedule_and_calibrated_copying.ipynb", ("def nn_dist",), ns)
    for excl in (None, snap_held):
        d_nb, j_nb = ns["nn_dist"](held, exclude_snap=excl, chunk=3)
        d, j = stats.nn_dist(held, TR, exclude_snap=excl, snap_train=snap_tr, chunk=3)
        np.testing.assert_allclose(d_nb, d, rtol=1e-6); np.testing.assert_array_equal(j_nb, j)


def test_stage3_cosine_schedule_and_stats_are_stage1_and_stage2_copies(ens):
    ns = dict(np=np)
    defs_from_notebook("03_cosine_schedule_and_calibrated_copying.ipynb", ("def cosine_beta",), ns)
    np.testing.assert_array_equal(ns["cosine_beta"](200), schedules.cosine_beta(200))
    u, n_x = ens["u"], ens["n_x"]
    ns3 = dict(np=np, n_x=n_x, dx=stats.LX / n_x)
    defs_from_notebook("03_cosine_schedule_and_calibrated_copying.ipynb", ("def spectrum",), ns3)
    np.testing.assert_allclose(ns3["spectrum"](u), stats.spectrum(u), rtol=1e-12)
    np.testing.assert_allclose(ns3["corr_x"](u, 2), stats.corr_x(u, 2), rtol=1e-12)
    assert ns3["first_below"](stats.corr_x(u, 2)) == stats.correlation_length(stats.corr_x(u, 2), stats.LX / n_x)


def test_denoiser_definition_is_identical_in_all_three_notebooks():
    """Stages 2 and 3 say the denoiser is copied unchanged from Stage 1; the package carries that copy."""
    tf = pytest.importorskip("tensorflow")
    from diffusion_flow import model as pkg

    shape = (16, 16, 1)
    x = tf.random.stateless_normal((2, *shape), seed=(9, 9)); t = tf.constant([3.0, 150.0])
    pkg_model = pkg.create_denoiser(shape, base_filters=4)
    weights = pkg_model.get_weights()
    ref = pkg_model([x, t], training=False).numpy()
    for name, markers in (("01_diffusion_channel_flow.ipynb", ("def sinusoidal_embedding", "def conv_block_t", "def create_denoiser")),
                          ("03_cosine_schedule_and_calibrated_copying.ipynb", ("def sinusoidal_embedding",))):
        ns = dict(tf=tf, np=np)
        defs_from_notebook(name, markers, ns)
        m = ns["create_denoiser"](shape, base_filters=4)
        assert m.count_params() == pkg_model.count_params()
        assert [w.shape for w in m.get_weights()] == [w.shape for w in weights]
        m.set_weights(weights)
        np.testing.assert_allclose(m([x, t], training=False).numpy(), ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ns["sinusoidal_embedding"](t, dim=16).numpy(), pkg.sinusoidal_embedding(t, dim=16).numpy())


def test_stage3_sampler_matches_package_sampler():
    """Stage 3's `sample` (also Stage 2's without clipping) and the package give identical samples."""
    tf = pytest.importorskip("tensorflow")
    from diffusion_flow.model import create_denoiser

    shape = (16, 16, 1); n_y_raw = 13; data_std, data_mean = 0.34, 0.78
    m = create_denoiser(shape, base_filters=4)
    ns = dict(tf=tf, np=np, n_y_padded=16, n_x=16, n_y_raw=n_y_raw, data_std=data_std, data_mean=data_mean, fn_cache={})
    defs_from_notebook("03_cosine_schedule_and_calibrated_copying.ipynb", ("def sample(model",), ns)
    beta = schedules.linear_beta(6)
    for kw in (dict(method="ddpm"), dict(method="ddpm", x0_clip=(-2.0, 1.5)), dict(method="ddim", steps=3), dict(method="ddim", steps=3, x0_clip=(-2.0, 1.5))):
        nb = ns["sample"](m, beta, 3, 11, chunk=2, **kw)                        # physical, cropped
        pk = data.crop_physical(sampling.sample(m, beta, 3, 11, shape, chunk=2, **kw), n_y_raw, data_std, data_mean)
        np.testing.assert_allclose(nb, pk, rtol=1e-6, atol=1e-6)
