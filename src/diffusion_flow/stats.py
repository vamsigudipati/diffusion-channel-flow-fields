"""Field statistics of Stages 1-3: spectra, correlations, PDFs, nearest-training distances.

The notebooks define these as closures over module-level constants (n_x, dx, the y+ rows,
the histogram bins). Here every such constant is an explicit argument; the arithmetic is
unchanged, and tests/test_notebook_equivalence.py checks that against the notebook source.
"""

import numpy as np

RE_TAU = 125.0                                  # nominal Re_tau of P125_83pi
LX = 8 * np.pi                                  # streamwise period of the box
BANDS = [(0, 10), (10, 30), (30, 70), (70, 126)]  # y+ bands, distance to the nearest wall
KMAX = 64                                       # modes 1..64 enter the spectral distance (Stage 2)


# ---------------------------------------------------------------- wall distance and rows
def yplus_bottom(y_grid, re_tau=RE_TAU):
    """y+ measured from the bottom wall."""
    half = (y_grid[-1] - y_grid[0]) / 2.0
    return (y_grid - y_grid[0]) / half * re_tau


def yplus_nearest(y_grid, re_tau=RE_TAU):
    """y+ measured from the nearest wall."""
    half = (y_grid[-1] - y_grid[0]) / 2.0
    return np.minimum(y_grid - y_grid[0], y_grid[-1] - y_grid) / half * re_tau


def row_at_yplus(y_grid, target, re_tau=RE_TAU):
    """Bottom-wall row closest to y+ = target (y+ ≈ 15 -> row 23 on this grid)."""
    return int(np.argmin(np.abs(yplus_bottom(y_grid, re_tau) - target)))


def band_rows(y_grid, bands=BANDS, re_tau=RE_TAU):
    yp = yplus_nearest(y_grid, re_tau)
    return [np.where((yp >= lo) & (yp < hi))[0] for lo, hi in bands]


# ---------------------------------------------------------------- spectra and correlations
def spectrum(u, chunk=64):
    """Streamwise energy spectrum per row: mean over samples of |FFT_x(u')|^2 / n_x^2, u' = u - <u>(y).

    u: (N, n_y, n_x). Returns (n_y, n_x // 2 + 1). Stage 1's `ex_spec` uses chunk=128,
    Stage 2's `spectrum` chunk=64; the chunking only affects summation order.
    """
    n_x = u.shape[-1]
    mu = u.mean(axis=(0, 2), keepdims=True)
    acc = np.zeros((u.shape[1], n_x // 2 + 1))
    for s in range(0, u.shape[0], chunk):
        acc += (np.abs(np.fft.rfft(u[s:s + chunk] - mu, axis=-1)) ** 2).sum(axis=0)
    return acc / u.shape[0] / n_x ** 2


def band_spectrum(E, rows):
    """Average a per-row spectrum over the rows of each band -> (n_bands, n_x // 2 + 1)."""
    return np.stack([E[r].mean(axis=0) for r in rows])


def corr_x(u, j):
    """Streamwise two-point correlation R_uu(r) at row j (periodic, via FFT), r = 0 .. n_x/2."""
    n_x = u.shape[-1]
    up = u[:, j, :] - u[:, j, :].mean()
    P = (np.abs(np.fft.rfft(up, axis=-1)) ** 2).mean(axis=0)
    R = np.fft.irfft(P, n=n_x)
    return (R / R[0])[: n_x // 2 + 1]


def corr_y(u, j0):
    """Wall-normal correlation coefficient of u' at zero streamwise separation, reference row j0."""
    up = u - u.mean(axis=(0, 2), keepdims=True)
    ref = up[:, j0, :]
    num = (up * ref[:, None, :]).mean(axis=(0, 2))
    return num / (up.std(axis=(0, 2)) * ref.std() + 1e-12)


def first_below(R, level=0.5):
    """Index of the first separation where R drops below `level` (len(R) if it never does)."""
    k = np.where(R < level)[0]
    return int(k[0]) if len(k) else len(R)


def correlation_length(R, dx, level=0.5):
    """Stage 2's `first_below`: the separation (in length units) where R first drops below `level`."""
    return first_below(R, level) * dx


def tau_bottom(u, y_grid, noslip=False):
    """tau_w(x) = du/dy at y[0], one-sided 2nd-order formula on the non-uniform grid.

    u: (N, n_y >= 3, n_x). Returns (N, n_x).
    """
    h1, h2 = y_grid[1] - y_grid[0], y_grid[2] - y_grid[1]
    c0 = -(2 * h1 + h2) / (h1 * (h1 + h2)); c1 = (h1 + h2) / (h1 * h2); c2 = -h1 / (h2 * (h1 + h2))
    u0 = 0.0 if noslip else u[:, 0, :]
    return c0 * u0 + c1 * u[:, 1, :] + c2 * u[:, 2, :]


# ---------------------------------------------------------------- Stage 1 quality measure
def log_spectral_distance(Eb, Eb_ref):
    """RMS of log10(E / E_ref) over bands and all modes >= 1 (Stage 1's `ex_quality`)."""
    return float(np.sqrt(np.mean(np.log10(Eb[:, 1:] / Eb_ref[:, 1:]) ** 2)))


def profile_mae(u, mean_profile_ref, std_profile_ref):
    """Mean- and std-profile mean absolute errors of an ensemble u (N, n_y, n_x)."""
    return (float(np.mean(np.abs(u.mean(axis=(0, 2)) - mean_profile_ref))),
            float(np.mean(np.abs(u.std(axis=(0, 2)) - std_profile_ref))))


# ---------------------------------------------------------------- Stage 2 statistics
def pdf_bins(held, train, j15, n_bins=61, margin=0.05):
    """Common histogram bins of u at row j15 over both real sets (Stage 2's `BINS`)."""
    lo = min(held[:, j15].min(), train[:, j15].min()) - margin
    hi = max(held[:, j15].max(), train[:, j15].max()) + margin
    return np.linspace(lo, hi, n_bins)


def pdf(u, j15, bins):
    h, _ = np.histogram(u[:, j15, :].ravel(), bins=bins, density=True)
    return h


def wasserstein_1d(p, q, bins):
    """Stage 2's 1-Wasserstein distance between two histogram densities on common bins."""
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) * np.diff(bins)[0] ** 2)


def reference(held, j15, jc, bins):
    """The held-out reference curves of Stage 2 (`REF`)."""
    return dict(mean=held.mean(axis=(0, 2)), rms=held.std(axis=(0, 2)), E=spectrum(held),
                R15=corr_x(held, j15), Rc=corr_x(held, jc), pdf=pdf(held, j15, bins))


def stage2_stats(u, ref, j15, jc, bins, dx, kmax=KMAX):
    """Every scalar of Stage 2's `stats(u)` against a reference from `reference(...)`.

    Returns (scalars dict, curves dict) with the column names of a_statistics_runs.csv.
    """
    E = spectrum(u)
    n_y = u.shape[1]

    def lsd(rows, k1=1, k2=kmax):
        return float(np.sqrt(np.mean(np.log10(E[rows][:, k1:k2 + 1].mean(0)
                                              / ref["E"][rows][:, k1:k2 + 1].mean(0)) ** 2)))

    R15, Rc = corr_x(u, j15), corr_x(u, jc); p = pdf(u, j15, bins)
    w1 = wasserstein_1d(p, ref["pdf"], bins)
    scalars = dict(
        mean_profile_mae=float(np.mean(np.abs(u.mean(axis=(0, 2)) - ref["mean"]))),
        rms_profile_mae=float(np.mean(np.abs(u.std(axis=(0, 2)) - ref["rms"]))),
        spectral_distance_y15=lsd([j15]), spectral_distance_centre=lsd([jc]),
        spectral_distance_all_rows=lsd(np.arange(n_y)),
        small_scale_energy_ratio_y15=float(E[j15, kmax + 1:].mean() / ref["E"][j15, kmax + 1:].mean()),
        corr_length_y15=correlation_length(R15, dx), corr_length_centre=correlation_length(Rc, dx),
        corr_mae_y15=float(np.mean(np.abs(R15 - ref["R15"]))),
        corr_mae_centre=float(np.mean(np.abs(Rc - ref["Rc"]))), pdf_wasserstein_y15=w1)
    curves = dict(mean=u.mean(axis=(0, 2)), rms=u.std(axis=(0, 2)), E15=E[j15], Ec=E[jc], R15=R15, Rc=Rc, pdf=p)
    return scalars, curves


# ---------------------------------------------------------------- copying test
def nn_dist(U, train_flat, exclude_snap=None, snap_train=None, chunk=64):
    """RMS distance of each field in U to its nearest training field (Stage 2 cell 12, Stage 3 cell 22).

    U: (N, n_y, n_x). train_flat: (M, n_y * n_x) float32. If `exclude_snap` (N,) and
    `snap_train` (M,) are given, training fields of the same snapshot are excluded (Stage 3).
    Returns (distances (N,), index into train_flat (N,)).
    """
    TR = np.asarray(train_flat, np.float32); tr_sq = (TR ** 2).sum(1)
    d, j = np.zeros(len(U)), np.zeros(len(U), int)
    for s in range(0, len(U), chunk):
        Q = U[s:s + chunk].reshape(-1, TR.shape[1]).astype(np.float32)
        d2 = (Q ** 2).sum(1)[:, None] + tr_sq[None, :] - 2.0 * Q @ TR.T
        if exclude_snap is not None:
            d2[snap_train[None, :] == exclude_snap[s:s + chunk, None]] = np.inf
        jj = np.argmin(d2, axis=1)
        d[s:s + chunk] = np.sqrt(np.maximum(d2[np.arange(len(Q)), jj], 0) / Q.shape[1]); j[s:s + chunk] = jj
    return d, j
