"""Pure functions extracted from the three notebooks of this repository.

Every function here is a verbatim extraction of a definition that a notebook makes
inline (the notebooks keep their own copies, so their outputs stay valid). The test
suite checks the package against the notebook definitions themselves, parsed from
the .ipynb JSON, so the two cannot drift apart unnoticed.

Modules
-------
schedules : linear and cosine beta schedules, alpha_bar, the forward process q_sample
data      : padding, normalisation, the train/held-out split, plane loading, file names
model     : the time-conditioned U-Net denoiser (TensorFlow/Keras)
stats     : spectra, two-point correlations, PDFs, wall-distance helpers, Stage 2 metrics
sampling  : DDPM and DDIM samplers (plain and clipped) for a trained denoiser
notebook  : the same functions under the names the notebooks use (ex_spec, ...)
"""

from . import data, sampling, schedules, stats  # noqa: F401

__all__ = ["data", "model", "sampling", "schedules", "stats", "notebook"]
__version__ = "0.1.0"


def __getattr__(name):
    # `model` imports TensorFlow, which is slow; load it only when asked for.
    if name in ("model", "notebook"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
