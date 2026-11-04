# Repository map

What each file does, the order the pieces run in, and where every number in the README comes from.
Nothing in `results/`, `models/` or `animations/` was produced in this repository layout by anyone but the
author's single run of each notebook (Apple M3 Max, Apple GPU, TensorFlow 2.18.1 / Keras 3.6.0); see the README.

## Pipeline order

```
release_assets/u_yx_planes_h5/*.h5        5,777 planes (not in git; release of the unet-turbulent-fields repository)
        │
        ▼
notebooks/01_diffusion_channel_flow.ipynb          Stage 1: train the DDPM denoiser (linear schedule, 40 epochs),
        │   writes models/01_.../diffusion_denoiser.keras, results/01_.../exercises/*, animations/01_*
        ▼
notebooks/02_field_statistics_and_memorisation.ipynb   Stage 2: no training; scores the Stage 1 denoiser against
        │   577 held-out fields, writes results/02_.../*   (reads the device_timing_*.csv already in that folder)
        ▼
notebooks/03_cosine_schedule_and_calibrated_copying.ipynb   Stage 3: trains the cosine-schedule denoiser (3 seeds),
            writes models/03_.../cosine_denoiser_seed0.keras and results/03_.../*; reads Stage 2's tables
```

Each stage needs the one before it: Stage 2 loads the Stage 1 model; Stage 3 loads the Stage 1 model and
reads Stage 2's `a_statistics_table.csv`, `b_nearest_distance_quantiles.csv` and `device_timing_*.csv`.
All notebook paths are relative to `notebooks/`.

## Files

### Top level

| File | What it is |
| --- | --- |
| `README.md` | The project write-up: data, setup, what the three stages show. |
| `requirements.txt` | The frozen laptop environment (Python 3.12, TensorFlow 2.18.1 + `tensorflow-metal`, Keras 3.6.0). macOS-only pins (`tensorflow-metal`, `pyobjc-*`, `appnope`) make it unusable on Linux; CI uses `requirements-test.txt`. |
| `requirements-test.txt` | CPU-only pins for the tests and CI: `tensorflow-cpu==2.18.1`, `keras==3.6.0`, `numpy==2.0.2`, `h5py`, `pandas`, `pytest`. |
| `pyproject.toml` | Makes `src/diffusion_flow` an installable package and configures pytest (`pythonpath = ["src"]`). |
| `.github/workflows/tests.yml` | Runs `pytest` on every push and pull request on `ubuntu-latest`, Python 3.12, TensorFlow from PyPI (CPU). |
| `.gitignore` | Ignores `release_assets/` (the data), model checkpoints, caches. |

### `notebooks/`

| File | What it does |
| --- | --- |
| `01_diffusion_channel_flow.ipynb` (53 cells) | Loads the planes, pads 201 → 208 rows, normalises, splits 5,200 / 577 with `default_rng(0)`. Defines the linear schedule (T = 200), the forward process, the time-conditioned U-Net (`create_denoiser`, 508,785 weights) and trains it 40 epochs (Adam 2e-4, batch 16) with a hook that records an animation and 5-epoch checkpoints. Samples with 200-step DDPM, checks the mean and std profiles of 64 samples. Then five experiments: (1) Gaussian baseline vs two-point correlations, (2) a denoiser conditioned on the wall shear stress (10 epochs), (3) DDIM with 10/20/50 steps, (4) linear vs cosine schedules at 10 epochs (T = 200 and 500), (5) streamwise energy spectra in four y+ bands. Renders the training dashboard video. Repeats the experiment trainings and samplers with seeds 1 and 2 and writes `seeds_runs.csv` / `seeds_summary.csv`. |
| `02_field_statistics_and_memorisation.ipynb` (15 cells) | Rebuilds the data and split exactly as Stage 1, imports Stage 1's denoiser definition by parsing the Stage 1 notebook's JSON (defs only) and loads its weights. Part (a): mean/RMS profiles, spectra, two-point correlations and the PDF of u at y+ ≈ 15, all against the 577 held-out fields, for 3 × 256 DDPM samples and the Gaussian baseline. Part (b): nearest-training-field distance (copying test) and sampling cost vs quality for DDPM 200 and DDIM 50/20/10. |
| `03_cosine_schedule_and_calibrated_copying.ipynb` (25 cells) | Same data and split. Copies the denoiser, the training loop, the cosine schedule and the samplers (plain and clipped DDPM, DDIM) and Stage 2's statistics. Trains the cosine-schedule denoiser for 40 epochs with model seeds 0, 1, 2 (one cell each), saves seed 0, scores every Stage 2 statistic, writes a verdict table. Part (b): the held-out fields' nearest-training distance with same-snapshot planes excluded, beside the cosine and base models' samples. Writes `run_summary.json`. |
| `*.html` | `nbconvert` exports of the three notebooks with all outputs; text-only edits applied to both formats. |

The notebooks keep their function definitions inline so that the stored outputs remain the outputs of the code
shown. `src/diffusion_flow/` carries copies of those definitions, and `tests/test_notebook_equivalence.py` checks
the two against each other by parsing the `.ipynb` JSON (the notebooks are never executed by the tests).

### `src/diffusion_flow/` (extracted from the notebooks, no behaviour change)

| Module | Extracted from | Contents |
| --- | --- | --- |
| `schedules.py` | Stage 1 cells 13, 40; Stage 3 cell 7 | `linear_beta`, `cosine_beta`, `alpha_bar`, `diffusion_constants`, `q_sample` |
| `data.py` | Stage 1 cells 9, 11, 32, 34; Stage 2 cell 4; Stage 3 cell 5 | `pad_amount`, `pad_field`, `normalise`, `split_indices`, `snapshot_and_plane`, `list_planes`, `load_planes`, `crop_physical` (= `ex_phys`), `gaussian_baseline` |
| `model.py` | Stage 1 cells 16–18 (copied unchanged in Stage 3 cell 7) | `sinusoidal_embedding`, `conv_block_t`, `encoder_block_t`, `decoder_block_t`, `create_denoiser`, `load_denoiser` |
| `stats.py` | Stage 1 cell 32; Stage 2 cells 7, 12; Stage 3 cells 11, 22 | y+ helpers and band rows, `spectrum`, `band_spectrum`, `corr_x`, `corr_y`, `first_below`, `correlation_length`, `tau_bottom`, `log_spectral_distance`, `profile_mae`, `pdf_bins`, `pdf`, `wasserstein_1d`, `reference`, `stage2_stats` (Stage 2's `stats`), `nn_dist` (with Stage 3's snapshot exclusion) |
| `sampling.py` | Stage 1 cell 32 (`ex_sample`), Stage 2 cell 5, Stage 3 cell 9 | `ddim_timesteps`, `ddpm_mean`, `ddim_step`, `make_caller`, `sample` |
| `notebook.py` | — | The same functions under the notebooks' names (`ex_spec`, `ex_corr_x`, `stats`, …) for a notebook that is re-executed with `from diffusion_flow.notebook import *`. |

Not extracted (not pure functions): the training loops (`train_step`, `ex_train`, `train_denoiser`), which
mutate a model and an optimiser; the animation hook; all plotting, which is written inline against notebook
state. There are no standalone scripts in this repository, so there is no argument-parser smoke test.

### `tests/` (47 tests, about 13 s on a laptop CPU)

| File | Tests |
| --- | --- |
| `test_schedules.py` | Shapes and dtypes; ᾱ_T = 0.132 for the linear schedule (the README's 36 % of the signal); ᾱ_T ≈ 6e-8 for the cosine schedule; the cosine formula before clipping; `q_sample` limits. |
| `test_data.py` | 201 → 208 padding; edge rows; normalisation round trip; the 5,777 → 577 / 5,200 split and its first index; file-name parsing; loading tiny HDF5 files written in a temp dir. |
| `test_stats.py` | Spectrum of a single cosine has all its energy in one mode; mode 0 is the variance of per-sample row means; `corr_x` of a cosine and of white noise; `tau_bottom` is exact for a quadratic; Stage 2's statistics of an ensemble against itself are zero; `nn_dist` is zero for a training field, 0.5 for a field shifted by 0.5, and respects the snapshot exclusion and chunking. |
| `test_model_and_sampling.py` | 508,785 and 508,929 parameters (1- and 2-channel input); a forward pass of a 2-sample batch on the CPU; the timestep changes the output; DDIM step grid; plain and clipped DDPM means agree when clipping does not bind; a DDIM step with the true noise recovers x0; samplers are deterministic in the seed. |
| `test_notebook_equivalence.py` | Parses the three `.ipynb` files, execs their `def`s, and checks that every extracted function (schedules, spectra, correlations, τ_w, Stage 2 statistics, `nn_dist`, the denoiser with shared weights, the Stage 3 sampler in all four modes) returns the same numbers as the package on synthetic data. |
| `test_results_consistency.py` | Reads every README number back from the results files listed below; checks the notebooks parse and still carry their outputs. |

### `models/`

| File | Size | Source |
| --- | --- | --- |
| `01_diffusion_channel_flow/diffusion_denoiser.keras` | 2.1 MB | Stage 1 cell 23 (`denoiser.save`). Linear schedule, 40 epochs, seed 0. Reload with `create_denoiser((208, 384, 1))` + `load_weights` (it contains a `Lambda` layer). |
| `03_cosine_schedule_and_calibrated_copying/cosine_denoiser_seed0.keras` | 2.1 MB | Stage 3 cell 16 (`run_cosine_seed(0)`). Cosine schedule, 40 epochs, seed 0. Seeds 1 and 2 were not committed. |

### `results/01_diffusion_channel_flow/exercises/` (Stage 1)

| File | Written by | Contents |
| --- | --- | --- |
| `ex1_baseline_vs_diffusion.csv`, `ex1_fields.png`, `ex1_correlations.png` | cell 34 | Correlation lengths, wall-normal extent, profile MAEs and log-spectral distance for real / diffusion / Gaussian baseline (64 samples). |
| `ex2_conditioning_scores.csv`, `ex2_conditional_samples.png` | cell 36 | Correlation and relative RMSE between the conditioning τ_w and the samples' own τ_w, conditional vs unconditional. |
| `ex3_ddim_vs_ddpm.csv`, `ex3_ddim_vs_ddpm.png` | cell 38 | Wall time for 64 samples and quality for DDPM 200 and DDIM 50/20/10 (seed 3). |
| `ex4_schedules.csv`, `ex4_schedules.png` | cell 40 | Linear T=200, cosine T=200, cosine T=500 at 10 epochs, plus the base model row. |
| `ex5_spectra_ratios.csv`, `ex5_spectra.png` | cell 42 | Generated/real energy ratios per y+ band at large and small scales. |
| `seeds_runs.csv`, `seeds_summary.csv` | cell 52 | Experiments 2, 3 and 4 for seeds 0, 1, 2; mean/std/min/max per configuration. |

### `results/02_field_statistics_and_memorisation/` (Stage 2)

| File | Written by | Contents |
| --- | --- | --- |
| `device_timing_cpu.csv`, `device_timing_gpu.csv` | before the run (read by cell 3; a timing script that was not committed) | Median ms per training step for the denoiser and the conditional denoiser on CPU and Apple GPU. |
| `a_statistics_runs.csv`, `a_statistics_table.csv`, `a_statistics.png` | cells 9–10 | Every statistic per ensemble and seed; mean/min/max over seeds. |
| `b_nearest_training_distance.csv`, `b_nearest_distance_quantiles.csv` | cell 12 | Nearest-training-field RMS distance for 577 held-out fields and 768 samples; quantiles. |
| `b_cost_vs_quality_runs.csv`, `b_cost_vs_quality_table.csv`, `b_copying_and_cost.png` | cells 13–14 | Seconds per sample and quality for DDPM 200, DDIM 50/20/10, three seeds. |

### `results/03_cosine_schedule_and_calibrated_copying/` (Stage 3)

| File | Written by | Contents |
| --- | --- | --- |
| `a_loss_cosine_seed{0,1,2}.csv` | cells 16–18 | Per-epoch training loss of each cosine training. |
| `a_statistics_runs.csv`, `a_statistics_table.csv`, `a_verdict.csv`, `a_profiles.png` | cell 20 | Every statistic per ensemble; median/min/max; the verdict (beats the base model beyond the seed range? gap closed?). |
| `b_nearest_training_distance.csv`, `b_nearest_distance_quantiles.csv`, `b_calibrated_copying.png` | cell 22 | Nearest-training distances: held-out (any / different snapshot only), cosine samples, base samples. |
| `run_summary.json` | cell 24 | Counts, normalisation constants, training minutes and final losses per seed, the verdict, the quantiles, wall time, library versions. |

### `animations/`

`01_training_dashboard.mp4` (81 frames, 6 fps) and `01_final_frame.png`, written by Stage 1 cell 46 from the
snapshots the training hook (cell 21) recorded every half epoch.

## Where each README number comes from

Checked by `tests/test_results_consistency.py` where a results file holds the number (✓). Numbers marked
"notebook output" are in a stored output cell only; "author's run" means they come from the author's timing of the
run and no file in the repository holds them.

| README statement | Source |
| --- | --- |
| 5,777 planes; 201 × 384; Re_τ ≈ 125; 577 held-out, 5,200 training | Stage 1 cell 9/11 outputs; `results/03/run_summary.json` (`fields`, `training`, `held_out`) ✓ |
| T = 200, linear schedule, 508,785 weights | Stage 1 cells 13, 18 outputs; `device_timing_*.csv` (`params`) ✓; `tests/test_model_and_sampling.py` rebuilds the model and counts ✓ |
| Model sizes 2.1 MB (Stage 1) and 2.1 MB (Stage 3 seed 0) | File sizes in `models/` (2.11 MB each). The README used to say 2.2 MB for the Stage 3 file; corrected to 2.1 MB. |
| Timing: denoiser 588.6 vs 87.1 ms, conditional 596.8 vs 91.2 ms, GPU 6.7× faster | `results/02/device_timing_cpu.csv`, `device_timing_gpu.csv` ✓ |
| Stage 1 took 92 minutes (22 for the denoiser's training); Stage 2 took 5 minutes | Author's run; no file holds them. Stage 1 cell 3 comment repeats the 22 minutes; the markdown of cell 48 says "about 19 minutes", an inconsistency inside the notebook that is left as written. |
| Stage 3 took 62 minutes, three trainings of 18.5–19.2 minutes | `results/03/run_summary.json` (`wall_min` 62.3, `train_min_per_seed` 18.57 / 18.53 / 19.24) ✓ |
| Stage 1: mean-profile error 0.19 over 64 samples | Stage 1 cell 29 output (0.1920); `results/01/exercises/ex1_baseline_vs_diffusion.csv` (`diffusion`, `mean_mae`) ✓ |
| "about 1.06 at the centre" | Stage 1 cell 29 figure (notebook output); no table holds the profile. |
| 36 % of the signal at the last noise level, ᾱ_T = 0.132 | Stage 1 cell 13 output (0.13218); √0.132 = 0.36; `tests/test_schedules.py` ✓ |
| Cosine at 10 epochs halves the error: 0.095 [0.079, 0.115] vs 0.187 [0.178, 0.201] | `results/01/exercises/seeds_summary.csv` (`mean_mae` mean/min/max of `cosine, T=200, 10 epochs` and `linear, T=200, 10 epochs`) ✓ |
| Stage 2: 768 samples, 200 steps, three seeds | `results/02/a_statistics_runs.csv` (3 × 256) ✓ |
| Spectral distance at y+ ≈ 15: 0.44 vs 1.33 | `results/02/a_statistics_table.csv` (`spectral_distance_y15`, mean) ✓ |
| Correlation length 0.39 vs 0.07, 1.18 in the real fields | same file, `corr_length_y15` ✓ |
| Mean-profile error 0.19 vs 0.0007; larger PDF distance | same file, `mean_profile_mae`, `pdf_wasserstein_y15` ✓ |
| Copying: 0.229 median vs 0.026 | `results/02/b_nearest_distance_quantiles.csv` (`50%`) ✓ |
| Cost: 0.014 s vs 0.255 s per sample, 18×; 0.215 vs 0.193; 10/20/50 within 0.001 | `results/02/b_cost_vs_quality_table.csv` (`seconds_per_sample`, `mean_profile_mae`, mean) ✓ |
| Stage 3: 0.19 → 0.068 (0.043–0.084), beyond the seed range, gap to 0.0007 not closed | `results/03/a_verdict.csv` (`mean_profile_mae` row) ✓ |
| RMS profile 0.052 vs 0.030 | `results/03/a_verdict.csv` (`rms_profile_mae` row) ✓ |
| Correlation and PDF improve beyond the range; spectra do not | `results/03/a_verdict.csv` (`cosine_beats_base` of `corr_mae_y15`, `pdf_wasserstein_y15`, `spectral_distance_*`) ✓ |
| Ten-step DDIM loses the gain (0.23) | `results/03/a_statistics_table.csv` (`cosine, DDIM 10 clipped`, `mean_profile_mae`, median) ✓ |
| New real field 0.109 (0.090–0.124) from the training set, four times 0.026 | `results/03/b_nearest_distance_quantiles.csv` (`different snapshot only` and `any training field`, `50%`/`min`/`max`) ✓ |
| Cosine samples 0.169, base 0.229; 450 distinct nearest fields in 768 vs 13 in Stage 2 | same file (`50%`); `results/03/b_nearest_training_distance.csv` and `results/02/b_nearest_training_distance.csv` (`nn_index.nunique()`) ✓ |
| Mean-profile error 0.22 in an earlier run of the base training, 0.19 in this one | Author's earlier run; not in the repository. |
| Seeded experiment trainings reproduced an earlier run to every printed digit | Author's statement; not checkable here. |
