# A diffusion model for channel-flow fields, held to held-out statistics

A regression network trained with a mean-squared error returns the average of every field consistent with its input, which is smoother than any real turbulent snapshot. A generative model should instead draw new fields, one realisation at a time. These three notebooks train a denoising diffusion probabilistic model (DDPM) on 5,777 y–x planes of the streamwise velocity u from a direct numerical simulation of turbulent channel flow (Re_τ ≈ 125, 201 × 384 points), then check what it actually produces. Stage 1 trains the noise-predicting U-Net (T = 200 noise levels, linear schedule, 508,785 weights) and solves five experiments: a Gaussian baseline, a denoiser conditioned on the wall shear stress, faster DDIM sampling, a cosine schedule and a longer chain, and energy spectra. Stage 2 scores the generator against 577 held-out fields it never saw. It looks at profiles, spectra, two-point correlations and the probability density, asks whether the samples are copies of training fields, and weighs the cost of sampling against quality. Stage 3 trains the denoiser with the cosine schedule for the full 40 epochs, three seeds, scores it on the same held-out statistics, and calibrates the copying test with the distance a genuinely new field has from the training set.

Portfolio page: https://vamsigudipati.github.io/diffusion-channel-flow-fields.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_diffusion_channel_flow.ipynb` | Stage 1 — A diffusion model for synthetic channel-flow fields | `results/01_diffusion_channel_flow/exercises/` | `models/01_diffusion_channel_flow/diffusion_denoiser.keras` (the trained denoiser, 2.1 MB; reload with `create_denoiser` + `load_weights`, since it contains a `Lambda` layer) | `animations/01_*` |
| `notebooks/02_field_statistics_and_memorisation.ipynb` | Stage 2 — Field statistics and memorisation | `results/02_field_statistics_and_memorisation/` | — (uses the Stage 1 denoiser; no training) | — |
| `notebooks/03_cosine_schedule_and_calibrated_copying.ipynb` | Stage 3 — The cosine schedule at full training, and a copying test with a calibrated distance | `results/03_cosine_schedule_and_calibrated_copying/` | `models/03_cosine_schedule_and_calibrated_copying/cosine_denoiser_seed0.keras` (the seed-0 cosine denoiser, 2.2 MB; reload as for Stage 1's) | — |

Each notebook is also exported as HTML next to it, with all outputs. `exercises/` holds the tables (CSV) and figures (PNG) of Stage 1's experiments and the three-seed repeats (`seeds_runs.csv`, `seeds_summary.csv`). Stage 2 writes its tables, its figures and the CPU/GPU timing files into its results folder; Stage 3 writes its tables, figures and per-seed loss histories into its own. `animations/` holds the training video and its final frame.

The outputs in the notebooks come from my single run of each notebook. For this repository I changed only text afterwards: a header cell and a paths cell in each notebook, the path lines (data, model, outputs), comments, labels and printed text. The notebooks were not executed again in this layout, so the paths cell is the one cell without an output. The title of the training video was replaced with a plain one; its frames are otherwise those of the run.

## Data

No new upload. The planes are the same 5,777 files as in my U-Net project and come with that repository's release, https://github.com/vamsigudipati/unet-turbulent-fields:

| Release file | Size | Goes to | SHA-256 |
| --- | --- | --- | --- |
| `u_yx_planes_h5.tar` (from the unet-turbulent-fields release) | 1.6 GB | extract into `release_assets/`, giving `release_assets/u_yx_planes_h5/` (5,777 files `P125_83pi.<snapshot>.<plane>.h5`) | `410518b833edc2fd5bf1b5c569fd2ed35ea15824ed924117958ef8c1edefadd5` |

`release_assets/` is ignored by git. The trained denoisers (Stage 1, 2.1 MB; Stage 3 seed 0, 2.2 MB) are in git; its 5-epoch checkpoints are not. My run trained every model here; no pretrained model is used.

## Setting up and running

I ran everything on a laptop (Apple M3 Max, 48 GB) with Python 3.12, TensorFlow 2.18.1, Keras 3.6 and `tensorflow-metal` for the Apple GPU. The frozen environment is in `requirements.txt`. The video uses the ffmpeg binary that `imageio-ffmpeg` installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name py312-tf-metal-studio
tar -xf u_yx_planes_h5.tar -C release_assets/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 01_diffusion_channel_flow.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_field_statistics_and_memorisation.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 03_cosine_schedule_and_calibrated_copying.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there; Stage 2 needs the denoiser that Stage 1 saves; Stage 3 needs that denoiser and Stage 2's tables. On the laptop, Stage 1 took 92 minutes (22 of them for the denoiser's training), Stage 2 took 5 minutes and Stage 3 took 62 minutes (three cosine trainings of 18.5–19.2 minutes).

**Why the Apple GPU.** Before training, I timed one training step of each network inside `model.fit` (batch 16, median of 30 steps after 10 warm-up steps) on the CPU and on the Apple GPU (`results/02_field_statistics_and_memorisation/device_timing_*.csv`). In milliseconds per step: the denoiser 588.6 vs 87.1, the conditional denoiser with two input channels 596.8 vs 91.2. The GPU is about 6.7 times faster, so all three notebooks run on it (Stage 3 reuses the timings; the environment is unchanged).

**Why three seeds, and where there is only one.** A single training gives one draw of the initial weights, of the batch order and of the noise. The denoiser's 40-epoch training took 22 minutes, so I ran it once (seed 0, the base run) and say so wherever its numbers appear. The experiment trainings (about 5 minutes each) and every sampler are repeated with the seeds 0, 1 and 2. I report the mean and the range, and a difference smaller than that range is not an effect. With a fixed seed, the seeded experiment trainings reproduced an earlier run of mine to every printed digit on the GPU. The base training, which the notebook seeds only once at the top rather than directly before training, did not (mean-profile error 0.22 in the earlier run, 0.19 in this one); I have not established why. In Stage 3 each cosine training (18.5–19.2 minutes) is repeated with three model seeds all the same, because the base model's seed spread was never measured and measuring it was part of the experiment; the ranges there are over model seeds, not sampling seeds.

## What the stages show

- Stage 1: the generated fields have turbulent-looking structure, but their mean profile is wrong. Over 64 samples from the 200-step sampler, the mean-profile error is 0.19, against a velocity that runs from 0 at the walls to about 1.06 at the centre. The linear schedule leaves 36 % of the signal at the last noise level (ᾱ_T = 0.132) while sampling starts from pure noise. A cosine schedule, which does reach pure noise, halves the error at 10 epochs: 0.095 [0.079, 0.115] against 0.187 [0.178, 0.201] over three seeds. That points to the schedule as a likely cause.
- Stage 2, statistics against held-out fields (768 samples, 200 steps, three seeds): the diffusion samples beat independent Gaussian pixels on spatial structure. The spectral distance at y+ ≈ 15 is 0.44 against 1.33, and the correlation length is 0.39 against 0.07, with 1.18 in the real fields. They lose on one-point statistics: mean-profile error 0.19 against 0.0007, and a larger distance between the probability densities of u. Both results hold beyond the seed range.
- Stage 2, copying: no generated field is a copy. The nearest training field is 0.229 away (median), against 0.026 for the held-out real fields. The held-out fields are that close because the random split puts neighbouring planes of the same snapshots into training. So the test shows the samples are not copies; it does not show they are realistic new fields.
- Stage 2, cost: 10-step DDIM takes 0.014 s per sample against 0.255 s for 200-step DDPM, 18 times faster. Its mean-profile error is slightly worse (0.215 against 0.193); the 10-, 20- and 50-step versions are within 0.001 of each other.
- Stage 3, the cosine schedule at 40 epochs: the mean-profile error falls from 0.19 to 0.068 (range 0.043–0.084 over three model seeds), beyond the seed range, and the generated mean profile now follows the real one from the walls to the centre. It does not close the gap to a real ensemble (0.0007), and the RMS profile is too high everywhere (0.052 against 0.030 for the base model). Two-point correlation and the density of u improve beyond the range; the spectra do not; ten-step DDIM loses the mean-profile gain (0.23).
- Stage 3, calibrated copying: a new real field lies 0.109 (range 0.090–0.124) from the training set once planes of its own snapshot are excluded, four times Stage 2's 0.026. The cosine model's samples sit at 0.169 and the base model's at 0.229: not copies, still farther from the training set than a new real field, and the cosine model's samples are closer and more diverse (450 distinct nearest training fields in 768 samples, against 13 in Stage 2).
