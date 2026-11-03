"""Every number quoted in README.md is read back from the results files it is traced to in docs/REPO_MAP.md."""

import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "results/01_diffusion_channel_flow/exercises"
R2 = ROOT / "results/02_field_statistics_and_memorisation"
R3 = ROOT / "results/03_cosine_schedule_and_calibrated_copying"


def table(path):
    return pd.read_csv(path, header=[0, 1], index_col=0)


def test_device_timing():
    t = pd.concat([pd.read_csv(R2 / "device_timing_cpu.csv"), pd.read_csv(R2 / "device_timing_gpu.csv")])
    tt = t.pivot(index="network", columns="device", values="median_ms")
    den = [r for r in tt.index if r.startswith("denoiser")][0]; cond = [r for r in tt.index if r.startswith("conditional")][0]
    assert (round(tt.loc[den, "cpu"], 1), round(tt.loc[den, "gpu"], 1)) == (588.6, 87.1)
    assert (round(tt.loc[cond, "cpu"], 1), round(tt.loc[cond, "gpu"], 1)) == (596.8, 91.2)
    assert tt.loc[den, "cpu"] / tt.loc[den, "gpu"] == pytest.approx(6.7, abs=0.1)
    assert set(t.params) == {508785, 508929}


def test_stage1_numbers():
    ex1 = pd.read_csv(R1 / "ex1_baseline_vs_diffusion.csv").set_index("ensemble")
    assert ex1.loc["diffusion", "mean_mae"] == pytest.approx(0.19, abs=0.005)           # "mean-profile error is 0.19"
    seeds = pd.read_csv(R1 / "seeds_summary.csv", header=[0, 1], index_col=[0, 1])
    cos = seeds.xs("cosine, T=200, 10 epochs", level=1).iloc[0]; lin = seeds.xs("linear, T=200, 10 epochs", level=1).iloc[0]
    assert (round(cos[("mean_mae", "mean")], 3), round(cos[("mean_mae", "min")], 3), round(cos[("mean_mae", "max")], 3)) == (0.095, 0.079, 0.115)
    assert (round(lin[("mean_mae", "mean")], 3), round(lin[("mean_mae", "min")], 3), round(lin[("mean_mae", "max")], 3)) == (0.187, 0.178, 0.201)
    runs = pd.read_csv(R1 / "seeds_runs.csv")
    assert sorted(runs.seed.unique()) == [0, 1, 2]
    ex4 = pd.read_csv(R1 / "ex4_schedules.csv")
    assert ex4.loc[0, "epochs"] == 40 and (ex4.loc[1:, "epochs"] == 10).all()


def test_stage2_numbers():
    a = table(R2 / "a_statistics_table.csv")
    diff = a.loc["diffusion (DDPM, 200 steps)"]; gauss = a.loc["Gaussian baseline (independent pixels)"]
    real = a.loc["training fields (real)"]
    assert round(diff[("spectral_distance_y15", "mean")], 2) == 0.44 and round(gauss[("spectral_distance_y15", "mean")], 2) == 1.33
    assert round(diff[("corr_length_y15", "mean")], 2) == 0.39 and round(gauss[("corr_length_y15", "mean")], 2) == 0.07
    assert round(real[("corr_length_y15", "mean")], 2) == 1.18
    assert round(diff[("mean_profile_mae", "mean")], 2) == 0.19 and round(gauss[("mean_profile_mae", "mean")], 4) == 0.0007
    assert diff[("pdf_wasserstein_y15", "mean")] > gauss[("pdf_wasserstein_y15", "mean")]
    # "beyond the seed range": the min–max ranges do not overlap
    assert diff[("spectral_distance_y15", "max")] < gauss[("spectral_distance_y15", "min")]
    assert diff[("corr_length_y15", "min")] > gauss[("corr_length_y15", "max")]
    runs = pd.read_csv(R2 / "a_statistics_runs.csv")
    assert runs[runs.ensemble.str.startswith("diffusion")].n.sum() == 768                 # 3 x 256 samples
    assert int(runs[runs.ensemble == "held-out fields (real)"].n.iloc[0]) == 577
    assert int(runs[runs.ensemble == "training fields (real)"].n.iloc[0]) == 5200

    q = pd.read_csv(R2 / "b_nearest_distance_quantiles.csv", index_col=0)
    assert round(q.loc["diffusion samples", "50%"], 3) == 0.229 and round(q.loc["held-out real fields", "50%"], 3) == 0.026
    nn = pd.read_csv(R2 / "b_nearest_training_distance.csv")
    assert nn[nn.ensemble == "diffusion samples"].nn_index.nunique() == 13                # "against 13 in Stage 2"

    cb = table(R2 / "b_cost_vs_quality_table.csv")
    cb.index = cb.index.get_level_values(0)
    assert round(cb.loc["DDIM 10", ("seconds_per_sample", "mean")], 3) == 0.014 and round(cb.loc["DDPM 200", ("seconds_per_sample", "mean")], 3) == 0.255
    assert cb.loc["DDPM 200", ("seconds_per_sample", "mean")] / cb.loc["DDIM 10", ("seconds_per_sample", "mean")] == pytest.approx(18, abs=0.5)
    assert round(cb.loc["DDIM 10", ("mean_profile_mae", "mean")], 3) == 0.215 and round(cb.loc["DDPM 200", ("mean_profile_mae", "mean")], 3) == 0.193
    ddim = cb.loc[["DDIM 10", "DDIM 20", "DDIM 50"], ("mean_profile_mae", "mean")]
    assert ddim.max() - ddim.min() < 0.001


def test_stage3_numbers():
    v = pd.read_csv(R3 / "a_verdict.csv").set_index("statistic")
    mp = v.loc["mean_profile_mae"]
    assert (round(mp.cosine_median, 3), round(mp.cosine_min, 3), round(mp.cosine_max, 3)) == (0.068, 0.043, 0.084)
    assert round(mp.base_model, 2) == 0.19 and round(mp.real_to_real, 4) == 0.0007
    assert mp.cosine_beats_base and not mp.gap_closed
    rms = v.loc["rms_profile_mae"]
    assert round(rms.cosine_median, 3) == 0.052 and round(rms.base_model, 3) == 0.030
    assert v.loc["corr_mae_y15"].cosine_beats_base and v.loc["pdf_wasserstein_y15"].cosine_beats_base
    assert not v.loc["spectral_distance_y15"].cosine_beats_base and not v.loc["spectral_distance_all_rows"].cosine_beats_base
    a = table(R3 / "a_statistics_table.csv")
    assert round(a.loc["cosine, DDIM 10 clipped", ("mean_profile_mae", "median")], 2) == 0.23

    q = pd.read_csv(R3 / "b_nearest_distance_quantiles.csv", index_col=0)
    diffsnap = q.loc["held-out real fields, different snapshot only"]
    assert (round(diffsnap["50%"], 3), round(diffsnap["min"], 3), round(diffsnap["max"], 3)) == (0.109, 0.090, 0.124)
    assert round(q.loc["held-out real fields, any training field", "50%"], 3) == 0.026
    assert round(q.loc["cosine model samples (DDPM 200 clipped)", "50%"], 3) == 0.169
    assert round(q.loc["base model samples (DDPM 200)", "50%"], 3) == 0.229
    assert diffsnap["50%"] / q.loc["held-out real fields, any training field", "50%"] == pytest.approx(4, abs=0.2)
    nn = pd.read_csv(R3 / "b_nearest_training_distance.csv")
    cos = nn[nn.ensemble.str.startswith("cosine")]
    assert len(cos) == 768 and cos.nn_index.nunique() == 450

    s = json.loads((R3 / "run_summary.json").read_text())
    assert (s["fields"], s["training"], s["held_out"], s["T"], s["epochs"]) == (5777, 5200, 577, 200, 40)
    assert min(s["train_min_per_seed"]) == pytest.approx(18.5, abs=0.05) and max(s["train_min_per_seed"]) == pytest.approx(19.2, abs=0.05)
    assert s["wall_min"] == pytest.approx(62, abs=1)
    assert s["tensorflow"] == "2.18.1" and s["keras"] == "3.6.0"
    for seed in (0, 1, 2):
        assert len(pd.read_csv(R3 / f"a_loss_cosine_seed{seed}.csv")) == 40


def test_notebooks_parse_and_declare_their_outputs():
    """The .ipynb files are valid JSON whose code cells carry stored outputs (nothing was cleared)."""
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        d = json.loads(nb.read_text())
        code = [c for c in d["cells"] if c["cell_type"] == "code"]
        with_out = sum(1 for c in code if c.get("outputs"))
        assert with_out >= 0.6 * len(code), nb.name
        assert (nb.with_suffix(".html")).exists(), f"{nb.name} has no HTML export"
