"""
render_test_sample_analysis.py — the FULL per-device analysis for every ML
test device, plus the model's prediction on the grid.

    python render_test_sample_analysis.py

The ML test devices (training_data/ml_test_split_n5_res100) keep only three
arrays per device — the bulk generator prunes every figure.  This program
brings back, for each test device, everything the original pipeline produced
for a device (the sample_4-style folder):

    charge_sensing.jpg, images/, rays + per-ray plots, peaks.json,
    summary.png, summary_total.png, summary_peaks_only.png,
    summary_total_all_crosses.png, cropped_results/ (peak crops + sweeps),
    gifs/, evaluation.txt ...

and ADDS the ML study's own panels, drawn in the same house style:

    ml_truth.png        ground-truth transition lines on the grid
    ml_prediction.png   what the trained model draws from 6 rays x 100 points
    ml_overlay.png      truth (black cells) + prediction (red x) together

HOW THE DEVICES COME BACK.  The test devices are a pure function of the
recorded settings: capacitance matrices are drawn from one seeded stream
(seed = SEED + 1000 = 1000, test half of the disjoint intervals).  Replaying
that stream gives device i the exact matrices it had, and the pipeline is run
on those via fixed_matrices.  The regenerated charge_sensing_data.npy is
CHECKED against the stored test array — the program refuses to continue if a
device does not reproduce bit-for-bit physics (tiny float noise allowed).

Everything goes into runs/<timestamp>_testanalysis/sample_<i>/ with a
config.json — nothing existing is touched.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from dqd.config import paths
from dqd.config.capacitance_config import CapacitanceConfig
from dqd.ml import grid_train, run_dir
from dqd.ml.train_test_config import split_configs
from dqd.pipeline.dataset_pipeline import DatasetPipeline
from dqd.simulation.matrix_generator import CapacitanceMatrixGenerator

from render_device_figures import render_sample

# ══════════════════════════════════════════════════════════════════════
#  SETTINGS — must match the run that made the test set (its config.json)
# ══════════════════════════════════════════════════════════════════════

TEST_DIR = paths.training_data("ml_test_split_n5_res100")
N_TEST = 5
SEED = 0                     # run_experiment's SEED; test stream is SEED+1000

# The measurement budget of both the pipeline rays and the ML panels.
N_RAYS = 6
N_POINTS = 100

# Checkpoint for ml_prediction.png / ml_overlay.png.  None = newest one any
# run produced for this budget.
MODEL_PATH = None

# Sweep GIFs are the slowest part of the analysis; 100 dpi keeps them cheap.
SAVE_GIFS = True
GIF_DPI = 100

# ══════════════════════════════════════════════════════════════════════


def replay_matrices(n_test=N_TEST, seed=SEED, disjoint_intervals=True):
    """The exact capacitance matrices of test devices 1..n_test, in order."""
    if disjoint_intervals:
        _, test_cfg = split_configs()
    else:
        test_cfg = CapacitanceConfig()
    gen = CapacitanceMatrixGenerator(seed=seed + 1000)
    return [gen.generate_all(test_cfg) for _ in range(n_test)]


def check_same_device(analysis_dir, test_sdir, i, test_dir=TEST_DIR):
    """Refuse to continue if the replayed device is not the stored one."""
    a = np.load(os.path.join(analysis_dir, "numpy", "simulation",
                             "charge_sensing_data.npy"))
    b = np.load(os.path.join(test_sdir, "numpy", "simulation",
                             "charge_sensing_data.npy"))
    if a.shape != b.shape or not np.allclose(a, b, atol=1e-10):
        sys.exit(f"sample_{i}: replayed device does not match the stored "
                 f"test device — the settings above disagree with the run "
                 f"that generated {os.path.abspath(test_dir)}")
    print(f"  sample_{i}: replayed device matches the stored test arrays")


def analyse_test_devices(out, test_dir, net, thr,
                         n_test=N_TEST, seed=SEED,
                         n_rays=N_RAYS, n_points=N_POINTS,
                         resolution=100,
                         voltage_window=(-1.0, 1.0, -1.0, 1.0),
                         coulomb_peak_width=0.01, temperature=0.00001,
                         disjoint_intervals=True,
                         save_gifs=SAVE_GIFS, gif_dpi=GIF_DPI):
    """
    The full per-device analysis for every test device, into
    <out>/sample_<i>/: the whole original-pipeline output (rays, peaks.json,
    summary*.png, cropped_results/, gifs/, evaluation.txt ...) plus the ML
    panels (ml_stability / ml_measurement / ml_truth / ml_prediction /
    ml_overlay).  Every replayed device is checked bit-for-bit against the
    stored test arrays before anything ML is drawn from them.
    """
    vx_min, vx_max, vy_min, vy_max = voltage_window
    # One pipeline, configured exactly like the original project's
    # run_simulation.py; fixed_matrices is swapped per device.
    pipeline = DatasetPipeline(
        base_save_dir=out,
        n_samples=n_test,
        num_angles=n_rays,
        ray_resolution=n_points,
        x_resolution=resolution,
        y_resolution=resolution,
        vx_min=vx_min, vx_max=vx_max, vy_min=vy_min, vy_max=vy_max,
        crop_size=1,
        col_buffer=2,
        coulomb_peak_width=coulomb_peak_width,
        temperature=temperature,
        plot_dpi=300,
        save_gifs=save_gifs,
        gif_dpi=gif_dpi,
        x_axis_name="P1", y_axis_name="P2",
        x_axis_unit="mV", y_axis_unit="mV",
        figure_width_in=12.0, figure_height_in=12.0,
        peak_neighbor_cols=2,
    )

    matrices_list = replay_matrices(n_test, seed, disjoint_intervals)
    for i, matrices in enumerate(matrices_list, 1):
        sample_out = os.path.join(out, f"sample_{i}")
        print(f"\n{'=' * 60}\n  test device {i}/{n_test}\n"
              f"  {os.path.abspath(sample_out)}\n{'=' * 60}")
        os.makedirs(sample_out, exist_ok=True)
        pipeline.fixed_matrices = matrices
        pipeline._run_sample(i, sample_out)

        test_sdir = os.path.join(test_dir, f"sample_{i}")
        check_same_device(sample_out, test_sdir, i, test_dir)

        # The ML panels, from the SAME stored arrays the model was scored on.
        render_sample(test_sdir, "ml", sample_out, net, thr,
                      n_rays=n_rays, n_points=n_points)


def main():
    if not os.path.isdir(TEST_DIR):
        sys.exit(f"no test set at {os.path.abspath(TEST_DIR)}")

    model_path = MODEL_PATH or run_dir.find_file(
        os.path.join("models", grid_train.checkpoint_name(N_RAYS, N_POINTS)))
    if not (model_path and os.path.isfile(model_path)):
        sys.exit(f"no checkpoint for {N_RAYS} rays x {N_POINTS} points — "
                 "run train_model.py first")
    net, ck = grid_train.load(model_path)
    thr = ck["threshold"]
    print(f"model: {os.path.abspath(model_path)}  (threshold {thr})")

    out = run_dir.new_run("testanalysis", {
        "test_dir": os.path.abspath(TEST_DIR), "n_test": N_TEST,
        "seed": SEED, "n_rays": N_RAYS, "n_points": N_POINTS,
        "model_path": os.path.abspath(model_path),
        "save_gifs": SAVE_GIFS, "gif_dpi": GIF_DPI})

    analyse_test_devices(out, TEST_DIR, net, thr)

    print(f"\neverything is in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()


