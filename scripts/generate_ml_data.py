"""
generate_ml_data.py — MAIN PROGRAM 1 of 4.  Make the devices.

    1. generate_ml_data.py    <- you are here
    2. train_model.py         train one budget
    3. evaluate_model.py      score a checkpoint
    4. run_budget_sweep.py    the rays x points table

Edit the settings block below and run it.  No command-line arguments —
everything you would want to change lives in main(), the same way
run_simulation.py works.

    python generate_ml_data.py

run_simulation.py runs the FULL analysis pipeline per sample: ray plots, peak
crops, four sweeps per peak, GIFs, overlays, summaries.  That is what you want
when you are inspecting one device.  It is far too slow when you need
thousands, and the ML study needs exactly two files per sample:

    numpy/simulation/charge_sensing_data.npy    the measurement
    numpy/simulation/ground_truth_labels.npy    the answer

So this runs only the simulator and the ground-truth extraction.  The rays are
cut later, offline, at whatever budget you ask for (dqd.ml.ray_peaks) — you
never regenerate data to change n_rays or n_points.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dqd.config.axis_labels import set_axis_labels
from dqd.config.capacitance_config import CapacitanceConfig
from dqd.ml.train_test_config import describe, split_configs
from dqd.simulation.dqd_simulator import DQDSimulator
from dqd.simulation.matrix_generator import CapacitanceMatrixGenerator
from dqd.visualization.overlay import OverlayRenderer

KEEP = ("charge_sensing_data.npy", "double_dot_data.npy")

# Fixed simulation settings.  Here rather than in the settings block because
# changing them changes what a "device" is, and every device in one study has
# to agree.  run_simulation.py uses the same values.
VMIN, VMAX = -1.0, 1.0
COULOMB_PEAK_WIDTH = 0.01
TEMPERATURE = 0.00001
SEED = 0                    # one reproducible RNG stream per split
KEEP_IMAGES = False         # True keeps the per-device .jpg previews


def generate(out_dir, n_samples, config, resolution, seed,
             vmin=VMIN, vmax=VMAX, coulomb_peak_width=COULOMB_PEAK_WIDTH,
             temperature=TEMPERATURE, keep_images=KEEP_IMAGES):
    set_axis_labels(x_name="P1", y_name="P2", x_unit="mV", y_unit="mV")
    gen = CapacitanceMatrixGenerator(seed=seed)      # one stream, reproducible
    os.makedirs(out_dir, exist_ok=True)

    t0, made = time.time(), 0
    for i in range(1, n_samples + 1):
        sample_dir = os.path.join(out_dir, f"sample_{i}")
        sim_dir = os.path.join(sample_dir, "numpy", "simulation")
        gt_path = os.path.join(sim_dir, "ground_truth_labels.npy")
        if os.path.isfile(gt_path):
            continue                                  # resume, don't redo
        os.makedirs(sim_dir, exist_ok=True)

        sim_params = {
            "save_dir": sample_dir,
            "capacitance": gen.generate_all(config),
            "model_params": {"coulomb_peak_width": coulomb_peak_width,
                             "T": temperature},
            "xlabel": "P1 (mV)", "ylabel": "P2 (mV)",
            "voltage_sweep": {"vx_min": vmin, "vx_max": vmax,
                              "vy_min": vmin, "vy_max": vmax,
                              "n_points_x": resolution,
                              "n_points_y": resolution},
            "optimal_Vg": [0.0, 0.0, 0.0],
            "plot_options": {
                "charge_sensing_save_path": os.path.join(
                    sample_dir, "charge_sensing.jpg"),
                "charge_sensing_grad_save_path": os.path.join(
                    sample_dir, "charge_sensing2.jpg"),
                "dpi": 60,          # the jpgs are a by-product; keep them cheap
            },
        }
        try:
            DQDSimulator(sim_params).run()
        except Exception as exc:
            print(f"[skip] sample {i}: {exc}")
            continue

        # The simulator writes into sample_dir; move the arrays we need into
        # the numpy/simulation/ layout the ML loaders expect.
        for name in KEEP:
            src = os.path.join(sample_dir, name)
            if os.path.isfile(src):
                os.replace(src, os.path.join(sim_dir, name))

        OverlayRenderer.generate_ground_truth_array(
            data_path=os.path.join(sim_dir, "double_dot_data.npy"),
            output_npy_path=gt_path,
            use_double_dot=True,
        )
        if not keep_images:
            _prune(sample_dir, sim_dir)

        made += 1
        if made % 25 == 0:
            rate = (time.time() - t0) / made
            print(f"  {i}/{n_samples}  {rate:.2f}s/sample  "
                  f"~{(n_samples - i) * rate / 60:.1f} min left")

    print(f"{made} new samples in {out_dir}  ({time.time() - t0:.0f}s)\n")


def _prune(sample_dir, sim_dir):
    """Delete everything except the arrays the ML study reads."""
    for root, dirs, files in os.walk(sample_dir, topdown=False):
        for f in files:
            p = os.path.join(root, f)
            if os.path.dirname(p) == sim_dir and f in (
                    "charge_sensing_data.npy", "double_dot_data.npy",
                    "ground_truth_labels.npy"):
                continue
            try:
                os.remove(p)
            except OSError:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except OSError:
                pass


def main():
    # ══════════════════════════════════════════════════════════════════
    #  SETTINGS
    # ══════════════════════════════════════════════════════════════════

    # How many devices to make.  Reconstructing lines from a few dozen peaks
    # needs on the order of a thousand devices before the numbers mean
    # anything.  At ~0.7 s/device, 2000 + 500 is about half an hour.
    N_TRAIN = 2000
    N_TEST = 500

    # Stability-diagram side length in pixels.  This fixes the network's
    # input size, so every device in one study must share it.
    RESOLUTION = 100

    # True  : train and test devices come from DISJOINT capacitance intervals
    #         (see dqd/ml/train_test_config.py), so the test devices have
    #         geometry the training devices could not have had.  This is the
    #         claim worth publishing.
    # False : both halves use the full intervals — only shows interpolation.
    DISJOINT_INTERVALS = True

    # ══════════════════════════════════════════════════════════════════

    if DISJOINT_INTERVALS:
        train_cfg, test_cfg = split_configs()
        print("disjoint capacitance intervals:\n" + describe() + "\n")
    else:
        train_cfg = test_cfg = CapacitanceConfig()
        print("train and test share the full capacitance intervals\n")

    out_root = os.path.join("..", "training_data")
    tag = "split" if DISJOINT_INTERVALS else "same"

    print(f"── TRAIN: {N_TRAIN} devices " + "─" * 40)
    generate(os.path.join(out_root, f"ml_train_{tag}_n{N_TRAIN}_res{RESOLUTION}"),
             N_TRAIN, train_cfg, RESOLUTION, seed=SEED)

    print(f"── TEST: {N_TEST} devices " + "─" * 41)
    # A different seed stream, so the two splits never line up device for
    # device on the entries that are NOT split.
    generate(os.path.join(out_root, f"ml_test_{tag}_n{N_TEST}_res{RESOLUTION}"),
             N_TEST, test_cfg, RESOLUTION, seed=SEED + 1000)


if __name__ == "__main__":
    main()
