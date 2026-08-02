"""
run_experiment.py — THE experiment, end to end, in one program.

    python run_experiment.py

Edit the SETTINGS block below and run it.  It does the whole study:

    STAGE 1  make the devices        (simulator)
    STAGE 2  the rays x points sweep (train + evaluate every budget)
    STAGE 3  summary table + figures (budget_sweep.csv, figures/*.png)
    STAGE 4  full analysis of every test device (sample_<i>/ folders)

The question it answers: how many rays, at what ray resolution, does it take
to recover the transition lines?  For every (rays, points) pair it trains the
model from scratch and scores it against the classical Hough baseline on the
held-out test devices.

WHERE THINGS GO — nothing an earlier run made is ever touched.

    runs/<timestamp>_experiment/     one folder per run of this program
      config.json                    every setting that produced it
      budget_sweep.csv               the results
      models/                        one checkpoint per budget
      figures/                       for make_figures.py
      sample_<i>/                    STAGE 4: the full per-device analysis of
                                     every test device (rays, peaks.json,
                                     summary*.png, cropped_results/, gifs/,
                                     ml_* panels) — render_test_sample_analysis
                                     output, inside the run itself

    training_data/ml_*_n*_res*/      the devices
    grid_cache/                      the measured rays

The devices and the cut rays deliberately live OUTSIDE the run folder,
because both are a pure function of the settings recorded in config.json:
the same settings always give the same devices, so a second run reuses them
instead of spending half an hour re-simulating identical data.  config.json
names the exact dataset folders used, so a result is still traceable back to
the devices it came from.  Change N_TRAIN, RESOLUTION or DISJOINT_INTERVALS
and you get a differently-named dataset folder — a run can never silently
inherit devices made under different settings.

What makes the sweep a fair comparison: the architecture and every training
setting are identical in every cell, because they are constants in src/dqd/ml/
rather than knobs.  Only the measurement changes.

The old separate programs still work and do exactly these stages:
generate_ml_data.py (stage 1), train_model.py / evaluate_model.py (one cell of
stage 2), make_figures.py (stage 3 on its own, for re-drawing an old trial).
The figure code itself lives in make_figures.py and is imported from there —
one copy, same plots either way.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")            # no display: this runs unattended for hours

from dqd.config.capacitance_config import CapacitanceConfig
from dqd.config import paths
from dqd.ml import grid_dataset, grid_train, run_dir
from dqd.ml.grid_baseline import score_hough
from dqd.ml.grid_metrics import evaluate
from dqd.ml.train_test_config import describe, split_configs
from dqd.simulation import device_factory

# The figures, from the program that owns them.  scripts/ is on sys.path when
# either program is run directly, so a sibling import is enough.
from make_figures import fig_budget_coverage, fig_budget_rays, fig_examples
from render_test_sample_analysis import analyse_test_devices

# ══════════════════════════════════════════════════════════════════════
#  SETTINGS — everything you can change is here
# ══════════════════════════════════════════════════════════════════════

# ---- STAGE 1: the devices --------------------------------------------

# How many devices to make.  Recovering lines from a few dozen peaks needs on
# the order of a thousand training devices before the numbers mean anything.
# At ~0.7 s/device, 2000 + 500 is about half an hour — paid once, then reused
# by every later run with the same settings.
N_TRAIN = 200
N_TEST = 10

# Stability-diagram side length in pixels.  Fixes the network's input size,
# so every device in one study must share it.
RESOLUTION = 100

# True  : train and test devices come from DISJOINT capacitance intervals
#         (dqd/ml/train_test_config.py), so the test devices have geometry the
#         training devices could not have had.  The publishable claim.
# False : both use the full intervals — only shows interpolation.
DISJOINT_INTERVALS = True

# Gate-voltage window swept on both axes.
VOLTAGE_WINDOW = (-1.0, 1.0, -1.0, 1.0)      # vx_min, vx_max, vy_min, vy_max

# Simulator physics.
COULOMB_PEAK_WIDTH = 0.01
TEMPERATURE = 0.00001

# One reproducible random stream per split.  Same seed = same devices.
SEED = 0

# True keeps the per-device .jpg previews.  Off for a bulk run — it saves a
# lot of disk, and nothing downstream reads them.
KEEP_IMAGES = False

# ---- STAGE 2: the sweep ----------------------------------------------

# Every combination of these is trained and scored, so this is
# len(RAYS) x len(POINTS) full trainings.  Start small, widen once the shape
# of the curve is clear.
RAYS = [5,6,7,8]
POINTS = [100]


# RAYS = [2, 4, 6, 8, 12]
# POINTS = [25, 50, 100]


# Fewer epochs than train_model.py, because this runs many times over.
EPOCHS = 40

# Probability threshold that turns the model's per-pixel probabilities into
# the binary line map.  None = pick automatically (the value maximising
# tolerant F1 on the validation split, the current behaviour).  A number,
# e.g. 0.5, uses exactly that threshold in every cell instead.
THRESHOLD = None

# ---- STAGE 3: the figures --------------------------------------------

# Test devices drawn in fig_examples.png — enough to see failure modes,
# few enough to read.
N_EXAMPLES = 3

# ---- STAGE 4: full per-device analysis of the test devices -----------

# True writes <run>/sample_<i>/ for every test device: the complete original
# pipeline output (rays + per-ray plots, peaks.json, summary*.png,
# cropped_results/ with the peak crops and sweeps, gifs/, evaluation.txt ...)
# plus the ML panels (ml_stability / ml_measurement / ml_truth /
# ml_prediction / ml_overlay), all from this run's best checkpoint.  It is by
# far the slowest stage — turn it off for bulk sweeps.
SAVE_SAMPLE_ANALYSIS = True

# Sweep GIFs are the slowest part of that analysis; 100 dpi keeps them cheap.
SAVE_GIFS = False
GIF_DPI = 100

# ══════════════════════════════════════════════════════════════════════

# The columns copied out of the metrics, for both the model and the baseline.
KEYS = ("f1@0", "f1@1", "f1@2", "f1@3", "iou")


def settings_record():
    """The SETTINGS block as a dict, for config.json inside the run folder."""
    return {
        "n_train": N_TRAIN, "n_test": N_TEST, "resolution": RESOLUTION,
        "disjoint_intervals": DISJOINT_INTERVALS,
        "voltage_window": VOLTAGE_WINDOW,
        "coulomb_peak_width": COULOMB_PEAK_WIDTH, "temperature": TEMPERATURE,
        "seed": SEED,
        "rays": RAYS, "points": POINTS, "epochs": EPOCHS,
        "threshold": THRESHOLD,
        "save_sample_analysis": SAVE_SAMPLE_ANALYSIS,
        "save_gifs": SAVE_GIFS, "gif_dpi": GIF_DPI,
    }


# ──────────────────────────────────────────────────────────────────────
#  STAGE 1 — the devices
# ──────────────────────────────────────────────────────────────────────

def dataset_dirs():
    """
    Where this run's devices live.  A pure function of the SETTINGS above —
    which is why the paths can go into config.json before anything is made,
    and why two runs with the same settings share one dataset.
    """
    return (paths.training_data(device_factory.dataset_name(
                "train", N_TRAIN, RESOLUTION, DISJOINT_INTERVALS)),
            paths.training_data(device_factory.dataset_name(
                "test", N_TEST, RESOLUTION, DISJOINT_INTERVALS)))


def make_devices(train_dir, test_dir):
    """
    Simulate (or reuse) the train and test datasets.

    device_factory.generate() skips any device already on disk, so this is a
    no-op the second time you run with the same settings, and a top-up if you
    only raised N_TRAIN.
    """
    if DISJOINT_INTERVALS:
        train_cfg, test_cfg = split_configs()
        print("disjoint capacitance intervals:\n" + describe() + "\n")
    else:
        train_cfg = test_cfg = CapacitanceConfig()
        print("train and test share the full capacitance intervals\n")

    common = dict(resolution=RESOLUTION, voltage_window=VOLTAGE_WINDOW,
                  coulomb_peak_width=COULOMB_PEAK_WIDTH,
                  temperature=TEMPERATURE, keep_images=KEEP_IMAGES)

    print(f"── STAGE 1  TRAIN: {N_TRAIN} devices " + "─" * 34)
    device_factory.generate(train_dir, N_TRAIN, train_cfg, SEED, **common)

    print(f"── STAGE 1  TEST: {N_TEST} devices " + "─" * 35)
    # A different seed stream, so the two splits never line up device for
    # device on the capacitance entries that are NOT split.
    device_factory.generate(test_dir, N_TEST, test_cfg, SEED + 1000, **common)


# ──────────────────────────────────────────────────────────────────────
#  STAGE 2 — the rays x points sweep
# ──────────────────────────────────────────────────────────────────────

def run_cell(out, train_samples, test_samples, n_rays, n_points):
    """
    One cell of the sweep: measure at this budget, train from scratch, score.

    Writes the checkpoint into the run folder and returns the csv row.  This
    is exactly what train_model.py + evaluate_model.py do for a single budget,
    so any cell can be reproduced on its own.
    """
    cache = run_dir.SHARED_CACHE
    Xtr, Ytr = grid_dataset.build_cached(train_samples, n_rays, n_points,
                                         cache_dir=cache, tag="train")
    Xte, Yte = grid_dataset.build_cached(test_samples, n_rays, n_points,
                                         cache_dir=cache, tag="test")

    net, thr = grid_train.train(Xtr, Ytr, epochs=EPOCHS)
    if THRESHOLD is not None:
        thr = THRESHOLD               # manual override from the settings
    ckpt = os.path.join(out, "models",
                        grid_train.checkpoint_name(n_rays, n_points))
    grid_train.save(net, thr, ckpt, n_rays, n_points,
                    extra={"n_train": len(Xtr)})
    print(f"  saved {os.path.abspath(ckpt)}")

    m = evaluate(grid_train.predict(net, Xte) > thr, Yte)
    h = score_hough(Xte, Yte)
    return {"n_rays": n_rays, "n_points": n_points,
            # measured pixels as a fraction of the grid: the honest x-axis
            # for "measurement budget" in the paper
            "coverage": float(Xte[:, 1].mean()),
            "peak_frac": float(Xte[:, 2].mean()),
            "threshold": thr,
            **{f"ml_{k}": m[k] for k in KEYS},
            **{f"hough_{k}": h[k] for k in KEYS}}


def run_sweep(out, train_dir, test_dir):
    """Every (rays, points) cell, writing the csv as it goes.  Returns rows."""
    train_samples = grid_dataset.find_samples([train_dir])
    test_samples = grid_dataset.find_samples([test_dir])
    if not train_samples or not test_samples:
        sys.exit(f"stage 1 produced no usable devices\n"
                 f"  train: {os.path.abspath(train_dir)}\n"
                 f"  test:  {os.path.abspath(test_dir)}")
    print(f"{len(train_samples)} train devices, {len(test_samples)} test "
          f"devices, {len(RAYS) * len(POINTS)} cells")

    out_csv = os.path.join(out, "budget_sweep.csv")
    rows = []
    for n_rays in RAYS:
        for n_points in POINTS:
            print(f"\n=== STAGE 2  {n_rays} rays x {n_points} points "
                  + "=" * 30)
            row = run_cell(out, train_samples, test_samples, n_rays, n_points)
            rows.append(row)
            print(f"  ML F1@1 {row['ml_f1@1']:.3f}   "
                  f"hough {row['hough_f1@1']:.3f}")

            # Rewritten every cell, so an interrupted sweep still leaves a
            # usable csv behind.
            write_csv(out_csv, rows)
    return rows, out_csv


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ──────────────────────────────────────────────────────────────────────
#  STAGE 3 — summary table and figures
# ──────────────────────────────────────────────────────────────────────

def summarise(rows, out_csv):
    print("\n" + "=" * 70)
    print("transition-line recovery vs measurement budget  (tolerant F1, tau=1)")
    print("=" * 70)
    print(f"{'rays':>5} {'points':>7} {'coverage':>9} {'ML':>7} {'hough':>7}")
    for r in rows:
        print(f"{r['n_rays']:>5} {r['n_points']:>7} "
              f"{100 * r['coverage']:>8.1f}% {r['ml_f1@1']:>7.3f} "
              f"{r['hough_f1@1']:>7.3f}")
    print(f"\nwrote {os.path.abspath(out_csv)}")


def make_figures(out, rows, test_dir):
    """
    The budget figures and example predictions, into <run>/figures/.

    The plotting functions are make_figures.py's own, so a figure drawn here
    and one re-drawn later from the csv are the same figure.  Example
    predictions use this run's best cell — the checkpoint worth looking at.
    """
    fig_dir = os.path.join(out, "figures")
    print("\n=== STAGE 3  figures " + "=" * 45)
    fig_budget_rays(rows, fig_dir)
    fig_budget_coverage(rows, fig_dir)

    best = max(rows, key=lambda r: r["ml_f1@1"])
    ckpt = os.path.join(out, "models", grid_train.checkpoint_name(
        int(best["n_rays"]), int(best["n_points"])))
    print(f"  examples from the best cell: {int(best['n_rays'])} rays x "
          f"{int(best['n_points'])} points (F1@1 {best['ml_f1@1']:.3f})")
    fig_examples(test_dir, ckpt, fig_dir, n_examples=N_EXAMPLES)


# ──────────────────────────────────────────────────────────────────────
#  STAGE 4 — full per-device analysis of the test devices
# ──────────────────────────────────────────────────────────────────────

def analyse_samples(out, rows, test_dir):
    """
    <run>/sample_<i>/ for every test device — everything the original
    pipeline produces for a device plus the ML panels, exactly what
    render_test_sample_analysis.py makes, but inside THIS run's folder and
    from this run's best checkpoint.  The code is that program's own
    (analyse_test_devices), so the two outputs can never drift apart.
    """
    best = max(rows, key=lambda r: r["ml_f1@1"])
    n_rays, n_points = int(best["n_rays"]), int(best["n_points"])
    ckpt = os.path.join(out, "models",
                        grid_train.checkpoint_name(n_rays, n_points))
    net, ck = grid_train.load(ckpt)
    print(f"\n=== STAGE 4  per-device analysis: {n_rays} rays x "
          f"{n_points} points " + "=" * 15)
    analyse_test_devices(
        out, test_dir, net, ck["threshold"],
        n_test=N_TEST, seed=SEED, n_rays=n_rays, n_points=n_points,
        resolution=RESOLUTION, voltage_window=VOLTAGE_WINDOW,
        coulomb_peak_width=COULOMB_PEAK_WIDTH, temperature=TEMPERATURE,
        disjoint_intervals=DISJOINT_INTERVALS,
        save_gifs=SAVE_GIFS, gif_dpi=GIF_DPI)


def main():
    train_dir, test_dir = dataset_dirs()

    # The run folder first, so config.json — settings AND the dataset folders
    # they name — exists from the very start.  A run killed halfway through
    # stage 1 still says exactly what it was trying to do.
    out = run_dir.new_run("experiment", {**settings_record(),
                                         "train_dir": os.path.abspath(train_dir),
                                         "test_dir": os.path.abspath(test_dir)})

    make_devices(train_dir, test_dir)
    rows, out_csv = run_sweep(out, train_dir, test_dir)
    summarise(rows, out_csv)
    make_figures(out, rows, test_dir)
    if SAVE_SAMPLE_ANALYSIS:
        analyse_samples(out, rows, test_dir)
    print(f"\neverything for this trial is in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
