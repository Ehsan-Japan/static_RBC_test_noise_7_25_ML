"""
make_figures.py — MAIN PROGRAM 6.  Turn the CSVs and a checkpoint into figures.

    python make_figures.py

By default it finds the newest sweep results under runs/ and writes the
figures into that same run folder, so a trial's figures live beside the CSVs
and checkpoints that produced them.

Writes into <run folder>/figures/ :

    fig_budget_rays.png       F1 vs number of rays, one line per ray resolution
    fig_budget_coverage.png   F1 vs how much of the diagram was measured
    fig_data_size.png         F1 vs training-set size (the learning curve)
    fig_examples.png          what the model actually draws, on test devices

Whatever exists is drawn; missing inputs are reported and skipped, so this is
safe to run at any point.

Figure conventions, applied everywhere so the set reads as one:
  * one measured quantity per axis, never two y-scales;
  * the learned model is a coloured line, the classical Hough baseline is
    neutral grey dashed — it is a reference, not a peer series;
  * colours identify the ray resolution and nothing else, in fixed order, so
    the same resolution is the same colour in every figure;
  * text is ink-coloured, never series-coloured; the legend carries identity.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqd.ml import grid_dataset, grid_train, run_dir

# ── Palette ───────────────────────────────────────────────────────────
# Categorical slots 1-3 of the reference palette, in fixed order.  This
# triple is the documented all-pairs-safe subset (worst-pair CVD dE 9.2,
# normal-vision 24.0 on a light surface), which is why the series count is
# capped at three ray resolutions per figure.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")      # blue, orange, aqua
BASELINE = "#8a8a85"                            # neutral grey: the baseline
INK = "#0b0b0b"                                 # primary text
INK_2 = "#52514e"                               # secondary text
GRID = "#e2e2de"                                # recessive gridlines
SURFACE = "#ffffff"

DPI = 300


def _style(ax, xlabel, ylabel, title=None):
    """House style: recessive frame and grid, ink-coloured text."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, color=GRID)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=10, loc="left")


def _save(fig, name, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _read_csv(path):
    if not os.path.isfile(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path, newline="") as f:
        rows = [{k: float(v) if _num(v) else v for k, v in r.items()}
                for r in csv.DictReader(f)]
    return rows or None


def _num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------
# Figure 1: F1 vs number of rays, one line per ray resolution
# ----------------------------------------------------------------------

def fig_budget_rays(rows, fig_dir):
    resolutions = sorted({int(r["n_points"]) for r in rows})
    if len(resolutions) > len(SERIES):
        # Three is the validated series cap for this palette; more than that
        # and adjacent resolutions stop being separable for CVD readers.
        print(f"  [note] {len(resolutions)} ray resolutions, plotting the "
              f"{len(SERIES)} extremes + middle")
        resolutions = [resolutions[0],
                       resolutions[len(resolutions) // 2],
                       resolutions[-1]]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _style(ax, "number of rays", "transition-line F1  (tolerance 1 px)",
           "Line recovery improves with more rays")

    for color, P in zip(SERIES, resolutions):
        sub = sorted((r for r in rows if int(r["n_points"]) == P),
                     key=lambda r: r["n_rays"])
        x = [r["n_rays"] for r in sub]
        y = [r["ml_f1@1"] for r in sub]
        ax.plot(x, y, "-o", color=color, linewidth=2, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=f"{P} points/ray", zorder=3)
        # Direct label at the line end, in ink — the legend gives identity,
        # this makes the ordering readable without crossing back to it.
        if x:
            ax.annotate(f"{P} pts", (x[-1], y[-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=8, color=INK_2)

        base = [r.get("hough_f1@1") for r in sub]
        if P == resolutions[-1] and all(b is not None for b in base):
            ax.plot(x, base, "--", color=BASELINE, linewidth=1.6, zorder=2,
                    label=f"Hough baseline ({P} points/ray)")

    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    _save(fig, "fig_budget_rays.png", fig_dir)


# ----------------------------------------------------------------------
# Figure 2: F1 vs measured coverage — the true cost axis
# ----------------------------------------------------------------------

def fig_budget_coverage(rows, fig_dir):
    """
    Rays and points both cost measurement time; coverage is what they buy.
    Plotting against coverage collapses the two knobs into the quantity an
    experimentalist actually pays for, and shows whether two different
    (rays, points) combinations that cost the same also score the same.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _style(ax, "fraction of the diagram measured  (%)",
           "transition-line F1  (tolerance 1 px)",
           "What the measurement budget buys")

    srt = sorted(rows, key=lambda r: r["coverage"])
    x = [100 * r["coverage"] for r in srt]
    ax.plot(x, [r["ml_f1@1"] for r in srt], "o", color=SERIES[0],
            markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="learned model", zorder=3)
    if all("hough_f1@1" in r for r in srt):
        ax.plot(x, [r["hough_f1@1"] for r in srt], "s", color=BASELINE,
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label="Hough baseline", zorder=2)

    for r in srt:
        ax.annotate(f"{int(r['n_rays'])}x{int(r['n_points'])}",
                    (100 * r["coverage"], r["ml_f1@1"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=INK_2)

    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    _save(fig, "fig_budget_coverage.png", fig_dir)


# ----------------------------------------------------------------------
# Figure 3: learning curve
# ----------------------------------------------------------------------

def fig_data_size(rows, fig_dir):
    srt = sorted(rows, key=lambda r: r["n_train"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _style(ax, "training devices", "transition-line F1  (tolerance 1 px)",
           "Is the result limited by data?")
    ax.plot([r["n_train"] for r in srt], [r["ml_f1@1"] for r in srt],
            "-o", color=SERIES[0], linewidth=2, markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    if srt:
        b = srt[0]
        ax.annotate(f"{int(b['n_rays'])} rays x {int(b['n_points'])} points",
                    (0.02, 0.95), xycoords="axes fraction",
                    fontsize=9, color=INK_2, va="top")
    _save(fig, "fig_data_size.png", fig_dir)


# ----------------------------------------------------------------------
# Figure 4: what the model actually draws
# ----------------------------------------------------------------------

def fig_examples(test_dir, model_path, fig_dir, n_examples=3):
    """
    One row per device: the measurement it was given, what it predicted, the
    truth, and the two overlaid.  A table of F1 values cannot show whether
    the failures are missing lines or displaced ones; this can.
    """
    net, ck = grid_train.load(model_path)
    R, P, thr = ck["n_rays"], ck["n_points"], ck["threshold"]
    samples = grid_dataset.find_samples([test_dir])[:n_examples]
    if not samples:
        print(f"  [skip] no test devices in {test_dir}")
        return
    X, Y = grid_dataset.build(samples, R, P, verbose=False)
    pred = grid_train.predict(net, X) > thr

    titles = ("measured rays + peaks", "model prediction",
              "ground truth", "overlay")
    fig, axes = plt.subplots(len(samples), 4,
                             figsize=(9.5, 2.5 * len(samples)))
    axes = np.atleast_2d(axes)

    for row in range(len(samples)):
        # 1. the measurement: where the rays went, and where peaks were found
        ax = axes[row, 0]
        ax.imshow(1 - 0.12 * X[row, 1], cmap="gray", vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ys, xs = np.nonzero(X[row, 0])
        ax.plot(xs, ys, "o", color=SERIES[1], markersize=2.5, linestyle="none")

        # 2/3. prediction and truth, black on white like the other figures
        axes[row, 1].imshow(1 - pred[row], cmap="gray", vmin=0, vmax=1,
                            origin="lower", interpolation="nearest")
        axes[row, 2].imshow(1 - Y[row], cmap="gray", vmin=0, vmax=1,
                            origin="lower", interpolation="nearest")

        # 4. overlay: truth in grey underneath, prediction in blue on top
        ax = axes[row, 3]
        ax.imshow(1 - 0.35 * Y[row], cmap="gray", vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ys, xs = np.nonzero(pred[row])
        ax.plot(xs, ys, "s", color=SERIES[0], markersize=1.2,
                linestyle="none", alpha=0.75)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for s in axes[row, col].spines.values():
                s.set_color(GRID)
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=9, color=INK,
                                         pad=6)
        axes[row, 0].set_ylabel(f"device {row + 1}", fontsize=9, color=INK_2)

    fig.suptitle(f"{R} rays x {P} points — grey: true lines, "
                 f"blue: predicted, orange: measured peaks",
                 fontsize=10, color=INK_2, y=1.0)
    fig.tight_layout()
    _save(fig, "fig_examples.png", fig_dir)


def main():
    # ══════════════════════════════════════════════════════════════════
    #  SETTINGS
    # ══════════════════════════════════════════════════════════════════

    # Leave these as None to use the newest run of each kind under runs/.
    # Set one to a path to re-draw an older trial instead.
    BUDGET_CSV = None
    DATA_SIZE_CSV = None
    MODEL_PATH = None

    TEST_DIR = os.path.join("..", "training_data", "ml_test_split_n500_res100")
    N_EXAMPLES = 3

    # ══════════════════════════════════════════════════════════════════

    budget_csv = BUDGET_CSV or run_dir.find_file("budget_sweep.csv", "sweep")
    size_csv = DATA_SIZE_CSV or run_dir.find_file("data_size_sweep.csv",
                                                  "datasize")
    model = MODEL_PATH or run_dir.find_file(
        os.path.join("models", "rays6_points100.pt"))

    # Figures go beside the results that produced them.  When several runs
    # contribute, the newest one hosts them.
    hosts = [os.path.dirname(p) for p in (budget_csv, size_csv, model) if p]
    if not hosts:
        sys.exit("nothing to plot yet — run a sweep or train_model.py first")
    host = max(hosts)
    if os.path.basename(host) == "models":
        host = os.path.dirname(host)
    fig_dir = os.path.join(host, "figures")
    print(f"figures -> {fig_dir}\n")

    print("budget sweep figures:")
    rows = _read_csv(budget_csv) if budget_csv else None
    if rows:
        fig_budget_rays(rows, fig_dir)
        fig_budget_coverage(rows, fig_dir)
    elif not budget_csv:
        print("  [skip] no budget_sweep.csv in runs/")

    print("data-size figure:")
    rows = _read_csv(size_csv) if size_csv else None
    if rows:
        fig_data_size(rows, fig_dir)
    elif not size_csv:
        print("  [skip] no data_size_sweep.csv in runs/")

    print("example predictions:")
    if model and os.path.isfile(model):
        fig_examples(TEST_DIR, model, fig_dir, N_EXAMPLES)
    else:
        print("  [skip] no checkpoint in runs/ — run train_model.py")


if __name__ == "__main__":
    main()
