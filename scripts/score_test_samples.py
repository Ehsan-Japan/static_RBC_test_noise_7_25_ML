"""
score_test_samples.py — per-sample scores for an experiment run, in ONE json.

    python score_test_samples.py [runs/<timestamp>_experiment]

budget_sweep.csv and the sample_<i>/ figures show the test split as a whole;
this program writes the numbers PER TEST DEVICE, so the samples can be
compared at one glance.  For the run's best cell (the same checkpoint the
sample_<i>/ ml_* panels were drawn from) it re-measures every test device,
predicts, and writes <run>/sample_results.json:

    per sample : ground-truth / predicted line-pixel counts, pixel accuracy,
                 strict tp/fp/fn, tolerant precision/recall/F1 at tau = 0..3,
                 IoU — for the model AND the Hough baseline
    overall    : the mean over samples (matches the budget_sweep.csv row)

Nothing is retrained and nothing existing is touched — it only reads the
run's config.json, budget_sweep.csv and checkpoint, and adds the json.
"""
import csv
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from dqd.ml import grid_dataset, grid_train, run_dir
from dqd.ml.grid_baseline import hough_predict, score_hough
from dqd.ml.grid_metrics import iou, tolerant_f1
from dqd.ml.ray_peaks import CH_PEAKS

from render_device_figures import render_binary_panels

TAUS = (0, 1, 2, 3)


def best_cell(run):
    """The (n_rays, n_points, threshold) of the run's best sweep cell."""
    with open(os.path.join(run, "budget_sweep.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r["ml_f1@1"]))
    return int(best["n_rays"]), int(best["n_points"])


def per_sample_metrics(pred, true):
    """Every number for one (prediction, truth) pair of binary maps."""
    pred, true = pred > 0.5, true > 0.5
    out = {
        "true_line_pixels": int(true.sum()),
        "pred_line_pixels": int(pred.sum()),
        "pixel_accuracy": float((pred == true).mean()),
        "tp_strict": int((pred & true).sum()),
        "fp_strict": int((pred & ~true).sum()),
        "fn_strict": int((~pred & true).sum()),
        "iou": iou(pred, true),
    }
    for t in TAUS:
        m = tolerant_f1(pred, true, t)
        out[f"precision@{t}"] = m["precision"]
        out[f"recall@{t}"] = m["recall"]
        out[f"f1@{t}"] = m["f1"]
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else run_dir.latest_run("experiment")
    if not (run and os.path.isfile(os.path.join(run, "config.json"))):
        sys.exit("no experiment run found — pass its folder explicitly")
    with open(os.path.join(run, "config.json")) as f:
        cfg = json.load(f)

    n_rays, n_points = best_cell(run)
    ckpt = os.path.join(run, "models",
                        grid_train.checkpoint_name(n_rays, n_points))
    net, ck = grid_train.load(ckpt)
    thr = ck["threshold"]
    print(f"run:   {os.path.abspath(run)}\n"
          f"model: {os.path.abspath(ckpt)}  (threshold {thr})")

    samples = grid_dataset.find_samples([cfg["test_dir"]])
    X, Y = grid_dataset.build(samples, n_rays, n_points, verbose=False)
    prob = grid_train.predict(net, X)
    preds = prob > thr

    # One line count for every sample, tuned exactly as the sweep tuned it,
    # so the per-sample baseline numbers average back to the csv row.
    n_lines = score_hough(X, Y)["n_lines"]
    hough = np.stack([hough_predict(x[CH_PEAKS], n_lines=n_lines) for x in X])

    per_sample = []
    for i, sdir in enumerate(samples):
        name = os.path.basename(sdir)
        fig_dir = os.path.join(run, name)
        if os.path.isdir(fig_dir):
            # hough_prediction.png / hough_overlay.png next to the ml_*
            # panels, so baseline and model can be compared side by side
            render_binary_panels(sdir, "hough", fig_dir, hough[i],
                                 "Hough Baseline")
        per_sample.append({
            "sample": name,
            "device_dir": os.path.abspath(sdir),
            "figures_dir": os.path.abspath(os.path.join(run, name)),
            "coverage": float(X[i, 1].mean()),
            "peak_pixels": int(X[i, 2].sum()),
            "ml": per_sample_metrics(preds[i], Y[i]),
            "hough": per_sample_metrics(hough[i], Y[i]),
        })

    def mean_block(key):
        keys = per_sample[0][key]
        return {k: float(np.mean([s[key][k] for s in per_sample]))
                for k in keys}

    result = {
        "run": os.path.abspath(run),
        "checkpoint": os.path.abspath(ckpt),
        "n_rays": n_rays, "n_points": n_points, "threshold": thr,
        "hough_hyperparameters": {
            "n_lines": n_lines,          # tuned on the test split (best F1@1)
            "n_lines_grid": list(inspect.signature(
                score_hough).parameters["n_lines_grid"].default),
            **{k: p.default for k, p in
               inspect.signature(hough_predict).parameters.items()
               if k in ("n_angles", "min_votes", "thickness")},
        },
        "taus": list(TAUS),
        "n_samples": len(per_sample),
        "note": ("headline metric is f1@1 (tolerant F1, 1-pixel slack); "
                 "pixel_accuracy is reported only for completeness — lines "
                 "cover a few percent of the map, so it is always high"),
        "samples": per_sample,
        "overall": {"ml": mean_block("ml"), "hough": mean_block("hough")},
    }

    out = os.path.join(run, "sample_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {os.path.abspath(out)}")

    print(f"\n{'sample':>9} {'truth px':>9} {'pred px':>8} "
          f"{'ML f1@1':>8} {'hough':>6} {'iou':>6} {'acc':>7}")
    for s in per_sample:
        m = s["ml"]
        print(f"{s['sample']:>9} {m['true_line_pixels']:>9} "
              f"{m['pred_line_pixels']:>8} {m['f1@1']:>8.3f} "
              f"{s['hough']['f1@1']:>6.3f} {m['iou']:>6.3f} "
              f"{m['pixel_accuracy']:>7.3f}")
    o = result["overall"]
    print(f"{'MEAN':>9} {'':>9} {'':>8} {o['ml']['f1@1']:>8.3f} "
          f"{o['hough']['f1@1']:>6.3f} {o['ml']['iou']:>6.3f} "
          f"{o['ml']['pixel_accuracy']:>7.3f}")


if __name__ == "__main__":
    main()
