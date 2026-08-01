"""
evaluate_model.py — MAIN PROGRAM 3 of 4.  Score a checkpoint.  Trains nothing.

    1. generate_ml_data.py    make the devices
    2. train_model.py         train one budget
    3. evaluate_model.py      <- you are here
    4. run_budget_sweep.py    the rays x points table

Edit the two settings below and run it:

    python evaluate_model.py

The measurement budget is read out of the checkpoint, not set here, so the
test devices are measured exactly the way the training devices were.

It also scores a classical Hough line fit on the SAME peaks.  That
comparison is the point: a number for the network alone does not show that
learning was necessary.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqd.ml import grid_dataset, grid_train
from dqd.ml.grid_baseline import score_hough
from dqd.ml.grid_metrics import evaluate

# f1@t is tolerant F1: a predicted line pixel counts if a true one lies within
# t pixels.  f1@0 is strict (a perfect line drawn one pixel over scores zero),
# f1@1 is the headline number, f1@3 shows how much of the error is sub-pixel.
KEYS = ("f1@0", "f1@1", "f1@2", "f1@3", "iou")


def main():
    # ══════════════════════════════════════════════════════════════════
    #  SETTINGS
    # ══════════════════════════════════════════════════════════════════

    # Devices to score on.  Must NOT be the ones the checkpoint trained on.
    TEST_DIR = os.path.join("..", "training_data", "ml_test_split_n500_res100")

    # The checkpoint written by train_model.py.
    MODEL_PATH = os.path.join("..", "models", "rays6_points100.pt")

    # ══════════════════════════════════════════════════════════════════

    if not os.path.isfile(MODEL_PATH):
        sys.exit(f"no checkpoint at {MODEL_PATH}\nrun train_model.py first")

    net, ck = grid_train.load(MODEL_PATH)
    R, P, thr = ck["n_rays"], ck["n_points"], ck["threshold"]
    print(f"{os.path.basename(MODEL_PATH)}: {R} rays x {P} points, "
          f"threshold {thr}, trained on {ck.get('n_train', '?')} devices")

    samples = grid_dataset.find_samples([TEST_DIR])
    if not samples:
        sys.exit(f"no usable samples in {TEST_DIR}")
    print(f"{len(samples)} test devices")

    X, Y = grid_dataset.build_cached(samples, R, P, tag="test",
                                     cache_dir=os.path.join("..", "grid_cache"))
    result = {"model": os.path.basename(MODEL_PATH),
              "n_rays": R, "n_points": P, "n_test": len(X), "threshold": thr,
              "coverage": float(X[:, 1].mean()),
              "peak_frac": float(X[:, 0].mean()),
              "ml": evaluate(grid_train.predict(net, X) > thr, Y),
              "hough": score_hough(X, Y)}

    print(f"\nmeasured coverage {100 * result['coverage']:.1f}% of the grid\n")
    print(f"{'':10s}" + "".join(f"{k:>9s}" for k in KEYS))
    for name in ("ml", "hough"):
        print(f"{name:10s}" + "".join(f"{result[name][k]:>9.3f}" for k in KEYS))

    out = os.path.join("..", "results_eval.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
