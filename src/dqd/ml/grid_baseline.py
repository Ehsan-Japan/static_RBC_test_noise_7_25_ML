"""
grid_baseline.py — reconstructing the lines from the SAME peaks, without
learning anything.  This is the control the paper needs.

The claim being tested is "a network can recover the transition lines from
ray peaks alone".  A reviewer's first objection is that the peaks already
determine the lines and any classical method would do — so the network has
to be measured against a classical method given exactly the same input.

The baseline is a Hough line fit: vote every peak into (angle, offset) space,
keep the strongest lines, draw them.  Honeycomb transition lines really are
straight and come in two slope families, and fitting straight lines to the
peaks is what the existing geometric pipeline does in spirit — so this is a
serious competitor rather than a straw man, and beating it is the result.

It reads the peak channel of the same X the network sees, so the comparison
is exactly at equal measurement budget.
"""
from typing import Sequence

import numpy as np

from .grid_metrics import evaluate


def hough_predict(peaks: np.ndarray,
                  n_lines: int = 12,
                  n_angles: int = 180,
                  min_votes: int = 3,
                  thickness: float = 1.0) -> np.ndarray:
    """
    Fit straight lines through the peak pixels and draw them.

    Every peak votes for all lines through it, parameterised as
    rho = x cos(theta) + y sin(theta).  The strongest `n_lines` bins become
    lines; a pixel is predicted if it lies within `thickness` of one.
    """
    H, W = peaks.shape
    ys, xs = np.nonzero(peaks > 0.5)
    out = np.zeros((H, W), dtype=bool)
    if len(xs) < 2:
        return out

    thetas = np.linspace(-np.pi / 2, np.pi / 2, n_angles, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    # Vote: (n_peaks, n_angles) offsets, binned to integer rho.
    rho = xs[:, None] * cos_t[None, :] + ys[:, None] * sin_t[None, :]
    rho_max = int(np.ceil(np.hypot(H, W)))
    rho_idx = np.rint(rho).astype(int) + rho_max              # >= 0
    acc = np.zeros((n_angles, 2 * rho_max + 1), dtype=int)
    for a in range(n_angles):
        np.add.at(acc[a], rho_idx[:, a], 1)

    # Strongest bins, best first.
    flat = np.argsort(acc, axis=None)[::-1][:n_lines]
    a_sel, r_sel = np.unravel_index(flat, acc.shape)

    yy, xx = np.mgrid[0:H, 0:W]
    for a, r in zip(a_sel, r_sel):
        if acc[a, r] < min_votes:
            break                                  # sorted, so the rest are worse
        d = np.abs(xx * cos_t[a] + yy * sin_t[a] - (r - rho_max))
        out |= d <= thickness
    return out


def score_hough(X: np.ndarray, Y: np.ndarray,
                n_lines_grid: Sequence[int] = (6, 12, 20)) -> dict:
    """
    Best line count on this split, and its metrics.

    The line count is tuned ON the split being scored, which flatters the
    baseline — deliberately.  A control that has been given every advantage
    and is still beaten is the one worth reporting.
    """
    best = None
    for n in n_lines_grid:
        preds = np.stack([hough_predict(x[0], n_lines=n) for x in X])
        m = evaluate(preds, Y)
        if best is None or m["f1@1"] > best[1]["f1@1"]:
            best = (n, m)
    return {"n_lines": best[0], **best[1]}
