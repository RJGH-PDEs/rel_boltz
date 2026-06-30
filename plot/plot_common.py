"""Shared helpers for the plotting scripts (plot.py, plot_heatmap.py).

Single source of truth for: where a run's figures go (the per-case experiment
folder, keyed by run_meta.json), which snapshots to plot, and the core
distribution-value evaluation reused by both the 1D axis and 2D plane plots.
"""
import os
import glob
import json
import numpy as np
from lc import linear_comb

# Iterations to plot, shared by plot.py and plot_heatmap.py. All are saved by
# time_ev.py (which stores iters 1..20 and every save_every-th step).
SNAPSHOTS = [0, 1, 2, 5, 10, 20, 100, 10000]


def available_snapshots(coeff_dir='coeff'):
    """All saved snapshot iterations, sorted (from coeff/<iter>.pkl filenames).

    Unlike the sparse SNAPSHOTS list (chosen for per-frame heatmaps), this returns
    every snapshot the run actually wrote — what a moments time series wants.
    """
    paths = glob.glob(os.path.join(coeff_dir, '*.pkl'))
    its = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return sorted(int(s) for s in its if s.isdigit())


def load_run_meta(coeff_dir='coeff'):
    """Load the run metadata written by time_ev.py. Raises if the run is missing."""
    path = os.path.join(coeff_dir, 'run_meta.json')
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path} — run time_ev.py (with save=True) first to produce "
            "the snapshots and run metadata."
        )
    with open(path) as f:
        return json.load(f)


def experiment_case_dir(case):
    """Per-case output folder, relative to the plot/ cwd (mirrors ../src convention)."""
    return os.path.join('..', 'time_evol', 'experiments', case)


def eval_point(coeff, r, theta, phi, N):
    """Distribution value f at one momentum-space point (r, theta, phi)."""
    if r == 0.0:
        return linear_comb(coeff, 0.0, 0.0, 0.0, N)
    return np.exp(-r / 2) * linear_comb(coeff, r, theta, phi, N)
