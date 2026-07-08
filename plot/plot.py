import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import curve_fit
sys.path.insert(0, '../src')
from plot_common import SNAPSHOTS, load_run_meta, experiment_case_dir, eval_point

# ── flags ────────────────────────────────────────────────────────────────────
show = False   # one figure per axis is saved instead; browse the files
save = True
# ─────────────────────────────────────────────────────────────────────────────

# Case and spectral order come from the run that produced the snapshots.
meta    = load_run_meta()
N       = meta['n']
case    = meta['case']
out_dir = os.path.join(experiment_case_dir(case, N), 'axis_plots')
os.makedirs(out_dir, exist_ok=True)

# ── evaluation grid ───────────────────────────────────────────────────────────
n_pts  = 200
coord_vals = np.linspace(-15.0, 15.0, n_pts)
r_vals     = np.abs(coord_vals)

# Evaluates f along a single Cartesian axis ('x', 'y', or 'z'). Each axis
# corresponds to a fixed (theta, phi) for positive/negative values of the
# coordinate, since on a coordinate axis only the sign and r matter:
#   +x: theta=pi/2, phi=0      -x: theta=pi/2, phi=pi
#   +y: theta=pi/2, phi=pi/2   -y: theta=pi/2, phi=-pi/2
#   +z: theta=0                -z: theta=pi          (phi irrelevant at poles)
def eval_axis(coeff_vec, axis):
    f = np.zeros(len(coord_vals))
    for i, (c, r) in enumerate(zip(coord_vals, r_vals)):
        if r == 0.0:
            theta, phi = 0.0, 0.0
        elif axis == 'x':
            theta, phi = np.pi / 2, (0.0 if c > 0 else np.pi)
        elif axis == 'y':
            theta, phi = np.pi / 2, (np.pi / 2 if c > 0 else -np.pi / 2)
        elif axis == 'z':
            theta, phi = (0.0 if c > 0 else np.pi), 0.0
        else:
            raise ValueError(f"unknown axis: {axis}")
        f[i] = eval_point(coeff_vec, r, theta, phi, N)
    return f

# ── load snapshots ────────────────────────────────────────────────────────────
coeffs = []
for it in SNAPSHOTS:
    path = f"coeff/{it}.pkl"
    if not os.path.exists(path):
        print(f"missing {path}, skipping")
        continue
    with open(path, 'rb') as file:
        coeffs.append((it, pickle.load(file)))

cmap   = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(coeffs)))

# For the radial case: fit Jüttner f_J(r) = A * exp(-r/T) from the final snapshot
# and overlay it as an analytical equilibrium reference.
juttner_params = None
if case == 'radial' and coeffs:
    _, final_coeff = coeffs[-1]
    r_fit = np.linspace(0.1, 15.0, 300)
    f_eq  = np.array([eval_point(final_coeff, r, 0.0, 0.0, N) for r in r_fit])
    try:
        popt, _ = curve_fit(lambda r, A, T: A * np.exp(-r / T), r_fit, f_eq, p0=[0.3, 2.5])
        juttner_params = popt
        print(f"Jüttner fit: A={popt[0]:.4f}, T={popt[1]:.4f}")
    except Exception as e:
        print(f"Jüttner fit failed: {e}")

num_iters = meta.get('num_iterations', 10000)

for axis in ['x', 'y', 'z']:
    fig, ax = plt.subplots(figsize=(9, 5))

    for color, (it, coeff) in zip(colors, coeffs):
        label = f'iter {it}' + (' (IC)' if it == 0 else '') + (' (final)' if it == num_iters else '')
        ax.plot(coord_vals, eval_axis(coeff, axis), color=color, label=label)

    if juttner_params is not None:
        A, T = juttner_params
        j_vals = A * np.exp(-r_vals / T)
        ax.plot(coord_vals, j_vals, 'k--', linewidth=1.5,
                label=f'Jüttner  A={A:.3f}, T={T:.3f}')

    ax.set_title(f'time evolution of f along the ξ_{axis} axis')
    ax.set_xlabel(f'ξ_{axis}')
    ax.set_ylabel('f')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()

    if save:
        figure_name = os.path.join(out_dir, f'xi_{axis}.png')
        plt.savefig(figure_name, dpi=150)
        print(f"saved {figure_name}")

    if show:
        plt.show()
    plt.close(fig)
