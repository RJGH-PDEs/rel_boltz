import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
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

# For the radial case: compute the Jüttner A and T analytically from the IC's
# raw phase-space moments (conserved throughout the evolution).
# For massless particles in 3D:
#   M = 4π ∫ f(r) r² dr,   E = 4π ∫ r f(r) r² dr
#   T = E / (3M),           A = M / (8π T³)
juttner_params = None
if case.startswith('radial') and any(it == 0 for it, _ in coeffs):
    _, ic_coeff = next((it, c) for it, c in coeffs if it == 0)
    r_int = np.linspace(0.0, 30.0, 5000)
    f_int = np.array([eval_point(ic_coeff, r, 0.0, 0.0, N) for r in r_int])
    M_raw = 4 * np.pi * np.trapezoid(f_int * r_int**2, r_int)
    E_raw = 4 * np.pi * np.trapezoid(f_int * r_int**3, r_int)
    T = E_raw / (3 * M_raw)
    A = M_raw / (8 * np.pi * T**3)
    juttner_params = (A, T)
    print(f"Jüttner from IC moments: M={M_raw:.4f}, E={E_raw:.4f}, T={T:.4f}, A={A:.4f}")

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
                label=f'Jüttner (from IC)  A={A:.3f}, T={T:.3f}')

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
