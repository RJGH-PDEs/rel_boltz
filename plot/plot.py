import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
sys.path.insert(0, '../src')
from sparse import ind
from lc import linear_comb

os.makedirs("figures/axis_plots", exist_ok=True)
os.makedirs("coeff", exist_ok=True)

# ── flags ────────────────────────────────────────────────────────────────────
N    = 3
show = False   # one figure per axis is saved instead; browse the files
save = True
# ─────────────────────────────────────────────────────────────────────────────

i0 = ind(0, 0, 0, N)   # = 0
i1 = ind(1, 0, 0, N)   # = 9
i2 = ind(2, 0, 0, N)   # = 18

# ── evaluation grid ───────────────────────────────────────────────────────────
n_pts  = 200
coord_vals = np.linspace(-10.0, 10.0, n_pts)
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
            f[i] = linear_comb(coeff_vec, 0.0, 0.0, 0.0, N)
            continue
        if axis == 'x':
            theta, phi = np.pi / 2, (0.0 if c > 0 else np.pi)
        elif axis == 'y':
            theta, phi = np.pi / 2, (np.pi / 2 if c > 0 else -np.pi / 2)
        elif axis == 'z':
            theta, phi = (0.0 if c > 0 else np.pi), 0.0
        else:
            raise ValueError(f"unknown axis: {axis}")
        f[i] = np.exp(-r / 2) * linear_comb(coeff_vec, r, theta, phi, N)
    return f

# ── load snapshots ────────────────────────────────────────────────────────────
snapshots = [0, 1, 2, 3, 5, 8, 12, 17, 20, 10000]   # iteration numbers

coeffs = []
for it in snapshots:
    path = f"coeff/{it}.pkl"
    if not os.path.exists(path):
        print(f"missing {path}, skipping")
        continue
    with open(path, 'rb') as file:
        coeffs.append((it, pickle.load(file)))

cmap   = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(coeffs)))

for axis in ['x', 'y', 'z']:
    fig, ax = plt.subplots(figsize=(9, 5))

    for color, (it, coeff) in zip(colors, coeffs):
        label = f'iter {it}' + (' (IC)' if it == 0 else '') + (' (final)' if it == 10000 else '')
        ax.plot(coord_vals, eval_axis(coeff, axis), color=color, label=label)

    ax.set_title(f'time evolution of f along the p_{axis} axis')
    ax.set_xlabel(f'p_{axis}')
    ax.set_ylabel('f')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()

    if save:
        figure_name = f'./figures/axis_plots/p_{axis}.png'
        plt.savefig(figure_name, dpi=150)
        print(f"saved {figure_name}")

    if show:
        plt.show()
    plt.close(fig)
