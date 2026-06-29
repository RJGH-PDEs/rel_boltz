import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
sys.path.insert(0, '../src')
from lc import linear_comb

# ── flags ────────────────────────────────────────────────────────────────────
N          = 3
snapshots  = [0, 1, 2, 5, 10, 20, 100, 10000]   # iterations to plot
extent     = 10.0    # plot the two in-plane axes in [-extent, extent]
n_grid     = 150      # grid resolution per axis
show       = False    # showing one-by-one in a loop isn't practical; browse the saved files instead
save       = True
shared_scale = True   # True: same color scale across all frames; False: each frame auto-scales
planes     = ['xy', 'xz', 'yz']   # all three are shown side-by-side in one figure per snapshot
# Both views are rendered each run (into separate dirs):
#   direct     — F(u,v), the raw distribution (viridis)
#   asymmetry  — F(u,v) - F(u,-v), isolating the part of f that's odd under
#                v -> -v (e.g. a z-axis dipole on the xz/yz planes), since the
#                dominant even/radial part exactly cancels and would otherwise
#                wash out a small asymmetry on a linear color scale (coolwarm).
# Note an l=1,m=0 (dipole-along-z) perturbation is exactly zero on the xy plane
# (z=0), so it shows up flat there and only on xz/yz.
# ─────────────────────────────────────────────────────────────────────────────

us = np.linspace(-extent, extent, n_grid)   # horizontal in-plane axis
vs = np.linspace(-extent, extent, n_grid)   # vertical in-plane axis

# For each plane, (u,v) are the two in-plane Cartesian coordinates and the
# third coordinate is held at 0. 'v' is always the axis flipped for the
# asymmetry difference F(u,v) - F(u,-v).
PLANE_INFO = {
    'xy': dict(xlabel='ξ_x', ylabel='ξ_y', desc='ξ_z=0'),
    'xz': dict(xlabel='ξ_x', ylabel='ξ_z', desc='ξ_y=0'),
    'yz': dict(xlabel='ξ_y', ylabel='ξ_z', desc='ξ_x=0'),
}

def eval_plane(coeff, plane):
    F = np.zeros((n_grid, n_grid))
    for i, v in enumerate(vs):
        for j, u in enumerate(us):
            r = np.hypot(u, v)
            if r == 0.0:
                F[i, j] = linear_comb(coeff, 0.0, 0.0, 0.0, N)
                continue
            if plane == 'xy':
                theta, phi = np.pi / 2, np.arctan2(v, u)
            elif plane == 'xz':
                theta, phi = np.arccos(v / r), (0.0 if u >= 0 else np.pi)
            elif plane == 'yz':
                theta, phi = np.arccos(v / r), (np.pi / 2 if u >= 0 else -np.pi / 2)
            else:
                raise ValueError(f"unknown plane: {plane}")
            F[i, j] = np.exp(-r / 2) * linear_comb(coeff, r, theta, phi, N)
    return F

# ── render one view (direct f, or the v -> -v asymmetry) for all snapshots ────
def render(show_asymmetry):
    out_dir = f'./figures/heatmap_evolution_{"asymmetry" if show_asymmetry else "direct"}'
    os.makedirs(out_dir, exist_ok=True)

    # compute all (iteration, plane) panels
    snapshot_panels = {}   # it -> {plane: F}
    for it in snapshots:
        path = f"coeff/{it}.pkl"
        if not os.path.exists(path):
            print(f"missing {path}, skipping")
            continue
        with open(path, 'rb') as file:
            coeff = pickle.load(file)
        plane_data = {}
        for plane in planes:
            F = eval_plane(coeff, plane)
            if show_asymmetry:
                F = F - F[::-1, :]   # F(u,v) - F(u,-v)
            plane_data[plane] = F
        snapshot_panels[it] = plane_data

    if shared_scale:
        all_F = [F for plane_data in snapshot_panels.values() for F in plane_data.values()]
        if show_asymmetry:
            vmax = max(np.abs(F).max() for F in all_F)
            vmin = -vmax
        else:
            vmin = min(F.min() for F in all_F)
            vmax = max(F.max() for F in all_F)

    cmap = 'coolwarm' if show_asymmetry else 'viridis'

    for it, plane_data in snapshot_panels.items():
        fig, axes = plt.subplots(1, len(planes), figsize=(5*len(planes), 4.5))

        if not shared_scale:
            if show_asymmetry:
                m = max(np.abs(F).max() for F in plane_data.values())
                vmin, vmax = -m, m
            else:
                vmin = min(F.min() for F in plane_data.values())
                vmax = max(F.max() for F in plane_data.values())

        for ax, plane in zip(axes, planes):
            F = plane_data[plane]
            info = PLANE_INFO[plane]
            im = ax.imshow(F, extent=[-extent, extent, -extent, extent], origin='lower',
                            cmap=cmap, vmin=vmin, vmax=vmax)
            title = f'asymmetry, {info["desc"]}' if show_asymmetry else f'f(ξ), {info["desc"]}'
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(info['xlabel'])
            ax.set_ylabel(info['ylabel'])
            fig.colorbar(im, ax=ax, label='f(u,v)-f(u,-v)' if show_asymmetry else 'f', shrink=0.85)

        label = f'iter {it}' + (' (IC)' if it == 0 else '') + (' (final)' if it == 10000 else '')
        fig.suptitle(label)
        plt.tight_layout()

        if save:
            figure_name = f'{out_dir}/{it}.png'
            plt.savefig(figure_name, dpi=150)
            print(f"saved {figure_name}")

        if show:
            plt.show()
        plt.close(fig)

# render both views each run
render(show_asymmetry=False)
render(show_asymmetry=True)
