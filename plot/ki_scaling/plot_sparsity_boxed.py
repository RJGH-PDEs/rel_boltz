"""
Reproduce the sparsity plot for sparse_n3_lag7_leb9 and overlay a red box
around the 3×5 block corresponding to l_i=2 test functions (columns 4–8,
all three k_i rows).  The first four columns (l_i=0 and l_i=1) are left
unboxed — those test functions couple to collision invariants and are
excluded from the k_i-scaling analysis.

Run from plot/ki_scaling/:
    ~/miniconda3/envs/ttenv/bin/python plot_sparsity_boxed.py
Writes: ../figures/sparsity_sparse_n3_lag7_leb9_boxed.png
"""

import sys
import os
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

sys.path.insert(0, '../../src')
from sparse import sparse_name

# ── config ────────────────────────────────────────────────────────────────────
n     = 3
n_lag = 7
n_leb = 9
# ─────────────────────────────────────────────────────────────────────────────

pkl_path = f'../../src/{sparse_name(n, n_lag, n_leb)}'
tag      = os.path.splitext(os.path.basename(pkl_path))[0]
with open(pkl_path, 'rb') as fh:
    tensor = pickle.load(fh)

labels = {}
for k in range(n):
    for l in range(n):
        for m in range(-l, l+1):
            labels[n*n*k + l*l + (m+l)] = (k, l, m)

n3            = n**3
total_nnz     = sum(mat.nnz for mat in tensor)
total_entries = n3 * n3 * n3

all_nonzero = np.concatenate([tensor[t].data for t in range(n3) if tensor[t].nnz > 0])
vmax      = np.abs(all_nonzero).max()
linthresh = np.abs(all_nonzero).min()
norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

cmap = mcm.RdBu_r.copy()
cmap.set_bad('white')

os.makedirs('../figures', exist_ok=True)

ncols = n3 // n   # 9
nrows = n         # 3

fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 4.5),
                         constrained_layout=True)

for t in range(n3):
    row = t // ncols
    col = t % ncols
    ax  = axes[row, col]

    arr    = tensor[t].toarray()
    masked = ma.array(arr, mask=(arr == 0))

    ax.imshow(masked, cmap=cmap, norm=norm,
              aspect='equal', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    k, l, m = labels[t]
    ax.set_title(f'({k},{l},{m})', fontsize=7, pad=2)

    nnz = tensor[t].nnz
    ax.text(0.97, 0.03, f'nnz {nnz}',
            transform=ax.transAxes, fontsize=6,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

fig.supxlabel(r'$\psi_t$', fontsize=9)
fig.supylabel(r'$\psi_s$', fontsize=9)

fig.suptitle(
    f'Sparsity pattern — {tag}   [panel title $(k,l,m)$ = test function]\n'
    f'{total_nnz} / {total_entries} nonzeros  ({100 * total_nnz / total_entries:.1f}%)',
    fontsize=9
)

sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, label='entry value')

# ── red box around the l_i=2 block (columns 4–8, all rows) ───────────────────
# constrained_layout finalises positions at draw time
fig.canvas.draw()

# top-left panel of the box: row=0, col=4  (k_i=0, l_i=2, m_i=-2)
# bottom-right panel:        row=2, col=8  (k_i=2, l_i=2, m_i=+2)
ax_tl = axes[0, 4]
ax_br = axes[2, 8]

bb_tl = ax_tl.get_position()   # in figure fraction
bb_br = ax_br.get_position()

pad_x   = 0.005
pad_bot = 0.018
pad_top = 0.038
x0 = bb_tl.x0 - pad_x
y0 = bb_br.y0 - pad_bot
x1 = bb_br.x1 + pad_x
y1 = bb_tl.y1 + pad_top

rect = mpatches.FancyBboxPatch(
    (x0, y0), x1 - x0, y1 - y0,
    boxstyle='square,pad=0',
    linewidth=2.5, edgecolor='red', facecolor='none',
    transform=fig.transFigure, clip_on=False, zorder=10
)
fig.add_artist(rect)

fig_path = f'../figures/sparsity_{tag}_boxed.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'saved {fig_path}')
