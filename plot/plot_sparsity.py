import sys
import os
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

sys.path.insert(0, '../src')
from sparse import sparse_name

# ── config ────────────────────────────────────────────────────────────────────
n     = 4
n_lag = 11
n_leb = 13
# ─────────────────────────────────────────────────────────────────────────────

pkl_path = f'../src/{sparse_name(n, n_lag, n_leb)}'
tag      = os.path.splitext(os.path.basename(pkl_path))[0]
with open(pkl_path, 'rb') as fh:
    tensor = pickle.load(fh)

# map flat index → (k, l, m) label
labels = {}
for k in range(n):
    for l in range(n):
        for m in range(-l, l+1):
            labels[n*n*k + l*l + (m+l)] = (k, l, m)

n3            = n**3
total_nnz     = sum(mat.nnz for mat in tensor)
total_entries = n3 * n3 * n3

# global color scale: symmetric around 0 so diverging map is centred at white
all_nonzero = np.concatenate([tensor[t].data for t in range(n3) if tensor[t].nnz > 0])
vmax      = np.abs(all_nonzero).max()
linthresh = np.abs(all_nonzero).min()   # linear window as tight as the data allows
norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

cmap = mcm.RdBu_r.copy()
cmap.set_bad('white')          # masked zeros → white

os.makedirs('./figures', exist_ok=True)

# 3 rows × 9 cols: row = radial index k, cols = all (l,m) for that k
ncols = n3 // n   # = 9
nrows = n         # = 3

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

# shared colorbar on the right
sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, label='entry value')

fig_path = f'./figures/sparsity_{tag}.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'saved {fig_path}')
