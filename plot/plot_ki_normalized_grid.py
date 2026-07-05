"""
3×5 sign grid for the l_i=2 operator block.

Rows: k_i = 0, 1, 2
Cols: m_i = -2, -1, 0, 1, 2

Each panel shows sign(Q_{k_i,2,m_i,s,t}) = Q / |Q| ∈ {-1, +1} with a
two-tone diverging colormap.  This demonstrates:
  - constant support  (identical nonzero positions in all 15 panels)
  - alternating sign  (row k_i=1 is entirely flipped vs rows 0 and 2)
  - m-independence    (all five columns in each row are identical)

Run from plot/:
    ~/miniconda3/envs/ttenv/bin/python plot_ki_normalized_grid.py
Writes: ../plot/figures/ki_normalized_grid.png
"""

import pickle
import sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

sys.path.insert(0, '../src')
from sparse import ind

L_I     = 2
M_VALS  = list(range(-L_I, L_I + 1))   # [-2, -1, 0, 1, 2]
KI_VALS = [0, 1, 2]

# ── load ──────────────────────────────────────────────────────────────────────
with open('../src/sparse_operators/sparse_n3_lag7_leb9.pkl', 'rb') as fh:
    raw = pickle.load(fh)
sparse_list = raw if isinstance(raw, list) else raw['results']
n = round(len(sparse_list) ** (1/3))
print(f"loaded  n={n}")

# ── build sign panels ─────────────────────────────────────────────────────────
sign_data = {}   # (ki_idx, mi_idx) -> masked array of ±1

for mi_idx, m_i in enumerate(M_VALS):
    # use k_i=0 support as the reference mask (support is constant)
    mat0 = sparse_list[ind(0, L_I, m_i, n)].toarray()
    mask = np.abs(mat0) < 1e-20

    for ki_idx, k_i in enumerate(KI_VALS):
        mat  = sparse_list[ind(k_i, L_I, m_i, n)].toarray()
        signs = np.sign(mat)              # +1 / -1 / 0
        sign_data[(ki_idx, mi_idx)] = ma.array(signs, mask=mask)

# ── figure ────────────────────────────────────────────────────────────────────
# sample lighter shades from RdBu_r so +1 and -1 are clearly distinct
_rdbu = mcm.RdBu_r
blue_color = _rdbu(0.18)   # medium-light blue
red_color  = _rdbu(0.82)   # medium-light red
cmap = mcolors.ListedColormap([blue_color, red_color])
cmap.set_bad('white')
norm = mcolors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)

fig, axes = plt.subplots(3, 5, figsize=(10, 6.5), constrained_layout=True)

fig.suptitle(
    r'$\ell_i=2$ block: sign of each entry  '
    r'$\mathrm{sgn}\,Q_{(k_i,2,m_i),s,t} \in \{+1,-1\}$'
    '\n'
    r'rows $= k_i \in \{0,1,2\}$,  columns $= m_i \in \{-2,\ldots,2\}$',
    fontsize=11
)

for ki_idx, k_i in enumerate(KI_VALS):
    for mi_idx, m_i in enumerate(M_VALS):
        ax = axes[ki_idx, mi_idx]
        ax.imshow(sign_data[(ki_idx, mi_idx)],
                  cmap=cmap, norm=norm,
                  aspect='equal', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])

        if mi_idx == 0:
            ax.set_ylabel(rf'$k_i = {k_i}$', fontsize=12)
        if ki_idx == 0:
            ax.set_title(rf'$m_i = {m_i:+d}$', fontsize=11)

fig.supxlabel(r'$\psi_t$', fontsize=10)

sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.02, ticks=[-0.5, 0.5])
cb.ax.set_yticklabels([r'$-1$', r'$+1$'], fontsize=11)
cb.set_label('sign', fontsize=10)

outpath = 'figures/ki_normalized_grid.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"saved {outpath}")
