"""
5-panel signed growth plot for l_i=2 test functions.

Each panel (one per m_i ∈ {-2,-1,0,1,2}) shows
    Q_{(k_i,2,m_i),s,t} / Q_{(0,2,m_i),s,t}
for k_i = 0,1,2, one curve per nonzero (s,t) entry, coloured by (l_s,l_t).

Unlike ki_growth_l2.py, no absolute value is taken, so the sign alternation
(+1 at k_i=0, negative at k_i=1, positive at k_i=2) is directly visible.

Run from plot/ki_scaling/:
    ~/miniconda3/envs/ttenv/bin/python plot_ki_growth_signed.py
Writes: ../plot/figures/ki_growth_signed.png
"""

import pickle
import sys
from math import isqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, '../../src')
from sparse import ind

L_I    = 2
M_VALS = list(range(-L_I, L_I + 1))   # [-2, -1, 0, 1, 2]
KI_VALS = np.array([0, 1, 2])

def ind_to_klm(idx, n):
    k  = idx // (n * n)
    lm = idx % (n * n)
    l  = isqrt(lm)
    m  = lm - l * l - l
    return k, l, m

# ── load ──────────────────────────────────────────────────────────────────────
with open('../../src/sparse_operators/sparse_n3_lag7_leb9.pkl', 'rb') as fh:
    raw = pickle.load(fh)
sparse_list = raw if isinstance(raw, list) else raw['results']
n = round(len(sparse_list) ** (1/3))
print(f"loaded  n={n}")

# discover unique (l_s, l_t) pairs for the legend
all_pairs = set()
for m_i in M_VALS:
    mat0 = sparse_list[ind(0, L_I, m_i, n)]
    for f, g in zip(*mat0.nonzero()):
        if abs(float(mat0[f, g])) > 1e-20:
            _, ls, _ = ind_to_klm(f, n)
            _, lt, _ = ind_to_klm(g, n)
            all_pairs.add((ls, lt))

unique_pairs = sorted(all_pairs)
palette      = plt.cm.tab10(np.linspace(0, 0.9, len(unique_pairs)))
pair_color   = {p: palette[i] for i, p in enumerate(unique_pairs)}

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), sharey=True)
fig.suptitle(
    r'Signed normalized growth for test functions $(\ell_i=2,\,m_i)$, all nonzero $(s,t)$'
    '\n'
    r'$Q_{(k_i,2,m_i),s,t}\,/\,Q_{(0,2,m_i),s,t}$  vs  $k_i$',
    fontsize=11
)

for ax, m_i in zip(axes, M_VALS):
    mats = [sparse_list[ind(ki, L_I, m_i, n)] for ki in [0, 1, 2]]
    mat0 = mats[0]
    rows, cols = mat0.nonzero()

    for f, g in zip(rows.tolist(), cols.tolist()):
        v0 = float(mat0[f, g])
        if abs(v0) < 1e-20:
            continue
        seq = np.array([float(mats[ki][f, g]) / v0 for ki in range(3)])
        _, ls, _ = ind_to_klm(f, n)
        _, lt, _ = ind_to_klm(g, n)
        ax.plot(KI_VALS, seq,
                color=pair_color[(ls, lt)], lw=1.2, alpha=0.7,
                marker='o', ms=3.5)

    ax.axhline(0, color='k', lw=0.8, ls='-', zorder=0)
    ax.axhline(1, color='k', lw=0.5, ls=':', zorder=0)
    ax.set_yscale('symlog', linthresh=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel(r'$k_i$', fontsize=11)
    ax.set_title(rf'$m_i = {m_i}$', fontsize=11)
    ax.grid(True, alpha=0.2, which='both')

axes[0].set_ylabel(r'$Q_{k_i}\,/\,Q_0$', fontsize=11)
axes[0].set_yticks([-10, -1, 0, 1, 10, 100])

# legend
handles = [
    mlines.Line2D([], [], color=pair_color[p], lw=2, marker='o', ms=5,
                  label=rf'$\ell_s={p[0]},\,\ell_t={p[1]}$')
    for p in unique_pairs
]
fig.legend(handles=handles, title=r'$(\ell_s,\,\ell_t)$',
           loc='lower center', ncol=len(unique_pairs),
           fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.1, 1, 1])
outpath = '../figures/ki_growth_signed.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"saved {outpath}")
