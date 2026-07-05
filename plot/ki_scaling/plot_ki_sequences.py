"""
Plot the k_i-sequence Q_{(k_i,l_i,m_i),s,t} for k_i = 0,1,2 with fixed (l_i,m_i=0)
and fixed (s,t), one curve per (k_s,l_s,k_t,l_t) block.  Uses m_i=0 and takes one
representative per block (m-independence verified).

Left panel:  raw Q values.
Right panel: normalized by Q_{(0,l_i,m_i),s,t} — all curves start at 1.

Run from plot/ki_scaling/:
    ~/miniconda3/envs/ttenv/bin/python plot_ki_sequences.py
Writes: ../plot/figures/ki_sequences.png
"""

import pickle
import sys
from math import isqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, '../../src')
from sparse import ind


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

# ── figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    r'$|Q_{(k_i,\,l_i,\,0),\,s,\,t}|\,/\,|Q_0|$ sequences for $k_i = 0,1,2$'
    '\none curve per $(k_s,l_s,k_t,l_t)$ block  [n=3 operator]',
    fontsize=12
)

ki_vals = np.array([0, 1, 2])

# color map: one color per (l_s, l_t) pair
# all possible pairs from nonzero entries, across both l_i values
all_lt_pairs = set()
for l_i in [1, 2]:
    m_i = 0
    mat0 = sparse_list[ind(0, l_i, m_i, n)]
    for f, g in zip(*mat0.nonzero()):
        _, ls, _ = ind_to_klm(f, n)
        _, lt, _ = ind_to_klm(g, n)
        all_lt_pairs.add((ls, lt))
lt_pairs_sorted = sorted(all_lt_pairs)
palette = plt.cm.tab10(np.linspace(0, 0.9, len(lt_pairs_sorted)))
pair_color = {p: palette[i] for i, p in enumerate(lt_pairs_sorted)}

for row, l_i in enumerate([1, 2]):
    m_i = 0
    mat0 = sparse_list[ind(0, l_i, m_i, n)]
    rows_nz, cols_nz = mat0.nonzero()

    # one representative per (k_s,l_s,k_t,l_t) block
    # prefer the entry with smallest |m_s|+|m_t| (most "clean")
    blocks = {}
    for f, g in zip(rows_nz.tolist(), cols_nz.tolist()):
        v0 = float(mat0[f, g])
        if abs(v0) < 1e-20:
            continue
        ks, ls, ms = ind_to_klm(f, n)
        kt, lt, mt = ind_to_klm(g, n)
        key = (ks, ls, kt, lt)
        score = abs(ms) + abs(mt)
        if key not in blocks or score < blocks[key][2]:
            blocks[key] = (f, g, score)

    ax_norm = axes[row]

    for (ks, ls, kt, lt), (f, g, _) in sorted(blocks.items()):
        seq = np.array([float(sparse_list[ind(ki, l_i, m_i, n)][f, g])
                        for ki in [0, 1, 2]])
        color = pair_color[(ls, lt)]

        if abs(seq[0]) > 1e-20:
            ax_norm.plot(ki_vals, np.abs(seq) / abs(seq[0]), color=color, lw=1.2, alpha=0.75,
                         marker='o', ms=4)

    ax_norm.set_yscale('log')
    ax_norm.set_xticks([0, 1, 2])
    ax_norm.set_xlabel(r'$k_i$', fontsize=11)
    ax_norm.grid(True, alpha=0.3, which='both')
    ax_norm.set_title(rf'$l_i={l_i}$, $m_i=0$', fontsize=11)
    ax_norm.set_ylabel(r'$|Q_{k_i}| \,/\, |Q_0|$', fontsize=10)
    ax_norm.axhline(1, color='k', lw=0.6, ls=':', zorder=0)

# shared legend on right side
legend_handles = [
    mlines.Line2D([], [], color=pair_color[(ls, lt)], lw=2, marker='o', ms=5,
                  label=rf'$l_s={ls},\,l_t={lt}$')
    for (ls, lt) in lt_pairs_sorted
]
fig.legend(handles=legend_handles, title='$(l_s, l_t)$', loc='lower center',
           ncol=len(lt_pairs_sorted), fontsize=9, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout(rect=[0, 0.08, 1, 1])
outpath = '../figures/ki_sequences.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"saved {outpath}")
