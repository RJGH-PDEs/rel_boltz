"""
Single-entry growth plot: |R_k| vs k_i for one representative entry,
with k_i^alpha overlaid for several values of alpha.

Entry: test function (k_i, l_i=2, m_i=0), trial functions s=(0,2,0), t=(0,2,0).
k_i=1,2 from the stored n=3 operator; k_i=3,4,5 via operator_numba (7,9).
The alpha curves are pinned to pass through the k_i=1 data point.

Run from plot/:
    ~/miniconda3/envs/ttenv/bin/python plot_single_entry_growth.py
Writes: ../plot/figures/single_entry_growth.png
"""

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, '../src')
from sparse import ind
from basis_numba import mu_const, spher_const
from quadrature import load_quad
from integrand_numba import operator_numba

# ── entry definition ──────────────────────────────────────────────────────────
L_I, M_I = 2, 0
KS, LS, MS = 0, 2, 0
KT, LT, MT = 0, 2, 0

KI_STORED = [0, 1, 2]
KI_NEW    = [3, 4, 5]
KI_ALL    = KI_STORED + KI_NEW
KI_PLOT   = np.array(KI_ALL[1:], dtype=float)   # k_i = 1..5

ALPHAS = [2.0, 2.5, 3.0, 3.5, 4.0]

# ── load ──────────────────────────────────────────────────────────────────────
with open('../src/sparse_operators/sparse_n3_lag7_leb9.pkl', 'rb') as fh:
    raw = pickle.load(fh)
sparse_list = raw if isinstance(raw, list) else raw['results']
n = round(len(sparse_list) ** (1/3))
print(f"loaded  n={n}")

print("loading (7,9) quadrature …", end='', flush=True)
raw_q, _, _ = load_quad('../src/quadratures/collision_lag7_leb9.pkl')
quad = np.array(raw_q, dtype=np.float64)
print(f"  {quad.shape[0]:,} points")

print("JIT warmup …", end='', flush=True)
_ = operator_numba(0,1,0,spher_const(1,0), 0,1,0,mu_const(0,1),spher_const(1,0),
                   0,2,0,mu_const(0,2),spher_const(2,0), quad)
print(" done")

# ── compute R_k ───────────────────────────────────────────────────────────────
c_i  = spher_const(L_I, M_I)
mu_s = mu_const(KS, LS);  c_s = spher_const(LS, MS)
mu_t = mu_const(KT, LT);  c_t = spher_const(LT, MT)

f_idx = ind(KS, LS, MS, n)
g_idx = ind(KT, LT, MT, n)

vals = [abs(float(sparse_list[ind(ki, L_I, M_I, n)][f_idx, g_idx]))
        for ki in KI_STORED]
for ki in KI_NEW:
    v = operator_numba(ki, L_I, M_I, c_i, KS, LS, MS, mu_s, c_s,
                       KT, LT, MT, mu_t, c_t, quad)
    vals.append(abs(v))

R = np.array(vals) / vals[0]   # normalize by k_i=0
R_plot = R[1:]                  # k_i = 1..5
print("\nR_k values:")
for ki, r in zip(KI_ALL, R):
    print(f"  k_i={ki}: R={r:.6f}")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title(
    r'Growth of $|R_k|$ for a single entry: $(\ell_i=2,\,m_i=0)$, '
    r'$s=(0,2,0)$, $t=(0,2,0)$'
    '\n'
    r'with $k_i^{\,\alpha}$ overlaid (pinned at $k_i=1$)',
    fontsize=11
)

# data points
ax.plot(KI_PLOT[:2], R_plot[:2], 'ko', ms=8, zorder=5, label='stored ($n=3$)')
ax.plot(KI_PLOT[2:], R_plot[2:], 'k*', ms=12, zorder=5, label='computed $(7,9)$')
ax.plot(KI_PLOT, R_plot, 'k-', lw=1.5, alpha=0.4)

# alpha reference curves, all pinned to pass through (1, R_1)
C = R_plot[0]   # = R_1, so C * 1^alpha = R_1 for any alpha
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(ALPHAS)))
for alpha, color in zip(ALPHAS, colors):
    curve = C * KI_PLOT ** alpha
    ax.plot(KI_PLOT, curve, color=color, lw=1.8, ls='--', alpha=0.85,
            label=rf'$k_i^{{{alpha}}}$')

ax.set_xscale('log');  ax.set_yscale('log')
ax.set_xticks(KI_PLOT);  ax.set_xticklabels([str(int(k)) for k in KI_PLOT])
ax.set_xlabel(r'$k_i$', fontsize=13)
ax.set_ylabel(r'$|R_{k_i}| = |Q_{k_i}|\,/\,|Q_0|$', fontsize=12)
ax.grid(True, alpha=0.25, which='both')
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
outpath = 'figures/single_entry_growth.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nsaved {outpath}")
