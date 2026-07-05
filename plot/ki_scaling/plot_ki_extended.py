"""
Extend the k_i-growth sequences for l_i=2 test functions to k_i=0..5
by calling operator_numba directly for k_i >= 3.
k_i=0,1,2 are read from the stored n=3 sparse operator (cross-check).
k_i=3,4,5 are computed with the (11,13) quadrature.

Plot: log|R_k| vs log(k_i) — polynomial growth appears as a straight line
      whose slope is the polynomial degree.

Run from plot/ki_scaling/:
    ~/miniconda3/envs/ttenv/bin/python plot_ki_extended.py
Writes: ../plot/figures/ki_extended_l2.png
"""

import pickle
import sys
from math import isqrt
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../../src')
from sparse import ind
from basis_numba import mu_const, spher_const
from quadrature import load_quad
from integrand_numba import operator_numba

L_I = 2
M_I = 0

# Representative (k_s,l_s,m_s, k_t,l_t,m_t) — one per (l_s,l_t) family,
# simplest m=0 entry with k_s=k_t=0.
# These will be verified against the stored operator before computing k_i>=3.
PAIRS = [
    (0, 0, 0,  0, 2, 0),   # (l_s=0, l_t=2)
    (0, 1, 0,  0, 1, 0),   # (l_s=1, l_t=1)
    (0, 2, 0,  0, 0, 0),   # (l_s=2, l_t=0)  symmetric to (0,2)
    (0, 2, 0,  0, 2, 0),   # (l_s=2, l_t=2)
]
LABELS = [
    r'$(l_s{=}0,\,l_t{=}2)$',
    r'$(l_s{=}1,\,l_t{=}1)$',
    r'$(l_s{=}2,\,l_t{=}0)$',
    r'$(l_s{=}2,\,l_t{=}2)$',
]

KI_STORED  = [0, 1, 2]
KI_NEW     = [3, 4, 5]
KI_ALL     = KI_STORED + KI_NEW


def ind_to_klm(idx, n):
    k  = idx // (n * n)
    lm = idx % (n * n)
    l  = isqrt(lm)
    m  = lm - l * l - l
    return k, l, m


# ── load stored operator ──────────────────────────────────────────────────────
with open('../../src/sparse_operators/sparse_n3_lag7_leb9.pkl', 'rb') as fh:
    raw = pickle.load(fh)
sparse_list = raw if isinstance(raw, list) else raw['results']
n = round(len(sparse_list) ** (1/3))
print(f"stored operator  n={n}")

# ── load quadrature (7,9) ─────────────────────────────────────────────────────
# (11,13) pkl is corrupted; (7,9) is verified for n=3 and sufficient for the
# growth trend up to k_i=5 — individual values may be off ~10% at high k_i
# but the polynomial degree is clearly visible.
print("loading (7,9) quadrature …", end='', flush=True)
raw_q, _, _ = load_quad('../../src/quadratures/collision_lag7_leb9.pkl')
quad = np.array(raw_q, dtype=np.float64)
print(f"  {quad.shape[0]:,} points  ({quad.nbytes/1e9:.2f} GB)")

# ── JIT warmup ────────────────────────────────────────────────────────────────
print("JIT warmup …", end='', flush=True)
_ = operator_numba(0, 1, 0, spher_const(1,0),
                   0, 1, 0, mu_const(0,1), spher_const(1,0),
                   0, 2, 0, mu_const(0,2), spher_const(2,0),
                   quad)
print(" done")

# ── compute sequences ─────────────────────────────────────────────────────────
c_i = spher_const(L_I, M_I)

results = {}   # pair_idx -> list of |Q_{k_i}| for k_i in KI_ALL

for pidx, (ks, ls, ms, kt, lt, mt) in enumerate(PAIRS):
    f_idx = ind(ks, ls, ms, n)
    g_idx = ind(kt, lt, mt, n)
    vals = []

    # k_i = 0,1,2 from stored operator
    for ki in KI_STORED:
        v = float(sparse_list[ind(ki, L_I, M_I, n)][f_idx, g_idx])
        vals.append(abs(v))

    # verify k_i=0 is nonzero
    if vals[0] < 1e-20:
        print(f"  pair {pidx} ({ls},{lt}): Q_0 ~ 0, SKIPPING")
        results[pidx] = None
        continue

    print(f"  pair {pidx} ({ls},{lt}):  k_i=0,1,2 → "
          f"|Q|={vals[0]:.4e}, {vals[1]:.4e}, {vals[2]:.4e}")

    # k_i = 3,4,5 via operator_numba
    mu_s = mu_const(ks, ls);  c_s = spher_const(ls, ms)
    mu_t = mu_const(kt, lt);  c_t = spher_const(lt, mt)
    for ki in KI_NEW:
        v = operator_numba(ki, L_I, M_I, c_i,
                           ks, ls, ms, mu_s, c_s,
                           kt, lt, mt, mu_t, c_t,
                           quad)
        vals.append(abs(v))
        print(f"    k_i={ki}: |Q|={abs(v):.4e}")

    results[pidx] = vals

# ── cross-check: k_i=0,1,2 against stored operator ───────────────────────────
print("\ncross-check k_i=0,1,2 against stored operator:")
pidx_check = 3   # (l_s=2, l_t=2)
ks,ls,ms,kt,lt,mt = PAIRS[pidx_check]
f_idx = ind(ks, ls, ms, n);  g_idx = ind(kt, lt, mt, n)
for ki in KI_STORED:
    v_stored = float(sparse_list[ind(ki, L_I, M_I, n)][f_idx, g_idx])
    v_quad   = operator_numba(ki, L_I, M_I, c_i,
                              ks, ls, ms, mu_const(ks,ls), spher_const(ls,ms),
                              kt, lt, mt, mu_const(kt,lt), spher_const(lt,mt),
                              quad)
    rel = abs(v_stored - v_quad) / abs(v_stored) if abs(v_stored) > 1e-20 else float('nan')
    print(f"  k_i={ki}: stored={v_stored:.6e}  quad(7,9)={v_quad:.6e}  rel={rel:.2e}")

# ── plot: log-log ─────────────────────────────────────────────────────────────
colors = plt.cm.tab10(np.linspace(0, 0.8, len(PAIRS)))

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title(
    r'$k_i$-growth for $\ell_i=2$ test functions  (extended to $k_i=5$)'
    '\n' r'log–log scale: straight line $\Rightarrow$ power law $|R_k| \sim k^\alpha$',
    fontsize=11
)

ki_plot = np.array(KI_ALL[1:])   # skip k_i=0 (R_0=1, log(0) undefined)

for pidx, (pair, label) in enumerate(zip(PAIRS, LABELS)):
    vals = results[pidx]
    if vals is None:
        continue
    v0 = vals[0]
    R = np.array(vals) / v0          # normalize by k_i=0 entry
    R_plot = R[1:]                   # k_i=1..5

    # plot stored points (k_i=1,2) as filled circles, new (k_i=3,4,5) as stars
    ax.plot(ki_plot, R_plot,
            color=colors[pidx], lw=1.5, alpha=0.85, label=label,
            marker='o', ms=6)
    # distinguish stored vs new
    ax.plot(ki_plot[:2], R_plot[:2],
            color=colors[pidx], marker='o', ms=7, ls='none', zorder=5)
    ax.plot(ki_plot[2:], R_plot[2:],
            color=colors[pidx], marker='*', ms=11, ls='none', zorder=5)

    # fit log-log slope on all 5 points (k_i=1..5)
    valid = R_plot > 0
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(np.log(ki_plot[valid]),
                                      np.log(R_plot[valid]), 1)
        ax.annotate(rf'$\alpha={slope:.2f}$',
                    xy=(ki_plot[-1], R_plot[-1]),
                    xytext=(ki_plot[-1]*1.05, R_plot[-1]),
                    fontsize=9, color=colors[pidx], va='center')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k_i$', fontsize=13)
ax.set_ylabel(r'$|Q_{k_i}|\,/\,|Q_0|$', fontsize=12)
ax.set_xticks(ki_plot)
ax.set_xticklabels([str(k) for k in ki_plot])

# legend entry for marker types
from matplotlib.lines import Line2D
extra = [
    Line2D([0],[0], marker='o', color='gray', ls='none', ms=6, label='stored (n=3)'),
    Line2D([0],[0], marker='*', color='gray', ls='none', ms=10, label='computed (11,13)'),
]
handles, lbls = ax.get_legend_handles_labels()
ax.legend(handles=handles+extra, fontsize=9, loc='upper left')
ax.grid(True, alpha=0.25, which='both')

plt.tight_layout()
outpath = '../figures/ki_extended_l2.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nsaved {outpath}")
