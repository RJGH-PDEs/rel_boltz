"""
Log-log growth plot R_k vs k_i for l_i=2, m_i=0, extended to k_i=5.

Shows:
  - (l_s=2, l_t=2): all 9 (k_s,k_t) blocks with k_s,k_t in {0,1,2}
  - (l_s=0, l_t=2), (l_s=1, l_t=1), (l_s=2, l_t=0): k_s=k_t=0 representative each

k_i=0,1,2 from stored n=3 sparse operator (circles).
k_i=3,4,5 from operator_numba with (7,9) quadrature (stars).

Run from plot/:
    ~/miniconda3/envs/ttenv/bin/python plot_ki_loglog_full.py
Writes: ../plot/figures/ki_loglog_full.png
"""

import pickle
import sys
from math import isqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, '../src')
from sparse import ind
from basis_numba import mu_const, spher_const
from quadrature import load_quad
from integrand_numba import operator_numba

L_I, M_I = 2, 0
KI_STORED = [0, 1, 2]
KI_NEW    = [3, 4, 5]
KI_ALL    = KI_STORED + KI_NEW
KI_PLOT   = np.array(KI_ALL[1:])   # k_i = 1..5
c_i = spher_const(L_I, M_I)

def ind_to_klm(idx, n):
    k  = idx // (n * n)
    lm = idx % (n * n)
    l  = isqrt(lm)
    m  = lm - l * l - l
    return k, l, m

# ── load ──────────────────────────────────────────────────────────────────────
with open('../src/sparse_operators/sparse_n3_lag7_leb9.pkl', 'rb') as fh:
    raw = pickle.load(fh)
sparse_list = raw if isinstance(raw, list) else raw['results']
n = round(len(sparse_list) ** (1/3))
print(f"stored operator  n={n}")

print("loading (7,9) quadrature …", end='', flush=True)
raw_q, _, _ = load_quad('../src/quadratures/collision_lag7_leb9.pkl')
quad = np.array(raw_q, dtype=np.float64)
print(f"  {quad.shape[0]:,} points")

print("JIT warmup …", end='', flush=True)
_ = operator_numba(0,1,0,spher_const(1,0), 0,1,0,mu_const(0,1),spher_const(1,0),
                   0,2,0,mu_const(0,2),spher_const(2,0), quad)
print(" done\n")

# ── define blocks ─────────────────────────────────────────────────────────────
# (l_s=2, l_t=2): all 9 (k_s, k_t), m_s=m_t=0
blocks_22 = [(ks,2,0, kt,2,0) for ks in range(3) for kt in range(3)]
# one k_s=k_t=0 representative per other (l_s,l_t) family
blocks_other = [
    (0,0,0, 0,2,0),   # (l_s=0, l_t=2)
    (0,1,0, 0,1,0),   # (l_s=1, l_t=1)
    (0,2,0, 0,0,0),   # (l_s=2, l_t=0)
]
all_blocks = blocks_22 + blocks_other

# ── compute R_k for each block ────────────────────────────────────────────────
def compute_Rk(ks, ls, ms, kt, lt, mt):
    f_idx = ind(ks, ls, ms, n);  g_idx = ind(kt, lt, mt, n)
    mu_s = mu_const(ks,ls);  c_s = spher_const(ls,ms)
    mu_t = mu_const(kt,lt);  c_t = spher_const(lt,mt)
    vals = [abs(float(sparse_list[ind(ki,L_I,M_I,n)][f_idx,g_idx]))
            for ki in KI_STORED]
    if vals[0] < 1e-20:
        return None
    for ki in KI_NEW:
        v = operator_numba(ki,L_I,M_I,c_i, ks,ls,ms,mu_s,c_s, kt,lt,mt,mu_t,c_t, quad)
        vals.append(abs(v))
    v0 = vals[0]
    return np.array(vals) / v0

print("computing:")
R_data   = {}
alpha_22 = np.full((3,3), np.nan)

for bidx, block in enumerate(all_blocks):
    ks,ls,ms,kt,lt,mt = block
    R = compute_Rk(ks,ls,ms,kt,lt,mt)
    if R is None:
        print(f"  ({ks},{ls},{kt},{lt}): Q_0~0, skip")
        continue
    R_plot = R[1:]
    slope, _ = np.polyfit(np.log(KI_PLOT), np.log(R_plot), 1)
    R_data[bidx] = (R, slope)
    print(f"  ({ks},{ls},{kt},{lt})  R1={R[1]:.3f}  alpha={slope:.3f}")
    if ls==2 and lt==2:
        alpha_22[ks, kt] = slope

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title(
    r'$k_i$-growth for $\ell_i=2$ test functions, varied trial functions'
    '\n'
    r'log–log: $|Q_{k_i}|/|Q_0| \sim k_i^{\,\alpha}$  '
    r'[circles = stored $n=3$,  stars = computed $(7,9)$]',
    fontsize=11
)

# (l_s=2, l_t=2): viridis shaded by k_s, line style by k_t
cmap_ks  = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
ls_styles = ['-', '--', '-.']
ks_colors = {0: cmap_ks[0], 1: cmap_ks[1], 2: cmap_ks[2]}

for bidx, block in enumerate(blocks_22):
    ks,ls,ms,kt,lt,mt = block
    if bidx not in R_data:
        continue
    R, slope = R_data[bidx]
    R_plot = R[1:]
    ax.plot(KI_PLOT, R_plot,
            color=ks_colors[ks], linestyle=ls_styles[kt],
            lw=1.6, alpha=0.85)
    ax.plot(KI_PLOT[:2], R_plot[:2],
            color=ks_colors[ks], marker='o', ms=6, ls='none', zorder=5)
    ax.plot(KI_PLOT[2:], R_plot[2:],
            color=ks_colors[ks], marker='*', ms=10, ls='none', zorder=5)
    # annotate alpha at k_i=5
    ax.annotate(rf'$\alpha\!=\!{slope:.2f}$',
                xy=(KI_PLOT[-1], R_plot[-1]),
                xytext=(5.15, R_plot[-1]),
                fontsize=7.5, color=ks_colors[ks], va='center')

# other families: distinct colours, solid line, k_s=k_t=0
other_colors = ['royalblue', 'crimson', 'darkorange']
other_labels = [r'$(l_s{=}0,l_t{=}2)$', r'$(l_s{=}1,l_t{=}1)$', r'$(l_s{=}2,l_t{=}0)$']
for i, block in enumerate(blocks_other):
    bidx = len(blocks_22) + i
    if bidx not in R_data:
        continue
    R, slope = R_data[bidx]
    R_plot = R[1:]
    ax.plot(KI_PLOT, R_plot,
            color=other_colors[i], lw=2.0, ls='-', alpha=0.9,
            label=f'{other_labels[i]}  ' + rf'$\alpha\!=\!{slope:.2f}$')
    ax.plot(KI_PLOT[:2], R_plot[:2],
            color=other_colors[i], marker='o', ms=7, ls='none', zorder=5)
    ax.plot(KI_PLOT[2:], R_plot[2:],
            color=other_colors[i], marker='*', ms=11, ls='none', zorder=5)

ax.set_xscale('log');  ax.set_yscale('log')
ax.set_xticks(KI_PLOT);  ax.set_xticklabels([str(k) for k in KI_PLOT])
ax.set_xlabel(r'$k_i$', fontsize=13)
ax.set_ylabel(r'$|Q_{k_i}|\,/\,|Q_0|$', fontsize=12)
ax.grid(True, alpha=0.2, which='both')

# legend
h_ks = [Line2D([],[],color=ks_colors[k],lw=2,label=rf'$k_s={k}$ (ls=lt=2)')
        for k in range(3)]
h_kt = [Line2D([],[],color='gray',lw=1.5,ls=ls_styles[k],label=rf'$k_t={k}$ (ls=lt=2)')
        for k in range(3)]
other_handles, other_lbls = ax.get_legend_handles_labels()
h_stored = Line2D([],[],color='gray',marker='o',ms=6,ls='none',label='stored (n=3)')
h_new    = Line2D([],[],color='gray',marker='*',ms=9,ls='none',label='computed (7,9)')
ax.legend(handles=h_ks+h_kt+other_handles+[h_stored,h_new],
          fontsize=8, ncol=2, loc='upper left')

plt.tight_layout()
outpath = 'figures/ki_loglog_full.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nsaved {outpath}")
