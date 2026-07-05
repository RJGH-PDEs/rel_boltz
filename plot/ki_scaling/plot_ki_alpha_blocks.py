"""
Check whether the power-law exponent alpha in R_k ~ k^alpha depends on
(k_s, k_t) within a fixed (l_s, l_t) family.

For l_i=2, m_i=0 and the (l_s=2, l_t=2) family: compute k_i=1..5 for all
9 (k_s, k_t) blocks with k_s,k_t in {0,1,2}.  Also check two (l_s=0, l_t=2)
blocks with k_s in {0,1} for comparison.

Fit log-log slope alpha for each block; display:
  Left  — log-log curves coloured by k_s (one curve per (k_s,k_t) block)
  Right — 3x3 heatmap of fitted alpha over (k_s,k_t) for l_s=l_t=2

Run from plot/ki_scaling/:
    ~/miniconda3/envs/ttenv/bin/python plot_ki_alpha_blocks.py
Writes: ../plot/figures/ki_alpha_blocks.png
"""

import pickle
import sys
from math import isqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, '../../src')
from sparse import ind
from basis_numba import mu_const, spher_const
from quadrature import load_quad
from integrand_numba import operator_numba

L_I = 2
M_I = 0
KI_STORED = [0, 1, 2]
KI_NEW    = [3, 4, 5]
KI_ALL    = KI_STORED + KI_NEW
KI_PLOT   = np.array(KI_ALL[1:])   # skip k_i=0 (R_0=1)

c_i = spher_const(L_I, M_I)


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
print(f"stored operator  n={n}")

print("loading (7,9) quadrature …", end='', flush=True)
raw_q, _, _ = load_quad('../../src/quadratures/collision_lag7_leb9.pkl')
quad = np.array(raw_q, dtype=np.float64)
print(f"  {quad.shape[0]:,} points")

print("JIT warmup …", end='', flush=True)
_ = operator_numba(0, 1, 0, spher_const(1,0),
                   0, 1, 0, mu_const(0,1), spher_const(1,0),
                   0, 2, 0, mu_const(0,2), spher_const(2,0), quad)
print(" done\n")

# ── define blocks to compute ──────────────────────────────────────────────────
# (l_s=2, l_t=2): all 9 (k_s,k_t) combinations with m_s=m_t=0
LS, LT = 2, 2
MS, MT = 0, 0

blocks_22 = [(ks, LS, MS, kt, LT, MT) for ks in range(3) for kt in range(3)]

# (l_s=0, l_t=2): k_s in {0,1}, k_t=0 for comparison
blocks_02 = [(ks, 0, 0, 0, 2, 0) for ks in range(2)]

all_blocks  = blocks_22 + blocks_02
all_labels  = [f'ks={b[0]},kt={b[3]}  (ls=lt=2)' for b in blocks_22] + \
              [f'ks={b[0]},kt={b[3]}  (ls=0,lt=2)' for b in blocks_02]

# ── compute R_k sequences ─────────────────────────────────────────────────────
def compute_Rk(ks, ls, ms, kt, lt, mt):
    f_idx = ind(ks, ls, ms, n)
    g_idx = ind(kt, lt, mt, n)
    mu_s = mu_const(ks, ls);  c_s = spher_const(ls, ms)
    mu_t = mu_const(kt, lt);  c_t = spher_const(lt, mt)
    vals = []
    for ki in KI_STORED:
        v = float(sparse_list[ind(ki, L_I, M_I, n)][f_idx, g_idx])
        vals.append(abs(v))
    if vals[0] < 1e-20:
        return None
    for ki in KI_NEW:
        v = operator_numba(ki, L_I, M_I, c_i,
                           ks, ls, ms, mu_s, c_s,
                           kt, lt, mt, mu_t, c_t, quad)
        vals.append(abs(v))
    v0 = vals[0]
    return np.array(vals) / v0   # R_k, normalized

print("computing blocks:")
results = {}
alphas_22 = np.full((3, 3), np.nan)

for bidx, (block, label) in enumerate(zip(all_blocks, all_labels)):
    ks, ls, ms, kt, lt, mt = block
    R = compute_Rk(ks, ls, ms, kt, lt, mt)
    if R is None:
        print(f"  {label}: Q_0~0, skip")
        continue
    R_plot = R[1:]   # k_i=1..5
    slope, _ = np.polyfit(np.log(KI_PLOT), np.log(R_plot), 1)
    results[bidx] = (R, slope)
    print(f"  {label}:  R1={R[1]:.3f}  R5={R[5]:.1f}  alpha={slope:.3f}")
    if ls == 2 and lt == 2:
        alphas_22[ks, kt] = slope

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_log, ax_heat) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    r'Does $\alpha$ depend on $(k_s, k_t)$?  [$\ell_i=2$, $m_i=0$, $(7,9)$ quadrature]',
    fontsize=12
)

# left: log-log curves for (l_s=2, l_t=2)
cmap_ks = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
ls_kt   = ['-', '--', '-.']
for bidx, (block, label) in enumerate(zip(blocks_22, all_labels[:9])):
    ks, ls, ms, kt, lt, mt = block
    if bidx not in results:
        continue
    R, slope = results[bidx]
    R_plot = R[1:]
    ax_log.plot(KI_PLOT, R_plot,
                color=cmap_ks[ks], linestyle=ls_kt[kt],
                lw=1.5, alpha=0.85, marker='o', ms=4)

# add (l_s=0, l_t=2) for comparison in gray
for bidx, (block, label) in enumerate(zip(blocks_02, all_labels[9:])):
    idx = 9 + bidx
    if idx not in results:
        continue
    R, slope = results[idx]
    ax_log.plot(KI_PLOT, R[1:], color='gray', lw=1.2, ls=':', alpha=0.7,
                marker='s', ms=4, label=label if bidx == 0 else '')

ax_log.set_xscale('log');  ax_log.set_yscale('log')
ax_log.set_xticks(KI_PLOT);  ax_log.set_xticklabels([str(k) for k in KI_PLOT])
ax_log.set_xlabel(r'$k_i$', fontsize=12)
ax_log.set_ylabel(r'$|Q_{k_i}|\,/\,|Q_0|$', fontsize=11)
ax_log.set_title(r'Log–log curves  ($\ell_s=\ell_t=2$ family)', fontsize=10)
ax_log.grid(True, alpha=0.2, which='both')

# legend for k_s colours and k_t line styles
from matplotlib.lines import Line2D
h_ks = [Line2D([],[],color=cmap_ks[k],lw=2,label=rf'$k_s={k}$') for k in range(3)]
h_kt = [Line2D([],[],color='k',lw=1.5,ls=ls_kt[k],label=rf'$k_t={k}$') for k in range(3)]
h_02 = [Line2D([],[],color='gray',lw=1.2,ls=':',marker='s',ms=4,
                label=r'$(l_s{=}0,l_t{=}2)$')]
ax_log.legend(handles=h_ks+h_kt+h_02, fontsize=8, ncol=2, loc='upper left')

# right: heatmap of alpha over (k_s, k_t) for l_s=l_t=2
im = ax_heat.imshow(alphas_22, origin='lower', cmap='plasma',
                    vmin=alphas_22[~np.isnan(alphas_22)].min() - 0.05,
                    vmax=alphas_22[~np.isnan(alphas_22)].max() + 0.05)
for ks in range(3):
    for kt in range(3):
        if not np.isnan(alphas_22[ks, kt]):
            ax_heat.text(kt, ks, f'{alphas_22[ks,kt]:.2f}',
                         ha='center', va='center', fontsize=11,
                         color='white' if alphas_22[ks,kt] > alphas_22.mean() else 'black')
ax_heat.set_xticks(range(3));  ax_heat.set_xticklabels(['0','1','2'])
ax_heat.set_yticks(range(3));  ax_heat.set_yticklabels(['0','1','2'])
ax_heat.set_xlabel(r'$k_t$', fontsize=12)
ax_heat.set_ylabel(r'$k_s$', fontsize=12)
ax_heat.set_title(r'Fitted $\alpha(k_s,k_t)$  for $\ell_s=\ell_t=2$, $\ell_i=2$',
                  fontsize=10)
plt.colorbar(im, ax=ax_heat, label=r'$\alpha$')

plt.tight_layout()
outpath = '../figures/ki_alpha_blocks.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nsaved {outpath}")
