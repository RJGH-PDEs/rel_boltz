"""
Check whether Q_{(k_i, l_i, m_i), s, t} / Q_{(0, l_i, m_i), s, t}
is a universal constant — independent of the field pair (s,t) — for
each (k_i, l_i).

Part 1: crude check — is R_k constant across ALL nonzero (s,t)?
Part 2: block check — is R_k constant within each (k_s,l_s,k_t,l_t) block?
        (i.e. does it depend on m values?)

Run from docs/:
    ~/miniconda3/envs/ttenv/bin/python verify_ki_scaling.py
"""

import pickle
import sys
from math import isqrt
from collections import defaultdict
import numpy as np

sys.path.insert(0, '../src')
from sparse import ind


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
print(f"loaded  n={n},  {len(sparse_list)} matrices")

# ── Part 1: crude check ───────────────────────────────────────────────────────

print()
print("=" * 72)
print("Part 1  R_k over ALL nonzero (s,t)")
print("=" * 72)

for l_i in range(1, n):
    m_i = 0
    idx0 = ind(0, l_i, m_i, n)
    mat0 = sparse_list[idx0]
    rows, cols = mat0.nonzero()
    if len(rows) == 0:
        continue

    print(f"\n  l_i={l_i}, m_i=0  ({len(rows)} nonzero in k_i=0 slice)")
    for k_i in (1, 2):
        R_vals = []
        for f, g in zip(rows.tolist(), cols.tolist()):
            v0 = float(mat0[f, g])
            if abs(v0) < 1e-20:
                continue
            vk = float(sparse_list[ind(k_i, l_i, m_i, n)][f, g])
            R_vals.append(vk / v0)
        arr = np.array(R_vals)
        rel = arr.std() / abs(arr.mean()) if abs(arr.mean()) > 1e-20 else np.inf
        print(f"    k_i={k_i}:  mean={arr.mean():+.4f}  std={arr.std():.3f}"
              f"  range=[{arr.min():.3f}, {arr.max():.3f}]  rel_std={rel:.2f}")

# ── Part 2: block check ───────────────────────────────────────────────────────

print()
print("=" * 72)
print("Part 2  R_k grouped by (k_s,l_s,k_t,l_t) block  (m_i=0)")
print("  Checks whether R_k depends on m_s,m_t within each block")
print("=" * 72)

for l_i in range(1, n):
    m_i = 0
    idx0 = ind(0, l_i, m_i, n)
    mat0 = sparse_list[idx0]
    rows, cols = mat0.nonzero()
    if len(rows) == 0:
        continue

    # group by (k_s, l_s, k_t, l_t)
    for k_i in (1, 2):
        mat_ki = sparse_list[ind(k_i, l_i, m_i, n)]
        blocks = defaultdict(list)  # (ks,ls,kt,lt) -> list of R values
        for f, g in zip(rows.tolist(), cols.tolist()):
            v0 = float(mat0[f, g])
            if abs(v0) < 1e-20:
                continue
            ks, ls, ms = ind_to_klm(f, n)
            kt, lt, mt = ind_to_klm(g, n)
            vk = float(mat_ki[f, g])
            blocks[(ks, ls, kt, lt)].append((ms, mt, vk / v0))

        print(f"\n  l_i={l_i}, m_i=0, k_i={k_i}  — R_{k_i} by (k_s,l_s,k_t,l_t) block:")
        print(f"    {'(ks,ls,kt,lt)':>18}  {'R mean':>10}  {'R std':>10}  {'m-indep?':>10}")
        print(f"    {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}")
        for key in sorted(blocks):
            vals = [r for _, _, r in blocks[key]]
            arr  = np.array(vals)
            mean = arr.mean()
            std  = arr.std()
            rel  = std / abs(mean) if abs(mean) > 1e-20 else np.inf
            ok   = "YES" if rel < 1e-8 else f"NO {rel:.1e}"
            ks, ls, kt, lt = key
            print(f"    ({ks},{ls},{kt},{lt}):  {mean:>10.4f}  {std:>10.2e}  {ok:>10}")
