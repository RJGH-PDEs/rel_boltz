"""
Verify the stripped-sequence formula from tensor_k_scaling.tex.

The formula predicts, for fixed test function i and varying k_s (with k_t=0):

    S_k / S_0 = (-1)^k [C(k+a-2,k) + C(k+a-4,k-2)],   a = 2*l_s + 2

where S_k = Q_{i,(k,l_s,m_s),(0,l_t,m_t)} / (mu_{k,l_s} * mu_{0,l_t}).
An analogous formula holds when k_t varies with k_s=0 fixed (roles of l_s,l_t swap).

Part 1 (comprehensive): scan every stored nonzero entry of the n=3 sparse operator
        with k_t=0 and l_s>=1 and report pass/fail counts.

Part 2 (k_s=3 extended): compute the k=3 entry via operator_numba for a broad set
        of cases — multiple (l_s,l_t), nonzero m, different test-function k_i,
        and the symmetric formula (vary k_t instead of k_s).

Run from docs/:
    ~/miniconda3/envs/ttenv/bin/python verify_k_scaling.py
"""

import pickle
import sys
from math import comb, isqrt

import numpy as np

sys.path.insert(0, '../src')

from sparse import ind
from basis_numba import mu_const, spher_const
from quadrature import load_quad
from integrand_numba import operator_numba


def formula_norm(k, l):
    """S_k/S_0 = (-1)^k [C(k+a-2,k) + C(k+a-4,k-2)],  a = 2l+2."""
    def bc(n, r):
        return comb(n, r) if (0 <= r <= n and n >= 0) else 0
    a = 2 * l + 2
    return (-1)**k * (bc(k + a - 2, k) + bc(k + a - 4, k - 2))


def ind_to_klm(idx, n):
    k  = idx // (n * n)
    lm = idx % (n * n)
    l  = isqrt(lm)
    m  = lm - l * l - l
    return k, l, m


# ── load sparse operator ──────────────────────────────────────────────────────

with open('../src/sparse_operators/sparse_n3_lag7_leb9.pkl', 'rb') as f:
    raw = pickle.load(f)
sparse_list = raw if isinstance(raw, list) else raw['results']
n = round(len(sparse_list) ** (1/3))
print(f"loaded sparse_n3_lag7_leb9.pkl  ({len(sparse_list)} matrices, n={n})")


# ── Part 1: scan ALL stored entries ──────────────────────────────────────────

print()
print("=" * 68)
print("Part 1  Comprehensive formula check  (n=3 stored operator)")
print("=" * 68)
print("  Checking every stored entry with k_t=0, l_s>=1, k_s in {1,2}.")
print()

total = passed = 0
failures = []

for t_idx in range(len(sparse_list)):
    k_i, l_i, m_i = ind_to_klm(t_idx, n)
    mat = sparse_list[t_idx]
    rows, cols = mat.nonzero()
    for f, g in zip(rows.tolist(), cols.tolist()):
        k_s, l_s, m_s = ind_to_klm(f, n)
        k_t, l_t, m_t = ind_to_klm(g, n)
        if k_t != 0 or l_s == 0 or k_s == 0:
            continue
        # Reference: k_s=0 entry with same l_s,m_s and same l_t,m_t
        f0  = ind(0, l_s, m_s, n)
        ref = mat[f0, g]
        if abs(ref) < 1e-20:
            continue
        # S_k/S_0 = (entry/ref) * (mu(0,l_s)/mu(k_s,l_s))
        ratio = float(mat[f, g] / ref) * (mu_const(0, l_s) / mu_const(k_s, l_s))
        pred  = formula_norm(k_s, l_s)
        total += 1
        if abs(ratio - pred) < 1e-5:
            passed += 1
        else:
            failures.append((k_i,l_i,m_i, k_s,l_s,m_s, k_t,l_t,m_t, ratio, pred))

print(f"  {total} entries checked: {passed} passed, {total - passed} failed")
if failures:
    print("  FAILURES:")
    for c in failures:
        print(f"    i=({c[0]},{c[1]},{c[2]}) s=({c[3]},{c[4]},{c[5]}) t=({c[6]},{c[7]},{c[8]})"
              f"  ratio={c[9]:.6f}  pred={c[10]}")
else:
    print("  All entries match formula exactly (float64 roundoff only).")


# ── Part 2: k=3 via operator_numba — extended cases ──────────────────────────

print()
print("=" * 68)
print("Part 2  k=3 via operator_numba — extended cases")
print("=" * 68)

print("loading (7,9) quadrature ...", end='', flush=True)
raw_q, _, _ = load_quad('../src/quadratures/collision_lag7_leb9.pkl')
quad = np.array(raw_q, dtype=np.float64)
print(f"  {quad.shape[0]:,} points")

print("JIT warmup ...", end='', flush=True)
_ = operator_numba(0, 1, 0, spher_const(1, 0),
                   0, 1, 0, mu_const(0,1), spher_const(1,0),
                   0, 2, 0, mu_const(0,2), spher_const(2,0),
                   quad)
print(" done\n")

# Case list: (description, k_i, l_i, m_i, l_s, m_s, l_t, m_t, vary)
# vary='s': k_s 0->3, k_t=0 fixed; formula uses l_s
# vary='t': k_t 0->3, k_s=0 fixed; formula uses l_t
# Test functions are chosen to satisfy the andrea and cai sparsity rules.
CASES = [
    # ── original two cases ────────────────────────────────────────────────
    ("ls=1,lt=2  m=0/0  ki=0",   0,1,0,  1,0, 2,0,  's'),
    ("ls=2,lt=1  m=0/0  ki=0",   0,1,0,  2,0, 1,0,  's'),
    # ── k_i independence: same (ls,lt,m) different radial test-func ───────
    ("ls=1,lt=2  m=0/0  ki=1",   1,1,0,  1,0, 2,0,  's'),
    ("ls=1,lt=2  m=0/0  ki=2",   2,1,0,  1,0, 2,0,  's'),
    # ── m independence ────────────────────────────────────────────────────
    # (ls=1,lt=2,ms=1,mt=0): andrea l_i=1, cai |m_i|=1 -> m_i=1
    ("ls=1,lt=2  ms=1,mt=0",     0,1,1,  1,1, 2,0,  's'),
    # (ls=1,lt=2,ms=1,mt=1): andrea l_i=1, cai |m_i|=|1-1|=0 -> m_i=0
    ("ls=1,lt=2  ms=1,mt=1",     0,1,0,  1,1, 2,1,  's'),
    # (ls=2,lt=1,ms=1,mt=1): andrea l_i=1, cai |m_i|=|1-1|=0 -> m_i=0
    ("ls=2,lt=1  ms=1,mt=1",     0,1,0,  2,1, 1,1,  's'),
    # ── other l_t values (l_s=1) ─────────────────────────────────────────
    # (ls=1,lt=0): andrea 1+0-l_i in [0,0] -> l_i=1
    ("ls=1,lt=0  m=0/0",         0,1,0,  1,0, 0,0,  's'),
    # (ls=1,lt=1): andrea 2-l_i in [0,2] even -> l_i in {0,2}, use l_i=0
    ("ls=1,lt=1  m=0/0",         0,0,0,  1,0, 1,0,  's'),
    # ── other l_t values (l_s=2) ─────────────────────────────────────────
    # (ls=2,lt=0): andrea 2+0-l_i in [0,0] -> l_i=2
    ("ls=2,lt=0  m=0/0",         0,2,0,  2,0, 0,0,  's'),
    # (ls=2,lt=2): andrea 4-l_i in [0,4] even -> l_i in {0,2}, use l_i=0
    ("ls=2,lt=2  m=0/0",         0,0,0,  2,0, 2,0,  's'),
    # ── symmetric formula: vary k_t instead of k_s ───────────────────────
    # Q_{i,(0,l_s,0),(3,l_t,0)} / Q_{i,(0,l_s,0),(0,l_t,0)}: should give formula(3,l_t)
    ("SYMM ls=2,lt=1  m=0/0",    0,1,0,  2,0, 1,0,  't'),  # pred formula(3,lt=1)=-13
    ("SYMM ls=1,lt=2  m=0/0",    0,1,0,  1,0, 2,0,  't'),  # pred formula(3,lt=2)=-40
]

print(f"  {'case':<28}  {'S_3/S_0':>10}  {'predicted':>10}  match?")
print(f"  {'-'*28}  {'-'*10}  {'-'*10}  ------")

for desc, k_i, l_i, m_i, l_s, m_s, l_t, m_t, vary in CASES:
    c_i = spher_const(l_i, m_i)
    # reference: k_s=k_t=0
    val0 = operator_numba(k_i, l_i, m_i, c_i,
                          0, l_s, m_s, mu_const(0,l_s), spher_const(l_s,m_s),
                          0, l_t, m_t, mu_const(0,l_t), spher_const(l_t,m_t),
                          quad)
    S0 = val0 / (mu_const(0,l_s) * mu_const(0,l_t))
    if abs(S0) < 1e-15:
        print(f"  {desc:<28}  {'S0~0, skip':>10}")
        continue

    # k=3 entry
    if vary == 's':
        val3 = operator_numba(k_i, l_i, m_i, c_i,
                              3, l_s, m_s, mu_const(3,l_s), spher_const(l_s,m_s),
                              0, l_t, m_t, mu_const(0,l_t),  spher_const(l_t,m_t),
                              quad)
        S3 = val3 / (mu_const(3,l_s) * mu_const(0,l_t))
        pred = formula_norm(3, l_s)
    else:  # vary == 't'
        val3 = operator_numba(k_i, l_i, m_i, c_i,
                              0, l_s, m_s, mu_const(0,l_s),  spher_const(l_s,m_s),
                              3, l_t, m_t, mu_const(3,l_t), spher_const(l_t,m_t),
                              quad)
        S3 = val3 / (mu_const(0,l_s) * mu_const(3,l_t))
        pred = formula_norm(3, l_t)

    ratio   = S3 / S0
    rel_err = abs(ratio - pred) / abs(pred) if pred != 0 else abs(ratio)
    match   = "YES" if rel_err < 0.01 else f"NO  ({rel_err:.2%})"
    print(f"  {desc:<28}  {ratio:>10.5f}  {pred:>10.0f}  {match}")

print("\nDone.")
