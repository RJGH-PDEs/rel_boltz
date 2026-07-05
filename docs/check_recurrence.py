"""
Check whether A_k = Q_{(k_i,2,0),(0,2,0),(0,2,0)} / Q_0 satisfies a
closed three-term linear recurrence with polynomial-in-k coefficients.

Tests two forms:
  (I)  (k+1) A_{k+1} = (2k+β) A_k  - (k+γ) A_{k-1}   [Laguerre-type, 2 params]
       fit β,γ from k=1,2; predict k=3,4,5

  (II) (k+1) A_{k+1} = (ak+b) A_k  - (ck+d) A_{k-1}   [general linear, 4 params]
       fit a,b,c,d exactly from k=1,2,3,4; predict k=5

Run from docs/.
"""
import pickle
import sys
import numpy as np

sys.path.insert(0, '../src')
from sparse import ind
from basis_numba import mu_const, spher_const
from quadrature import load_quad
from integrand_numba import operator_numba

L_I, M_I = 2, 0
KS, LS, MS = 0, 2, 0
KT, LT, MT = 0, 2, 0

# ── Load stored values k_i = 0, 1, 2 ─────────────────────────────────────────
with open('../src/sparse_operators/sparse_n3_lag7_leb9.pkl', 'rb') as fh:
    raw = pickle.load(fh)
sparse_list = raw if isinstance(raw, list) else raw['results']
n = round(len(sparse_list) ** (1/3))

f_idx = ind(KS, LS, MS, n)
g_idx = ind(KT, LT, MT, n)

vals_stored = [float(sparse_list[ind(ki, L_I, M_I, n)][f_idx, g_idx])
               for ki in range(3)]
A0_raw = vals_stored[0]
A = [v / A0_raw for v in vals_stored]   # A[0] = 1 (signed normalization)

print("Stored (k_i = 0, 1, 2):")
for ki, a in enumerate(A):
    print(f"  A_{ki} = {a:+.10f}")

# ── Compute k_i = 3, 4, 5 via operator_numba ─────────────────────────────────
print("\nLoading collision_lag7_leb9 quadrature ...", end='', flush=True)
raw_q, _, _ = load_quad('../src/quadratures/collision_lag7_leb9.pkl')
quad = np.array(raw_q, dtype=np.float64)
print(f" {quad.shape[0]:,} points")

c_i  = spher_const(L_I, M_I)
mu_s = mu_const(KS, LS); c_s = spher_const(LS, MS)
mu_t = mu_const(KT, LT); c_t = spher_const(LT, MT)

# JIT warmup
_ = operator_numba(0, 1, 0, spher_const(1, 0),
                   0, 1, 0, mu_const(0, 1), spher_const(1, 0),
                   0, 2, 0, mu_const(0, 2), spher_const(2, 0), quad)
print("JIT warmed up")

for ki in [3, 4, 5]:
    v = operator_numba(ki, L_I, M_I, c_i,
                       KS, LS, MS, mu_s, c_s,
                       KT, LT, MT, mu_t, c_t, quad)
    A.append(v / A0_raw)
    print(f"  A_{ki} = {A[-1]:+.10f}")

A = np.array(A)   # shape (6,)

print("\nFull signed sequence A_k:")
for k, a in enumerate(A):
    print(f"  A_{k} = {a:+.8f}   |A_k|^(1/k) = {abs(a)**(1/max(k,1)):.4f}")

# ── (I) Laguerre-type 2-parameter recurrence ─────────────────────────────────
# (k+1) A_{k+1} = (2k+β) A_k - (k+γ) A_{k-1}
# Rearrange: β A_k - γ A_{k-1} = (k+1) A_{k+1} - 2k A_k
# k=1: β A_1 - γ A_0 = 2 A_2 - 2 A_1
# k=2: β A_2 - γ A_1 = 3 A_3 - 4 A_2
M2 = np.array([[A[1], -A[0]],
               [A[2], -A[1]]])
rhs2 = np.array([2*A[2] - 2*A[1],
                 3*A[3] - 4*A[2]])
beta, gamma = np.linalg.solve(M2, rhs2)

print(f"\n── (I) Laguerre-type fit (from k=1,2) ─────────────────────────────")
print(f"   β = {beta:.8f}   (Laguerre: 2ℓ_i+3 - r_eff = {2*L_I+3} - r_eff)")
print(f"   γ = {gamma:.8f}   (Laguerre: 2ℓ_i+2 = {2*L_I+2})")
print(f"\n   k  | A_k (data)        | A_k (predicted)   | rel. error")
print(f"   ---+-------------------+-------------------+-----------")
for k in range(3, 6):
    pred = ((2*(k-1) + beta)*A[k-1] - ((k-1) + gamma)*A[k-2]) / k
    err  = abs(pred - A[k]) / abs(A[k])
    print(f"   {k}  | {A[k]:+16.8f}  | {pred:+16.8f}  | {err:.2e}")

# ── (II) General 4-parameter recurrence ──────────────────────────────────────
# (k+1) A_{k+1} = (ak+b) A_k - (ck+d) A_{k-1}
# → row for each k: [k A_k,  A_k,  -k A_{k-1},  -A_{k-1}] · [a,b,c,d]^T = (k+1) A_{k+1}
rows, rhs4 = [], []
for k in range(1, 5):   # k=1..4, needs A[k+1]=A[5] at most
    rows.append([k*A[k], A[k], -k*A[k-1], -A[k-1]])
    rhs4.append((k+1)*A[k+1])
M4   = np.array(rows)
rhs4 = np.array(rhs4)

# Fit exactly on k=1,2,3,4 (4 equations, 4 unknowns)
a4, b4, c4, d4 = np.linalg.solve(M4[:4], rhs4[:4])

print(f"\n── (II) General 4-param fit (from k=1,2,3,4) ───────────────────────")
print(f"   a={a4:.8f}  b={b4:.8f}")
print(f"   c={c4:.8f}  d={d4:.8f}")

# Predict k=5 and check
pred5 = (4*a4 + b4)*A[4] - (4*c4 + d4)*A[3]   # (k=4): 5 A_5 = (4a+b)A_4-(4c+d)A_3
# wait: (k+1)A_{k+1}=... so with k=4: 5 A_5 = (4a+b)A_4 - (4c+d)A_3
pred5_val = pred5 / 5
err5 = abs(pred5_val - A[5]) / abs(A[5])
print(f"\n   Prediction at k=5:")
print(f"   A_5 data      = {A[5]:+.8f}")
print(f"   A_5 predicted = {pred5_val:+.8f}")
print(f"   rel. error    = {err5:.2e}")

# Also check consistency: least-squares over k=1..4 (same as exact solve)
# and separately confirm the residual is near machine precision
pred4_all = M4 @ np.array([a4, b4, c4, d4])
errs4 = np.abs(pred4_all - rhs4) / np.abs(rhs4)
print(f"   Max rel. residual (fit equations k=1..4): {errs4.max():.2e}")
