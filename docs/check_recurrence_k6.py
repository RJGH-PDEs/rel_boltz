"""
Compute A_6 at two quadrature sizes and check against the 4-parameter
recurrence prediction fitted from k=1..4.

Entry: l_i=2, m_i=0, s=(0,2,0), t=(0,2,0).

Run from docs/.
"""
import pickle, sys
import numpy as np

sys.path.insert(0, '../src')
from sparse import ind
from basis_numba import mu_const, spher_const
from quadrature import load_quad
from integrand_numba import operator_numba

L_I, M_I = 2, 0
KS, LS, MS = 0, 2, 0
KT, LT, MT = 0, 2, 0

# ── Sequence from previous run ────────────────────────────────────────────────
# A_0..A_5 from stored operator + operator_numba(7,9), normalized to A_0=1.
A_known = np.array([1.0, -11.0, 56.8, -196.0, 520.4, -1135.6])

# 4-parameter recurrence coefficients, fit from k=1..4
rows, rhs4 = [], []
for k in range(1, 5):
    rows.append([k*A_known[k], A_known[k], -k*A_known[k-1], -A_known[k-1]])
    rhs4.append((k+1)*A_known[k+1])
a, b, c, d = np.linalg.solve(np.array(rows), np.array(rhs4))
print(f"Recurrence coefficients (fitted from k=1..4):")
print(f"  a={a:.8f}  b={b:.8f}")
print(f"  c={c:.8f}  d={d:.8f}")

# Predicted A_6: (k+1)A_{k+1} = (ak+b)A_k - (ck+d)A_{k-1}  at k=5
A6_pred = ((5*a + b)*A_known[5] - (5*c + d)*A_known[4]) / 6
print(f"\nRecurrence prediction:  A_6 = {A6_pred:.8f}")

# ── Load normalisation constant from stored operator ─────────────────────────
with open('../src/sparse_operators/sparse_n3_lag7_leb9.pkl', 'rb') as fh:
    raw = pickle.load(fh)
sparse_list = raw if isinstance(raw, list) else raw['results']
n = round(len(sparse_list) ** (1/3))
A0_raw = float(sparse_list[ind(0, L_I, M_I, n)][ind(KS, LS, MS, n),
                                                   ind(KT, LT, MT, n)])

c_i  = spher_const(L_I, M_I)
mu_s = mu_const(KS, LS); c_s = spher_const(LS, MS)
mu_t = mu_const(KT, LT); c_t = spher_const(LT, MT)

# ── JIT warmup (reuse cheap entry) ───────────────────────────────────────────
print("\nJIT warmup ...", end='', flush=True)
raw_q0, _, _ = load_quad('../src/quadratures/collision_lag7_leb9.pkl')
quad79 = np.array(raw_q0, dtype=np.float64)
_ = operator_numba(0,1,0,spher_const(1,0), 0,1,0,mu_const(0,1),spher_const(1,0),
                   0,2,0,mu_const(0,2),spher_const(2,0), quad79)
print(" done")

# ── A_6 with (7,9) ────────────────────────────────────────────────────────────
v79 = operator_numba(6, L_I, M_I, c_i,
                     KS, LS, MS, mu_s, c_s,
                     KT, LT, MT, mu_t, c_t, quad79)
A6_79 = v79 / A0_raw
print(f"\nA_6 with (lag=7, leb=9):   {A6_79:.8f}")

# ── A_6 with (11,13) ─────────────────────────────────────────────────────────
print("Loading (lag=11, leb=13) quadrature ...", end='', flush=True)
raw_q1, _, _ = load_quad('../src/quadratures/collision_lag11_leb13.pkl')
quad1113 = np.array(raw_q1, dtype=np.float64)
print(f" {quad1113.shape[0]:,} points")

v1113 = operator_numba(6, L_I, M_I, c_i,
                       KS, LS, MS, mu_s, c_s,
                       KT, LT, MT, mu_t, c_t, quad1113)
A6_1113 = v1113 / A0_raw
print(f"A_6 with (lag=11, leb=13): {A6_1113:.8f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n── Summary ─────────────────────────────────────────────────────────")
print(f"  Recurrence prediction:      {A6_pred:+.6f}")
print(f"  Computed (7,9):             {A6_79:+.6f}")
print(f"  Computed (11,13):           {A6_1113:+.6f}")
print(f"  (7,9) vs (11,13) rel diff:  {abs(A6_79 - A6_1113)/abs(A6_1113):.2e}")
print(f"  (7,9) vs prediction rel err:{abs(A6_79 - A6_pred)/abs(A6_pred):.2e}")
print(f"  (11,13) vs prediction rel err:{abs(A6_1113 - A6_pred)/abs(A6_pred):.2e}")
