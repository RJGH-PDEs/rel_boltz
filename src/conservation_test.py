"""
Test conservation law accuracy vs quadrature size.

Particle number conservation: <phi_{0,0,0}, Q(f_tilde, g_tilde)> = 0 for all f, g
Energy conservation:          <phi_{1,0,0}, Q(f_tilde, g_tilde)> = 0 for all f, g

We test the worst-case trial functions (k=2, l=2) which had the largest errors.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from quadrature import collision_quadrature
from integrand_numba import operator_numba
from basis_numba import spher_const, mu_const


def compute_entry(select, quad_np):
    k,  l,  m  = select[0]; c   = spher_const(l,  m)
    k1, l1, m1 = select[1]; mu1 = mu_const(k1, l1); c1 = spher_const(l1, m1)
    k2, l2, m2 = select[2]; mu2 = mu_const(k2, l2); c2 = spher_const(l2, m2)
    return operator_numba(k,l,m,c, k1,l1,m1,mu1,c1, k2,l2,m2,mu2,c2, quad_np)


# Trial functions that gave the largest conservation errors
test_cases = [
    # (label, select)
    ("mass [0,0,0] x [2,2,-2] x [2,2,-2]", [[0,0,0],[2,2,-2],[2,2,-2]]),
    ("mass [0,0,0] x [2,2, 0] x [2,2, 0]", [[0,0,0],[2,2, 0],[2,2, 0]]),
    ("energy [1,0,0] x [2,2,-2] x [2,2,-2]", [[1,0,0],[2,2,-2],[2,2,-2]]),
    ("energy [1,0,0] x [2,2, 0] x [2,2, 0]", [[1,0,0],[2,2, 0],[2,2, 0]]),
]

n_lebedev  = 7   # keep fixed, vary laguerre
laguerre_orders = [7, 8, 9, 10, 11]

# warmup JIT
print("warming up JIT...")
_wq = np.array(collision_quadrature(3, 3))
compute_entry([[0,0,0],[0,0,0],[0,0,0]], _wq)
print("done\n")

print(f"{'n_lag':>6}  {'n_pts':>10}  " + "  ".join(f"{label:>40}" for label, _ in test_cases))
print("-" * (6 + 10 + 3 + 42 * len(test_cases)))

for n_lag in laguerre_orders:
    quad    = collision_quadrature(n_lag, n_lebedev)
    quad_np = np.array(quad)
    n_pts   = len(quad)

    values = [compute_entry(sel, quad_np) for _, sel in test_cases]

    row = f"{n_lag:>6}  {n_pts:>10}  " + "  ".join(f"{v:>40.6e}" for v in values)
    print(row, flush=True)
