import numpy as np
from quadrature import load_quad

# Worker state: quadrature loaded once per process via init_worker
_quad_np = None

def init_worker(quad_path):
    global _quad_np
    quad, _, _ = load_quad(quad_path)
    _quad_np = np.array(quad)

def operator_parallel_numba(select):
    from integrand_numba import operator_numba
    from basis_numba import spher_const, mu_const
    k,  l,  m  = select[0]; c   = spher_const(l,  m)
    k1, l1, m1 = select[1]; mu1 = mu_const(k1, l1); c1 = spher_const(l1, m1)
    k2, l2, m2 = select[2]; mu2 = mu_const(k2, l2); c2 = spher_const(l2, m2)
    result = operator_numba(k,l,m,c, k1,l1,m1,mu1,c1, k2,l2,m2,mu2,c2, _quad_np)
    return [select, result]
