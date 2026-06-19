import numpy as np
from quadrature import load_quad
from integrand import pieces, integrand

# The Boltzmann collision operator: loops over quadrature and accumulates
def operator(phi, ft, gt, quad):
    result = 0
    for pt in quad:
        w     = pt[-1]
        point = pt[:-1]
        result += w * integrand(phi, ft, gt, point)
    return result

# Worker for parallel computation: uses a global quadrature loaded once per
# worker process via init_worker — avoids repeated disk reads per task.
_quad    = None
_quad_np = None

def init_worker(quad_path):
    global _quad, _quad_np
    _quad    = load_quad(quad_path)
    _quad_np = np.array(_quad)

def operator_parallel(select):
    from basis import basis, f_tilde
    phi    = basis(  *select[0])
    ft     = f_tilde(*select[1])
    gt     = f_tilde(*select[2])
    result = operator(phi, ft, gt, _quad)
    return [select, result]

def operator_parallel_numba(select):
    from integrand_numba import operator_numba
    from basis_numba import spher_const, mu_const
    k,  l,  m  = select[0]; c   = spher_const(l,  m)
    k1, l1, m1 = select[1]; mu1 = mu_const(k1, l1); c1 = spher_const(l1, m1)
    k2, l2, m2 = select[2]; mu2 = mu_const(k2, l2); c2 = spher_const(l2, m2)
    result = operator_numba(k,l,m,c, k1,l1,m1,mu1,c1, k2,l2,m2,mu2,c2, _quad_np)
    return [select, result]

# End-to-end test
# select = [[k, l, m], [k1, l1, m1], [k2, l2, m2]]
#   [k,  l,  m ]: test function phi
#   [k1, l1, m1]: trial function f_tilde
#   [k2, l2, m2]: trial function g_tilde
def operator_test(select):
    quad = load_quad('./quadratures/collision.pkl')
    print("quadrature points:", len(quad))

    phi, ft, gt = pieces(select)
    result      = operator(phi, ft, gt, quad)

    print("select:", select)
    print("result:", result)

def test():
    select = [[1, 0, 0], [0, 0, 0], [1, 0, 0]]
    operator_test(select)

def main():
    test()

if __name__ == "__main__":
    main()
