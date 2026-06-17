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

# Worker for parallel computation: builds basis callables from raw indices,
# runs the full quadrature sum, returns [select, value]
def operator_parallel(select, quad):
    from basis import basis, f_tilde
    phi = basis(  *select[0])
    ft  = f_tilde(*select[1])
    gt  = f_tilde(*select[2])
    result = operator(phi, ft, gt, quad)
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
