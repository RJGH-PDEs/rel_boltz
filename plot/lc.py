import sys
sys.path.insert(0, '../src')

from basis_numba import mu_const, spher_const, basis_eval
from sparse import ind


def linear_comb(coefficients, r, t, p, n):
    result = 0
    for k in range(n):
        for l in range(n):
            for m in range(-l, l+1):
                mu = mu_const(k, l)
                c  = spher_const(l, m)
                idx = ind(k, l, m, n)
                result += coefficients[idx] * mu * basis_eval(k, l, m, c, r, t, p)
    return result
