import sys
sys.path.insert(0, '../src')

from basis import f_tilde
from sparse import ind


def linear_comb(coefficients, r, t, p, n):
    result = 0
    for k in range(n):
        for l in range(n):
            for m in range(-l, l+1):
                ft  = f_tilde(k, l, m)
                idx = ind(k, l, m, n)
                result += coefficients[idx] * ft(r, t, p)
    return result
