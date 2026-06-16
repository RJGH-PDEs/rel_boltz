import sympy as sp
import numpy as np
from scipy.special import factorial, gamma

# The constant for the spherical harmonic
def spher_const(l, m):
    result = (2*l+1)/(2*np.pi)
    if m == 0:
        return np.sqrt(result/2)

    result = result*factorial(l-np.abs(m))
    result = result/factorial(l+np.abs(m))
    return np.sqrt(result)

# mu_{k,l} = sqrt( k! / Gamma(k + 2l + 3) )
def mu_const(k, l):
    result = factorial(k) / gamma(k + 2*l + 3)
    return np.sqrt(result)

# Test function (no weight, no mu): phi_{k,l,m}(p) = L_k^{2l+2}(r) * r^l * Y_l^m
def basis(k, l, m):
    r = sp.symbols('r')
    t = sp.symbols('t')
    p = sp.symbols('p')

    # Spherical harmonic
    sphr = sp.simplify(sp.assoc_legendre(l, abs(m), sp.cos(t)))
    sphr = sp.refine(sphr, sp.Q.positive(sp.sin(t)))

    if m >= 0:
        sphr = sphr * sp.cos(m*p)
    else:
        sphr = sphr * sp.sin(abs(m)*p)

    sphr = sphr * spher_const(l, m)

    # Radial part: L_k^{2l+2}(r) * r^l
    a      = 2*l + 2
    radial = sp.assoc_laguerre(k, a, r) * r**l

    return sphr * radial

# Trial function (with weight and mu): psi_{k,l,m}(p) = mu_{k,l} * e^{-r/2} * phi_{k,l,m}
def trial(k, l, m):
    r  = sp.symbols('r')
    mu = mu_const(k, l)
    return mu * sp.exp(-r/2) * basis(k, l, m)

# A test
def test():
    k, l, m = 0, 1, 1

    print("basis (test function):")
    print(basis(k, l, m))
    print()
    print("trial function (with weight and mu):")
    print(trial(k, l, m))
    print()

def main():
    test()

if __name__ == "__main__":
    main()