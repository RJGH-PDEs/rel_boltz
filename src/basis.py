import numpy as np
from scipy.special import factorial, gamma, eval_genlaguerre, lpmv

# Spherical harmonic normalization constant
def spher_const(l, m):
    result = (2*l+1) / (2*np.pi)
    if m == 0:
        return np.sqrt(result / 2)
    result = result * factorial(l - np.abs(m)) / factorial(l + np.abs(m))
    return np.sqrt(result)

# mu_{k,l} = sqrt( k! / Gamma(k + 2l + 3) )
def mu_const(k, l):
    return np.sqrt(factorial(k) / gamma(k + 2*l + 3))

# Test function (no weight, no mu): phi_{k,l,m}(r, t, p) = L_k^{2l+2}(r) * r^l * Y_l^m(t, p)
# Returns a callable f(r, t, p)
def basis(k, l, m):
    c = spher_const(l, m)
    a = 2*l + 2

    def f(r, t, p):
        lag = eval_genlaguerre(k, a, r)
        leg = lpmv(abs(m), l, np.cos(t))
        ang = np.cos(m * p) if m >= 0 else np.sin(abs(m) * p)
        return c * leg * ang * lag * r**l

    return f

# f_tilde: mu_{k,l} * phi_{k,l,m}  (no exponential — weight absorbed into quadrature)
# Returns a callable f(r, t, p)
def f_tilde(k, l, m):
    mu  = mu_const(k, l)
    phi = basis(k, l, m)

    def f(r, t, p):
        return mu * phi(r, t, p)

    return f

# Full trial function: e^{-r/2} * f_tilde  (= the actual solution ansatz)
# Returns a callable f(r, t, p)
def trial(k, l, m):
    ft = f_tilde(k, l, m)

    def f(r, t, p):
        return np.exp(-r / 2) * ft(r, t, p)

    return f

# A test
def test():
    k, l, m = 1, 1, 0

    phi = basis(k, l, m)
    psi = trial(k, l, m)

    r, t, p = 1.5, np.pi/4, np.pi/3
    print(f"basis  ({k},{l},{m}) at ({r},{t:.3f},{p:.3f}): {phi(r, t, p):.8f}")
    print(f"trial  ({k},{l},{m}) at ({r},{t:.3f},{p:.3f}): {psi(r, t, p):.8f}")

def main():
    test()

if __name__ == "__main__":
    main()
