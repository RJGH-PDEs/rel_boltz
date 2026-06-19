import numpy as np
from math import sqrt, cos, sin, factorial, gamma, log
from numba import njit

# --- Generalized Laguerre polynomial L_k^alpha(x) via three-term recurrence ---
# L_0^alpha = 1
# L_1^alpha = 1 + alpha - x
# L_{n+1}^alpha = ((2n + alpha + 1 - x) * L_n - (n + alpha) * L_{n-1}) / (n + 1)
@njit
def gen_laguerre(k, alpha, x):
    if k == 0:
        return 1.0
    p0 = 1.0
    p1 = 1.0 + alpha - x
    if k == 1:
        return p1
    for n in range(1, k):
        p2 = ((2*n + alpha + 1 - x) * p1 - (n + alpha) * p0) / (n + 1)
        p0 = p1
        p1 = p2
    return p1


# --- Associated Legendre polynomial P_l^|m|(x) via recurrence ---
# Starts from P_m^m, steps up to P_l^m using:
# P_{m}^{m}(x)   = (-1)^m (2m-1)!! (1-x^2)^{m/2}
# P_{m+1}^{m}(x) = x (2m+1) P_m^m(x)
# P_{l+1}^{m}(x) = ((2l+1) x P_l^m - (l+m) P_{l-1}^m) / (l - m + 1)
@njit
def assoc_legendre(l, m, x):
    am = abs(m)
    # compute P_{am}^{am}
    pmm = 1.0
    if am > 0:
        factor = 1.0
        s = sqrt(max(0.0, 1.0 - x*x))
        for i in range(1, am + 1):
            pmm *= -(2*i - 1) * s
    if l == am:
        return pmm
    # step up to P_{am+1}^{am}
    pm1 = x * (2*am + 1) * pmm
    if l == am + 1:
        return pm1
    # continue stepping up
    p0 = pmm
    p1 = pm1
    for ll in range(am + 1, l):
        p2 = ((2*ll + 1) * x * p1 - (ll + am) * p0) / (ll - am + 1)
        p0 = p1
        p1 = p2
    return p1


# --- Spherical harmonic normalization constant ---
# spher_const(l, m) — precomputed as a plain Python float (called outside njit)
def spher_const(l, m):
    result = (2*l + 1) / (2 * np.pi)
    if m == 0:
        return sqrt(result / 2)
    result = result * factorial(l - abs(m)) / factorial(l + abs(m))
    return sqrt(result)


# --- mu_{k,l} = sqrt(k! / Gamma(k + 2l + 3)) ---
def mu_const(k, l):
    return sqrt(factorial(k) / gamma(k + 2*l + 3))


# --- Core evaluation functions (njit-compatible) ---

@njit
def basis_eval(k, l, m, c, r, t, p):
    """Evaluate phi_{k,l,m}(r,t,p) = c * L_k^{2l+2}(r) * r^l * P_l^|m|(cos t) * azimuth(m,p)"""
    alpha = 2*l + 2
    lag   = gen_laguerre(k, alpha, r)
    leg   = assoc_legendre(l, m, cos(t))
    ang   = cos(m * p) if m >= 0 else sin(-m * p)
    return c * leg * ang * lag * r**l


@njit
def f_tilde_eval(k, l, m, mu, c, r, t, p):
    """Evaluate f_tilde_{k,l,m} = mu * phi_{k,l,m}"""
    return mu * basis_eval(k, l, m, c, r, t, p)


# --- Python-level wrappers that precompute constants and return callables ---
# These match the interface of basis.py so they can be used as drop-in replacements.

def basis(k, l, m):
    c = spher_const(l, m)
    def f(r, t, p):
        return basis_eval(k, l, m, c, r, t, p)
    return f

def f_tilde(k, l, m):
    mu = mu_const(k, l)
    c  = spher_const(l, m)
    def f(r, t, p):
        return f_tilde_eval(k, l, m, mu, c, r, t, p)
    return f


# --- Verification against basis.py ---
def test():
    import basis as basis_orig
    import numpy as np

    test_cases = [
        (0, 0, 0), (1, 0, 0), (2, 0, 0),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, 2, -1), (2, 2, 2),
    ]

    # test at multiple random points to rule out coincidental agreement
    n_points = 10
    points = [
        (np.random.uniform(0.1, 5.0),
         np.random.uniform(0.01, np.pi - 0.01),
         np.random.uniform(0.0, 2*np.pi))
        for _ in range(n_points)
    ]

    all_ok = True

    print(f"Testing basis at {n_points} random points...")
    for klm in test_cases:
        k, l, m = klm
        phi_orig = basis_orig.basis(k, l, m)
        phi_new  = basis(k, l, m)
        for r, t, p in points:
            ref = phi_orig(r, t, p)
            new = phi_new(r, t, p)
            ok  = np.isclose(ref, new, rtol=1e-10)
            if not ok:
                all_ok = False
                print(f"  FAIL basis {klm} at ({r:.3f},{t:.3f},{p:.3f}): ref={ref:.8f} new={new:.8f}")
    print(f"  basis: {'All OK' if all_ok else 'SOME FAILURES'}")

    print(f"Testing f_tilde at {n_points} random points...")
    ft_ok = True
    for klm in test_cases:
        k, l, m = klm
        ft_orig = basis_orig.f_tilde(k, l, m)
        ft_new  = f_tilde(k, l, m)
        for r, t, p in points:
            ref = ft_orig(r, t, p)
            new = ft_new(r, t, p)
            ok  = np.isclose(ref, new, rtol=1e-10)
            if not ok:
                ft_ok = False
                all_ok = False
                print(f"  FAIL f_tilde {klm} at ({r:.3f},{t:.3f},{p:.3f}): ref={ref:.8f} new={new:.8f}")
    print(f"  f_tilde: {'All OK' if ft_ok else 'SOME FAILURES'}")

    print()
    print("All OK" if all_ok else "SOME FAILURES")

if __name__ == "__main__":
    test()
