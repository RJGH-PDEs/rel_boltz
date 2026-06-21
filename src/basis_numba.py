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
#
# BUG (found and fixed): s = sin(theta) must be passed in directly rather
# than derived from x = cos(theta) via sqrt(1 - x^2). Near a pole
# (theta -> 0 or pi), cos(theta) rounds to exactly +-1.0 in float64 once
# theta < ~1.5e-8, which collapses 1 - x^2 to exactly 0 by catastrophic
# cancellation, even though sin(theta) itself is still accurately
# representable. The old code computed s internally from x and silently
# returned 0.0 for any (l,m) with m!=0 evaluated that close to a pole.
# Verified against mpmath (50-digit precision) to be correct after the fix,
# and verified that no quadrature node in our actual collision/mass
# quadratures (n_lebedev=7,9) fell in the affected band, so prior tensors
# computed with the old code are unaffected.
@njit
def assoc_legendre(l, m, x, s):
    am = abs(m)
    # compute P_{am}^{am}
    pmm = 1.0
    if am > 0:
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
    leg   = assoc_legendre(l, m, cos(t), sin(t))
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

