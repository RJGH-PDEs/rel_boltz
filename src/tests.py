"""
Verification tests for the collision tensor pipeline.

Sections:
  1. Quadrature tests       — radial, Lebedev, collision, mass
  2. Basis tests            — numba basis vs analytical spot checks
  3. Conservation tests     — Lebedev sweep for conservation entries
  4. Quadrature convergence — Laguerre/Lebedev order sweep for a hard entry
  5. Compare tensors        — diff two saved tensor pkl files
"""

import numpy as np


# ── 1. Quadrature tests ──────────────────────────────────────────────────────

def test_radial(n):
    from scipy.special import roots_genlaguerre
    x, w = roots_genlaguerre(n, 2)
    pts, wts = 2 * x, 8 * w
    result = np.sum(wts * pts)
    print(f"Radial test (n={n}): {result:.10f}  (expected 96)")

def test_lebedev(n):
    from pylebedev import PyLebedev
    _, w = PyLebedev().get_points_and_weights(n)
    result = np.sum(4 * np.pi * w)
    print(f"Lebedev test (n={n}): {result:.10f}  (expected {4*np.pi:.10f})")

def test_collision_quad(path='./quadratures/collision.pkl'):
    from quadrature import load_quad
    quad     = load_quad(path)
    result   = sum(pt[-1] for pt in quad)
    expected = 256 * (4*np.pi)**3
    print(f"Collision quad: {result:.6f}  (expected {expected:.6f})  {'OK' if np.isclose(result, expected) else 'FAIL'}")

def test_mass_quad(path='./quadratures/mass.pkl'):
    from quadrature import load_quad
    quad     = load_quad(path)
    result   = sum(pt[-1] for pt in quad)
    expected = 16 * 4 * np.pi
    print(f"Mass quad:      {result:.6f}  (expected {expected:.6f})  {'OK' if np.isclose(result, expected) else 'FAIL'}")

def run_quadrature_tests():
    print("=== Quadrature tests ===")
    for n in [5, 9, 15]:
        test_radial(n)
    for n in [7, 11, 15]:
        test_lebedev(n)
    test_collision_quad()
    test_mass_quad()


# ── 2. Basis tests ───────────────────────────────────────────────────────────

def run_basis_tests(n_random=10):
    """Compare numba basis against scipy at random points."""
    from basis_numba import gen_laguerre, assoc_legendre, spher_const, mu_const, basis_eval_full
    from scipy.special import eval_genlaguerre, lpmv
    import math

    print("=== Basis tests (numba vs scipy) ===")
    rng = np.random.default_rng(42)
    cases = [(k, l, m) for k in range(3) for l in range(3) for m in range(-l, l+1)]
    for k, l, m in cases[:8]:
        for _ in range(n_random):
            r = rng.uniform(0.1, 5.0)
            x = rng.uniform(-1.0, 1.0)
            lag_ref = eval_genlaguerre(k, 2*l+2, r)
            lag_nb  = gen_laguerre(k, 2*l+2, r)
            leg_ref = lpmv(abs(m), l, x)
            leg_nb  = assoc_legendre(l, m, x)
            assert np.isclose(lag_nb, lag_ref, rtol=1e-10), f"Laguerre mismatch k={k} l={l} r={r}"
            assert np.isclose(leg_nb, leg_ref, rtol=1e-10), f"Legendre mismatch l={l} m={m} x={x}"
    print("All basis checks passed.")


# ── 3. Conservation tests ────────────────────────────────────────────────────

def run_conservation_tests(n_laguerre=7, lebedev_orders=None):
    """
    Sweep n_lebedev and print conservation-violating entries.
    These should go to zero as n_lebedev increases.
    """
    from quadrature import collision_quadrature
    from integrand_numba import operator_numba
    from basis_numba import spher_const, mu_const

    if lebedev_orders is None:
        lebedev_orders = [7, 9, 11]

    cases = [
        ("mass   [0,0,0] x [2,2,-2] x [2,2,-2]", [[0,0,0],[2,2,-2],[2,2,-2]]),
        ("mass   [0,0,0] x [2,2, 0] x [2,2, 0]", [[0,0,0],[2,2, 0],[2,2, 0]]),
        ("energy [1,0,0] x [2,2,-2] x [2,2,-2]", [[1,0,0],[2,2,-2],[2,2,-2]]),
    ]

    def eval_entry(select, quad_np):
        k,  l,  m  = select[0]; c   = spher_const(l,  m)
        k1, l1, m1 = select[1]; mu1 = mu_const(k1,l1); c1 = spher_const(l1,m1)
        k2, l2, m2 = select[2]; mu2 = mu_const(k2,l2); c2 = spher_const(l2,m2)
        return operator_numba(k,l,m,c, k1,l1,m1,mu1,c1, k2,l2,m2,mu2,c2, quad_np)

    # warmup JIT
    _wq = np.array(collision_quadrature(3, 3))
    eval_entry([[0,0,0],[0,0,0],[0,0,0]], _wq)

    print("=== Conservation tests ===")
    for label, sel in cases:
        print(f"\n  {label}")
        for n_leb in lebedev_orders:
            quad = np.array(collision_quadrature(n_laguerre, n_leb))
            val  = eval_entry(sel, quad)
            print(f"    n_lebedev={n_leb:2d}  n_pts={len(quad):>10}  val={val:.4e}")


# ── 4. Quadrature convergence ────────────────────────────────────────────────

def run_convergence_test(select=None, configs=None):
    """Check that a hard entry converges as quadrature is refined."""
    from quadrature import collision_quadrature
    from integrand_numba import operator_numba
    from basis_numba import spher_const, mu_const

    if select is None:
        select = [[2,2,0],[2,2,0],[2,0,0]]
    if configs is None:
        configs = [(7,7), (7,9), (8,9)]

    def eval_entry(sel, quad_np):
        k,  l,  m  = sel[0]; c   = spher_const(l,  m)
        k1, l1, m1 = sel[1]; mu1 = mu_const(k1,l1); c1 = spher_const(l1,m1)
        k2, l2, m2 = sel[2]; mu2 = mu_const(k2,l2); c2 = spher_const(l2,m2)
        return operator_numba(k,l,m,c, k1,l1,m1,mu1,c1, k2,l2,m2,mu2,c2, quad_np)

    # warmup
    _wq = np.array(collision_quadrature(3, 3))
    eval_entry([[0,0,0],[0,0,0],[0,0,0]], _wq)

    print(f"=== Convergence test  select={select} ===")
    for n_lag, n_leb in configs:
        quad = np.array(collision_quadrature(n_lag, n_leb))
        val  = eval_entry(select, quad)
        print(f"  n_laguerre={n_lag}  n_lebedev={n_leb}  n_pts={len(quad):>10}  val={val:.10f}")


# ── 5. Compare tensors ───────────────────────────────────────────────────────

def compare_tensors(path_a, path_b, tol=1e-6):
    """Diff two saved collision tensor pkl files entry by entry."""
    import pickle

    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def to_dict(data):
        return {(tuple(e[0][0]), tuple(e[0][1]), tuple(e[0][2])): e[1] for e in data}

    a, b = to_dict(load(path_a)), to_dict(load(path_b))
    common   = set(a) & set(b)
    only_a   = set(a) - set(b)
    only_b   = set(b) - set(a)

    print(f"=== Compare tensors ===")
    print(f"  {path_a}: {len(a)} entries")
    print(f"  {path_b}: {len(b)} entries")
    print(f"  common={len(common)}  only_a={len(only_a)}  only_b={len(only_b)}")

    mismatches = [(k, a[k], b[k]) for k in common if not np.isclose(a[k], b[k], rtol=tol)]
    if mismatches:
        print(f"  MISMATCHES ({len(mismatches)}):")
        for k, va, vb in mismatches:
            print(f"    {list(k)}  a={va:.6f}  b={vb:.6f}  diff={abs(va-vb):.2e}")
    else:
        print(f"  All {len(common)} common entries match to rtol={tol}  OK")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_quadrature_tests()
    print()
    run_conservation_tests()
    print()
    run_convergence_test()
