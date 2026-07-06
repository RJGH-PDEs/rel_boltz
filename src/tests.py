"""
Verification tests for the collision tensor pipeline.

Sections:
  1. Quadrature tests       — radial, Lebedev, collision, mass
  2. Basis tests            — numba basis vs scipy spot checks
  3. Conservation tests     — Lebedev sweep for conservation entries
  4. Quadrature convergence — Laguerre/Lebedev order sweep for a hard entry
  5. Compare tensors        — diff two saved tensor pkl files
  6. Quadrature build verify — weight sum, point-wise, operator sweep, conservation
"""

import numpy as np


# ── 1. Quadrature tests ──────────────────────────────────────────────────────

def test_radial(n):
    from scipy.special import roots_genlaguerre
    x, w = roots_genlaguerre(n, 2)
    result = np.sum(8 * w * 2 * x)
    print(f"Radial test (n={n}): {result:.10f}  (expected 96)")

def test_lebedev(n):
    from pylebedev import PyLebedev
    _, w = PyLebedev().get_points_and_weights(n)
    result = np.sum(4 * np.pi * w)
    print(f"Lebedev test (n={n}): {result:.10f}  (expected {4*np.pi:.10f})")

def test_collision_quad(n_laguerre=7, n_lebedev=7):
    from quadrature import load_quad, quad_name
    quad, _, _ = load_quad(quad_name('collision', n_laguerre, n_lebedev))
    result   = sum(pt[-1] for pt in quad)
    expected = 256 * (4*np.pi)**3
    print(f"Collision quad: {result:.6f}  (expected {expected:.6f})  {'OK' if np.isclose(result, expected) else 'FAIL'}")

def test_mass_quad(n_laguerre=7, n_lebedev=7):
    from quadrature import load_quad, quad_name
    quad, _, _ = load_quad(quad_name('mass', n_laguerre, n_lebedev))
    result   = sum(pt[-1] for pt in quad)
    expected = 16 * 4 * np.pi
    print(f"Mass quad:      {result:.6f}  (expected {expected:.6f})  {'OK' if np.isclose(result, expected) else 'FAIL'}")

def run_quadrature_tests(n_laguerre=7, n_lebedev=7):
    print("=== Quadrature tests ===")
    for n in [5, 9, 15]:
        test_radial(n)
    for n in [7, 11, 15]:
        test_lebedev(n)
    test_collision_quad(n_laguerre, n_lebedev)
    test_mass_quad(n_laguerre, n_lebedev)


# ── 2. Basis tests ───────────────────────────────────────────────────────────

def run_basis_tests(n_random=10):
    from basis_numba import gen_laguerre, assoc_legendre
    from scipy.special import eval_genlaguerre, lpmv

    print("=== Basis tests (numba vs scipy) ===")
    rng = np.random.default_rng(42)
    cases = [(k, l, m) for k in range(3) for l in range(3) for m in range(-l, l+1)]
    for k, l, m in cases[:8]:
        for _ in range(n_random):
            r = rng.uniform(0.1, 5.0)
            x = rng.uniform(-1.0, 1.0)
            assert np.isclose(gen_laguerre(k, 2*l+2, r), eval_genlaguerre(k, 2*l+2, r), rtol=1e-10), \
                f"Laguerre mismatch k={k} l={l} r={r}"
            s = np.sqrt(1.0 - x*x)
            assert np.isclose(assoc_legendre(l, m, x, s), lpmv(abs(m), l, x), rtol=1e-10), \
                f"Legendre mismatch l={l} m={m} x={x}"
    print("All basis checks passed.")


# ── 3. Conservation tests ────────────────────────────────────────────────────

def run_conservation_tests(n_laguerre=7, lebedev_orders=None):
    """Sweep n_lebedev — conservation entries should go to zero as n_lebedev increases."""
    from quadrature import collision_quadrature
    from integrand_numba import operator_numba
    from basis_numba import spher_const, mu_const

    if lebedev_orders is None:
        lebedev_orders = [9, 11, 13]

    cases = [
        ("mass   [0,0,0] x [2,2,-2] x [2,2,-2]", [[0,0,0],[2,2,-2],[2,2,-2]]),
        ("mass   [0,0,0] x [2,2, 0] x [2,2, 0]", [[0,0,0],[2,2, 0],[2,2, 0]]),
        ("energy [1,0,0] x [2,2,-2] x [2,2,-2]", [[1,0,0],[2,2,-2],[2,2,-2]]),
        # l=3 trial functions: the worst case once n=4 is in play (n=3 only
        # ever has l up to 2). Included whenever this is run for n=4.
        ("mass   [0,0,0] x [3,3,-3] x [3,3,-3]", [[0,0,0],[3,3,-3],[3,3,-3]]),
        ("mass   [0,0,0] x [3,3, 0] x [3,3, 0]", [[0,0,0],[3,3, 0],[3,3, 0]]),
        ("energy [1,0,0] x [3,3,-3] x [3,3,-3]", [[1,0,0],[3,3,-3],[3,3,-3]]),
        ("mass     [0,0,0] x [3,3, 1] x [3,3, 1]", [[0,0,0],[3,3, 1],[3,3, 1]]),
        ("energy   [1,0,0] x [3,3, 0] x [3,3, 0]", [[1,0,0],[3,3, 0],[3,3, 0]]),
        ("momentum [0,1,0] x [3,2,-1] x [2,3,-1]", [[0,1,0],[3,2,-1],[2,3,-1]]),
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
    # build the quadrature once per n_lebedev order and reuse it across all
    # cases, instead of rebuilding it (expensive!) once per case
    for n_leb in lebedev_orders:
        import time
        t0 = time.time()
        quad = np.array(collision_quadrature(n_laguerre, n_leb))
        build_time = time.time() - t0
        print(f"\n--- n_lebedev={n_leb}  n_pts={len(quad):>10}  (built in {build_time:.1f}s) ---")
        for label, sel in cases:
            val = eval_entry(sel, quad)
            print(f"  {label}  val={val:.4e}")


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
    from sparse import load_operator

    def to_dict(results):
        return {(tuple(e[0][0]), tuple(e[0][1]), tuple(e[0][2])): e[1] for e in results}

    results_a, n_a, lag_a, leb_a = load_operator(path_a)
    results_b, n_b, lag_b, leb_b = load_operator(path_b)
    a, b   = to_dict(results_a), to_dict(results_b)
    common = set(a) & set(b)

    print(f"=== Compare tensors ===")
    print(f"  {path_a}: n={n_a} lag={lag_a} leb={leb_a}  ({len(a)} entries)")
    print(f"  {path_b}: n={n_b} lag={lag_b} leb={leb_b}  ({len(b)} entries)")
    print(f"  common={len(common)}  only_a={len(set(a)-set(b))}  only_b={len(set(b)-set(a))}")

    mismatches = [(k, a[k], b[k]) for k in common if not np.isclose(a[k], b[k], rtol=tol)]
    if mismatches:
        print(f"  MISMATCHES ({len(mismatches)}):")
        for k, va, vb in mismatches:
            print(f"    {list(k)}  a={va:.6f}  b={vb:.6f}  diff={abs(va-vb):.2e}")
    else:
        print(f"  All {len(common)} common entries match to rtol={tol}  OK")


# ── 6. Quadrature build verification ────────────────────────────────────────

# Operator entries for the sweep: (k_i,l_i,m_i, k_s,l_s,m_s, k_t,l_t,m_t).
# Covers: k_i=0..2, l_i=0..2, k_s/k_t variants, several m≠0, mixed radial.
_SWEEP_ENTRIES = [
    (0, 2, 0,  0, 2, 0,  0, 2, 0),   # k_i=0 base entry
    (1, 2, 0,  0, 2, 0,  0, 2, 0),   # k_i=1
    (2, 2, 0,  0, 2, 0,  0, 2, 0),   # k_i=2
    (0, 0, 0,  0, 0, 0,  0, 0, 0),   # l_i=0
    (0, 1, 0,  0, 1, 0,  0, 2, 0),   # l_i=1
    (0, 2, 0,  1, 2, 0,  0, 2, 0),   # k_s=1
    (0, 2, 0,  2, 2, 0,  0, 2, 0),   # k_s=2
    (0, 2, 0,  0, 2, 0,  1, 2, 0),   # k_t=1
    (0, 2, 0,  0, 2, 0,  2, 2, 0),   # k_t=2
    (0, 1, 0,  0, 2, 0,  0, 1, 0),   # l_s≠l_t
    (0, 2, 0,  0, 1, 0,  0, 1, 0),
    (0, 2,  2,  0, 2,  2,  0, 2, 0), # m≠0
    (0, 2, -2,  0, 2, -2,  0, 2, 0),
    (0, 1,  1,  0, 1,  1,  0, 2, 0),
    (0, 2,  1,  0, 2,  0,  0, 2, 1),
    (1, 2, 0,  1, 2, 0,  0, 2, 0),   # mixed radial
    (1, 1, 0,  0, 1, 0,  1, 2, 0),
    (0, 0, 0,  1, 1, 0,  1, 1, 0),   # l_s=l_t=1, l_i=0
]

# Conservation entries (l≤2) — should be ~0 with n_lebedev=9.
_CONSERVED_ENTRIES = [
    ("mass   i=(0,0,0) s=(0,2,-2) t=(0,2,-2)", (0,0,0,  0,2,-2,  0,2,-2)),
    ("mass   i=(0,0,0) s=(0,2, 0) t=(0,2, 0)", (0,0,0,  0,2, 0,  0,2, 0)),
    ("energy i=(1,0,0) s=(0,2,-2) t=(0,2,-2)", (1,0,0,  0,2,-2,  0,2,-2)),
]


def verify_quadrature_build(n_laguerre, n_lebedev, existing_path=None):
    """
    Four-level verification of collision_quadrature(n_laguerre, n_lebedev).

    1. Analytical weight sum  — must equal 256*(4π)³ to float64 precision.
    2. Point-wise match       — if existing_path given, every column must match
       the saved reference file exactly (max diff = 0).
    3. Operator entry sweep   — 18 entries; compared against existing_path if
       given, otherwise checked for finiteness.
    4. Conservation sweep     — 3 invariant entries (l≤2) should be ~0.

    Run from src/.  existing_path is optional; skip Check 2 if omitted.
    """
    import time
    from quadrature import collision_quadrature, load_quad
    from basis_numba import mu_const, spher_const
    from integrand_numba import operator_numba

    print("=== Quadrature build verification ===")
    print(f"  n_laguerre={n_laguerre}  n_lebedev={n_lebedev}")

    t0 = time.time()
    quad_new = collision_quadrature(n_laguerre, n_lebedev)
    print(f"  built in {time.time()-t0:.1f}s  shape={quad_new.shape}")

    quad_ref = None
    if existing_path is not None:
        raw, _, _ = load_quad(existing_path)
        quad_ref = np.array(raw, dtype=np.float64)
        assert quad_ref.shape == quad_new.shape, \
            f"shape mismatch: ref {quad_ref.shape} vs new {quad_new.shape}"

    # ── Check 1: analytical weight sum ───────────────────────────────────
    expected = 256.0 * (4.0 * np.pi) ** 3
    wdiff = abs(quad_new[:, 8].sum() - expected) / expected
    c1 = wdiff < 1e-12
    print(f"  [1] weight sum rel err: {wdiff:.2e}  {'PASS' if c1 else 'FAIL'}")

    # ── Check 2: point-wise match ─────────────────────────────────────────
    c2 = True
    if quad_ref is not None:
        col_names = ['rp','tp','pp','rq','tq','pq','tw','pw','w']
        diffs = [np.max(np.abs(quad_new[:, j] - quad_ref[:, j])) for j in range(9)]
        overall = max(diffs)
        c2 = overall == 0.0
        detail = '  '.join(f"{n}:{d:.0e}" for n, d in zip(col_names, diffs))
        print(f"  [2] point-wise ({existing_path}): {detail}"
              f"  {'PASS' if c2 else 'FAIL'}")
    else:
        print(f"  [2] point-wise: skipped (no existing_path provided)")

    # ── JIT warmup ────────────────────────────────────────────────────────
    operator_numba(0,1,0,spher_const(1,0), 0,1,0,mu_const(0,1),spher_const(1,0),
                   0,2,0,mu_const(0,2),spher_const(2,0), quad_new)

    def ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad):
        return operator_numba(
            ki, li, mi, spher_const(li, mi),
            ks, ls, ms, mu_const(ks, ls), spher_const(ls, ms),
            kt, lt, mt, mu_const(kt, lt), spher_const(lt, mt),
            quad)

    # ── Check 3: operator entry sweep ─────────────────────────────────────
    n3_pass = n3_fail = 0
    for entry in _SWEEP_ENTRIES:
        ki, li, mi, ks, ls, ms, kt, lt, mt = entry
        v_new = ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad_new)
        if quad_ref is not None:
            v_ref = ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad_ref)
            rd = abs(v_new - v_ref) / abs(v_ref) if abs(v_ref) > 1e-20 else abs(v_new)
            ok = rd < 1e-12
        else:
            ok = np.isfinite(v_new)
        if ok: n3_pass += 1
        else:  n3_fail += 1
    c3 = n3_fail == 0
    print(f"  [3] operator sweep: {n3_pass}/{len(_SWEEP_ENTRIES)}"
          f"  {'PASS' if c3 else 'FAIL'}")

    # ── Check 4: conservation sweep ───────────────────────────────────────
    n4_pass = n4_fail = 0
    for label, (ki, li, mi, ks, ls, ms, kt, lt, mt) in _CONSERVED_ENTRIES:
        v = ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad_new)
        ok = abs(v) < 1e-6
        if ok: n4_pass += 1
        else:  n4_fail += 1
        print(f"  [4] {label}  {v:.4e}  {'PASS' if ok else 'FAIL'}")
    c4 = n4_fail == 0

    all_pass = c1 and c2 and c3 and c4
    print(f"  {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 42)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_laguerre, n_lebedev = 7, 7
    run_quadrature_tests(n_laguerre, n_lebedev)
    print()
    run_basis_tests()
    print()
    run_conservation_tests(n_laguerre)
    print()
    run_convergence_test()
    # To run the build verifier:
    # verify_quadrature_build(7, 9, existing_path='quadratures/collision_lag7_leb9.pkl')
