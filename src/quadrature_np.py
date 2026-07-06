"""
Numpy-vectorised replacement for collision_quadrature() in quadrature.py.

The original pure-Python nested loop is too slow and memory-heavy at large
quadrature orders (n_laguerre=11, n_lebedev=13 requires ~49M points, which
the Python list-of-lists approach cannot handle without ~15 GB RAM).

This file is SEPARATE from quadrature.py — the original is not touched.
The output format (dict with keys 'quad', 'n_laguerre', 'n_lebedev') is
identical, so load_quad() from quadrature.py works on the files produced here.

Usage:
    Run from src/ to verify against (7,9) then build (11,13):
        ~/miniconda3/envs/ttenv/bin/python quadrature_np.py

Key idea: the 8D collision quadrature is a Cartesian product of 5 factors:
    (r_p, e_p, r_q, e_q, omega)
Each output column depends on exactly one factor, so it can be built via
np.repeat / np.tile in O(N) time and O(N) peak memory — no meshgrid needed.
"""
import pickle
import time
import numpy as np
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev


# ── helpers (mirror quadrature.py, no shared state) ──────────────────────────

def _radial_nodes_weights(n_laguerre):
    """Gauss-Laguerre nodes/weights for int e^{-r/2} r^2 f(r) dr."""
    x, w = roots_genlaguerre(n_laguerre, 2)
    return 2.0 * x, 8.0 * w            # r = 2x, w_r = 8 w_GL


def _lebedev_nodes_weights(n_lebedev):
    """Lebedev nodes (theta, phi) and weights (including 4π factor)."""
    leblib = PyLebedev()
    s, w = leblib.get_points_and_weights(n_lebedev)
    # s: (n_leb, 3) unit vectors
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi   = np.arctan2(y, x)
    return theta, phi, 4.0 * np.pi * w


def _cartesian_factor(values, axis, sizes):
    """
    For a Cartesian product with factor sizes `sizes`, return the flat array
    of `values` corresponding to axis `axis`.

    Each value[i] repeats prod(sizes[axis+1:]) consecutive times, and the
    full pattern tiles prod(sizes[:axis]) times.
    """
    repeat = int(np.prod(sizes[axis + 1:]))
    tile   = int(np.prod(sizes[:axis]))
    return np.tile(np.repeat(values, repeat), tile)


# ── main build function ───────────────────────────────────────────────────────

def collision_quadrature_np(n_laguerre, n_lebedev):
    """
    Build the 8D collision quadrature as a (N, 9) float64 array.
    Column order: [rp, tp, pp, rq, tq, pq, tw, pw, w]
    identical to the quadrature.py convention.

    Point order matches the original nested-loop order:
    outermost→innermost: r_p, e_p, r_q, e_q, omega.
    """
    r_nodes, r_weights          = _radial_nodes_weights(n_laguerre)
    leb_theta, leb_phi, leb_w   = _lebedev_nodes_weights(n_lebedev)

    n_lag = len(r_nodes)
    n_leb = len(leb_w)
    # Factor order: (r_p, e_p, r_q, e_q, omega)
    sizes = [n_lag, n_leb, n_lag, n_leb, n_leb]
    N     = int(np.prod(sizes))

    print(f"  {n_laguerre} Laguerre pts × {n_leb} Lebedev pts³ "
          f"= {N:,} quadrature points  ({N*9*8/1e9:.2f} GB)")

    quad = np.empty((N, 9), dtype=np.float64)

    # ── geometry columns ─────────────────────────────────────────────────────
    quad[:, 0] = _cartesian_factor(r_nodes,   0, sizes)   # rp
    quad[:, 1] = _cartesian_factor(leb_theta, 1, sizes)   # tp
    quad[:, 2] = _cartesian_factor(leb_phi,   1, sizes)   # pp
    quad[:, 3] = _cartesian_factor(r_nodes,   2, sizes)   # rq
    quad[:, 4] = _cartesian_factor(leb_theta, 3, sizes)   # tq
    quad[:, 5] = _cartesian_factor(leb_phi,   3, sizes)   # pq
    quad[:, 6] = _cartesian_factor(leb_theta, 4, sizes)   # tw
    quad[:, 7] = _cartesian_factor(leb_phi,   4, sizes)   # pw

    # ── weight column (product of all 5 component weights) ───────────────────
    w = _cartesian_factor(r_weights, 0, sizes)
    w = w * _cartesian_factor(leb_w,    1, sizes)
    w = w * _cartesian_factor(r_weights, 2, sizes)
    w = w * _cartesian_factor(leb_w,    3, sizes)
    w = w * _cartesian_factor(leb_w,    4, sizes)
    quad[:, 8] = w

    return quad


def save_quad_np(quad, path, n_laguerre, n_lebedev):
    data = {'quad': quad, 'n_laguerre': n_laguerre, 'n_lebedev': n_lebedev}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  saved → {path}  ({quad.shape[0]:,} pts, "
          f"{quad.nbytes/1e9:.2f} GB on disk before compression)")


# ── comprehensive verification ────────────────────────────────────────────────

# 18 operator entries for the sweep: (k_i,l_i,m_i, k_s,l_s,m_s, k_t,l_t,m_t)
# Chosen to cover: varying k_i (0-2), l_i (0-2), k_s/k_t, l_s/l_t, and m≠0.
_SWEEP = [
    # vary k_i  (l_i=2, m_i=0, s=(0,2,0), t=(0,2,0)) — A_k sequence base
    (0, 2, 0,  0, 2, 0,  0, 2, 0),
    (1, 2, 0,  0, 2, 0,  0, 2, 0),
    (2, 2, 0,  0, 2, 0,  0, 2, 0),
    # vary l_i  (k_i=0, m_i=0)
    (0, 0, 0,  0, 0, 0,  0, 0, 0),
    (0, 1, 0,  0, 1, 0,  0, 2, 0),
    # vary k_s  (k_i=0, l_i=2, k_t=0)
    (0, 2, 0,  1, 2, 0,  0, 2, 0),
    (0, 2, 0,  2, 2, 0,  0, 2, 0),
    # vary k_t  (k_i=0, l_i=2, k_s=0)
    (0, 2, 0,  0, 2, 0,  1, 2, 0),
    (0, 2, 0,  0, 2, 0,  2, 2, 0),
    # different l_s, l_t
    (0, 1, 0,  0, 2, 0,  0, 1, 0),
    (0, 2, 0,  0, 1, 0,  0, 1, 0),
    # nonzero m
    (0, 2,  2,  0, 2,  2,  0, 2, 0),
    (0, 2, -2,  0, 2, -2,  0, 2, 0),
    (0, 1,  1,  0, 1,  1,  0, 2, 0),
    (0, 2,  1,  0, 2,  0,  0, 2, 1),
    # mixed radial indices
    (1, 2, 0,  1, 2, 0,  0, 2, 0),
    (1, 1, 0,  0, 1, 0,  1, 2, 0),
    # l_s=l_t=1, l_i=0
    (0, 0, 0,  1, 1, 0,  1, 1, 0),
]

# Conservation entries (l≤2) that should evaluate to ~0 with the (7,9) quadrature.
# Using the same cases as run_conservation_tests() in tests.py.
_CONSERVED = [
    ("mass   i=(0,0,0) s=(0,2,-2) t=(0,2,-2)", (0,0,0,  0,2,-2,  0,2,-2)),
    ("mass   i=(0,0,0) s=(0,2, 0) t=(0,2, 0)", (0,0,0,  0,2, 0,  0,2, 0)),
    ("energy i=(1,0,0) s=(0,2,-2) t=(0,2,-2)", (1,0,0,  0,2,-2,  0,2,-2)),
]


def verify_full(n_laguerre, n_lebedev, existing_path):
    """
    Four-level verification of collision_quadrature_np against an existing file.

    1. Analytical weight sum  — must equal 256*(4π)³ to float64 precision.
    2. Per-column point-wise  — every column of new must match old exactly (max
       diff should be 0; floating-point order of operations is identical).
    3. Operator entry sweep   — 18 entries spanning k_i, l_i, k_s, k_t, m;
       old and new must agree to <1e-12 relative error on every entry.
    4. Conservation sweep     — 3 invariant entries (mass/energy, l≤2) must be
       ~0 (quadrature accuracy) and old/new must agree to <1e-12.

    Returns the newly built quadrature array.
    """
    import sys
    sys.path.insert(0, '.')
    from quadrature import load_quad
    from basis_numba import mu_const, spher_const
    from integrand_numba import operator_numba

    W  = '='
    print(f"\n{W*64}")
    print(f"  verify_full  ({n_laguerre},{n_lebedev})  vs  {existing_path}")
    print(f"{W*64}")

    # ── build ────────────────────────────────────────────────────────────────
    t0 = time.time()
    quad_new = collision_quadrature_np(n_laguerre, n_lebedev)
    print(f"  built in {time.time()-t0:.1f}s")

    raw_old, _, _ = load_quad(existing_path)
    quad_old = np.array(raw_old, dtype=np.float64)
    assert quad_old.shape == quad_new.shape, \
        f"shape mismatch: old {quad_old.shape} vs new {quad_new.shape}"

    # ── Check 1: analytical weight sum ───────────────────────────────────────
    expected = 256.0 * (4.0 * np.pi) ** 3
    w_new = quad_new[:, 8].sum()
    w_old = quad_old[:, 8].sum()
    wdiff_new = abs(w_new - expected) / expected
    wdiff_old = abs(w_old - expected) / expected
    c1 = wdiff_new < 1e-12
    print(f"\n── Check 1: analytical weight sum  (expected {expected:.6f})")
    print(f"  old: {w_old:.10f}  rel err {wdiff_old:.2e}")
    print(f"  new: {w_new:.10f}  rel err {wdiff_new:.2e}  {'PASS' if c1 else 'FAIL'}")

    # ── Check 2: per-column point-wise match ─────────────────────────────────
    col_names = ['rp', 'tp', 'pp', 'rq', 'tq', 'pq', 'tw', 'pw', 'w']
    col_diffs = [np.max(np.abs(quad_new[:, j] - quad_old[:, j]))
                 for j in range(9)]
    overall = max(col_diffs)
    c2 = overall == 0.0
    print(f"\n── Check 2: per-column point-wise match")
    for name, d in zip(col_names, col_diffs):
        print(f"  {name}: {d:.2e}")
    print(f"  overall max diff: {overall:.2e}  {'PASS' if c2 else 'FAIL'}")

    # ── JIT warmup ───────────────────────────────────────────────────────────
    print("\n  JIT warmup ...", end='', flush=True)
    operator_numba(0, 1, 0, spher_const(1, 0),
                   0, 1, 0, mu_const(0, 1), spher_const(1, 0),
                   0, 2, 0, mu_const(0, 2), spher_const(2, 0), quad_old)
    print(" done")

    def ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad):
        return operator_numba(
            ki, li, mi, spher_const(li, mi),
            ks, ls, ms, mu_const(ks, ls), spher_const(ls, ms),
            kt, lt, mt, mu_const(kt, lt), spher_const(lt, mt),
            quad)

    # ── Check 3: operator entry sweep ────────────────────────────────────────
    print(f"\n── Check 3: operator entry sweep  ({len(_SWEEP)} entries)")
    hdr = f"  {'entry':<38}  {'old':>13}  {'rel diff':>10}  status"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    n3_pass = n3_fail = 0
    for entry in _SWEEP:
        ki, li, mi, ks, ls, ms, kt, lt, mt = entry
        v_old = ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad_old)
        v_new = ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad_new)
        rd = abs(v_old - v_new) / abs(v_old) if abs(v_old) > 1e-20 else abs(v_new)
        ok = rd < 1e-12
        if ok: n3_pass += 1
        else:  n3_fail += 1
        label = f"i=({ki},{li},{mi:+d}) s=({ks},{ls},{ms:+d}) t=({kt},{lt},{mt:+d})"
        print(f"  {label:<38}  {v_old:>13.6e}  {rd:>10.2e}  {'PASS' if ok else 'FAIL'}")
    c3 = n3_fail == 0
    print(f"  {n3_pass}/{len(_SWEEP)} passed")

    # ── Check 4: conservation sweep ──────────────────────────────────────────
    print(f"\n── Check 4: conservation entries  (should be ~0, old ≈ new)")
    hdr4 = f"  {'label':<44}  {'old':>11}  {'new':>11}  {'|Δ|/|old|':>10}"
    print(hdr4)
    print("  " + "-" * (len(hdr4) - 2))
    n4_pass = n4_fail = 0
    for label, (ki, li, mi, ks, ls, ms, kt, lt, mt) in _CONSERVED:
        v_old = ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad_old)
        v_new = ev(ki, li, mi, ks, ls, ms, kt, lt, mt, quad_new)
        rd = abs(v_old - v_new) / abs(v_old) if abs(v_old) > 1e-20 else abs(v_new)
        ok = rd < 1e-12
        if ok: n4_pass += 1
        else:  n4_fail += 1
        print(f"  {label:<44}  {v_old:>11.3e}  {v_new:>11.3e}  {rd:>10.2e}  {'PASS' if ok else 'FAIL'}")
    c4 = n4_fail == 0

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  Check 1  weight sum (analytical):  {'PASS' if c1 else 'FAIL'}")
    print(f"  Check 2  point-wise match:          {'PASS' if c2 else 'FAIL'}")
    print(f"  Check 3  operator sweep:            {'PASS' if c3 else 'FAIL'}  ({n3_pass}/{len(_SWEEP)})")
    print(f"  Check 4  conservation entries:      {'PASS' if c4 else 'FAIL'}  ({n4_pass}/{len(_CONSERVED)})")
    all_pass = c1 and c2 and c3 and c4
    print(f"  {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'─'*64}")

    return quad_new


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from quadrature import quad_name

    verify_full(7, 9, existing_path='quadratures/collision_lag7_leb9.pkl')

    print("\n── (11,13) already built; skipping rebuild ─────────────────────")
    print("   To rebuild: uncomment the block below and re-run.")
    # t0 = time.time()
    # quad_1113 = collision_quadrature_np(11, 13)
    # out_path = quad_name('collision', 11, 13)
    # save_quad_np(quad_1113, out_path, 11, 13)
    # print(f"  done in {time.time()-t0:.1f}s")
