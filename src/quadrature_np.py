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


# ── verification against existing (7,9) quadrature ───────────────────────────

def verify_against_existing(n_laguerre, n_lebedev, existing_path):
    """
    Build the quadrature with collision_quadrature_np and compare against
    a pre-existing file by computing one operator entry with each and
    checking they agree to near float64 precision.
    """
    import sys
    sys.path.insert(0, '.')
    from quadrature import load_quad
    from basis_numba import mu_const, spher_const
    from integrand_numba import operator_numba

    print(f"\n── Verification: ({n_laguerre},{n_lebedev}) vs {existing_path} ──")

    # Build new quadrature
    t0 = time.time()
    quad_new = collision_quadrature_np(n_laguerre, n_lebedev)
    print(f"  built in {time.time()-t0:.1f}s")

    # Load existing
    raw_old, _, _ = load_quad(existing_path)
    quad_old = np.array(raw_old, dtype=np.float64)
    print(f"  existing: {quad_old.shape[0]:,} pts  new: {quad_new.shape[0]:,} pts")

    assert quad_old.shape == quad_new.shape, "point counts differ!"

    # Weight-sum sanity check
    w_old = quad_old[:, 8].sum()
    w_new = quad_new[:, 8].sum()
    print(f"  weight sum old: {w_old:.10f}")
    print(f"  weight sum new: {w_new:.10f}")
    print(f"  weight sum rel diff: {abs(w_old-w_new)/abs(w_old):.2e}")

    # Point-wise max difference
    max_diff = np.max(np.abs(quad_new - quad_old))
    print(f"  max pointwise diff (all columns): {max_diff:.2e}")

    # Compute one operator entry with each and compare
    print("  JIT warmup ...", end='', flush=True)
    _ = operator_numba(0,1,0,spher_const(1,0), 0,1,0,mu_const(0,1),spher_const(1,0),
                       0,2,0,mu_const(0,2),spher_const(2,0), quad_old)
    print(" done")

    # Entry: test (0,2,0), field s=(0,2,0), t=(0,2,0)
    c_i = spher_const(2, 0)
    mu_s = mu_const(0, 2); c_s = spher_const(2, 0)
    mu_t = mu_const(0, 2); c_t = spher_const(2, 0)

    v_old = operator_numba(0, 2, 0, c_i, 0, 2, 0, mu_s, c_s, 0, 2, 0, mu_t, c_t, quad_old)
    v_new = operator_numba(0, 2, 0, c_i, 0, 2, 0, mu_s, c_s, 0, 2, 0, mu_t, c_t, quad_new)
    print(f"  operator entry (old): {v_old:.10f}")
    print(f"  operator entry (new): {v_new:.10f}")
    print(f"  rel diff: {abs(v_old-v_new)/abs(v_old):.2e}")

    return quad_new


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from quadrature import quad_name

    # Step 1: verify (7,9) against existing
    quad_79 = verify_against_existing(
        7, 9,
        existing_path='quadratures/collision_lag7_leb9.pkl'
    )

    print("\n── Verification passed.  Building (11,13) ──────────────────────")
    t0 = time.time()
    quad_1113 = collision_quadrature_np(11, 13)
    print(f"  built in {time.time()-t0:.1f}s")

    out_path = quad_name('collision', 11, 13)
    save_quad_np(quad_1113, out_path, 11, 13)
    print(f"  done in {time.time()-t0:.1f}s total")
