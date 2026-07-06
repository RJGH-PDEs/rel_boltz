import pickle
import numpy as np
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev


# ── helpers used by mass_quadrature ──────────────────────────────────────────

def radial_quad(n):
    """Gauss-Laguerre nodes/weights for int e^{-r/2} r^2 f(r) dr as (r, w) pairs."""
    x, w = roots_genlaguerre(n, 2)
    return list(zip(2 * x, 8 * w))

def lebedev_quad(n):
    """Lebedev quadrature over S^2, weights include 4π factor. Returns (point_3d, w) pairs."""
    leblib = PyLebedev()
    s, w = leblib.get_points_and_weights(n)
    return list(zip(s, 4 * np.pi * w))

def cart_to_ang(point):
    """Convert Cartesian unit vector to (theta, phi)."""
    x, y, z = point
    return np.arccos(np.clip(z, -1.0, 1.0)), np.arctan2(y, x)


# ── numpy helpers for collision_quadrature ────────────────────────────────────

def _radial_nodes_weights(n_laguerre):
    x, w = roots_genlaguerre(n_laguerre, 2)
    return 2.0 * x, 8.0 * w

def _lebedev_nodes_weights(n_lebedev):
    leblib = PyLebedev()
    s, w = leblib.get_points_and_weights(n_lebedev)
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    return np.arccos(np.clip(z, -1.0, 1.0)), np.arctan2(y, x), 4.0 * np.pi * w

def _cartesian_factor(values, axis, sizes):
    """
    Return the flat column of `values` for `axis` in a Cartesian product
    with per-factor sizes `sizes`, using repeat/tile without meshgrid.
    """
    repeat = int(np.prod(sizes[axis + 1:]))
    tile   = int(np.prod(sizes[:axis]))
    return np.tile(np.repeat(values, repeat), tile)


# ── collision quadrature ──────────────────────────────────────────────────────

def collision_quadrature(n_laguerre, n_lebedev):
    """
    Build the 8D collision quadrature as a (N, 9) float64 array.

    Point layout: Cartesian product (r_p, e_p, r_q, e_q, omega), outermost
    to innermost.  Column order: [rp, tp, pp, rq, tq, pq, tw, pw, w].

    Implemented with numpy repeat/tile — O(N) time and memory, no nested
    loops.  Handles large orders (e.g. n_laguerre=11, n_lebedev=13, ~49M pts)
    that the old pure-Python loop could not.
    """
    r_nodes, r_weights        = _radial_nodes_weights(n_laguerre)
    leb_theta, leb_phi, leb_w = _lebedev_nodes_weights(n_lebedev)

    n_lag = len(r_nodes)
    n_leb = len(leb_w)
    sizes = [n_lag, n_leb, n_lag, n_leb, n_leb]
    N     = int(np.prod(sizes))

    quad = np.empty((N, 9), dtype=np.float64)

    quad[:, 0] = _cartesian_factor(r_nodes,   0, sizes)   # rp
    quad[:, 1] = _cartesian_factor(leb_theta, 1, sizes)   # tp
    quad[:, 2] = _cartesian_factor(leb_phi,   1, sizes)   # pp
    quad[:, 3] = _cartesian_factor(r_nodes,   2, sizes)   # rq
    quad[:, 4] = _cartesian_factor(leb_theta, 3, sizes)   # tq
    quad[:, 5] = _cartesian_factor(leb_phi,   3, sizes)   # pq
    quad[:, 6] = _cartesian_factor(leb_theta, 4, sizes)   # tw
    quad[:, 7] = _cartesian_factor(leb_phi,   4, sizes)   # pw

    w = _cartesian_factor(r_weights, 0, sizes)
    w = w * _cartesian_factor(leb_w,     1, sizes)
    w = w * _cartesian_factor(r_weights, 2, sizes)
    w = w * _cartesian_factor(leb_w,     3, sizes)
    w = w * _cartesian_factor(leb_w,     4, sizes)
    quad[:, 8] = w

    return quad


# ── save / load ───────────────────────────────────────────────────────────────

def save_quad(quad, path, n_laguerre, n_lebedev):
    data = {'quad': quad, 'n_laguerre': n_laguerre, 'n_lebedev': n_lebedev}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    n = quad.shape[0] if hasattr(quad, 'shape') else len(quad)
    print(f"quadrature saved to {path}  ({n:,} points)")

def load_quad(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['quad'], data['n_laguerre'], data['n_lebedev']


# ── mass quadrature ───────────────────────────────────────────────────────────

def mass_quadrature(n_laguerre, n_lebedev):
    """3D quadrature for the mass matrix: one radial × one Lebedev. Each point: [r, t, p, w]."""
    rlag = radial_quad(n_laguerre)
    leb  = lebedev_quad(n_lebedev)
    quad = []
    for r, wr in rlag:
        for e, we in leb:
            t, p = cart_to_ang(e)
            quad.append([r, t, p, wr * we])
    return quad


# ── naming / main ─────────────────────────────────────────────────────────────

def quad_name(kind, n_laguerre, n_lebedev, tag=''):
    return f'./quadratures/{kind}_lag{n_laguerre}_leb{n_lebedev}{tag}.pkl'

def main():
    n_laguerre = 7
    n_lebedev  = 9

    print("building collision quadrature...")
    quad = collision_quadrature(n_laguerre, n_lebedev)
    save_quad(quad, quad_name('collision', n_laguerre, n_lebedev), n_laguerre, n_lebedev)

    print("building mass quadrature...")
    mquad = mass_quadrature(n_laguerre, n_lebedev)
    save_quad(mquad, quad_name('mass', n_laguerre, n_lebedev), n_laguerre, n_lebedev)

if __name__ == "__main__":
    main()
