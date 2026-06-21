import pickle
import numpy as np
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev

# Radial quadrature for int_0^inf e^{-r/2} r^2 f(r) dr
# Change of variables r' = r/2: points -> 2*r', weights -> 8*w
def radial_quad(n):
    x, w = roots_genlaguerre(n, 2)
    points  = 2 * x
    weights = 8 * w
    return list(zip(points, weights))

# Lebedev quadrature over S^2, weights include 4*pi factor
def lebedev_quad(n):
    leblib = PyLebedev()
    s, w = leblib.get_points_and_weights(n)
    weights = 4 * np.pi * w
    return list(zip(s, weights))

# Convert Cartesian point on S^2 to (theta, phi)
def cart_to_ang(point):
    x, y, z = point
    t = np.arccos(np.clip(z, -1.0, 1.0))
    p = np.arctan2(y, x)
    return t, p

# Full 8D quadrature for the collision operator:
# integrates over (r_p, e_p, r_q, e_q, omega) with combined weight
# each point is stored as [rp, tp, pp, rq, tq, pq, tw, pw, w]
def collision_quadrature(n_laguerre, n_lebedev):
    rlag = radial_quad(n_laguerre)
    leb  = lebedev_quad(n_lebedev)

    quad = []
    for rp, wrp in rlag:
        for ep, wep in leb:
            tp, pp = cart_to_ang(ep)
            for rq, wrq in rlag:
                for eq, weq in leb:
                    tq, pq = cart_to_ang(eq)
                    for ew, wew in leb:
                        tw, pw = cart_to_ang(ew)
                        # combined weight: radial_p * angular_p * radial_q * angular_q * omega
                        w = wrp * wep * wrq * weq * wew
                        quad.append([rp, tp, pp, rq, tq, pq, tw, pw, w])

    return quad

# Save/load — stored as dict with metadata so parameters are self-describing
def save_quad(quad, path, n_laguerre, n_lebedev):
    data = {'quad': quad, 'n_laguerre': n_laguerre, 'n_lebedev': n_lebedev}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"quadrature saved to {path}  ({len(quad)} points)")

def load_quad(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['quad'], data['n_laguerre'], data['n_lebedev']

# 3D quadrature for the mass matrix: one radial + one Lebedev
# each point stored as [r, t, p, w]
def mass_quadrature(n_laguerre, n_lebedev):
    rlag = radial_quad(n_laguerre)
    leb  = lebedev_quad(n_lebedev)

    quad = []
    for r, wr in rlag:
        for e, we in leb:
            t, p = cart_to_ang(e)
            w    = wr * we
            quad.append([r, t, p, w])

    return quad

def quad_name(kind, n_laguerre, n_lebedev, tag=''):
    return f'./quadratures/{kind}_lag{n_laguerre}_leb{n_lebedev}{tag}.pkl'

def main():
    n_laguerre = 7
    n_lebedev  = 7

    print("building collision quadrature...")
    quad = collision_quadrature(n_laguerre, n_lebedev)
    save_quad(quad, quad_name('collision', n_laguerre, n_lebedev), n_laguerre, n_lebedev)

    print("building mass quadrature...")
    mquad = mass_quadrature(n_laguerre, n_lebedev)
    save_quad(mquad, quad_name('mass', n_laguerre, n_lebedev), n_laguerre, n_lebedev)

if __name__ == "__main__":
    main()
