import numpy as np
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev

# Radial quadrature for int_0^inf e^{-r/2} r^2 f(r) dr
# Change of variables r' = r/2: points -> 2*r', weights -> 4*w
def radial_quad(n):
    x, w = roots_genlaguerre(n, 2)
    points  = 2 * x
    weights = 8 * w
    return points, weights

# Lebedev quadrature over S^2
def lebedev_quad(n):
    leblib = PyLebedev()
    s, w = leblib.get_points_and_weights(n)
    # weights scaled by 4*pi for integration over S^2
    return s, 4 * np.pi * w

# Test 1: int_0^inf e^{-r/2} r^2 (1) dr = 16
def test_radial(n):
    pts, wts = radial_quad(n)
    result = np.sum(wts * pts)   # f(r) = r
    print(f"Radial test (n={n}): {result:.10f}  (expected 96)")

# Test 2: int_{S^2} (1) domega = 4*pi
def test_lebedev(n):
    _, wts = lebedev_quad(n)
    result = np.sum(wts)
    print(f"Lebedev test (n={n}): {result:.10f}  (expected {4*np.pi:.10f})")

# Test 3: integrate f=1 over the full collision quadrature
# expected: 16 * 4pi * 16 * 4pi * 4pi = 256*(4pi)^3
def test_collision_quad(path):
    from quadrature import load_quad
    quad     = load_quad(path)
    result   = sum(pt[-1] for pt in quad)   # last entry is the combined weight
    expected = 256 * (4*np.pi)**3
    ok       = np.isclose(result, expected)
    print(f"Collision quad test: {result:.6f}  (expected {expected:.6f})  {'OK' if ok else 'FAIL'}")

# Test 4: integrate f=1 over the mass quadrature
# expected: 16 * 4pi
def test_mass_quad(path):
    from quadrature import load_quad
    quad     = load_quad(path)
    result   = sum(pt[-1] for pt in quad)
    expected = 16 * 4 * np.pi
    ok       = np.isclose(result, expected)
    print(f"Mass quad test: {result:.6f}  (expected {expected:.6f})  {'OK' if ok else 'FAIL'}")

# Test 5: integrate sin(theta) over S^2
# expected: pi^2  (note: sin(theta) is not polynomial in x,y,z so Lebedev won't be exact)
def test_lebedev_sin(n):
    from quadrature import lebedev_quad, cart_to_ang
    leb    = lebedev_quad(n)
    result = sum(w * np.sin(cart_to_ang(e)[0]) for e, w in leb)
    expected = np.pi**2
    ok = np.isclose(result, expected)
    print(f"Lebedev sin(theta) test (n={n}): {result:.10f}  (expected {expected:.10f})  {'OK' if ok else 'FAIL'}")

# Test 8: integrate cos(phi)sin(theta) = x over S^2
# expected: 0 by symmetry
def test_lebedev_x(n):
    from quadrature import lebedev_quad
    leb    = lebedev_quad(n)
    result = sum(w * e[0] for e, w in leb)
    ok = np.isclose(result, 0.0)
    print(f"Lebedev x test (n={n}): {result:.2e}  (expected 0)  {'OK' if ok else 'FAIL'}")

# Test 7: integrate sin^2(theta)cos^2(phi) = x^2 over S^2
# expected: 4*pi/3
def test_lebedev_x2(n):
    from quadrature import lebedev_quad
    leb    = lebedev_quad(n)
    result = sum(w * e[0]**2 for e, w in leb)
    expected = 4 * np.pi / 3
    ok = np.isclose(result, expected)
    print(f"Lebedev x^2 test (n={n}): {result:.10f}  (expected {expected:.10f})  {'OK' if ok else 'FAIL'}")

# Test 6: integrate sin^2(theta) = 1 - z^2 over S^2
# expected: 8*pi/3  (polynomial in Cartesian coords, Lebedev should be exact)
def test_lebedev_sin2(n):
    from quadrature import lebedev_quad
    leb    = lebedev_quad(n)
    result = sum(w * (1 - e[2]**2) for e, w in leb)
    expected = 8 * np.pi / 3
    ok = np.isclose(result, expected)
    print(f"Lebedev sin^2(theta) test (n={n}): {result:.10f}  (expected {expected:.10f})  {'OK' if ok else 'FAIL'}")

def main():
    print("--- Radial quadrature ---")
    for n in [5, 9, 15]:
        test_radial(n)

    print()
    print("--- Lebedev quadrature ---")
    for n in [7, 11, 15]:
        test_lebedev(n)

    print()
    print("--- Collision quadrature ---")
    test_collision_quad('./quadratures/collision.pkl')

    print()
    print("--- Mass quadrature ---")
    test_mass_quad('./quadratures/mass.pkl')

    print()
    print("--- Lebedev sin(theta) ---")
    for n in [3, 5, 7]:
        test_lebedev_sin(n)

    print()
    print("--- Lebedev sin^2(theta) ---")
    for n in [3, 5, 7]:
        test_lebedev_sin2(n)

    print()
    print("--- Lebedev cos(phi)sin(theta) = x ---")
    for n in [3, 5, 7]:
        test_lebedev_x(n)

    print()
    print("--- Lebedev sin^2(theta)cos^2(phi) = x^2 ---")
    for n in [3, 5, 7]:
        test_lebedev_x2(n)

# Quadrature convergence test for n=3 truncation.
# We evaluate a single high-degree collision tensor entry [[2,2,2],[2,2,2],[2,2,2]]
# at increasing quadrature resolutions. This is the hardest entry to integrate
# accurately (highest k and l in all three indices), so if the value stabilizes
# here, the quadrature is sufficient for the full n=3 tensor.
def test_quadrature_convergence():
    import pickle
    import os
    from quadrature import collision_quadrature, save_quad, load_quad
    from boltzmann import operator_test

    select = [[2, 2, 0], [2, 2, 0], [2, 0, 0]]
    configs = [
        (6, 5),
        (7, 7),
        (8, 9),
    ]

    print("--- Quadrature convergence test (select = [[2,2,2],[2,2,2],[2,2,2]]) ---")
    for n_lag, n_leb in configs:
        path = f'./quadratures/conv_lag{n_lag}_leb{n_leb}.pkl'
        if not os.path.exists(path):
            quad = collision_quadrature(n_lag, n_leb)
            save_quad(quad, path)
        from boltzmann import operator
        from integrand import pieces
        quad = load_quad(path)
        phi, ft, gt = pieces(select)
        result = operator(phi, ft, gt, quad)
        print(f"  n_laguerre={n_lag}, n_lebedev={n_leb}:  {result:.10f}  ({len(quad)} points)")


if __name__ == "__main__":
    test_quadrature_convergence()
