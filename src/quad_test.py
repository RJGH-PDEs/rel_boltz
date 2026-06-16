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

if __name__ == "__main__":
    main()
