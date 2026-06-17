import numpy as np
import pickle
from scipy.special import roots_genlaguerre, eval_genlaguerre
from basis import basis, f_tilde, mu_const
from quadrature import load_quad
from sparse import ind

# Integrand for one (i, j) pair at a single quadrature point
# point = [r, t, p]  (weight excluded)
def integrand(phi, ft, point):
    r, t, p = point
    return phi(r, t, p) * ft(r, t, p)

# Compute one mass matrix entry M[i, j]
def entry(phi, ft, quad):
    result = 0.0
    for pt in quad:
        w     = pt[-1]
        point = pt[:-1]
        result += w * integrand(phi, ft, point)
    return result

# Build the full mass matrix for a given n
# indices run over k in [0,n), l in [0,n), m in [-l, l]
def mass_matrix(n, quad):
    N = n**3
    M = np.zeros((N, N))

    for ki in range(n):
        for li in range(n):
            for mi in range(-li, li+1):
                phi = basis(ki, li, mi)
                i   = ind(ki, li, mi, n)

                for kj in range(n):
                    for lj in range(n):
                        for mj in range(-lj, lj+1):
                            ft = f_tilde(kj, lj, mj)
                            j  = ind(kj, lj, mj, n)

                            M[i, j] = entry(phi, ft, quad)

    return M

# 1D radial mass matrix using spherical harmonic orthogonality:
# M[i,j] = 0 unless l_i == l_j and m_i == m_j
# M[i,j] = mu_{kj,l} * 2^{2l+3} * sum_s w_s * L_{ki}^{2(l+1)}(2x_s) * L_{kj}^{2(l+1)}(2x_s)
def mass_matrix_1d(n, n_laguerre=15):
    N = n**3
    M = np.zeros((N, N))

    for ki in range(n):
        for li in range(n):
            for mi in range(-li, li+1):
                i = ind(ki, li, mi, n)

                for kj in range(n):
                    for lj in range(n):
                        for mj in range(-lj, lj+1):
                            j = ind(kj, lj, mj, n)

                            # orthogonality: zero unless l and m match
                            if li != lj or mi != mj:
                                M[i, j] = 0.0
                                continue

                            l     = li
                            alpha = 2*(l + 1)
                            x, w  = roots_genlaguerre(n_laguerre, alpha)
                            factor = mu_const(kj, l) * 2**(2*l + 3)

                            M[i, j] = factor * np.sum(
                                w * eval_genlaguerre(ki, alpha, 2*x)
                                  * eval_genlaguerre(kj, alpha, 2*x)
                            )

    return M

def save_mass(M, path):
    with open(path, 'wb') as f:
        pickle.dump(M, f)
    print(f"mass matrix saved to {path}")

def test():
    n    = 3
    quad = load_quad('./quadratures/mass.pkl')

    print(f"computing 3D mass matrix (n={n})...")
    M3d = mass_matrix(n, quad)

    print(f"computing 1D mass matrix (n={n})...")
    M1d = mass_matrix_1d(n)

    print("\nM 3D =")
    print(np.round(M3d, 4))

    print("\nM 1D =")
    print(np.round(M1d, 4))

    diff = np.max(np.abs(M3d - M1d))
    print(f"\nmax difference: {diff:.2e}  {'OK' if diff < 1e-6 else 'FAIL'}")

def main():
    test()

if __name__ == "__main__":
    main()
