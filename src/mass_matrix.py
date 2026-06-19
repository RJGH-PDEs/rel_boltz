import os
import pickle
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre

from basis_numba import mu_const
from quadrature import load_quad, quad_name
from sparse import ind


def mass_matrix_1d(n, n_laguerre=15):
    """
    Exploits spherical harmonic orthogonality: M[i,j] = 0 unless l_i==l_j and m_i==m_j.
    Only the radial Laguerre integral needs to be computed numerically.
    """
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

                            if li != lj or mi != mj:
                                continue

                            alpha  = 2*(li + 1)
                            x, w   = roots_genlaguerre(n_laguerre, alpha)
                            factor = mu_const(kj, li) * 2**(2*li + 3)
                            M[i, j] = factor * np.sum(
                                w * eval_genlaguerre(ki, alpha, 2*x)
                                  * eval_genlaguerre(kj, alpha, 2*x)
                            )
    return M


def mass_name(n, n_laguerre, n_lebedev):
    return f'./mass/mass_n{n}_lag{n_laguerre}_leb{n_lebedev}.pkl'

def save_mass(M, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(M, f)
    print(f"mass matrix saved to {path}")

def load_mass(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def generate(n, quad_path):
    _, n_laguerre, n_lebedev = load_quad(quad_path)
    print(f"computing mass matrix (n={n})...")
    M = mass_matrix_1d(n)
    save_mass(M, mass_name(n, n_laguerre, n_lebedev))
    return M


if __name__ == "__main__":
    quad_path = quad_name('mass', 7, 7)
    generate(n=2, quad_path=quad_path)
