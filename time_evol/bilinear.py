import sys
import numpy as np
import pickle
sys.path.insert(0, '../src')

from sparse import sparse_name


def load_sparse(n, n_laguerre, n_lebedev):
    path = '../src/' + sparse_name(n, n_laguerre, n_lebedev)
    with open(path, 'rb') as f:
        return pickle.load(f)


# The bilinear Boltzmann collision operator Q(f, f).
# tensor: list of sparse matrices, one per test function (from sparse.py)
# f:      coefficient vector of the distribution function
# result: output vector — result[i] = f^T @ tensor[i] @ f
def boltzmann(tensor, f, result):
    for i, mat in enumerate(tensor):
        result[i] = f @ mat.dot(f)


def fmt(result, tol=1e-3):
    return np.where(np.abs(result) < tol, 0.0, result)


def test(n=3, n_laguerre=7, n_lebedev=9):
    so     = load_sparse(n, n_laguerre, n_lebedev)
    size   = n**3
    result = np.zeros(size)

    for trial in range(5):
        f = np.random.uniform(-1, 1, size)
        boltzmann(so, f, result)
        print(f"trial {trial+1}  f: {np.round(f, 2)}")
        print(f"          Q(f,f): {fmt(result)}")
        print()


def test_specific(n=3, n_laguerre=7, n_lebedev=9):
    so     = load_sparse(n, n_laguerre, n_lebedev)
    size   = n**3
    result = np.zeros(size)

    f    = np.zeros(size)
    f[0] = 1.0
    boltzmann(so, f, result)
    print("--- Juttner equilibrium (f[0]=1, rest 0) ---")
    print(f"Q(f,f): {fmt(result)}")
    print()


def main():
    n, n_laguerre, n_lebedev = 3, 7, 9
    test(n, n_laguerre, n_lebedev)
    print()
    test_specific(n, n_laguerre, n_lebedev)


if __name__ == "__main__":
    main()
