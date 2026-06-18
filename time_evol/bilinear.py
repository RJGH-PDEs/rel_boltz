import numpy as np
import pickle


# The bilinear Boltzmann collision operator Q(f, f).
# tensor: list of sparse matrices, one per test function (from sparse.py)
# f:      coefficient vector of the distribution function
# result: output vector — result[i] = f^T @ tensor[i] @ f
def boltzmann(tensor, f, result):
    for i, mat in enumerate(tensor):
        result[i] = f @ mat.dot(f)


# Copy vector b into a
def update(a, b):
    for i, val in enumerate(b):
        a[i] = val


def test():
    n = 2
    size = n**3  # number of basis functions

    # load sparse collision tensor
    with open('../src/sparse_operators/collision_tensor.pkl', 'rb') as file:
        so = pickle.load(file)

    result = np.zeros(size)

    # evaluate Q(f, f) for random coefficient vectors in [-1, 1]
    for trial in range(5):
        f = np.random.uniform(-1, 1, size)
        boltzmann(so, f, result)
        print(f"trial {trial+1}  f: {np.round(f, 2)}")
        print(f"          Q(f,f): {result}")
        print()


def main():
    test()


if __name__ == "__main__":
    main()
