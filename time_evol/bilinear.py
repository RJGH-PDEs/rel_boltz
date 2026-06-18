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


def test_specific():
    n = 2
    size = n**3

    with open('../src/sparse_operators/collision_tensor.pkl', 'rb') as file:
        so = pickle.load(file)

    result = np.zeros(size)

    # index reminder for n=2:
    # 0=[0,0,0]  1=[0,1,-1]  2=[0,1,0]  3=[0,1,1]
    # 4=[1,0,0]  5=[1,1,-1]  6=[1,1,0]  7=[1,1,1]

    candidates = [
        ("equilibrium + l=1 perturbation",     [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("equilibrium + k=1,l=0 perturbation", [1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        ("equilibrium + k=1,l=1 perturbation", [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        ("mix l=0 and l=1 modes",              [1.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0]),
        ("all modes equally perturbed",        [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ]

    for name, coeffs in candidates:
        f = np.array(coeffs)
        boltzmann(so, f, result)
        print(f"--- {name} ---")
        print(f"Q(f,f): {result}")
        print()

def main():
    test()
    print()
    test_specific()


if __name__ == "__main__":
    main()
