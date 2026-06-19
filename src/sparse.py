import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix


# ── Sparsity rules ───────────────────────────────────────────────────────────

def cai(select):
    m, m1, m2 = select[0][2], select[1][2], select[2][2]
    return (abs(m) == abs(m1 + m2)) or (abs(m) == abs(m1 - m2))

def andrea(select):
    l, l1, l2 = select[0][1], select[1][1], select[2][1]
    rule = l1 + l2 - l
    return (0 <= rule <= 2*min(l1, l2)) and (rule % 2 == 0)


# ── Index maps ───────────────────────────────────────────────────────────────

def lm_index(l, m):
    return l*l + (m + l)

def ind(k, l, m, n):
    return (n*n)*k + lm_index(l, m)


# ── I/O ──────────────────────────────────────────────────────────────────────

def sparse_name(n, n_laguerre, n_lebedev):
    return f'sparse_operators/sparse_n{n}_lag{n_laguerre}_leb{n_lebedev}.pkl'

def load_operator(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['n'], data['n_laguerre'], data['n_lebedev']

def save_sparse_op(path, operator):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(operator, f)
    print(f"saved sparse operator to {path}")


# ── Pipeline steps ───────────────────────────────────────────────────────────

def non_zeros(operator, tol):
    nz = [entry for entry in operator if np.abs(entry[1]) > tol]
    print(f"non-zeros: {len(nz)}")
    return nz

def check_sparsity(nz):
    failures = 0
    for entry in nz:
        t, f1, f2 = entry[0]
        val = entry[1]

        m_t, m_1, m_2 = t[2], f1[2], f2[2]
        cai = (abs(m_t) == abs(m_1 + m_2)) or (abs(m_t) == abs(m_1 - m_2))

        l_t, l_1, l_2 = t[1], f1[1], f2[1]
        rule = l_1 + l_2 - l_t
        andrea = (0 <= rule <= 2*min(l_1, l_2)) and (rule % 2 == 0)

        if not (cai and andrea):
            failures += 1

        print(f"  {[t, f1, f2]}  val={val:.6e}  Cai={cai}  Andrea={andrea}")

    print(f"sparsity rule failures: {failures}")

def simple_index(nz, n):
    return [[ind(*t, n), ind(*f, n), ind(*g, n), val]
            for (t, f, g), val in nz]

def build_sparse(si, n):
    size  = n**3
    dense = [np.zeros((size, size)) for _ in range(size)]
    for t, f, g, val in si:
        dense[t][f][g] = val
    sparse = [csr_matrix(m) for m in dense]
    print(f"sparse matrices: {len(sparse)}  nnz per matrix: {[m.nnz for m in sparse]}")
    return sparse


# ── Full pipeline ─────────────────────────────────────────────────────────────

def analyze(tensor_path, tol=1e-1):
    op, n, n_laguerre, n_lebedev = load_operator(tensor_path)

    nz = non_zeros(op, tol)
    check_sparsity(nz)
    si = simple_index(nz, n)
    so = build_sparse(si, n)

    save_sparse_op(sparse_name(n, n_laguerre, n_lebedev), so)


if __name__ == "__main__":
    from collision_tensor import tensor_name
    analyze(tensor_name(2, 7, 7, use_sparsity=False))
