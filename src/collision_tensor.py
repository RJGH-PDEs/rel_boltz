import multiprocessing
import os
import pickle
import time

from boltzmann import operator_parallel_numba as operator_parallel, init_worker
from sparse_rules import andrea, cai
from quadrature import quad_name


# Build the flat list of all index triples (select) to compute.
# Each entry is [[k,l,m], [k1,l1,m1], [k2,l2,m2]] where:
#   [k,  l,  m ]: test function phi
#   [k1, l1, m1]: trial function f_tilde (first argument of Q)
#   [k2, l2, m2]: trial function g_tilde (second argument of Q)
# k, l run over [0, n); m runs over [-l, l].
# No sparsity skipping — all entries included so we can inspect the full pattern.
def create_param_iterable(n, use_sparsity=True):
    param = []
    for k in range(n):
        for l in range(n):
            for m in range(-l, l+1):
                for k1 in range(n):
                    for l1 in range(n):
                        for m1 in range(-l1, l1+1):
                            for k2 in range(n):
                                for l2 in range(n):
                                    for m2 in range(-l2, l2+1):
                                        select = [[k,l,m],[k1,l1,m1],[k2,l2,m2]]

                                        # Sparsity rules — skip known zeros to avoid
                                        # computing entries that vanish by symmetry.
                                        if use_sparsity and not (andrea(select) and cai(select)):
                                            continue

                                        param.append(select)
    print(f"total entries to compute: {len(param)}  (sparsity={'on' if use_sparsity else 'off'})")
    return param



# Compute the full collision tensor Q_{i,jk} in parallel.
# Each worker loads the quadrature independently to avoid serialization overhead.
# Results are saved as a list of [select, value].
def tensor_name(n, n_laguerre, n_lebedev, use_sparsity):
    sp = 'sparse' if use_sparsity else 'dense'
    return f'./results/tensor_n{n}_lag{n_laguerre}_leb{n_lebedev}_{sp}.pkl'

def compute_tensor(n, quad_path, use_sparsity=True):
    from quadrature import load_quad
    _, n_laguerre, n_lebedev = load_quad(quad_path)
    out_path = tensor_name(n, n_laguerre, n_lebedev, use_sparsity)
    params = create_param_iterable(n, use_sparsity)
    total  = len(params)

    print(f"workers: {multiprocessing.cpu_count()}")

    start = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(),
                              initializer=init_worker,
                              initargs=(quad_path,)) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(operator_parallel, params)):
            results.append(result)
            print(f"  {i+1}/{total}  elapsed: {time.time()-start:.1f}s", flush=True)

    elapsed = time.time() - start
    print(f"elapsed: {elapsed:.2f}s")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'results': results, 'n': n,
                     'n_laguerre': n_laguerre, 'n_lebedev': n_lebedev,
                     'use_sparsity': use_sparsity}, f)
    print(f"saved to {out_path}  ({len(results)} entries)")

    return results


if __name__ == "__main__":
    quad_path = quad_name('collision', 7, 7)
    compute_tensor(n=2, quad_path=quad_path, use_sparsity=False)
