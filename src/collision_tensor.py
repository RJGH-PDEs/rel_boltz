import multiprocessing
import os
import pickle
import time
from functools import partial

from quadrature import load_quad
from boltzmann import operator_parallel


# Build the flat list of all index triples (select) to compute.
# Each entry is [[k,l,m], [k1,l1,m1], [k2,l2,m2]] where:
#   [k,  l,  m ]: test function phi
#   [k1, l1, m1]: trial function f_tilde (first argument of Q)
#   [k2, l2, m2]: trial function g_tilde (second argument of Q)
# k, l run over [0, n); m runs over [-l, l].
# No sparsity skipping — all entries included so we can inspect the full pattern.
def create_param_iterable(n):
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
                                        # Uncomment when dense verification is complete.
                                        # from sparse_rules import andrea, cai
                                        # if not (andrea(select) and cai(select)):
                                        #     continue

                                        param.append(select)
    print(f"total entries to compute: {len(param)}")
    return param


# Single-argument worker for imap_unordered (quad is fixed via partial)
def _worker(select, quad):
    return operator_parallel(select, quad)


# Compute the full collision tensor Q_{i,jk} in parallel.
# The quadrature is loaded once in the parent process and passed to all workers
# to avoid redundant disk reads. Results are saved as a list of [select, value].
def compute_tensor(n, quad_path='./quadratures/collision.pkl',
                   out_path='./results/collision_tensor.pkl'):
    # load quadrature once — shared across all workers
    quad   = load_quad(quad_path)
    params = create_param_iterable(n)

    print(f"quadrature points: {len(quad)}")
    print(f"workers: {multiprocessing.cpu_count()}")

    worker = partial(_worker, quad=quad)
    total  = len(params)
    mile   = max(1, total // 100)

    start = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(worker, params)):
            results.append(result)
            if (i + 1) % mile == 0:
                pct = 100 * (i + 1) / total
                print(f"  {pct:.0f}%  ({i+1}/{total})  elapsed: {time.time()-start:.1f}s")

    elapsed = time.time() - start
    print(f"elapsed: {elapsed:.2f}s")

    # save as list of [select, value] for later sparsity analysis
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"saved to {out_path}  ({len(results)} entries)")

    return results


if __name__ == "__main__":
    compute_tensor(n=2)
