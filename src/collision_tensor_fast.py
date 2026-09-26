"""
Prototype: table-based collision-tensor assembly.

`collision_tensor.py` computes the tensor with `multiprocessing.Pool`, one
task per (test, f, g) select entry; each task's `operator_numba` call
re-derives, from scratch, at every quadrature point:
  - the kernel value and the post-collision coordinates (depend ONLY on the
    quadrature point, never on the entry's (k,l,m) triples), and
  - the basis-function values for that entry's own (k,l,m) triples (redundant
    across every other entry that happens to share one of those triples).

This module instead precomputes, once per run:
  1. `precompute_points` — kernel value + post-collision coordinates at every
     quadrature point (`N_quad` of them).
  2. `build_basis_tables` — `basis_eval` at every one of the `n**3` distinct
     (k,l,m) triples, at each of the 4 coordinate roles a tensor entry can
     need (pre-p, post-p, pre-q, post-q), i.e. tables of shape (n**3, N_quad).

Assembly (`assemble`) then reduces every tensor entry to pure table lookups +
multiply-accumulate inside a single `@njit(parallel=True)` kernel that
`prange`s over entries — no re-derivation, no multiprocessing/IPC overhead.

Adapted from the same "precompute once, lookup per entry" design used in the
sibling project `numba_landau/{kernel,basis_tables,assemble}.py` for the
Landau collision tensor.

Math must match `integrand_numba.operator_numba` exactly:
    result[e] = sum_i w[i] * kern[i] * phi_test(p_pre)
                * mu_f1*mu_g2 * ( phi_f1(p_post)*phi_g2(q_post)
                                  - phi_f1(p_pre)*phi_g2(q_pre) )
"""
import os
import pickle
import time

import numpy as np
from numba import njit, prange

from quadrature import load_quad
from basis_numba import basis_eval, spher_const, mu_const
from integrand_numba import kernel, postcoll_F, postcoll_G, sph_to_cart
from sparse import ind
from collision_tensor import create_param_iterable


@njit(parallel=True)
def precompute_points(quad):
    """
    quad: (N, 9) array [rp,tp,pp, rq,tq,pq, tw,pw, w].

    Returns (kern, rp_,tp_,pp_, rq_,tq_,pq_), each length N: the kernel value
    and post-collision coordinates at every quadrature point, independent of
    any (k,l,m) triple.
    """
    N = quad.shape[0]
    kern_arr = np.empty(N, dtype=np.float64)
    rp_ = np.empty(N, dtype=np.float64)
    tp_ = np.empty(N, dtype=np.float64)
    pp_ = np.empty(N, dtype=np.float64)
    rq_ = np.empty(N, dtype=np.float64)
    tq_ = np.empty(N, dtype=np.float64)
    pq_ = np.empty(N, dtype=np.float64)

    for i in prange(N):
        rp, tp, pp = quad[i, 0], quad[i, 1], quad[i, 2]
        rq, tq, pq = quad[i, 3], quad[i, 4], quad[i, 5]
        tw, pw = quad[i, 6], quad[i, 7]

        px, py, pz = sph_to_cart(rp, tp, pp)
        qx, qy, qz = sph_to_cart(rq, tq, pq)
        k = kernel(px, py, pz, qx, qy, qz)
        kern_arr[i] = k

        if k == 0.0:
            # matches operator_numba's "continue": the point contributes
            # nothing regardless of the post-collision coordinates, so any
            # placeholder value here is safe.
            rp_[i], tp_[i], pp_[i] = rp, tp, pp
            rq_[i], tq_[i], pq_[i] = rq, tq, pq
            continue

        rp2, tp2, pp2 = postcoll_F(rp, tp, pp, rq, tq, pq, tw, pw)
        rq2, tq2, pq2 = postcoll_G(rp, tp, pp, rq, tq, pq, tw, pw)
        rp_[i], tp_[i], pp_[i] = rp2, tp2, pp2
        rq_[i], tq_[i], pq_[i] = rq2, tq2, pq2

    return kern_arr, rp_, tp_, pp_, rq_, tq_, pq_


def basis_table_index_arrays(n):
    """ks, ls, ms, cs: parallel arrays over the n**3 distinct (k,l,m) triples,
    indexed by sparse.ind(k,l,m,n) so they line up with select-entry indices."""
    n_basis = n ** 3
    ks = np.empty(n_basis, dtype=np.int64)
    ls = np.empty(n_basis, dtype=np.int64)
    ms = np.empty(n_basis, dtype=np.int64)
    cs = np.empty(n_basis, dtype=np.float64)
    for k in range(n):
        for l in range(n):
            for m in range(-l, l + 1):
                idx = ind(k, l, m, n)
                ks[idx] = k
                ls[idx] = l
                ms[idx] = m
                cs[idx] = spher_const(l, m)
    return ks, ls, ms, cs


@njit(parallel=True)
def build_basis_tables(ks, ls, ms, cs,
                        rp, tp, pp, rp_, tp_, pp_,
                        rq, tq, pq, rq_, tq_, pq_):
    """
    Returns (B_p_pre, B_p_post, B_q_pre, B_q_post), each shape (n**3, N_quad):
    basis_eval(k,l,m,c, ...) for every distinct (k,l,m) triple (row), at the
    four coordinate roles a tensor entry can need (column = quadrature point).
    """
    n_basis = ks.shape[0]
    N = rp.shape[0]
    B_p_pre = np.zeros((n_basis, N), dtype=np.float64)
    B_p_post = np.zeros((n_basis, N), dtype=np.float64)
    B_q_pre = np.zeros((n_basis, N), dtype=np.float64)
    B_q_post = np.zeros((n_basis, N), dtype=np.float64)

    for idx in prange(n_basis):
        k, l, m, c = ks[idx], ls[idx], ms[idx], cs[idx]
        for i in range(N):
            B_p_pre[idx, i] = basis_eval(k, l, m, c, rp[i], tp[i], pp[i])
            B_p_post[idx, i] = basis_eval(k, l, m, c, rp_[i], tp_[i], pp_[i])
            B_q_pre[idx, i] = basis_eval(k, l, m, c, rq[i], tq[i], pq[i])
            B_q_post[idx, i] = basis_eval(k, l, m, c, rq_[i], tq_[i], pq_[i])

    return B_p_pre, B_p_post, B_q_pre, B_q_post


@njit(parallel=True)
def assemble(test_idx, f1_idx, f2_idx, mu_prod,
             B_p_pre, B_p_post, B_q_pre, B_q_post,
             w, kern_arr):
    """One prange over select entries; inner loop is pure table lookups."""
    n_entries = test_idx.shape[0]
    N = w.shape[0]
    results = np.zeros(n_entries, dtype=np.float64)

    for e in prange(n_entries):
        ti = test_idx[e]
        fi1 = f1_idx[e]
        fi2 = f2_idx[e]
        mu12 = mu_prod[e]

        acc = 0.0
        for i in range(N):
            wk = w[i] * kern_arr[i]
            if wk == 0.0:
                continue
            gain = B_p_post[fi1, i] * B_q_post[fi2, i]
            loss = B_p_pre[fi1, i] * B_q_pre[fi2, i]
            acc += wk * B_p_pre[ti, i] * mu12 * (gain - loss)
        results[e] = acc

    return results


def compute_tensor_fast(n, quad_path, use_sparsity=True):
    quad, n_lag, n_leb = load_quad(quad_path)
    quad = np.ascontiguousarray(np.asarray(quad, dtype=np.float64))
    N = quad.shape[0]
    print(f"quadrature: {N:,} points (n_laguerre={n_lag}, n_lebedev={n_leb})")

    t0 = time.time()
    kern_arr, rp_, tp_, pp_, rq_, tq_, pq_ = precompute_points(quad)
    t1 = time.time()
    print(f"point precompute (kernel + post-collision coords): {t1 - t0:.2f}s")

    rp, tp, pp = quad[:, 0], quad[:, 1], quad[:, 2]
    rq, tq, pq = quad[:, 3], quad[:, 4], quad[:, 5]
    w = quad[:, 8]

    ks, ls, ms, cs = basis_table_index_arrays(n)
    B_p_pre, B_p_post, B_q_pre, B_q_post = build_basis_tables(
        ks, ls, ms, cs,
        rp, tp, pp, rp_, tp_, pp_,
        rq, tq, pq, rq_, tq_, pq_,
    )
    t2 = time.time()
    print(f"basis table build (n**3={n**3} triples): {t2 - t1:.2f}s")

    selects = create_param_iterable(n, use_sparsity)
    n_e = len(selects)
    test_idx = np.empty(n_e, dtype=np.int64)
    f1_idx = np.empty(n_e, dtype=np.int64)
    f2_idx = np.empty(n_e, dtype=np.int64)
    mu_prod = np.empty(n_e, dtype=np.float64)
    for e, select in enumerate(selects):
        (k, l, m), (k1, l1, m1), (k2, l2, m2) = select
        test_idx[e] = ind(k, l, m, n)
        f1_idx[e] = ind(k1, l1, m1, n)
        f2_idx[e] = ind(k2, l2, m2, n)
        mu_prod[e] = mu_const(k1, l1) * mu_const(k2, l2)

    t3 = time.time()
    values = assemble(test_idx, f1_idx, f2_idx, mu_prod,
                       B_p_pre, B_p_post, B_q_pre, B_q_post,
                       w, kern_arr)
    t4 = time.time()
    print(f"assembly ({n_e} entries): {t4 - t3:.2f}s")
    print(f"TOTAL (excluding quadrature load): {t4 - t0:.2f}s")

    results = [[select, float(v)] for select, v in zip(selects, values)]
    return results


def compute_tensor_fast_chunked(n, quad_path, use_sparsity=True, chunk_size=1_500_000,
                                 checkpoint_path=None):
    """
    Same math as compute_tensor_fast, but the basis tables (n**3, N_quad) are
    never materialized for the full quadrature at once -- only for one
    quadrature chunk of chunk_size points at a time. Partial per-entry
    contributions accumulate across chunks (plain float addition, associative
    up to summation-order roundoff -- same order of error already seen
    between the unchunked prototype and the baseline).

    Table memory per chunk is 4 * n**3 * chunk_size * 8 bytes; pick
    chunk_size to fit comfortably in RAM (n**3 grows fast: 27 at n=3, 64 at
    n=4, 125 at n=5), e.g. via table_memory_budget_chunk_size below.

    checkpoint_path: if given, the running per-entry totals (plus which chunk
    to resume from) are saved after every chunk, and resumed from
    automatically if the file already exists at start and matches this run's
    (n, chunk_size, N_quad, entry count) -- so a run killed partway through
    (walltime limit, preemption, ...) loses at most one in-progress chunk, not
    everything. Only the small totals array is checkpointed; selects/tables
    are cheap to re-derive deterministically on resume, not worth saving.
    Written atomically (write to a temp file, then os.replace) so a kill
    mid-write can't corrupt the checkpoint.
    """
    quad, n_lag, n_leb = load_quad(quad_path)
    quad = np.ascontiguousarray(np.asarray(quad, dtype=np.float64))
    N = quad.shape[0]
    n_chunks = int(np.ceil(N / chunk_size))
    print(f"quadrature: {N:,} points (n_laguerre={n_lag}, n_lebedev={n_leb}), "
          f"{n_chunks} chunks of <= {chunk_size:,}")

    selects = create_param_iterable(n, use_sparsity)
    n_e = len(selects)
    test_idx = np.empty(n_e, dtype=np.int64)
    f1_idx = np.empty(n_e, dtype=np.int64)
    f2_idx = np.empty(n_e, dtype=np.int64)
    mu_prod = np.empty(n_e, dtype=np.float64)
    for e, select in enumerate(selects):
        (k, l, m), (k1, l1, m1), (k2, l2, m2) = select
        test_idx[e] = ind(k, l, m, n)
        f1_idx[e] = ind(k1, l1, m1, n)
        f2_idx[e] = ind(k2, l2, m2, n)
        mu_prod[e] = mu_const(k1, l1) * mu_const(k2, l2)

    ks, ls, ms, cs = basis_table_index_arrays(n)
    table_gb = 4 * (n ** 3) * chunk_size * 8 / 1e9
    print(f"basis table memory per chunk: {table_gb:.2f} GB")

    totals = np.zeros(n_e, dtype=np.float64)
    start_chunk = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        matches = (ckpt.get('n') == n and ckpt.get('chunk_size') == chunk_size
                   and ckpt.get('n_quad') == N and len(ckpt.get('totals', [])) == n_e)
        if matches:
            totals = ckpt['totals']
            start_chunk = ckpt['next_chunk']
            print(f"resuming from checkpoint {checkpoint_path}: "
                  f"chunks 1-{start_chunk}/{n_chunks} already done", flush=True)
        else:
            print(f"checkpoint {checkpoint_path} doesn't match this run's "
                  f"parameters -- ignoring it, starting from chunk 1", flush=True)

    t0 = time.time()
    for c in range(start_chunk, n_chunks):
        lo = c * chunk_size
        hi = min(N, lo + chunk_size)
        qc = quad[lo:hi]

        kern_arr, rp_, tp_, pp_, rq_, tq_, pq_ = precompute_points(qc)
        rp, tp, pp = qc[:, 0], qc[:, 1], qc[:, 2]
        rq, tq, pq = qc[:, 3], qc[:, 4], qc[:, 5]
        w = qc[:, 8]

        B_p_pre, B_p_post, B_q_pre, B_q_post = build_basis_tables(
            ks, ls, ms, cs,
            rp, tp, pp, rp_, tp_, pp_,
            rq, tq, pq, rq_, tq_, pq_,
        )

        chunk_values = assemble(test_idx, f1_idx, f2_idx, mu_prod,
                                 B_p_pre, B_p_post, B_q_pre, B_q_post,
                                 w, kern_arr)
        totals += chunk_values
        print(f"  chunk {c + 1}/{n_chunks}  ({hi:,}/{N:,} points)  "
              f"elapsed: {time.time() - t0:.1f}s", flush=True)

        if checkpoint_path:
            tmp_path = checkpoint_path + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump({'totals': totals, 'next_chunk': c + 1,
                             'n': n, 'chunk_size': chunk_size, 'n_quad': N}, f)
            os.replace(tmp_path, checkpoint_path)

    t1 = time.time()
    print(f"TOTAL assembly across all chunks: {t1 - t0:.2f}s")

    results = [[select, float(v)] for select, v in zip(selects, totals)]
    return results


if __name__ == "__main__":
    from quadrature import quad_name
    n = 3
    n_laguerre = 7
    n_lebedev = 9
    compute_tensor_fast(n, quad_path=quad_name('collision', n_laguerre, n_lebedev))
