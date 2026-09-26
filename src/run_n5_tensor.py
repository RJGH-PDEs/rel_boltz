"""
Production driver for the n=5 collision tensor, at the quadrature order
pinned in docs/claude_memory/project_quadrature_params.md's n=5 section:
n_laguerre=11, n_lebedev=17 (161,051,000 points), via the table-based
collision_tensor_fast.py prototype (see
docs/claude_memory/project_fast_tensor_computation.md).

Meant to be run as a batch job (see ../job_n5.sh), not interactively --
projected wall-clock at 128 workers is ~1 hour (calibrated from the real n=3
and n=4 runs at ~0.9ns per (entry x quadrature-point) pair; entries=285,750,
N_quad=161,051,000).

chunk_size below targets ~40 GB of basis-table memory per chunk (plus the
~11.6 GB quadrature array held for the whole run) -- safe on any node with
>=64 GB RAM. Raise it (fewer, larger chunks -> less per-chunk overhead) if
the actual node memory is confirmed larger, e.g. via a job_n5.sh comment or
`free -h` at job start.
"""
import time

from quadrature import collision_quadrature, load_quad, save_quad, quad_name
from collision_tensor import tensor_name
from collision_tensor_fast import compute_tensor_fast_chunked

import os
import pickle


def get_or_build_quadrature(n_laguerre, n_lebedev):
    path = quad_name('collision', n_laguerre, n_lebedev)
    if os.path.exists(path):
        print(f"quadrature already cached at {path}")
        return path
    print(f"building quadrature (n_laguerre={n_laguerre}, n_lebedev={n_lebedev})...")
    t0 = time.time()
    quad = collision_quadrature(n_laguerre, n_lebedev)
    print(f"built {quad.shape[0]:,} points in {time.time() - t0:.1f}s, {quad.nbytes / 1e9:.2f} GB")
    save_quad(quad, path, n_laguerre, n_lebedev)
    return path


def main():
    n = 5
    n_laguerre = 11
    n_lebedev = 17
    use_sparsity = True
    chunk_size = 10_000_000  # ~40 GB basis-table memory per chunk; see module docstring

    quad_path = get_or_build_quadrature(n_laguerre, n_lebedev)

    t0 = time.time()
    results = compute_tensor_fast_chunked(
        n, quad_path=quad_path, use_sparsity=use_sparsity, chunk_size=chunk_size,
    )
    elapsed = time.time() - t0
    print(f"WALL TIME: {elapsed:.2f}s")

    out_path = tensor_name(n, n_laguerre, n_lebedev, use_sparsity, tag='_fast')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'results': results, 'n': n,
            'n_laguerre': n_laguerre, 'n_lebedev': n_lebedev,
            'use_sparsity': use_sparsity, 'wall_time': elapsed,
        }, f)
    print(f"saved to {out_path}  ({len(results)} entries)")


if __name__ == "__main__":
    main()
