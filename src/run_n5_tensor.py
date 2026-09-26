"""
Production driver for the n=5 collision tensor, at the quadrature order
pinned in docs/claude_memory/project_quadrature_params.md's n=5 section:
n_laguerre=11, n_lebedev=17 (161,051,000 points), via the table-based
collision_tensor_fast.py prototype (see
docs/claude_memory/project_fast_tensor_computation.md).

Meant to be run as a batch job (see ../job_n5.sh), not interactively.

**Measured, not projected, timing (from an actual idev run on ls6, 2026-09-26):**
chunk 1/17 (chunk_size=10,000,000) took 874.8s. Extrapolated: 17 * 874.8s =~
4.1 hours total -- much worse than an earlier cross-machine projection of
~1 hour, because per-core throughput on this node turned out to be roughly
~6.6x slower than the M4 laptop this was validated on for this specific
workload (building/reducing over 40GB of tables with 128-way parallelism is
memory-bandwidth-heavy, which doesn't scale with core count the way
compute-bound work does) -- not just "~2x slower cores" as first guessed.
job_n5.sh's walltime has been bumped accordingly; see
docs/claude_memory/project_fast_tensor_computation.md for the full story.

chunk_size below targets ~40 GB of basis-table memory per chunk (plus the
~11.6 GB quadrature array held for the whole run) -- safe on any node with
>=64 GB RAM. A *smaller* chunk_size is also worth trying if a future run has
time to experiment -- it may parallelize better if this workload really is
memory-bandwidth-bound (more, smaller chunks means less bandwidth
contention per chunk, at the cost of more per-chunk overhead) -- this has
not yet been tested.

checkpoint_path is set below so a killed/timed-out run only loses at most
one in-progress chunk's work, not everything -- rerunning this script picks
up automatically from the last completed chunk.
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

    os.makedirs('results', exist_ok=True)  # checkpoint_path below needs this to exist up front
    quad_path = get_or_build_quadrature(n_laguerre, n_lebedev)
    checkpoint_path = f'results/n{n}_lag{n_laguerre}_leb{n_lebedev}_checkpoint.pkl'

    t0 = time.time()
    results = compute_tensor_fast_chunked(
        n, quad_path=quad_path, use_sparsity=use_sparsity, chunk_size=chunk_size,
        checkpoint_path=checkpoint_path,
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
