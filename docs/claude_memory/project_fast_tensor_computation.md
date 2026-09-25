---
name: project_fast_tensor_computation
description: Table-based collision-tensor prototype (src/collision_tensor_fast.py) — ~20x faster than the production pipeline, validated at n=3/n=4, n=5 quadrature order pinned but full n=5 run not yet attempted
metadata:
  node_type: memory
  type: project
---

## Where this idea came from

Sibling project `/Users/rjgh/Documents/Research/Projects/numba_landau` (a
numba reimplementation of the *Landau* collision tensor, not this project's
Boltzmann one) uses a "precompute once, lookup per entry" design: a kernel
table indexed only by quadrature node (`kernel.py:build_u_kernel_table`) and a
basis-value table indexed by `(k,l,m)` triple, not by tensor entry
(`basis_tables.py:build_basis_tables`), so the final tensor assembly
(`assemble.py:assemble_tensor`) is pure table lookups inside one
`@njit(parallel=True)` kernel. `rel_landau` (the sympy-based project
`numba_landau` was built to replace) is *not* where the speed trick lives —
it's actually slower than this project's existing numba pipeline (bare Python
loop, symbolic `lambdify` per entry, no vectorization at all).

## The waste this fixes in `src/collision_tensor.py` / `integrand_numba.py`

`collision_tensor.py` computes the tensor via `multiprocessing.Pool`, one task
per `(test, f, g)` select entry. Each task's `operator_numba` call
re-derives, from scratch, at every quadrature point:
- `kernel()` and `postcoll_F`/`postcoll_G` — depend **only on the quadrature
  point**, never on the entry's `(k,l,m)` triples, yet get recomputed
  identically for every one of the (thousands to hundreds of thousands of)
  entries.
- `f_tilde_eval`/`basis_eval` at a given `(k,l,m)` — recomputed independently
  by every entry that happens to share that triple (many do, since entries
  are built from combinations of only `n**3` distinct triples).

## The fix: `src/collision_tensor_fast.py` (prototype, not yet wired into the
production pipeline stage 3 in place of `collision_tensor.py`)

1. `precompute_points(quad)` — one `@njit(parallel=True)` pass over all
   `N_quad` points: kernel value + post-collision coordinates. Because this
   project's quadrature is already one flat `N_quad`-length list (unlike
   `numba_landau`'s `(p,u)` outer-product scheme, where `q=p-u` is a genuinely
   new point per pair), the post-collision points can be tabled at the same
   cost as the pre-collision ones — no "evaluate fresh per entry" fallback
   needed, unlike the Landau version.
2. `build_basis_tables(...)` — `basis_eval(k,l,m,c,...)` for every one of the
   `n**3` distinct triples (not per entry), at the 4 coordinate roles a tensor
   entry can need: pre-p, post-p, pre-q, post-q. Shape `(n**3, N_quad)` each.
3. `assemble(...)` — single `@njit(parallel=True)` kernel, `prange` over
   select entries, inner loop is pure table lookups + multiply-accumulate:
   `result[e] = sum_i w[i]*kern[i]*B_p_pre[test_idx,i]*mu1*mu2*(B_p_post[f1_idx,i]*B_q_post[f2_idx,i] - B_p_pre[f1_idx,i]*B_q_pre[f2_idx,i])`.

**Memory forced chunking:** materializing all 4 basis tables for the full
quadrature at once costs `4 * n**3 * N_quad * 8` bytes — **~100 GB at n=4**
(64 triples x 49M points), far beyond this machine's 17 GB RAM, even though
n=3 (27 triples x 2.7M points, ~2.3 GB) fit fine unchunked. `compute_tensor_fast_chunked(n, quad_path, chunk_size)` processes the
quadrature in blocks (table memory = `4*n**3*chunk_size*8` bytes, tunable),
accumulating each chunk's partial per-entry sum into a running total — same
total FLOP count as the unchunked version, just memory-bounded. This is what
made n=4 (previously TACC-only) run locally at all.

## Validation (both against real, previously-computed tensors — not just
self-consistency)

**n=3** (`results/tensor_n3_lag7_leb9_sparse_baseline.pkl` — a fresh rerun of
the existing `collision_tensor.py` — vs `..._fast.pkl`): 4185/4185 entries
match. Max absolute difference `2.1e-6` on a value of order `1.6e9` (relative
`~1e-15`, plain summation-order floating-point noise). The only large
*relative* differences are confined to entries of magnitude `~1e-8` — already
at the quadrature/sparsity noise floor CLAUDE.md's own conservation tests
treat as "zero".

**n=4** (existing TACC-computed `results/tensor_n4_lag11_leb13_sparse.pkl` vs
the new chunked run's `..._fast.pkl`): 44480/44480 entries match. Restricting
to the 19,241 entries with `|value| > 1.0` (i.e. excluding entries that are
themselves near-zero quadrature noise — median entry magnitude across all
44,480 is only `2.8e-6`, so more than half the sparsity-surviving entries are
numerically negligible once actually computed): **max relative difference
1.0e-11** — floating-point roundoff, full stop.

## Performance measured (this machine: 10 cores, 17 GB RAM)

| n | entries (sparse) | N_quad | baseline (`collision_tensor.py`) | fast/chunked |
|---|---|---|---|---|
| 3 | 4,185 | 2,688,728 | 358.75s (10-way multiprocessing) | 17.9s (unchunked) / 12.6s (chunked, `chunk_size=500k`) |
| 4 | 44,480 | 49,032,104 | not attempted locally (TACC-scale) | **2039.3s (~34 min)**, `chunk_size=1e6` (~2.05 GB/chunk) |

Calibrated rate from both real data points: **~0.9 ns per (entry x
quadrature-point) pair** on this 10-core machine (`total_time ≈ entries *
N_quad * 9e-10` seconds) — used for the n=5 projection below.

## n=5 status: quadrature order pinned, full tensor run **not yet attempted**

Quadrature order confirmed via [[project_quadrature_params]]'s n=5 section:
`n_laguerre=11, n_lebedev=17` → **161,051,000 points**. Sparse entry count
(pure combinatorics, already checked): **285,750** (`n**3=125` basis
triples), vs n=4's 44,480 — entries grew `6.4x`, quadrature points grew
`3.28x`, combined **~21x more total work than n=4**.

**Time projection** (using the calibrated 0.9ns/pair rate — NOT yet an actual
measured run): `285,750 * 161,051,000 * 9e-10 ≈ 41,400s ≈ 11.5 hours`
locally on this 10-core machine; `≈ 1 hour` on a ~128-core TACC node. Much
better than a first (wrong) speculative guess of 21-50 hours made before the
real n=5 quadrature order was known (that guess assumed both n_laguerre and
n_lebedev would need to grow further; n_laguerre in fact did not).

**Open items before attempting the full n=5 run:**
1. **Memory risk not yet resolved:** `compute_tensor_fast_chunked` currently
   loads the *entire* quadrature into memory (via `load_quad` +
   `np.asarray`) before slicing it into chunks for the table-build step. At
   n=5 that quadrature array alone is `~11.6 GB`; adding a few GB of
   per-chunk basis tables on top pushes close to this machine's 17 GB
   ceiling. Needs the quadrature *generation/loading* itself chunked (or
   built directly per-chunk from `collision_quadrature`'s node/weight
   factors, never materializing the full `(161M, 9)` array), not just the
   basis-table step, before a real n=5 run is attempted.
2. The n=5 quadrature has not yet been built/saved to
   `src/quadratures/collision_lag11_leb17.pkl` — only individual sweep
   evaluations were run so far (not saved to disk).
3. `collision_tensor_fast.py` is a validated **prototype**, not wired into
   the production pipeline in place of `src/collision_tensor.py` — stage 4
   (`src/sparse.py`) and downstream stages still expect
   `collision_tensor.py`'s output format (`{'results': [...], 'n', 'n_laguerre', 'n_lebedev', 'use_sparsity', 'tag'}`, which `compute_tensor_fast[_chunked]`'s
   return value already matches structurally, but this hasn't been exercised
   end-to-end through `sparse.py` yet).

**How to apply:** before launching an 11.5-hour local (or TACC) n=5 run,
resolve open item 1 (chunk the quadrature load itself), then build/save the
`(11,17)` quadrature once, then re-run `src/tests.py`'s conservation sweep
against the *saved* quadrature file as a final sanity check before spending
the compute budget on the full tensor.
