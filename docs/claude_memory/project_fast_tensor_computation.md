---
name: project_fast_tensor_computation
description: Table-based collision-tensor prototype (src/collision_tensor_fast.py) — ~20x faster than the production pipeline, validated at n=3/n=4/n=5-shape; n=5 quadrature saved and gated, TACC job (job_n5.sh) ready to submit
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

## n=5 status: quadrature built+saved+re-verified, code smoke-tested at n=5
shape, TACC job written — **full production run not yet submitted**

Quadrature order confirmed via [[project_quadrature_params]]'s n=5 section:
`n_laguerre=11, n_lebedev=17` → **161,051,000 points**. Sparse entry count
(pure combinatorics): **285,750** (`n**3=125` basis triples), vs n=4's
44,480 — entries grew `6.4x`, quadrature points grew `3.28x`, combined
`~21x` more total work than n=4.

**Quadrature built and saved:** `src/quadratures/collision_lag11_leb17.pkl`
(161,051,000 pts, ~11.6 GB, built in 24.4s via the already-vectorized
`collision_quadrature`). Re-ran the l=2/l=3/l=4 conservation cases against
this *saved* file (not just the in-memory sweep from the day before): all 8
cases still machine-zero (`1e-9` to `1e-11`), identical to the sweep —
confirms the saved file is correct and is the final go/no-go gate this
project's convention calls for before spending real compute budget.

**Full n=5 pipeline smoke-tested at real n=5 shape** (125 basis triples,
285,750 entries — untested shapes before today) on a 200,000-point slice of
the real n=5 quadrature (0.12% of the full 161M): `precompute_points`,
`build_basis_tables`, `assemble` all ran without error, no NaN/Inf, sane
output values. This is a shape/crash smoke test only — the *values* aren't
meaningful at this point count, only that the n=5-sized arrays/kernels work.

**Refined time projection** (from this real n=5 slice, not cross-n
extrapolation): 35.5s total for 285,750 entries x 200,000 points → scales
linearly (by design — no data-dependent branching that changes cost with
point count) to `35.5 * (161,051,000/200,000) ≈ 28,590s ≈ 7.9 hours` locally
(10 cores); `≈ 37 minutes` on a 128-core TACC node. Better than both the
initial wrong guess (21-50h, before the real quadrature order was known) and
the cross-n-calibrated guess (11.5h locally / ~1h TACC) from the day before.

**TACC job written:** `job_n5.sh` (repo root) + `src/run_n5_tensor.py`,
modeled structurally on the sibling `numba_landau` project's `job_n6.sh` (a
proven working TACC template for the same kind of numba-`prange`-parallel
workload) — **not a dependency on that project**, fully self-contained to
rel_boltz. `run_n5_tensor.py` builds-or-loads the `(11,17)` quadrature, calls
`compute_tensor_fast_chunked` with `chunk_size=10_000_000` (targets ~40 GB
basis-table memory per chunk, safe on any node >=64 GB RAM — tune upward if
the actual TACC node memory is confirmed larger), and saves to
`results/tensor_n5_lag11_leb17_sparse_fast.pkl` via `collision_tensor.py`'s
own `tensor_name()` naming convention. `job_n5.sh` requests `-N 1 -n 128`,
allocation `DMS23021`, `-p normal`, `-t 2:00:00` (comfortable margin over the
~37min projection), `cd`s into
`/work/09611/rodrigojosegonzalez/ls6/research/rel_boltz/src` — **this exact
path is an assumption by analogy with numba_landau's TACC layout, not
verified; confirm/adjust before submitting.**

**Resolved from the prior open-items list:**
1. The local-machine memory risk (quadrature array + chunked tables
   approaching this 17GB laptop's ceiling) is **moot for the TACC run** — ls6
   `normal`-queue nodes have far more RAM than the ~11.6GB quadrature array
   alone, so `compute_tensor_fast_chunked`'s full-quadrature-in-memory
   loading is not a blocker there. (Still worth fixing eventually for
   portability to smaller machines, but not gating this run.)
2. n=5 quadrature: built, saved, re-verified against the saved file (above).
3. `sparse.py` compatibility: confirmed by reading `sparse.py` directly —
   `load_operator`/`non_zeros`/`check_sparsity`/`simple_index`/`build_sparse`
   all consume exactly the `{'results': [[select, val], ...], 'n', 'n_laguerre', 'n_lebedev'}`
   shape that `compute_tensor_fast[_chunked]` already produces. No code
   changes needed. (Minor, non-blocking: `check_sparsity` prints one line per
   non-zero entry, which will be a lot of output at n=5's entry counts —
   cosmetic only.)

**How to apply:** confirm the `$WORK` path (and email/allocation if they've
changed) in `job_n5.sh`, `git push` from here, pull on TACC, `sbatch job_n5.sh`
from the repo root.
