---
name: project_quadrature_build_performance
description: collision_quadrature() is impractically slow/memory-heavy at n=4 scale; needs a numpy-vectorized rewrite
metadata: 
  node_type: memory
  type: project
  originSessionId: 12cac65b-881c-4cee-9420-9f57a1d42b0f
---

`collision_quadrature()` in `src/quadrature.py` builds the quadrature with a
5-fold nested **pure Python loop** (`rp, ep, rq, eq, ew`), appending a 9-element
Python list per point. For `n=4`'s chosen parameters (`n_laguerre=11,
n_lebedev=13`), that's `11² × 74³ ≈ 49 million` iterations — this hung for
6+ minutes and climbed past ~3GB of memory with no end in sight before being
killed. See [[project_quadrature_params]] for how `(11, 13)` was determined.

**Why it's slow/memory-heavy (two symptoms, one root cause):**
- **Speed**: 49M Python-interpreter-level loop iterations — same "pure Python
  loop" regime benchmarked earlier in the project at ~150x slower than
  compiled/vectorized code (see basis_numba.py history — same lesson, applied
  to a different piece of code that never got the numba/vectorization
  treatment).
- **Memory**: each `quad.append([...])` creates a Python list of 9 Python
  float objects — ~400 bytes/row of actual heap overhead instead of the
  `9 floats × 8 bytes = 72 bytes/row` a packed array would need. At 49M rows
  that's ~19GB instead of the ~3.5GB the data actually needs.

**Suggested fix (not yet implemented):** rewrite `collision_quadrature()` to
build the Cartesian product of the 5 independent factors (`rp, ep, rq, eq, ew`)
directly as numpy arrays via broadcasting/`repeat`/`tile` (or `np.meshgrid`),
so the "loop" happens in numpy's C internals rather than the Python
interpreter. Output should be flat `float64` numpy arrays (one per column)
at the honest ~3.5GB total, built in seconds rather than many minutes.

**Workaround used for now:** decreased radial quadrature order is acceptable
as an interim approximation — accept a bit of imprecision on n_laguerre for
the first n=4 attempt, revisit precision later once the vectorized build
exists. (Recall from [[project_quadrature_params]]: n_laguerre=7 is NOT fully
converged for n=4's worst-case entries, off by ~8%; n_laguerre=11 is the
verified-converged choice, but a lower value may be used pragmatically for an
initial pass.)

**How to apply:** before attempting a full n=4 tensor run, either (a) write
the vectorized quadrature builder, or (b) accept a smaller n_laguerre (with
documented imprecision) as a stopgap. Don't reattempt the current
nested-Python-loop builder at n=4 scale — it will not finish in reasonable
time/memory.
