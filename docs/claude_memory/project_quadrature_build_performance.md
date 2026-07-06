---
name: project_quadrature_build_performance
description: collision_quadrature() was rewritten to numpy repeat/tile — n=4 scale (49M pts) now builds in ~2.4s at ~3.5GB RAM
metadata: 
  node_type: memory
  type: project
  originSessionId: 12cac65b-881c-4cee-9420-9f57a1d42b0f
---

`collision_quadrature()` in `src/quadrature.py` was rewritten from a 5-level
pure-Python nested loop to a numpy-vectorized implementation using
`np.repeat`/`np.tile`. The old loop was unusable at n=4 scale
(`n_laguerre=11, n_lebedev=13`, ~49M pts): >6 min and >3 GB with no end in
sight. See [[project_quadrature_params]] for how `(11,13)` was determined.

**Fix implemented:** `collision_quadrature()` now builds the Cartesian product
via `_cartesian_factor(values, axis, sizes)` (repeat/tile pattern) — O(N) time
and memory, no Python-level loop. At `(11,13)`: builds in ~2.4s, ~3.3 GB.
The return value is a `(N, 9) float64` numpy array (was a list of lists);
callers that wrapped with `np.array()` are unaffected.

**Verification (four levels, all pass):**
1. Analytical weight sum matches `256*(4π)³` to float64 precision.
2. Per-column point-wise diff against original (7,9) file: exactly 0 on all 9 columns.
3. 18-entry operator sweep (varying k_i, l_i, k_s, k_t, m): all match to 0.00e+00 rel diff.
4. Conservation entries (mass/energy, l≤2): ~1e-10 as expected, old/new agree exactly.

`src/quadrature_np.py` remains as a standalone verification script containing
`verify_full()` (the four-level suite above) and the (11,13) rebuild
infrastructure. It is no longer needed as a replacement — the vectorized
implementation now lives directly in `quadrature.py`.

**How to apply:** `collision_quadrature()` is now ready for n=4 use.
Run `src/quadrature_np.py` (from `src/`) any time the quadrature file
needs to be rebuilt or re-verified.
