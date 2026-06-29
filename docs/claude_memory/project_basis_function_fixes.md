---
name: project_basis_function_fixes
description: assoc_legendre pole-collapse bug fix and basis_eval/f_tilde_eval deduplication
metadata: 
  node_type: memory
  type: project
  originSessionId: 12cac65b-881c-4cee-9420-9f57a1d42b0f
---

Two fixes applied to `src/basis_numba.py` / `src/integrand_numba.py`:

1. **`assoc_legendre` pole-collapse bug**: it derived `sin(theta)` internally via
   `sqrt(1 - cos(theta)^2)`, which collapses to exactly `0.0` for `theta` within
   `~1.5e-8` rad of a pole due to catastrophic cancellation (`cos(theta)` rounds to
   exactly `±1.0` there), silently returning `0.0` for `m≠0` even though the true
   value is small but nonzero. Fixed by passing `sin(theta)` directly as an
   explicit argument (computed via `sin(t)`, never derived from `cos(t)`).
   Verified against `mpmath` (50-digit precision) to match to machine epsilon
   across the full angular range after the fix; also discovered `scipy.special.lpmv`
   has the identical bug internally (it gave the wrong answer in the same regime).

2. **Duplicate `basis_eval`/`f_tilde_eval`**: `integrand_numba.py` had its own
   copy of these functions instead of importing from `basis_numba.py`, which is
   why the pole bug had to be patched in two places. Removed the duplicate;
   `integrand_numba.py` now imports both functions from `basis_numba.py`.

**Impact on existing results: none.** Checked that no quadrature node in our
actual `n_lebedev=7` or `n_lebedev=9` grids falls within the dangerous pole band
(closest was `~3.4e-4` rad away, ~4 orders of magnitude outside it). Recomputed
the n=3 collision tensor (dense and sparse) after both fixes and compared
entry-by-entry against the pre-fix tensor: all physically significant entries
(magnitude > 0.1) agree to `~1e-13` relative precision; all flagged "mismatches"
are noise-level (`<1e-5` magnitude) differences in entries that are analytically
zero by the Cai/Andrea sparsity rules anyway. See [[project-time-evolution]] for
the re-validated time-evolution run using the regenerated tensor.

**How to apply:** no action needed on prior n=3 results — they're confirmed
correct. Worth re-checking quadrature-node proximity to poles if `n_lebedev` is
ever increased significantly, or if evaluation points stop being tied to a fixed
quadrature grid (e.g. in the Landau-equation project, which uses arbitrary
gradient evaluation points rather than fixed Lebedev nodes).
