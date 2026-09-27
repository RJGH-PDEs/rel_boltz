# Memory Index

- [Quadrature parameters for n=3, n=4, n=5](project_quadrature_params.md) — n_lebedev minimum: 9/13/17, n_laguerre: 7/11/11 (verified via conservation tests)
- [Initial condition for time evolution](project_initial_condition.md) — "hot" radial, coeff[0]=2.0, coeff[9]=-0.8, all others zero
- [Time evolution runs](project_time_evolution.md) — stable run at t0=1.0, dt=1e-3; stability rule dt ∝ t0^{3/2}
- [Basis function bug fixes](project_basis_function_fixes.md) — assoc_legendre pole bug + basis_eval dedup, no impact on n=3 results
- [Quadrature build performance](project_quadrature_build_performance.md) — collision_quadrature() FIXED, now numpy-vectorized (~2.4s at n=4 scale)
- [Fast table-based tensor computation](project_fast_tensor_computation.md) — src/collision_tensor_fast.py, ~20x faster; n=3/n=4/n=5 tensors computed, sparsified, and rigorously validated (n=5 on TACC, 4.38h) via full-slice antisymmetry check on momentum/energy conservation matrices
