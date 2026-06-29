# Memory Index

- [Quadrature parameters for n=3 and n=4](project_quadrature_params.md) — n_lebedev minimum: 9 for n=3, 13 for n=4 (verified via conservation tests)
- [Initial condition for time evolution](project_initial_condition.md) — "hot" radial, coeff[0]=2.0, coeff[9]=-0.8, all others zero
- [Time evolution runs](project_time_evolution.md) — stable run at t0=1.0, dt=1e-3; stability rule dt ∝ t0^{3/2}
- [Basis function bug fixes](project_basis_function_fixes.md) — assoc_legendre pole bug + basis_eval dedup, no impact on n=3 results
- [Quadrature build performance](project_quadrature_build_performance.md) — collision_quadrature() pure-Python loop too slow/memory-heavy at n=4 scale, needs numpy vectorization
