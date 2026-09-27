---
name: project-time-evolution
description: Successful time evolution runs and stability parameters; n=3/n=4 and now n=5 (radial + new k=4 mode, validated against the analytical Jüttner equilibrium)
metadata: 
  node_type: memory
  type: project
  originSessionId: 12cac65b-881c-4cee-9420-9f57a1d42b0f
---

## Successful run: t0=1.0

- `t0=1.0`, `dt=1e-3`, `NUM_ITERATIONS=10000`, `save_every=100`
- Initial condition: hot radial, `f[0]=2.0, f[9]=-0.8` (see [[project-initial-condition]])
- Stable: `|f|` converges to 2.410626 by ~iter 100 (t≈1.1), stays flat after
- Final state: `f[0]=2.378, f[9]=-0.363, f[18]=0.154` — Jüttner equilibrium in our basis
- `Q(f,f)` at final state: ~1e-9 (machine zero) ✓
- Plot shows two symmetric blobs merging into equilibrium peak — visually confirmed thermalization

**Why:** At t0=1.0 the prefactor `0.5 * t0^{-3/2} = 0.5`, effective step `dt * prefactor = 5e-4` — well within Euler stability.

**Stability rule:** `dt ∝ t0^{3/2}`. To go to t0=0.1 need dt~3e-5, to t0=0.01 need dt~1e-6.

## Successful run: hot + dipole perturbation, t0=1.0

Same `t0=1.0, dt=1e-3, NUM_ITERATIONS=10000, save_every=100`, but IC adds a small
l=1 dipole on top of the hot radial IC:

```python
f[0] = 2.0    # ind(0,0,0,3)
f[9] = -0.8   # ind(1,0,0,3)
f[2] = 0.1    # ind(0,1,0,3)  dipole perturbation
```

- Stable, converges by ~iter 100 as before
- Dipole mode does **not** decay to zero — locks at `f[2]≈0.0578` (down from 0.1, but plateaus)
- `(M f)[2]` (the actual conserved z-momentum moment) is **exactly constant** across the whole run (verified to 8 decimal digits, iter 0 through 10000): momentum conservation holds exactly even with angular perturbation
- All 5 conserved quantities (mass, p_x, p_y, p_z, energy) exactly constant throughout
- System converges to a **boosted** Jüttner equilibrium (not the isotropic one), consistent with nonzero net momentum — see Strain–Taylor–Velozo Ruiz orbital-stability theory (our run sits at 𝔮=1/2>1/3, so this is expected, not a bug)
- Final state: `f[0]=2.380, f[2]=0.0578, f[9]=-0.362, f[11]=-0.00691, f[18]=0.155, f[20]=0.00691` (rest ≈0)

**Why this matters:** confirms the discretized collision operator preserves momentum exactly even for non-radial initial data — a stronger conservation-law check than the radial-only run.

## Re-validated post bug-fix (assoc_legendre pole-collapse bug + basis_eval dedup)

Re-ran the hot-radial-only IC (t0=1.0, dt=1e-3, same as above) using the canonical
`tensor_n3_lag7_leb9_dense.pkl` → `sparse_n3_lag7_leb9.pkl` regenerated after fixing
the `assoc_legendre` pole bug and removing the duplicate `basis_eval`/`f_tilde_eval`
in `integrand_numba.py` (see [[project_basis_function_fixes]] if that memory exists,
otherwise see code comments in `src/basis_numba.py`). Result is bit-for-bit identical
to the original run above: `|f|→2.410626`, final `f[0]=2.378, f[9]=-0.363, f[18]=0.154`,
`Q(f,f)≈1e-9`. Confirms the whole pipeline (basis functions → quadrature → tensor →
sparse operator → time evolution) is unaffected by either fix.

## Successful run: zero-momentum "heat-flux"-like perturbation, t0=1.0

Same `t0=1.0, dt=1e-3, NUM_ITERATIONS=10000, save_every=100`. IC combines the hot
radial part with an `l=1, m=0` angular perturbation built from `k=1,2` radial
modes (not `k=0`), with the ratio between them chosen so the net `p_z` momentum
is **exactly zero**:

```python
f[0]  = 2.0                 # ind(0,0,0,3)
f[9]  = -0.8                # ind(1,0,0,3)
f[11] = 0.1                 # ind(1,1,0,3)
f[20] = 0.1 / sqrt(3)        # ind(2,1,0,3)  ratio f[20]/f[11] = 1/sqrt(3) found by
                             # solving M[2,11]*f[11] + M[2,20]*f[20] = 0 using the
                             # l=1,m=0 block of the mass matrix (mass_n3_lag7.pkl)
```

Verified `(M f)[2] = 0.0` exactly before running (mass and energy unaffected too,
since l=1 is mass-matrix-orthogonal to l=0 by spherical harmonic orthogonality).

**Result — sharp contrast with the genuine dipole run:**
- This perturbation is *not* protected by any conservation law (it's purely in
  the microscopic part `{I-P}f`, not the macroscopic/conserved part `Pf`)
- `f[2]` (and the whole perturbation) decays smoothly from its initial transient
  value down to **machine zero** (`~1e-15`) by iteration ~100, and stays there
- System relaxes all the way back to the **isotropic, unboosted** Jüttner
  equilibrium — same final state as the plain hot-radial-only run
  (`f[0]=2.378, f[9]=-0.363, f[18]=0.154`), confirming the asymmetry was
  genuinely erased, not just suppressed
- Visually: heatmap asymmetry plot (`F(u,v)-F(u,-v)` on the `p_x`-`p_z` plane)
  shows a clear two-lobed pattern at `t=0` (max asymmetry `~3.7e-2`) collapsing
  to pure floating-point noise (`~1.3e-15`) by iteration 100 — a 13-order-of-
  magnitude drop. 1D plot along the polar (`p_z`) axis shows the same: lopsided
  curve at `t=0` straightening into mirror symmetry by later snapshots.

**Why this matters:** direct numerical demonstration of the macro (`Pf`,
conserved, survives forever) vs micro (`{I-P}f`, no conservation law, decays to
zero) split central to the Strain–Taylor–Velozo Ruiz orbital-vs-asymptotic
stability theory — built from scratch in our own truncated basis, not just
read about. Contrast directly with the dipole run above, which has the *same*
angular shape (`l=1,m=0`) but carries real net momentum and therefore locks in
forever instead of decaying.

## Successful run: n=5, hot radial + new k=4 mode, t0=1.0 (2026-09-28)

First time-evolution run at n=5, using the freshly-computed and validated
n=5 tensor/sparse operator (see [[project_fast_tensor_computation]]). Same
`t0=1.0, dt=1e-3, NUM_ITERATIONS=10000, save_every=100` as every n=4 run
above — no stability issue found, despite n=5's mass matrix being far worse
conditioned (`cond=6.67e11` vs n=4's `4.8e8`, n=3's `4.3e5` — geometrically
worsening, but conservation held just as tight in practice; see below).

Generated fresh for this run: `mass/mass_n5_lag11.pkl` (mass matrix, `cond`
above), and copied the already-validated `sparse_operators/sparse_n5_lag11_leb17_fast.pkl`
to the canonical untagged name `sparse_n5_lag11_leb17.pkl` (matching n=3/n=4's
convention — tag is for *alternate* runs, and n=5 only has the one, now-canonical
result).

**Bug found and fixed while wiring this up:** `time_evol/time_ev.py` had
hardcoded n=4-specific flat indices (`f[16]`, `f[32]`, `f[48]`, `f[2]`,
`f[18]`, `f[34]`) instead of computing them via `ind(k,l,m,n)` — since the
flat index's per-`k` stride is `n**2`, every one of those literals is wrong
at any `n != 4`, silently. Generalized all IC-setting code (base radial part
and every named case: `dipole`, `zero_momentum`, `radial_T2`,
`radial_T2_full`) to use `ind(k,l,m,n)` — verified this reproduces the exact
same flat indices at n=4 (no change to any prior published result), and is
now correct at n=5 (and any future n).

**IC:** hot radial base (`k=0..3`: `2.0, -0.8, -0.1, -0.05`, same as every
prior `radial`-family case) plus a new small `k=4` perturbation (`-0.02`) —
the first exercise, dynamically (not just as a static tensor entry), of the
radial mode n=5 newly unlocks. Case name kept as `'radial'` (same physical
setup as the n=3/n=4 `radial` case, just with n=5's extra available mode),
gated behind `if n >= 5` so n=4 behavior is unchanged.

**Result:** stable, no blow-up. `|f|` climbs from `2.157` (IC) to `2.392814`
by iteration ~100 and stays *exactly* frozen through iteration 10000
(`Q(f,f)` there is `~1e-7` to `1e-8`, i.e. at equilibrium). Final radial
(`l=0`) coefficients: `k=0=+2.3582, k=1=-0.40102, k=2=+0.060024,
k=3=-0.0045970, k=4=+0.0018075`. **None of the higher modes vanish** — this
is expected and correct, not a bug: unlike `radial_T2`/`radial_T2_full`,
this IC was never T-tuned to make the equilibrium land exactly on the `k=0`
basis function, so the natural equilibrium temperature this IC implies
projects onto every radial mode, including the new `k=4` one.

**Independent equilibrium check:** the final state along the `ξ_x` axis
*exactly* overlays the analytical Jüttner curve (`A=0.352, T=2.444`)
computed independently from the IC's conserved mass/energy moments — visual
confirmation the solver converges to the physically correct equilibrium
implied by conservation, not just to *some* fixed point of the discrete
operator.

**Conservation** (`plot_moments.py`, all 10,000 saved snapshots): mass
spread `7.25e-13`, energy spread `1.03e-8`, momentum exactly `0.0` throughout
(purely radial IC, no angular content to carry momentum) — same tightness as
every n=3/n=4 run, despite n=5's far worse mass-matrix conditioning. The
conditioning concern (worth watching for n=6+) did not manifest as any
numerical degradation here.

Full export at `time_evol/experiments/radial_n5/`.

**Why this matters:** first full pipeline validation (quadrature -> tensor ->
sparse operator -> time evolution -> conservation/equilibrium check) at n=5,
end to end, and the first *dynamical* exercise of the n=5-only k=4 radial
mode (previously only checked as a static tensor entry).

## Plotting note

`plot/plot.py` reads whichever snapshots exist in `plot/coeff/` and plots `f`
along the **polar axis (`p_z`)**, not `p_x` — its `eval_polar_axis` function sets
`theta=0` for `x>0` and `theta=pi` for `x<0`, which is the `z`-axis, mislabeled
as `p_x` until this was caught and fixed. It has no hardcoded assumption about
which experiment produced the snapshots, so don't bake experiment-specific
titles/labels into it; it's meant to be reused across IC variants (radial-only,
dipole, zero-momentum heat-flux, etc.) without editing.

`plot/plot_heatmap.py` slices either the `p_x`-`p_y` plane (`plane='xy'`,
`p_z=0`) or the `p_x`-`p_z` plane (`plane='xz'`, `p_y=0`). An `l=1,m=0` (z-axis)
perturbation is exactly zero on the `xy` slice (proportional to `cos(theta)`,
which vanishes at `theta=pi/2`), so use `xz` to see z-axis asymmetries. Has a
`show_asymmetry` flag that plots `F(u,v)-F(u,-v)` instead of `F(u,v)` directly —
isolates the odd-in-`v` part exactly, since the dominant even/radial part
cancels in the difference. Use `shared_scale=True` when comparing magnitude
across snapshots (e.g. watching a perturbation decay) — per-frame auto-scaling
will misleadingly rescale pure floating-point noise to look like a real signal.
