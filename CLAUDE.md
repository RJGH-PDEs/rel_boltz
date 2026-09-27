# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Numerical experiments solving the **relativistic Boltzmann equation** with a Petrov–Galerkin
spectral method. The distribution function is expanded in basis functions defined on momentum
space `(r, θ, φ)`: generalized Laguerre polynomials in the radial direction times real spherical
harmonics in the angular directions. A function is represented by a coefficient vector of length
`n³`, indexed by `(k, l, m)` where `k, l ∈ [0, n)` and `m ∈ [-l, l]` (see `ind(k, l, m, n)` in
`src/sparse.py` — this index convention is used everywhere).

The end goal is to time-evolve `df/dt = ½ t^{-3/2} M⁻¹ Q(f, f)` and watch the distribution relax to
a Jüttner equilibrium.

## Related project — the mathematical write-up (read-only reference)

This code implements the scheme derived in a **separate** LaTeX project:

- **Path:** `/Users/rjgh/Documents/Research/Latex/relativistic_boltzmann`
- It is its own git repository with its own `CLAUDE.md` (`main.tex` = the paper *"Galerkin-Petrov
  Approach for the relativistic Boltzmann Equation"*, Gonzalez & Gamba).

Treat that directory as **read-only context from here**: consult it for the derivations, notation,
and definitions (the `ξ = tᵅ p` change of variables — which is why the plots use `ξ` axes — the
basis functions `φ_{klm}`/`ψ_{klm}`, the `μ_{k,l}` constants, the mass matrix, the `t^{-3/2}`
prefactor, the sparsity-inducing rotational symmetry). Do **not** edit or commit in that repo from a
session rooted in this code project — it has its own git history and its own Claude guidance. The
`time_evol/experiments/<case>/` exports are meant to be copied into that paper by hand.

## Environment

There is no `requirements.txt`. Code runs under a conda env named **`ttenv`**
(`~/miniconda3/envs/ttenv/bin/python`). Dependencies: `numpy`, `numba`, `scipy`, `matplotlib`, and
**`pylebedev`** (Lebedev quadrature on S²). Numba is required — the hot integrand loops are
`@njit`-compiled.

## Running things — working directory matters

Scripts use **relative paths** and `sys.path.insert(0, '../src')`, so each must be run from its own
directory, not the repo root:

- `src/*.py` scripts write to / read from `./quadratures/`, `./results/`, `./mass/`,
  `./sparse_operators/` — run them with cwd = `src/`.
- `plot/*.py` and `time_evol/*.py` reach back into `../src/` for the operator/mass files — run them
  with cwd = `plot/` and `time_evol/` respectively.

Each script has a `__main__` block with hardcoded parameters (`n`, `n_laguerre`, `n_lebedev`) — set
parameters by editing those blocks, not via CLI args. The `tag` parameter (e.g. `'_tacc'`) threads
through filenames to keep alternate runs side by side.

## Pipeline (run in this order)

The build is a chain of pickled artifacts; each stage consumes the previous stage's output.

1. **`src/quadrature.py`** → builds the 8D collision quadrature and 3D mass quadrature
   (`quadratures/*.pkl`). The collision quadrature is the cost bottleneck: point count scales as
   `n_laguerre² × n_lebedev_points³`.
2. **`src/mass_matrix.py`** → mass matrix `M` (`mass/*.pkl`). Block-diagonal by spherical-harmonic
   orthogonality; only the radial Laguerre integral is done numerically.
3. **`src/collision_tensor.py`** → the dense collision tensor `Q_{i,jk}` (`results/*.pkl`). Computed
   in parallel across CPUs via `multiprocessing.Pool`, with the quadrature held in a single
   **shared-memory block** (`src/boltzmann.py`) so workers don't each copy it. This is the
   expensive compute step (TACC-scale at `n=4`).
4. **`src/sparse.py`** → thresholds the tensor, verifies the analytic sparsity rules, and writes a
   list of `csr_matrix` operators, one per test function (`sparse_operators/*.pkl`).
5. **`time_evol/time_ev.py`** → loads `M` and the sparse operator, sets an initial coefficient
   vector, and forward-Euler integrates. Saves snapshots to `plot/coeff/<iter>.pkl` and a
   `plot/coeff/run_meta.json` describing the run (case, `t0`, `dt`, iterations, nonzero IC entries).
   The IC is chosen by the `CASE` flag — see "Initial-condition cases" below.
6. **`plot/plot.py`** (1D along x/y/z axes) and **`plot/plot_heatmap.py`** (2D slices) read those
   snapshots and `run_meta.json`, and write figures **directly** into the per-case experiment folder
   `time_evol/experiments/<case>_n<n>/` — `axis_plots/`, and (from `plot_heatmap.py`, both views every
   run) `heatmaps_direct/` (raw `f`, viridis) and `heatmaps_asymmetry/` (`F(u,v)−F(u,−v)`, coolwarm).
   The folder includes `n` in its name so n=3 and n=4 results never share a directory.
   Shared helpers (`SNAPSHOTS`, `available_snapshots`, `load_run_meta`, `experiment_case_dir`,
   `eval_point`) live in `plot/plot_common.py` so the scripts don't duplicate the
   evaluation/output logic.
6b. **`plot/plot_moments.py`** → conservation-law diagnostic. Reads `run_meta.json`, loads the same
   mass matrix `M` used by `time_ev.py` (`mass_name`/`load_mass`), and for **every** saved snapshot
   (`available_snapshots`, not the sparse `SNAPSHOTS`) computes the moments `M @ f` and pulls the five
   collision-invariant entries via `ind(k,l,m,n)`: mass `(0,0,0)`, momentum `(0,1,-1/0/1)`, energy
   `(1,0,0)`. The moment of `f` against test function `φ_i` is exactly `(M f)_i` (the
   "Checking Conservation Laws" section of the LaTeX writeup). Writes `moments/moments.csv`
   (columns `iteration, time, mass, mom_m-1, mom_m0, mom_m1, energy`) and `moments/moments.png`
   (absolute values vs time, three panels). These quantities are conserved by the collision operator,
   so they stay flat in time to quadrature/roundoff (mass/energy spread ~1e-9, momentum ~1e-12);
   `mom_m0` (net p_z) is the case discriminator — see "Initial-condition cases".
7. **`time_evol/export_experiment.py`** → final packaging step: reads `run_meta.json`, checks the
   figures (axis, heatmap, and moments) are present, and writes the LaTeX-ready `README.md` into
   `time_evol/experiments/<case>_n<n>/`. Run from `time_evol/` after the plot scripts. It no longer copies
   figures — they are already in place. The `experiments/` tree is gitignored; it is export output
   meant to be copied into the LaTeX writeup.

## Initial-condition cases

`time_ev.py` defines five ICs via the `CASE` flag; `CASE_INFO` holds each one's physical significance
(the single source of truth, also dumped into `run_meta.json`). Run the
`time_ev.py → plot.py + plot_heatmap.py + plot_moments.py + plot_coeff_evolution.py → export_experiment.py`
chain once per case (clear `plot/coeff/*.pkl` between runs so stale snapshots don't linger):

- **`radial`** — no angular perturbation; isotropic control. Thermalizes to the isotropic Jüttner
  equilibrium; the asymmetry diagnostic stays at float64 roundoff (`~1e-16`). Moments: all three
  `mom_*` identically 0.
- **`dipole`** — adds an `l=1,m=0` dipole (`f[2]`) carrying net `p_z`. Momentum is conserved, so the
  `p_z → −p_z` asymmetry **persists** (the equilibrium is boosted along z). Moments: `mom_m0` is a
  nonzero constant (`~15.68` for the standard IC).
- **`zero_momentum`** — adds an `l=1,m=0` perturbation (`f[11]`, `f[20]` in ratio `1/√3`) with net
  `p_z = 0`. With no conserved momentum protecting it, the asymmetry **decays** back to `~0`. Moments:
  `mom_m0` stays at `~0` (roundoff) the whole run, even though the distribution is asymmetric — this
  is the key check that the IC really carries zero net momentum.
- **`radial_T2`** — radial IC tuned so E_raw/(3·M_raw) = 2 (c0=2, c1=−0.4, c2=−0.142199, c3=0).
  The T=2 Jüttner A·exp(−r/2) is exactly the k=0 basis function → zero spectral truncation error.
  Higher modes (k=1, k=2) decay to near-zero; analytical Jüttner from IC moments coincides exactly
  with the computed equilibrium.
- **`radial_T2_full`** — same T=2 target but all four radial modes active (c0=2, c1=−0.4,
  c2=−0.331730, c3=−0.1; c2 found via brentq). Produces a W-shaped double-hump IC (origin
  suppressed, peaks at |ξ|≈2). All three higher modes (k=1,2,3) decay to near-zero within ~20
  iterations; most striking demonstration that the solver finds the exact equilibrium.

## Key architecture details

- **`src/integrand_numba.py`** is the physics core: `operator_numba` is the `@njit` quadrature loop
  evaluating `Q`. `postcoll_F`/`postcoll_G` compute post-collision momenta; `kernel` is the
  relativistic collision kernel. All basis evaluation goes through `src/basis_numba.py`
  (`basis_eval`, `f_tilde_eval`, `gen_laguerre`, `assoc_legendre`) — `integrand_numba.py` imports
  these rather than duplicating them (a past duplication caused a bug to need patching twice).
- **Sparsity rules** (`cai`, `andrea` in `src/sparse.py`) are analytic selection rules predicting
  which tensor entries vanish by symmetry. Used both to skip computation (`use_sparsity=True` in
  `collision_tensor.py`) and to validate computed nonzeros. A structural zero stays zero at any
  quadrature order, so check these before trusting a quadrature sweep.
- **`assoc_legendre` pole bug (fixed):** `sin(θ)` must be passed in explicitly, never derived from
  `cos(θ)` via `sqrt(1 - cos²)` — near a pole that collapses to exactly 0 in float64. See the long
  comment in `src/basis_numba.py`.
- **Plotting module structure (single output pipeline — keep it DRY).** Each plotting file has one
  job and nothing is duplicated; don't reintroduce a second copy:
  - `plot/plot_common.py` — the only home for logic shared by the plot scripts: `eval_point`
    (the `exp(-r/2)·linear_comb` value with the `r==0` case), the sparse `SNAPSHOTS` list (frames for
    axis/heatmap plots), `available_snapshots` (every saved snapshot — what `plot_moments.py` wants),
    `load_run_meta`, and `experiment_case_dir`. Add anything multiple scripts need here, not in each.
  - `plot/plot.py` — 1D line plots along the x/y/z axes (`eval_axis`).
  - `plot/plot_heatmap.py` — 2D plane slices, both `direct` and `asymmetry` views (`eval_plane`).
  - `plot/plot_moments.py` — conserved-moment time series (`M @ f`); CSV + figure, no field evaluation.
  - `plot/plot_coeff_evolution.py` — time evolution of the radial l=0 coefficients (k=0..n-1); log x-axis zoomed to iter 0–100. Writes `coeff_evolution.png` to the experiment folder. Run from `plot/`.
  - `plot/plot_sparsity.py` — operator diagnostic (not case-specific): loads a sparse operator pkl
    via `sparse.sparse_name`, and renders each test-function slice as a panel in a 3×9 grid (rows =
    radial index `k`, columns = `(l,m)` pairs). Each panel title is the test-function `(k,l,m)`;
    axes are `ψ_s` (row) and `ψ_t` (col); a `nnz N` annotation shows the non-zero count per slice.
    Writes to `plot/figures/` (not `time_evol/experiments/` — this is operator structure, not a
    time-evolution result). Run from `plot/`.
  - `plot/ki_scaling/plot_sparsity_boxed.py` — variant of the above with a red box highlighting the ℓ_i=2 block.
  - `plot/ki_scaling/plot_ki_*.py` and `plot/ki_scaling/plot_single_entry_growth.py` — analysis
    scripts for the `docs/tensor_k_scaling.tex` write-up (k_i scaling study). Run from
    `plot/ki_scaling/`; write to `plot/figures/`.
  - `time_evol/export_experiment.py` — README only; it does **not** plot or copy figures.
  The experiment plot scripts write figures **directly** into `time_evol/experiments/<case>/` (one
  copy, in its final home) and read `n`/`case` from `run_meta.json` rather than hardcoding them.
  There is no `plot/figures/` staging dir for experiment outputs — adding one back would recreate
  the duplicate-output problem this layout was built to remove. (`plot/figures/` is used only by
  `plot_sparsity.py` for operator diagnostics, which are not case-specific.)

## Verification

`src/tests.py` is the verification suite (run from `src/`): quadrature sanity checks against known
integrals, numba-vs-scipy basis spot checks, a **conservation sweep** (`run_conservation_tests` —
conserved-quantity entries should → 0 as `n_lebedev` rises; this is how quadrature orders are
chosen), a convergence sweep, and `compare_tensors` to diff two saved tensors entry-by-entry.

**Conservation-entry subtlety (found while validating the n=5 tensor):** mass conservation holds
*per tensor entry*, for any `(f1, f2)` pair — `T[mass, f1, f2]` is analytically zero regardless of
whether `f1 == f2`. Momentum and energy are **not** individually zero per entry in general; the
conserved identity is `T[test, f1, f2] = -T[test, f2, f1]` (antisymmetric under swapping the two
trial functions), which only forces `T` to zero when `f1 == f2`. Every hand-picked case in
`run_conservation_tests` happens to use `f1 == f2` (e.g. `[4,4,-4]x[4,4,-4]`), so historically the
sweep has only ever exercised the `f1==f2` special case — genuinely diagnostic for mass and for
picking quadrature order overall, but **not** capable of catching a momentum/energy convergence
problem that only shows up for `f1 != f2` pairs. Confirmed on the actual n=5 tensor: `f1 != f2`
momentum/energy entries reach magnitudes up to `~1.6e6` individually, but `T[test,f1,f2] +
T[test,f2,f1]` is `~1e-12` relative — the physically meaningful quantity (since the PDE only ever
evaluates `Q(f,f)`, i.e. the same coefficient vector in both trial slots, which sums exactly these
antisymmetric pairs to zero). When adding conservation cases for a future `n`, include at least one
`f1 != f2` momentum/energy pair and check the antisymmetrized sum, not just individual-entry
convergence to zero.

## Choosing quadrature orders (verified — see `docs/claude_memory/`)

Quadrature orders are not free parameters; they were pinned by the conservation/convergence sweeps:

- **n=3:** `n_laguerre=7`, `n_lebedev=9` (at `n_lebedev=7`, conservation fails for `l=2` entries).
- **n=4:** `n_laguerre=11`, `n_lebedev=13` (the `l=3` entries need both; intermediate Lebedev orders
  are non-nested and can get *worse*, so don't trust them).
- **n=5:** `n_laguerre=11` (unchanged from n=4), `n_lebedev=17` (up from 13, for the new `l=4`
  entries) — 161,051,000 quadrature points, built and saved to
  `src/quadratures/collision_lag11_leb17.pkl`. Full tensor **computed on TACC** (job 3472070,
  4.38h, `results/tensor_n5_lag11_leb17_sparse_fast.pkl`, 285,750 entries) and validated (see
  `docs/claude_memory/project_fast_tensor_computation.md`).

`docs/claude_memory/` is a versioned backup of this project's accumulated findings — verified
quadrature parameters, the chosen "hot radial" initial conditions, time-evolution run results and
the `dt ∝ t0^{3/2}` stability rule, and the basis-function fixes. Consult it before re-deriving any
of these. (`collision_quadrature()`'s pure-Python-loop performance issue is already fixed — it's now
a numpy-vectorized build, ~2.4s at n=4 scale.)

## Fast table-based tensor computation (prototype — not yet the production path)

`src/collision_tensor_fast.py` is a validated but not-yet-wired-in alternative to
`collision_tensor.py`: instead of every `(test, f, g)` tensor entry independently re-deriving the
kernel, post-collision kinematics, and basis values at every quadrature point (what
`collision_tensor.py`/`integrand_numba.py` do today), it precomputes those once — a per-point table
(kernel + post-collision coords, size `N_quad`) and a per-`(k,l,m)`-triple basis-value table (size
`n**3 x N_quad`, only `n**3` distinct triples instead of re-deriving per entry) — then assembles
every tensor entry as pure table lookups inside one `@njit(parallel=True)` kernel.
`compute_tensor_fast_chunked` processes the quadrature in memory-bounded blocks (table memory doesn't
fit in RAM unchunked past n=3). Validated to floating-point roundoff against both the n=3 and n=4
production tensors, ~20x faster, and turned n=4 (previously TACC-only) into a ~34-minute local run.
The n=5 tensor has been computed on TACC (`job_n5.sh` + `src/run_n5_tensor.py`, adapted structurally
from the sibling `numba_landau` project's working template — not a dependency on it; ran 4.38h on
128 cores, checkpointing after every chunk via `checkpoint_path` so a killed/timed-out run resumes
instead of restarting) and validated: exact combinatorial match to the expected select set, zero
sparsity-rule violations, no NaN/Inf, mass conservation at the roundoff floor, and momentum/energy
conservation confirmed via the antisymmetric-pair check described in "Verification" above. See
`docs/claude_memory/project_fast_tensor_computation.md` for full numbers.

## k_i scaling write-up (`docs/tensor_k_scaling.tex`)

`docs/tensor_k_scaling.tex` is a standalone document recording numerical observations about how
the collision operator entries scale with the test-function radial index `k_i`. Build it with:
```
cd docs && tectonic tensor_k_scaling.tex
```
Build artifacts (`.pdf`, `.aux`, `.log`, `.synctex.gz`) are gitignored inside `docs/`.

Supporting analysis scripts (run from `docs/`):
- `docs/check_recurrence.py` — tests whether A_k satisfies a closed three-term recurrence
- `docs/check_recurrence_k6.py` — extends the check to k_i=6 with two quadrature sizes
- `docs/verify_ki_scaling.py` — spot-checks specific k_i entries
- `docs/verify_k_scaling.py` — verifies the k_s combinatorial formula against the stored n=3 operator and extended k=3 cases

`src/tests.py` section 6 (`verify_quadrature_build`) — four-level correctness check for
`collision_quadrature()`: analytical weight sum, point-wise match against a reference file,
18-entry operator sweep, and conservation entries. Run from `src/`.
