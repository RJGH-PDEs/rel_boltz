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
   `time_evol/experiments/<case>/` — `axis_plots/`, and (from `plot_heatmap.py`, both views every
   run) `heatmaps_direct/` (raw `f`, viridis) and `heatmaps_asymmetry/` (`F(u,v)−F(u,−v)`, coolwarm).
   Shared helpers (`SNAPSHOTS`, `load_run_meta`, `experiment_case_dir`, `eval_point`) live in
   `plot/plot_common.py` so the two scripts don't duplicate the evaluation/output logic.
7. **`time_evol/export_experiment.py`** → final packaging step: reads `run_meta.json`, checks the
   figures are present, and writes the LaTeX-ready `README.md` into `time_evol/experiments/<case>/`.
   Run from `time_evol/` after the plot scripts. It no longer copies figures — they are already in
   place. The `experiments/` tree is gitignored; it is export output meant to be copied into the
   LaTeX writeup.

## Initial-condition cases

`time_ev.py` defines three ICs via the `CASE` flag, all sharing the "hot radial" base
(`f[0]=2.0`, `f[9]=-0.8`); `CASE_INFO` holds each one's physical significance (the single source of
truth, also dumped into `run_meta.json`). To run all three, edit `CASE`, then run the
`time_ev.py → plot.py + plot_heatmap.py → export_experiment.py` chain once per case (clear
`plot/coeff/*.pkl` between runs so stale snapshots don't linger):

- **`radial`** — no angular perturbation; isotropic control. Thermalizes to the isotropic Jüttner
  equilibrium; the asymmetry diagnostic stays at float64 roundoff (`~1e-16`).
- **`dipole`** — adds an `l=1,m=0` dipole (`f[2]`) carrying net `p_z`. Momentum is conserved, so the
  `p_z → −p_z` asymmetry **persists** (the equilibrium is boosted along z).
- **`zero_momentum`** — adds an `l=1,m=0` perturbation (`f[11]`, `f[20]` in ratio `1/√3`) with net
  `p_z = 0`. With no conserved momentum protecting it, the asymmetry **decays** back to `~0`.

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
    (the `exp(-r/2)·linear_comb` value with the `r==0` case), the unified `SNAPSHOTS` list,
    `load_run_meta`, and `experiment_case_dir`. Add anything both scripts need here, not in both.
  - `plot/plot.py` — 1D line plots along the x/y/z axes (`eval_axis`).
  - `plot/plot_heatmap.py` — 2D plane slices, both `direct` and `asymmetry` views (`eval_plane`).
  - `time_evol/export_experiment.py` — README only; it does **not** plot or copy figures.
  The plot scripts write figures **directly** into `time_evol/experiments/<case>/` (one copy, in its
  final home) and read `n`/`case` from `run_meta.json` rather than hardcoding them. There is no
  `plot/figures/` staging dir — adding one back would recreate the duplicate-output problem this
  layout was built to remove.

## Verification

`src/tests.py` is the verification suite (run from `src/`): quadrature sanity checks against known
integrals, numba-vs-scipy basis spot checks, a **conservation sweep** (`run_conservation_tests` —
conserved-quantity entries should → 0 as `n_lebedev` rises; this is how quadrature orders are
chosen), a convergence sweep, and `compare_tensors` to diff two saved tensors entry-by-entry.

## Choosing quadrature orders (verified — see `docs/claude_memory/`)

Quadrature orders are not free parameters; they were pinned by the conservation/convergence sweeps:

- **n=3:** `n_laguerre=7`, `n_lebedev=9` (at `n_lebedev=7`, conservation fails for `l=2` entries).
- **n=4:** `n_laguerre=11`, `n_lebedev=13` (the `l=3` entries need both; intermediate Lebedev orders
  are non-nested and can get *worse*, so don't trust them).

`docs/claude_memory/` is a versioned backup of this project's accumulated findings — verified
quadrature parameters, the chosen "hot radial" initial conditions, time-evolution run results and
the `dt ∝ t0^{3/2}` stability rule, the basis-function fixes, and the known performance issue that
`collision_quadrature()` needs a numpy-vectorized rewrite before `n=4` scale. Consult it before
re-deriving any of these.
