# Claude Code project memory (backup)

These Markdown files are a versioned backup of the Claude Code memory for this
project. The live copies that Claude reads each session live **outside** the repo at:

```
~/.claude/projects/-Users-rjgh-Documents-Research-Projects-rel-boltz/memory/
```

That location is keyed to the project's absolute folder path (not to any login
account), and is not covered by git — hence this in-repo backup.

`MEMORY.md` is the index; each other file holds one fact:

- `project_quadrature_params.md` — verified quadrature orders (n=3: n_laguerre=7,
  n_lebedev=9; n=4: n_laguerre=11, n_lebedev=13; n=5: n_laguerre=11, n_lebedev=17)
  with conservation-test reasoning.
- `project_initial_condition.md` — the "hot" radial initial condition
  (coeff[0]=2.0, coeff[9]=-0.8, rest zero).
- `project_time_evolution.md` — successful time-evolution runs, stability rule
  dt ∝ t0^{3/2}, and plotting notes. Includes the first n=5 run (radial + new
  k=4 mode), validated against the analytical Jüttner equilibrium.
- `project_basis_function_fixes.md` — assoc_legendre pole-collapse fix + basis_eval dedup.
- `project_quadrature_build_performance.md` — collision_quadrature() was
  rewritten to a numpy-vectorized build (fix already implemented, not just
  planned — builds n=4 scale in ~2.4s).
- `project_fast_tensor_computation.md` — table-based collision-tensor
  prototype (`src/collision_tensor_fast.py`), ~20x faster than the production
  `collision_tensor.py`. n=3/n=4/n=5 tensors all computed and validated (n=5
  on TACC, job 3472070, 4.38h) — includes the momentum/energy
  antisymmetric-conservation subtlety (mass vanishes per-entry; momentum/
  energy only vanish under the `f1<->f2` antisymmetrized sum).

## Restoring

To restore into the live location after a fresh machine / re-clone:

```sh
cp docs/claude_memory/*.md \
   ~/.claude/projects/-Users-rjgh-Documents-Research-Projects-rel-boltz/memory/
```

These are point-in-time notes; verify any file:line citations against current code.
