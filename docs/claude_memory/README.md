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
  n_lebedev=9; n=4: n_laguerre=11, n_lebedev=13) with conservation-test reasoning.
- `project_initial_condition.md` — the "hot" radial initial condition
  (coeff[0]=2.0, coeff[9]=-0.8, rest zero).
- `project_time_evolution.md` — successful time-evolution runs, stability rule
  dt ∝ t0^{3/2}, and plotting notes.
- `project_basis_function_fixes.md` — assoc_legendre pole-collapse fix + basis_eval dedup.
- `project_quadrature_build_performance.md` — collision_quadrature() needs a
  numpy-vectorized rewrite before n=4 scale.

## Restoring

To restore into the live location after a fresh machine / re-clone:

```sh
cp docs/claude_memory/*.md \
   ~/.claude/projects/-Users-rjgh-Documents-Research-Projects-rel-boltz/memory/
```

These are point-in-time notes; verify any file:line citations against current code.
