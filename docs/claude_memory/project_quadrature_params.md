---
name: project-quadrature-params
description: Verified quadrature parameters for the n=3 collision tensor — why n_lebedev=9 is the minimum
metadata: 
  node_type: memory
  type: project
  originSessionId: 12cac65b-881c-4cee-9420-9f57a1d42b0f
---

For the n=3 relativistic Boltzmann collision tensor, the verified quadrature parameters are:

- `n_laguerre = 7`
- `n_lebedev = 9`

**Why:** `n_lebedev=7` causes mass conservation to fail for entries with (k=2, l=2) trial functions — the value converges to ~284 instead of zero. At `n_lebedev=9` it drops to machine precision (~1e-12). The issue is that f̃_{2,2,m}(p'(p,q,ω)) has high angular frequency in ω, requiring more Lebedev points to resolve the gain integral accurately.

Increasing `n_laguerre` beyond 7 does not help — the conservation error is purely in the angular (Lebedev) direction.

**How to apply:** Always use at least `n_lebedev=9` for n=3.

## n=4: confirmed n_lebedev=13 (not 11)

Re-ran the same diagnostic for n=4 (introduces l=3 trial functions) using
`run_conservation_tests` in `src/tests.py`. Tested 9 cases total: the original
3 `l=2` cases (still machine-zero at every order, as a sanity baseline) plus 6
new `l=3` cases spanning different `m` values and one mixed `l1=3,l2=2`
momentum case. All cases first checked against the Cai/Andrea sparsity rules
(`src/sparse.py`) to confirm they're not trivially zero by selection rules
before trusting the sweep — important, since a structural zero would stay
zero at any quadrature order and give false confidence.

Results:
- `n_lebedev=9`: l=3 cases are badly wrong (`~35` to `~155`, should be 0)
- `n_lebedev=11`: l=3 cases are **still** badly wrong, and non-monotonic —
  some got *worse* (e.g. one energy case went from `+155` to `-2025`).
  Lebedev rules aren't nested, so intermediate orders aren't guaranteed to
  trend toward the right answer. (One mixed-l momentum case was already at
  machine zero here, interestingly — only the l1=l2=3 mass/energy cases
  needed the jump to 13.)
- `n_lebedev=13`: **all 9 cases** machine zero (`1e-10` to `1e-8`)

**Practical note:** also discovered (and fixed) that `run_conservation_tests`
originally rebuilt the (expensive) quadrature once per *case* instead of once
per *order* — wasteful, since multiple cases share the same order. Fixed to
build once per order and loop cases inside. At `n_lebedev=13`, quadrature has
~19.9M points and takes ~60s to build (vs ~2.7M points / ~7s at n_lebedev=9
for n=3) — the n=4 tensor computation will be substantially more expensive
than n=3 was; worth planning compute budget (TACC) before launching it.

**How to apply:** use `n_lebedev=13` for n=4 (see n_laguerre finding below —
n_laguerre also needs to increase, contrary to first guess).

## n=4: n_laguerre also needs to increase, to 11 (not 7)

Checked radial (n_laguerre) convergence separately from angular, using a cheap
fixed `n_lebedev=7` while sweeping n_laguerre — radial and angular convergence
are independent, so there's no need to pay for the expensive n_lebedev=13
quadrature while diagnosing n_laguerre alone.

First attempt picked `[[0,0,0],[3,0,0],[3,0,0]]` as a "large nonzero" entry to
track — but `[0,0,0]` (k=0,l=0,m=0) is the **mass-conservation test function**,
so that entry is forced to zero by physics for any trial pair, at any
n_laguerre. Not useful for this check; switched to non-conserved test
directions like `[2,0,0]`.

Checked several non-conserved entries spanning a range of (k,l,m). Most
converged fine already at n_laguerre=7 (e.g. `[2,0,0]x[3,0,0]x[3,0,0]` and
`[2,1,0]x[3,1,0]x[3,0,0]` agree to ~9 sig figs from n_laguerre=7 onward). But
the worst case — stacking high k AND high l simultaneously on both trial
functions, e.g. `[3,3,2]x[3,3,1]x[1,2,1]` — was **not** converged at
n_laguerre=7 (off by a factor of ~2 from n_laguerre=5, and still off by ~8%
from the converged value). Swept further:

```
n_laguerre=7:   54849306988    (off by ~8% from converged value)
n_laguerre=9:   50330766771    (still off by ~8% from converged value!)
n_laguerre=11:  50330760273    (Δ from 9:  -6498, rel. ~1.3e-7)
n_laguerre=13:  50330758091    (Δ from 11: -2182, rel. ~4.3e-8)
n_laguerre=15:  50330757235    (Δ from 13:  -856,  rel. ~1.7e-8)
```

Increments shrink geometrically from n_laguerre=11 onward (genuine
convergence, not noise) — `n_laguerre=11` already gets ~1e-7 relative
accuracy, a good practical stopping point (13/15 are tighter but likely
unnecessary given the cost).

**How to apply:** for n=4, use `n_laguerre=11` and `n_lebedev=13` together
(both confirmed necessary, independently, via cheap one-parameter-at-a-time
sweeps before combining for the production tensor run). This is a real
increase from n=3's `(7, 9)`, not just on the angular side as first assumed.

**Cost warning:** quadrature point count scales roughly as
`n_laguerre^2 * n_lebedev_points^3`. Going from `(7,9)` to `(11,13)` is a
substantial jump — worth estimating total n=4 tensor compute cost (point
count increase x larger sparse index set, since n=4 has more (k,l,m) triples
than n=3) before launching on TACC.
