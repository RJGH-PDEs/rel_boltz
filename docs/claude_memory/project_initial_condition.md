---
name: project-initial-condition
description: "Chosen initial condition for time evolution — \"hot\" radial distribution, n=3"
metadata: 
  node_type: memory
  type: project
  originSessionId: 12cac65b-881c-4cee-9420-9f57a1d42b0f
---

Chosen initial condition: "hot" radial distribution for n=3 time evolution.

```python
coeff = np.zeros(N**3)   # N=3, size=27
coeff[ind(0, 0, 0, 3)] = coeff[0]  =  2.0
coeff[ind(1, 0, 0, 3)] = coeff[9]  = -0.8
```

All other coefficients zero (pure l=0, radial).

**Why:** Shifts the peak of f(r) = e^{-r/2} * linear_comb outward relative to the Jüttner equilibrium, giving a "hotter" distribution that stays non-negative. Chosen by visual inspection in plot/plot.py.

**How to apply:** Use these as the starting coefficients for the time evolution ODE.
