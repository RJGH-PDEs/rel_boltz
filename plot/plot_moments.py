"""Conservation-law diagnostic: collision-invariant moments vs time.

Post-processes a finished time-evolution run. For each saved coefficient snapshot
f(t) it computes the moments by testing against every test function, which is just
M @ f (see the "Checking Conservation Laws" section of the write-up). The five
collision invariants — mass, the three momentum components, and energy — should
stay constant in time, so their entries of M @ f are the conservation check.

Run from plot/ (same cwd convention as plot.py / plot_heatmap.py). Writes a CSV
time series and a figure into time_evol/experiments/<case>/moments/.
"""
import os
import sys
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../src')
from plot_common import load_run_meta, experiment_case_dir, available_snapshots
from mass_matrix import mass_name, load_mass
from sparse import ind

# Collision invariants and their (k, l, m) triplets. mom_m0 is the net p_z carried
# by the l=1, m=0 mode (the dipole / zero-momentum experiment direction).
INVARIANTS = [
    ('mass',    (0, 0, 0)),
    ('mom_m-1', (0, 1, -1)),
    ('mom_m0',  (0, 1,  0)),
    ('mom_m1',  (0, 1,  1)),
    ('energy',  (1, 0, 0)),
]

# Case and spectral order come from the run that produced the snapshots.
meta       = load_run_meta()
n          = meta['n']
n_laguerre = meta['n_laguerre']
t0         = meta['t0']
dt         = meta['dt']
case       = meta['case']
label      = meta.get('label', case)
tag        = meta.get('tag', '')

# Same mass-matrix path convention as time_ev.py.
M   = load_mass('../src/' + mass_name(n, n_laguerre, tag=tag))
idx = {name: ind(k, l, m, n) for name, (k, l, m) in INVARIANTS}

iters = available_snapshots()
if not iters:
    raise SystemExit("no coeff/<iter>.pkl snapshots found — run time_ev.py first.")

times = []
series = {name: [] for name, _ in INVARIANTS}
for it in iters:
    with open(f"coeff/{it}.pkl", 'rb') as f:
        coeff = pickle.load(f)
    moments = M @ coeff
    times.append(t0 + it * dt)
    for name, _ in INVARIANTS:
        series[name].append(moments[idx[name]])

out_dir = os.path.join(experiment_case_dir(case), 'moments')
os.makedirs(out_dir, exist_ok=True)

# ── CSV time series ───────────────────────────────────────────────────────────
csv_path = os.path.join(out_dir, 'moments.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'time'] + [name for name, _ in INVARIANTS])
    for row, (it, t) in enumerate(zip(iters, times)):
        writer.writerow([it, t] + [series[name][row] for name, _ in INVARIANTS])
print(f"saved {csv_path}")

# ── figure: absolute moment values vs time ────────────────────────────────────
# Three stacked panels (mass / momentum / energy) so the differing scales stay
# legible; flat curves = conserved.
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))

axes[0].plot(times, series['mass'], 'o-', color='tab:blue')
axes[0].set_ylabel('mass  (Mf)[0,0,0]')

for name, marker in (('mom_m-1', 'o-'), ('mom_m0', 's-'), ('mom_m1', '^-')):
    axes[1].plot(times, series[name], marker, label=name)
axes[1].set_ylabel('momentum  (Mf)[0,1,m]')
axes[1].legend()

axes[2].plot(times, series['energy'], 'o-', color='tab:red')
axes[2].set_ylabel('energy  (Mf)[1,0,0]')
axes[2].set_xlabel('time  t')

axes[0].set_title(f'conserved moments vs time — {label} ({case})')
fig.tight_layout()
fig_path = os.path.join(out_dir, 'moments.png')
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"saved {fig_path}")
