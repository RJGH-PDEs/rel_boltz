"""Plot the time evolution of the four radial l=0 basis coefficients.

For a radially-symmetric run only the (k,0,0) modes are ever nonzero.
Produces a single-panel log-x plot zoomed to the first 100 iterations,
where all the interesting dynamics happen.

Run from plot/.
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../src')
from plot_common import available_snapshots, load_run_meta, experiment_case_dir
from sparse import ind

show = False
save = True

meta = load_run_meta()
n    = meta['n']
case = meta['case']

iters = available_snapshots()
its = []
cs  = [[] for _ in range(n)]   # one list per k

for it in iters:
    with open(f'coeff/{it}.pkl', 'rb') as f:
        coeff = pickle.load(f)
    its.append(it)
    for k in range(n):
        cs[k].append(coeff[ind(k, 0, 0, n)])

its    = np.array(its)
its_log = np.where(its == 0, 0.5, its)   # shift iter 0 off log(0)

fig, ax = plt.subplots(figsize=(8, 5))
for k in range(n):
    ax.plot(its_log, cs[k], label=f'k={k}  ({k},0,0)', linewidth=2.5 if k == 0 else 2)
ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
ax.set_xscale('log')
ax.set_xlim(0.5, 100)
ax.set_xlabel('iteration (log scale)')
ax.set_ylabel('coefficient value')
ax.set_title(f'radial l=0 coefficient evolution  ({case}, n={n})')
ax.legend()
ax.grid(True, which='both', alpha=0.4)
plt.tight_layout()

if save:
    out_dir = experiment_case_dir(case, n)
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'coeff_evolution.png')
    plt.savefig(path, dpi=150)
    print(f'saved {path}')

if show:
    plt.show()
plt.close(fig)
