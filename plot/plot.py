import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
sys.path.insert(0, '../src')
from sparse import ind
from lc import linear_comb

os.makedirs("figures", exist_ok=True)
os.makedirs("coeff", exist_ok=True)

# ── flags ────────────────────────────────────────────────────────────────────
N    = 3
show = True
save = True
# ─────────────────────────────────────────────────────────────────────────────

i0 = ind(0, 0, 0, N)   # = 0
i1 = ind(1, 0, 0, N)   # = 9
i2 = ind(2, 0, 0, N)   # = 18

# ── evaluation grid ───────────────────────────────────────────────────────────
n_pts  = 200
x_vals = np.linspace(-10.0, 10.0, n_pts)
r_vals = np.abs(x_vals)

def eval_radial(coeff_vec):
    f = np.zeros(len(x_vals))
    for i, (x, r) in enumerate(zip(x_vals, r_vals)):
        if r == 0.0:
            f[i] = linear_comb(coeff_vec, 0.0, 0.0, 0.0, N)
        else:
            theta = 0.0 if x > 0 else np.pi
            f[i] = np.exp(-r / 2) * linear_comb(coeff_vec, r, theta, 0.0, N)
    return f

# ── load snapshots ────────────────────────────────────────────────────────────
snapshots = [0, 1, 2, 3, 5, 8, 12, 17, 20, 10000]   # iteration numbers

fig, ax = plt.subplots(figsize=(9, 5))

cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(snapshots)))

for color, it in zip(colors, snapshots):
    path = f"coeff/{it}.pkl"
    if not os.path.exists(path):
        print(f"missing {path}, skipping")
        continue
    with open(path, 'rb') as f:
        coeff = pickle.load(f)
    label = f'iter {it}' + (' (IC)' if it == 0 else '') + (' (final)' if it == 10000 else '')
    ax.plot(x_vals, eval_radial(coeff), color=color, label=label)

ax.set_title('time evolution: hot IC → equilibrium')
ax.set_xlabel('p_x')
ax.set_ylabel('f')
ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()

if save:
    plt.savefig('./figures/evolution.png', dpi=150)
    print("saved ./figures/evolution.png")

if show:
    plt.show()
