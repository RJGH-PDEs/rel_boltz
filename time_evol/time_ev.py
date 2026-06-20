import sys
import numpy as np
import pickle
sys.path.insert(0, '../src')
from bilinear import boltzmann
from mass_matrix import mass_name, load_mass
from sparse import sparse_name

# ── parameters ────────────────────────────────────────────────────────────────
n          = 3
n_laguerre = 7
n_lebedev  = 9

t0             = 0.01
dt             = 1e-6
NUM_ITERATIONS = 10000
save_every     = 100
save           = True

size = n**3

# ── load ──────────────────────────────────────────────────────────────────────
M  = load_mass('../src/' + mass_name(n, n_laguerre))
Mi = np.linalg.inv(M)

with open('../src/' + sparse_name(n, n_laguerre, n_lebedev), 'rb') as f:
    so = pickle.load(f)

# ── initial condition ─────────────────────────────────────────────────────────
f    = np.zeros(size)
f[0] = 2.0   # ind(0,0,0,3)
f[9] = -0.8  # ind(1,0,0,3)

# ── helpers ───────────────────────────────────────────────────────────────────
def save_coeff(i, coeff):
    name = f"../plot/coeff/{i}.pkl"
    with open(name, 'wb') as file:
        pickle.dump(coeff, file)

# ── time evolution ────────────────────────────────────────────────────────────
# df/dt = (1/2) t^{-3/2} M^{-1} Q(f, f)
# forward Euler: f^{n+1} = f^n + dt * (1/2) * t^{-3/2} * M^{-1} Q(f^n, f^n)

if save:
    save_coeff(0, f)

result = np.zeros(size)
t      = t0

for i in range(1, NUM_ITERATIONS + 1):
    prefactor = 0.5 * t**(-1.5)

    boltzmann(so, f, result)
    f = f + dt * prefactor * (Mi @ result)

    t += dt

    if i <= 20 or (save and i % save_every == 0):
        print(f"iter {i:6d}  t={t:.6f}  |f|={np.linalg.norm(f):.6f}")
    if save and (i <= 20 or i % save_every == 0):
        save_coeff(i, f)

print("\nfinal f:")
print(f)
boltzmann(so, f, result)
print("Q(f,f):")
print(result)
