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
tag        = ''   # e.g. '_postfix', '_tacc' — to load tensors/quadratures from a tagged run

t0             = 1.0
dt             = 1e-3
NUM_ITERATIONS = 10000
save_every     = 100
save           = True

size = n**3

# ── load ──────────────────────────────────────────────────────────────────────
M  = load_mass('../src/' + mass_name(n, n_laguerre, tag=tag))
Mi = np.linalg.inv(M)

with open('../src/' + sparse_name(n, n_laguerre, n_lebedev, tag=tag), 'rb') as f:
    so = pickle.load(f)

# ── initial condition ─────────────────────────────────────────────────────────
# hot radial IC
f    = np.zeros(size)
f[0] = 2.0    # ind(0,0,0,3)
f[9] = -0.8   # ind(1,0,0,3)

# equilibrium from previous runs
f_eq    = np.zeros(size)
f_eq[0]  =  2.37806729
f_eq[9]  = -0.36344549
f_eq[18] =  0.15434533

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
        print(f"iter {i:6d}  t={t:.6f}  |f|={np.linalg.norm(f):.6f}  f[2]={f[2]:.6e}")
    if save and (i <= 20 or i % save_every == 0):
        save_coeff(i, f)

print("\nfinal f:")
print(f)
print(f"f[2] (dipole mode): {f[2]:.6e}")
boltzmann(so, f, result)
print("Q(f,f):")
print(result)
