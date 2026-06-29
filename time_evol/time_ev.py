import sys
import json
import numpy as np
import pickle
from math import sqrt
sys.path.insert(0, '../src')
from bilinear import boltzmann
from mass_matrix import mass_name, load_mass
from sparse import sparse_name, ind

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
# All three cases share the same "hot radial" base (f[0]=2.0, f[9]=-0.8); they
# differ only in the angular perturbation added on top. CASE_INFO is the single
# source of truth for each case's physical significance (also dumped to the
# run-meta file so the exported writeup stays in sync with the actual run).
CASE_INFO = {
    'radial': dict(
        label='Hot radially-symmetric',
        significance=(
            "Isotropic hot initial state with no angular perturbation. The "
            "distribution stays spherically symmetric and thermalizes to the "
            "isotropic Jüttner equilibrium. Serves as the control: the "
            "asymmetry diagnostic is identically zero throughout."
        ),
    ),
    'dipole': dict(
        label='Dipole (net momentum)',
        significance=(
            "Hot radial base plus an l=1, m=0 dipole (k=0) that carries a net "
            "p_z momentum. Because total momentum is conserved, the asymmetry "
            "cannot relax away — the system thermalizes to a Jüttner "
            "distribution boosted along z, so the p_z -> -p_z asymmetry persists."
        ),
    ),
    'zero_momentum': dict(
        label='Zero-momentum asymmetry',
        significance=(
            "Hot radial base plus an l=1, m=0 perturbation built from the k=1 "
            "and k=2 radial modes in the ratio f[20]/f[11] = 1/sqrt(3), chosen "
            "so the net p_z momentum (Mf)[2] is exactly zero while the state is "
            "still asymmetric under p_z -> -p_z. With no conserved momentum to "
            "protect it, the asymmetry decays back to ~0 as the system relaxes "
            "to the isotropic equilibrium."
        ),
    ),
}
CASE = 'zero_momentum'   # 'radial' | 'dipole' | 'zero_momentum'

f    = np.zeros(size)
f[0] = 2.0    # ind(0,0,0,3)  hot radial base
f[9] = -0.8   # ind(1,0,0,3)  (shared by all three cases)

if CASE not in CASE_INFO:
    raise ValueError(f"unknown CASE: {CASE}")

if CASE == 'dipole':
    f[2] = 0.1             # ind(0,1,0,3)  dipole — carries net p_z, persists
elif CASE == 'zero_momentum':
    f[11] = 0.1            # ind(1,1,0,3)
    f[20] = 0.1 / sqrt(3)  # ind(2,1,0,3)  ratio cancels net p_z -> asymmetry decays

# ── run metadata (consumed by ../time_evol/export_experiment.py) ────────────────
# Decode each nonzero IC index back to (k, l, m) using the project index map.
_klm = {ind(k, l, m, n): (k, l, m)
        for k in range(n) for l in range(n) for m in range(-l, l + 1)}
ic_entries = [[int(i), list(_klm[i]), float(f[i])]
              for i in np.nonzero(f)[0]]
run_meta = dict(
    case=CASE,
    label=CASE_INFO[CASE]['label'],
    significance=CASE_INFO[CASE]['significance'],
    n=n, n_laguerre=n_laguerre, n_lebedev=n_lebedev,
    t0=t0, dt=dt, num_iterations=NUM_ITERATIONS, save_every=save_every,
    ic_entries=ic_entries,
)
if save:
    with open('../plot/coeff/run_meta.json', 'w') as mf:
        json.dump(run_meta, mf, indent=2)
    print(f"wrote run_meta.json for case '{CASE}'")

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
