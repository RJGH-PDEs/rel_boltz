import numpy as np
import pickle
from bilinear import boltzmann, update

# TODO: the handling of t=0 singularity needs to be made more rigorous,
# in consultation with Shu. For now we start at a small t_0 > 0 and
# use a fixed timestep. The prefactor t^{-3/2} is applied explicitly.

# Parameters
n              = 2          # truncation (must match collision tensor and mass matrix)
alpha          = 1.0        # alpha = 2*q, q = 1/2
t0             = 0.01       # starting time (small but nonzero to avoid singularity)
dt             = 1e-5       # timestep (small due to t^{-3/2} prefactor near t=0)
NUM_ITERATIONS = 10000      # number of timesteps
save_every     = 100        # save coefficients every this many steps
save           = True

size = n**3

# Load mass matrix inverse and sparse collision tensor
with open('../src/mass/mass.pkl', 'rb') as f:
    M = pickle.load(f)
    Mi = np.linalg.inv(M)

with open('../src/sparse_operators/collision_tensor.pkl', 'rb') as f:
    so = pickle.load(f)

# Save coefficient vector at iteration i
def save_coeff(i, coeff):
    name = f"../plot/coeff/{i}.pkl"
    with open(name, 'wb') as f:
        pickle.dump(coeff, f)

# Initial condition: near-equilibrium perturbation
# f[0] = 1 is the equilibrium (Juttner), small perturbation in f[1]
f      = np.zeros(size)
f[0]   = 1.0
f[1]   = 0.1

if save:
    save_coeff(0, f)

result = np.zeros(size)
t      = t0

# Time evolution: forward Euler with explicit t^{-3/2} prefactor
# G^{n+1} = G^n + dt * t^{-3/2} * M^{-1} Q(G^n, G^n)
for i in range(1, NUM_ITERATIONS + 1):
    prefactor = t**(-3 * alpha / 2)

    boltzmann(so, f, result)
    f = f + dt * prefactor * (Mi @ result)

    t += dt

    if save and i % save_every == 0:
        save_coeff(i, f)
        print(f"iter {i:6d}  t={t:.6f}  |f|={np.linalg.norm(f):.6f}")

print("\nfinal f:")
print(f)
boltzmann(so, f, result)
print("Q(f, f):")
print(result)
