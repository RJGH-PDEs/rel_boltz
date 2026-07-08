"""
Search for a double-hump T=2 radial IC.

With n=4 and T=2, the constraint on (c1,c2,c3) leaves 2 free parameters.
We fix c0=2, sweep (c1,c3), compute c2 from the T=2 constraint numerically,
then check for:  (1) double hump  (2) f >= 0 everywhere.

Run from src/.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, '.')
from basis_numba import mu_const, spher_const, basis_eval
from sparse import ind

n  = 4
N4 = n**3

# ── helpers ───────────────────────────────────────────────────────────────────

def lc(coeff, r, t, p):
    s = 0.0
    for k in range(n):
        for l in range(n):
            for m in range(-l, l + 1):
                s += coeff[ind(k,l,m,n)] * mu_const(k,l) * basis_eval(k,l,m,spher_const(l,m),r,t,p)
    return s

def f_val(coeff, r):
    return np.exp(-r/2) * lc(coeff, r, 0.0, 0.0) if r > 0 else lc(coeff, 0.0, 0.0, 0.0)

def make_coeff(c0, c1, c2, c3):
    f = np.zeros(N4)
    f[ind(0,0,0,n)] = c0; f[ind(1,0,0,n)] = c1
    f[ind(2,0,0,n)] = c2; f[ind(3,0,0,n)] = c3
    return f

def temperature(c0, c1, c2, c3):
    r  = np.linspace(0.0, 30.0, 3000)
    fv = np.array([f_val(make_coeff(c0,c1,c2,c3), ri) for ri in r])
    M  = 4 * np.pi * np.trapezoid(fv * r**2, r)
    E  = 4 * np.pi * np.trapezoid(fv * r**3, r)
    return E / (3 * M)

def find_c2(c0, c1, c3, n_scan=40):
    """Scan c2 to find a bracket where T crosses 2, then brentq."""
    c2_scan = np.linspace(-6.0, 6.0, n_scan)
    T_scan  = []
    for c2 in c2_scan:
        try:
            T_scan.append(temperature(c0, c1, c2, c3))
        except Exception:
            T_scan.append(np.nan)
    for i in range(len(T_scan) - 1):
        a, b = T_scan[i], T_scan[i+1]
        if np.isfinite(a) and np.isfinite(b) and (a - 2.0) * (b - 2.0) < 0:
            try:
                return brentq(lambda c2: temperature(c0,c1,c2,c3) - 2.0,
                              c2_scan[i], c2_scan[i+1], xtol=1e-5)
            except Exception:
                pass
    return None

def hump_info(r_arr, f_arr):
    """Return (is_double_hump, r_dip, r_peak2) if f has dip then recovery."""
    local_min, local_max = [], []
    for i in range(1, len(f_arr) - 1):
        if f_arr[i] < f_arr[i-1] and f_arr[i] < f_arr[i+1]:
            local_min.append(i)
        elif f_arr[i] > f_arr[i-1] and f_arr[i] > f_arr[i+1]:
            local_max.append(i)
    for i_dip in local_min:
        later = [j for j in local_max if j > i_dip]
        if later:
            return True, r_arr[i_dip], r_arr[later[0]], f_arr[i_dip], f_arr[later[0]]
    return False, None, None, None, None

# ── 2-D sweep over (c1, c3) ───────────────────────────────────────────────────

r_eval  = np.linspace(0.0, 20.0, 600)
c0      = 2.0
c1_vals = np.linspace(-2.0, 2.0, 41)
c3_vals = np.linspace(-1.5, 1.5, 31)

candidates = []
print("Searching (c1, c3) grid …")
for c1 in c1_vals:
    for c3 in c3_vals:
        c2 = find_c2(c0, c1, c3)
        if c2 is None:
            continue
        coeff = make_coeff(c0, c1, c2, c3)
        fv    = np.array([f_val(coeff, r) for r in r_eval])
        fmin  = fv.min()
        dh, r_dip, r_peak, f_dip, f_peak = hump_info(r_eval, fv)
        if dh:
            candidates.append(dict(c1=c1, c2=c2, c3=c3, fmin=fmin,
                                   r_dip=r_dip, r_peak=r_peak,
                                   f_dip=f_dip, f_peak=f_peak, fv=fv))

print(f"\nFound {len(candidates)} double-hump candidates.")
pos = [c for c in candidates if c['fmin'] >= 0]
print(f"{len(pos)} are positive (f >= 0 everywhere).\n")

if pos:
    print(f"{'c1':>6}  {'c2':>7}  {'c3':>6}  {'fmin':>10}  {'r_dip':>6}  {'r_peak':>7}  {'f_peak':>8}")
    for c in pos:
        print(f"{c['c1']:6.3f}  {c['c2']:7.4f}  {c['c3']:6.3f}  "
              f"{c['fmin']:10.2e}  {c['r_dip']:6.2f}  {c['r_peak']:7.2f}  {c['f_peak']:8.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────

if not candidates:
    print("Nothing to plot.")
    sys.exit(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: all double-hump candidates (positive = solid, negative = dashed)
ax = axes[0]
ax.set_title(f'All double-hump candidates (T=2, n={n})')
cmap   = plt.cm.coolwarm
c1s    = [c['c1'] for c in candidates]
lo, hi = min(c1s), max(c1s)
for c in candidates:
    col = cmap((c['c1'] - lo) / (hi - lo + 1e-12))
    lw  = 1.8 if c['fmin'] >= 0 else 0.6
    ls  = '-'  if c['fmin'] >= 0 else '--'
    ax.plot(r_eval, c['fv'], color=col, linewidth=lw, linestyle=ls)
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlabel('r'); ax.set_ylabel('f(r)'); ax.grid(True)
ax.text(0.98, 0.95, 'solid = f≥0', transform=ax.transAxes,
        ha='right', va='top', fontsize=9)

# Right: best positive candidate vs Jüttner
ax = axes[1]
if pos:
    # pick the one with the largest second peak / most pronounced hump
    best = max(pos, key=lambda c: c['f_peak'])
    c1b, c2b, c3b = best['c1'], best['c2'], best['c3']
    ax.set_title(f'Best positive double-hump  c1={c1b:.3f}, c2={c2b:.4f}, c3={c3b:.3f}')
    ax.plot(r_eval, best['fv'], 'b-', linewidth=2, label='IC')
    r_ref = np.linspace(0.0, 20.0, 400)
    c_best = make_coeff(c0, c1b, c2b, c3b)
    fv_ref = np.array([f_val(c_best, r) for r in r_ref])
    M  = 4*np.pi*np.trapezoid(fv_ref*r_ref**2, r_ref)
    E  = 4*np.pi*np.trapezoid(fv_ref*r_ref**3, r_ref)
    T  = E/(3*M); A = M/(8*np.pi*T**3)
    ax.plot(r_ref, A*np.exp(-r_ref/T), 'k--', linewidth=1.5,
            label=f'Jüttner A={A:.3f}, T={T:.3f}')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(best['r_dip'],  color='r', linewidth=0.8, linestyle=':', label=f"dip r={best['r_dip']:.1f}")
    ax.axvline(best['r_peak'], color='g', linewidth=0.8, linestyle=':', label=f"peak r={best['r_peak']:.1f}")
    ax.set_xlabel('r'); ax.set_ylabel('f(r)'); ax.legend(fontsize=8); ax.grid(True)
else:
    # closest to positive
    best = min(candidates, key=lambda c: abs(c['fmin']))
    ax.set_title(f'Closest-to-positive  fmin={best["fmin"]:.2e}')
    ax.plot(r_eval, best['fv'], 'r--', linewidth=2,
            label=f"c1={best['c1']:.3f}, c3={best['c3']:.3f}")
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel('r'); ax.set_ylabel('f(r)'); ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig('double_hump_candidates.png', dpi=150)
print("\nsaved double_hump_candidates.png")
plt.show()
