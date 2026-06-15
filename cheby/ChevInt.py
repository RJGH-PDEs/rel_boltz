import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev
import pickle

# Define the function to interpolate
def func(x):
    return np.sqrt(1+x*x)

# Gaussian weight
def gaussWeight(x):
	return np.exp(-x*x)

# Interpolate using Chebyshev polynomials
L = 8
deg = 12
cheb_poly = Chebyshev.interpolate(func, deg, [-L,L])

# Convert to standard polynomial form
cheb_poly_standard = cheb_poly.convert(kind=np.polynomial.Polynomial)
print(cheb_poly_standard)

# Save the standard form in sympy
r = sp.symbols('r')
coeffs = cheb_poly_standard.coef
expr = sum(c * r**i for i, c in enumerate(coeffs))
print(expr)

with open('eh.pkl', 'wb') as f:
    pickle.dump(expr, f)

# Evaluate the polynomial at a new set of x values
L_g = 1.1*L
x_new = np.linspace(-L_g, L_g, 500)  # New points to evaluate the interpolation
y_new = cheb_poly(x_new)

# Evaluate the original function at the new points for comparison
y_exact = func(x_new)
y_weight = gaussWeight(x_new)

# Plot the results
plt.plot(x_new, y_new, label='E_h')
plt.plot(x_new, y_exact, '--', label='E', color='green')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.suptitle('Chebyshev Interpolation with polynomial of degree '+str(deg))
#plt.title('Both weighted by the Gaussian')
plt.show()
