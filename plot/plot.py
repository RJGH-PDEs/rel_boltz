import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from lc import linear_comb

os.makedirs("figures", exist_ok=True)
os.makedirs("coeff", exist_ok=True)

'''
flags
'''
# truncation parameter (must match the tensor used)
N = 2
# time
time = 1000
# flag to save the figure
save = True
# either to use hard-coded coefficients, or open file
hard_coded_coeff = True
# show the plot, or just save it
show = False

'''
end of flags
'''
# generate cartesian points
n = 40
x = np.linspace(-5, 5, n)
y = np.zeros(n)
z = np.zeros(n)

# generate spherical cordinates
def theta(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)

    if r == 0:
        return 0
    else:
        return np.arccos(z/ r)
    
def phi(x, y):
    r = np.sqrt(x**2 + y**2)

    if r == 0:
        return 0
    elif y == 0:
        return np.arccos(x/r) # new discovery, this might be wrong
    else:
        return np.sign(y)*np.arccos(x/r)
    
def radius(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

# spherical points
r = np.zeros(n)
t = np.zeros(n)
p = np.zeros(n)

# counter
i = 0
for point in x:
    # compute the points
    r[i] = radius(x[i], y[i], z[i])
    t[i] = theta(x[i], y[i], z[i])
    p[i] = phi(x[i], y[i])
    # advance the counter
    i = i + 1

# print the points
'''
print(r)
print(t)
print(p)
'''

# will store the function
f = np.zeros(n)

if hard_coded_coeff:
    # coefficients: equilibrium [0,0,0]  radial perturbation [1,0,0]
    coeff = np.zeros(N**3)
    coeff[0] =   2   # [0,0,0]: equilibrium
    coeff[4] =  -1   # [1,0,0]: radial perturbation
    plt_name = "equilibrium + radial perturbation"
else: 
    file_name = "coeff/" + str(time) + ".pkl"
    with open(file_name, 'rb') as file:
        coeff = pickle.load(file)
    plt_name = 'Solution at after ' + str(time) + ' iterations' 

for i in range(n):
    func_val = linear_comb(coeff, r[i], t[i], p[i], N)
    f[i] = np.exp(-r[i] / 2) * func_val

# Create the plot
plt.plot(x, f, marker='o')  # marker='o' will put points at each (x, y)
plt.title(plt_name)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
   
# show it
if show:
    plt.show()

# save it
if save:
    # Save the figure
    if hard_coded_coeff:
        figure_name = "./figures/hardcoded.png"
    else:
        figure_name = "./figures/" + str(time) + ".png"
    
    plt.savefig(figure_name)
    print("plot ", figure_name, " has been saved.")
    

