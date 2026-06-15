import matplotlib.pyplot as plt
import numpy as np
from lc import linear_comb
import pickle

'''
flags
'''
# time 
time = 1000
# flag to save the figure
save = True
# either to use hard-coded coefficients, or open file
hard_coded_coeff = False
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
    # coefficients
    coeff = np.zeros(27)
    coeff[0] = 1
    coeff[1] = 0
    coeff[9] = 0
    plt_name = "hard-coded coefficients"
else: 
    file_name = "coeff/" + str(time) + ".pkl"
    with open(file_name, 'rb') as file:
        coeff = pickle.load(file)
    plt_name = 'Solution at after ' + str(time) + ' iterations' 

# counter
for i in range(n):
    func_val = linear_comb(coeff, r[i], t[i], p[i])
    f[i] = np.exp((-r[i]**2)/2)*func_val

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
    

