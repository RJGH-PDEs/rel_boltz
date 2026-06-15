# numpy, sympy
import numpy as np
import sympy as sp
import pickle
# basis
from basis import basis, mu_const
# integration
from quadrature import unpack_mass_quad, load_mass_quad
# index
from sparse import ind

# the integrand for the mass matrix
def integrand(f, phi, point):
    # the evaluation point
    r = point[0]
    t = point[1]
    p = point[2]

    # parts
    trial   = f(r, t, p)
    test    = phi(r, t, p)

    return trial*test

# produces the pieces given a selection of the indices
def pieces(select):
    # test function
    ki = select[0][0]
    li = select[0][1]
    mi = select[0][2]
    # print(kj, lj, mj)

    # trial function
    kj = select[1][0]
    lj = select[1][1]
    mj = select[1][2]
    # print(ki, li, mi)

    # produce symbolic pieces
    test_sym    = basis(ki, li, mi)
    # print("test  function: ", test_sym)

    # the two basis functiosn will include the mu constant
    f_sym       = basis(kj, lj, mj)*mu_const(kj, lj)
    # print("basis function: ", f_sym) 

    # lambdafy
    r, t, p = sp.symbols('r t p')

    # numpy pieces
    f       = sp.lambdify((r, t, p), f_sym, modules = 'numpy')              # trial function
    test    = sp.lambdify((r, t, p), test_sym, modules = 'numpy')           # test function
    
    # return the pieces
    return f, test

# coefficient
def coefficient(select, quad):
    # produce the numpy pieces
    f, test = pieces(select)

    # numerical integration
    partial_sum = 0
    for quad in quad:
        # unpack the quadrature 
        weight, points = unpack_mass_quad(quad)
        # perform the partial sum

        sample = integrand(f, test, points)
        partial_sum = partial_sum + weight*sample

    return partial_sum

# test a coefficient
def coeff_test():
    # select
    test   = [0, 1, 1]
    basis  = [0, 1, 1]
    select = [test, basis]
    
    # obtain the mass quadrature
    quad = load_mass_quad()

    # test
    coeff = coefficient(select, quad)
    print("for: ", select, " the coeff is: ", coeff)

# builds the mass matrix 
def mass_matrix(n):
    # number of degrees of freedom, max value "l" can take
    N = n**3

    # obtain the mass quadrature
    quad = load_mass_quad()

    # build the empty matrix
    M = np.zeros((N,N))

    # iterate over test functions
    for k in range(0, n):
        for l in range(0, n):
            for m in range(-l, l+1):

                # iterate over all basis functions
                for k1 in range(0, n):
                    for l1 in range(0, n):
                        for m1 in range(-l1, l1+1):

                            # select
                            test   = [k, l, m]
                            basis  = [k1, l1, m1]
                            select = [test, basis]
                            
                            # indices
                            i = ind(k, l, m, n)     # test 
                            j = ind(k1, l1, m1, n)  # trial
                            M[i][j] = coefficient(select, quad)
 
    return M

# saves the mass inverse as pkl
def save_inv_mass():
    # value of n, determines the number of dof
    n = 3

    # coeff_test()
    m = mass_matrix(n)

    # invert
    m_inv = np.linalg.inv(m)

    # check that these are inverses
    # print(np.dot(m, m_inv))

    # save full quadrature
    with open('./mass/mass_inv.pkl', 'wb') as file:
        pickle.dump(m_inv, file)

    print("mass inverse has been saved.")
 
# The main function
def main():
    # coeff_test()
    save_inv_mass() # save mass matrix

if __name__ == "__main__":
    main()
