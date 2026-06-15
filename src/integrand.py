import sympy as sp
import numpy as np

# symbolic parts
from kern import kernel
from basis import basis, grad_weighted, gradient, mu_const

# the integrand
def integrand(k, f, g, gt, points):
    '''
    Input:
        - k:    the kernel
        - f:    basis function (p)
        - g:    gradient of basis function  -- Gaussian weight (q)
        - gt:   gradient of test function   -- no weight (p - q)
        - pts:   coordinates of p, q in spherical coordinates

    Output: 
        - the integrand, evaluated at p, q
    '''
    '''
    the evaluation points, 
    just be careful about this ordering. 
    The order comes from unpack_quadrature.
    points = [r_p, t_p, p_p, r_q, t_q, p_q]
    '''
    # p
    rp = points[0]
    tp = points[1]
    pp = points[2]
    # q
    rq = points[3]
    tq = points[4]
    pq = points[5]

    # obtain parts
    a = f(rp, tp, pp)*g(rq, tq, pq)
    b = k(rp, tp, pp, rq, tq, pq)@(gt(rp, tp, pp) - gt(rq, tq, pq))
    
    # inner product
    result = np.dot(a.flatten(),b.flatten())

    return result

# produces the pieces given a selection of the indices
def pieces(select, k_sym):
    # test function
    k = select[0][0]
    l = select[0][1]
    m = select[0][2]
    
    # f
    k1 = select[1][0]
    l1 = select[1][1]
    m1 = select[1][2]
    
    # g
    k2 = select[2][0]
    l2 = select[2][1]
    m2 = select[2][2]

    # produce symbolic pieces
    test_sym    = gradient(basis(k, l, m))
    # print("test: ", test_sym)
    # print()
    # the two basis functiosn will include the mu constant
    f_sym       = basis(k1, l1, m1)*mu_const(k1, l1)
    # print("f: ", f_sym)
    # print()
    g_sym       = grad_weighted(basis(k2, l2, m2))*mu_const(k2, l2)
    # print("g: ", g_sym)
    # print()
    
    # lambdafy
    rp, tp, pp, rq, tq, pq = sp.symbols('rp tp pp rq tq pq')
    r, t, p = sp.symbols('r t p')

    # numpy pieces
    k       = sp.lambdify((rp, tp, pp, rq, tq, pq), k_sym, modules='numpy') # kernel
    f       = sp.lambdify((r, t, p), f_sym, modules = 'numpy')              # trial function
    g       = sp.lambdify((r, t, p), g_sym, modules = 'numpy')              # gradient trial function
    test    = sp.lambdify((r, t, p), test_sym, modules = 'numpy')           # test function
    
    # return the pieces
    return k, f, g, test

# testing the integrand
def test():
    # choose the three functions
    # test function, phi
    k = 1
    l = 0
    m = 0
    
    # trial function f(p), no gradient
    k1 = 0
    l1 = 1
    m1 = 0
    
    # trial function \nabla g(p), gradient
    k2 = 2
    l2 = 0
    m2 = 0

    # package 
    select = [[k,l,m],[k1,l1,m1],[k2,l2,m2]]

    # define the points
    rp = 4
    tp = np.pi/6
    pp = np.pi/3

    rq = 1
    tq = np.pi/5
    pq = np.pi/6

    points = [rp, tp, pp, rq, tq, pq] 

    # produce the sympy parts
    # radial symbol
    r = sp.symbols('r')
    # energy = sp.sqrt(1+r**2)  # relativistic
    energy = (r**2)/2 # non-relativistic

    # symbolic kernel 
    rel  = False
    verb = False
    kern = kernel(energy, verb, rel)

    # construct the pieces
    k, f, g, test = pieces(select, kern)

    # call the integrand
    print(integrand(k, f, g, test, points))

# The main function
def main():
    test()

if __name__ == "__main__":
    main()   
