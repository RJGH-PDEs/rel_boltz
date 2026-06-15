# Import
import numpy as np
import sympy as sp

from scipy.special import genlaguerre
from scipy.special import lpmv
from scipy.special import factorial
from scipy.special import gamma

# the mu_kl constant that makes the basis functions be an orthonormal system (see p. 348 of paper)
def mu_const(k, l):
    '''
    Goes in front of only the basis functions, not the test functions
    '''
    # compute the constant    
    result = 2 * factorial(k)
    result = result/gamma(k + l + 3/2)
    result = np.sqrt(result)
    
    # return result
    return result 

# The constant for the spherical harmonic
def spher_const(l,m):
    """
    The constant that goes in front of the Legendre polynomial to produce a spherical harmonic.
    """
    result = 0

    result = (2*l+1)/(2*np.pi)
    if m == 0:
        return np.sqrt(result/2)

    result = result*factorial(l-np.abs(m))
    # print(factorial(l-np.abs(m)))
    result = result/factorial(l+np.abs(m))
    # print(factorial(l+np.abs(m)))
    return np.sqrt(result)


# Phi(r): the radial part of the test functions
def Phi(l, k, r):
    """
    radial part of the test functions
    """
    result  = 0

    # Parametes for the Laguerre
    x       = r**2
    n       = k
    alpha   = l + 1/2

    result = genlaguerre(n, alpha)(x)*(r**(l))

    return result

# Legendre polynomial
def Leg(m, l, t):
    """
    Computes the Laguerre polynomial evaluated
    at x = cos(theta)
    """
    result = 0

    x = np.cos(t)
    result = lpmv(np.abs(m), l, x)

    return result 

# The part of the spherical harmonic that depend on phi (the azimuth)
def azimuth(m, p):
    """
    the part of the spherical harmonic that depends on phi.
    It should handle all possible cases wrt m
    """
    if m > 0:
        return np.cos(m*p)
    elif m == 0:
        return 1
    else:
        return np.sin(np.abs(m)*p)

# Test function (without the exponential weight)
def test(k, l, m, r, theta ,phi):
    """
    The test function
    """
    result = azimuth(m, phi)            # azimunth:     phi
    # print('azimuth (phi): ', azimuth(m, phi))

    result = result*Leg(m, l, theta)    # Legendre:     theta
    # print('Legendre (theta): ', Leg(m, l, theta))

    result = result*Phi(l, k, r)        # Phi:          r
    # print('radial: ', Phi(l, k, r))

    # print(' -> without constant: ', result) # before multiplying by constant
    
    result = result*spher_const(l, m)   # Constant      

    '''
    here we may or may not include the following constant,
    so that we can use it for the test functions as well
    '''
    result = result*mu_const(k, l) # the other constant

    return result

# symbolic test function (without the exponential weight)
def sym_test(k, l, m, rad, the, phi):
    result = 0

    # symbols
    r = sp.symbols('r')
    t = sp.symbols('t')
    p = sp.symbols('p')

    # alpha
    a = l + 1/2

    # Spherical harmonic
    sphr = sp.simplify(sp.assoc_legendre(l,abs(m), sp.cos(t)))
    sphr = sp.refine(sphr, sp.Q.positive(sp.sin(t)))

    if m >= 0:
        sphr = sphr*sp.cos(m*p)
    else:
        sphr = sphr*sp.sin(abs(m)*p)

    # Radial part
    radial = 1
    if k > 0:
        radial = sp.assoc_laguerre(k, a, r**2)
    radial = radial*r**l

    # the test function
    f = sphr*radial

    # print the test function
    print("test function: ", f)

    # evaluation at point
    point = {"r":rad,"t":the,"p":phi}
    result = f.subs(point)

    # multiply by constant 
    result = result*spher_const(l, m)
    
    # return
    return result

# function to be integrated
def f_integrated(select, rp, tp, pp, rq, tq, pq):
    # parameters for first function
    k_1 = select[0]
    l_1 = select[1]
    m_1 = select[2]
    # parameters for second function
    k_2 = select[3]
    l_2 = select[4]
    m_2 = select[5]

    # print(k_1, l_1, m_1, k_2, l_2, m_2)

    # return sym_test(k_1, l_1, m_1, rp, tp, pp)*sym_test(k_2, l_2, m_2, rq, tq, pq)
    return test(k_1, l_1, m_1, rp, tp, pp)*test(k_2, l_2, m_2, rq, tq, pq)

# The main function
def main():
    # Parameters
    k = 1
    l = 1
    m = 1
    
    # Coefficients
    r = 5
    theta = np.pi/5
    phi = np.pi/3.4

    # printing
    print('test function: ', test(k, l, m, r, theta, phi))
    print()
    print('symbolic test function: ', sym_test(k, l, m, r, theta, phi))

# Main function
if __name__ == "__main__":
    main()
