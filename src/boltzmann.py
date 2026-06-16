import sympy as sp
# import quadrature unpacker
from quadrature import unpack_quad, load_quad
# import integrand
from integrand import pieces, integrand
# symbolic parts
from kern import kernel

# The Landau Operator
def operator(k, f, g, test, quadrature):
    # numerical integration
    integral = 0

    for q in quadrature:
        # unpack quadrature 
        weight, points = unpack_quad(q)
        # sample the function
        sample = integrand(k, f, g, test, points)
        # update sum
        integral = integral + weight*sample
        # print(sum)

    return integral

# test the operator
def operator_parallel(select, shared_data):
    # unpack shared data
    quad        = shared_data[0]
    sym_kern    = shared_data[1]

    # produce the numpy pieces
    k, f, g, test = pieces(select, sym_kern)

    # compute the landau operator
    result = operator(k, f, g, test, quad)

    # print results
    print("select: ", select, "result: ", result)

    return [select, result]

# test the operator
def operator_test(select, energy, rel):
    # load the quadrature
    quad = load_quad()

    # print the size of the quadrature
    print("quadrature length: ", len(quad))
    print()

    # produce the symbolic pieces
    verbose         = True
    sym_kern        = kernel(energy, verbose, rel)
    # print(sym_kern)
    k, f, g, test   = pieces(select, sym_kern)

    # compute the landau operator
    result = operator(k, f, g, test, quad)

    # print results
    print("select: ", select)
    print("result: ", result ) 

# The main function
def test():
   # test function
    k = 2
    l = 2
    m = 2
    
    # f
    k1 = 2
    l1 = 2
    m1 = 2
    
    # g
    k2 = 2
    l2 = 2
    m2 = 0

    select = [[k,l,m],[k1,l1,m1],[k2,l2,m2]]

    '''
    Choose the energy
    '''
    # radial symbol
    r = sp.symbols('r')
 
    # relativistic flag
    rel = True
    if rel:
        energy = sp.sqrt(1+r**2)    # relativistic
    else:
        energy = (1/2)*r**2         # non-relativistic

    # test the operator
    operator_test(select, energy, rel)

# main function
def main():
    test()

if __name__ == "__main__":
    main()
