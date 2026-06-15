import sympy as sp

# energy gradient
def energy_grad(energy):
    # Define symbols
    r = sp.symbols('r')
    t = sp.symbols('t')
    p = sp.symbols('p')
    r_hat = sp.Matrix([sp.sin(t)*sp.cos(p), sp.sin(t)*sp.sin(p), sp.cos(t)])

    # take partial
    partial     = sp.diff(energy, r)
    eg          = partial*r_hat

    # return 
    return eg

# kernel (projector)
def kernel(energy, verbose=False, rel=False):
    # symbols
    r, t, p     = sp.symbols('r t p')
    rp, tp, pp  = sp.symbols('rp tp pp')
    rq, tq, pq  = sp.symbols('rq tq pq')

    # obtain the energy
    eg = energy_grad(energy)
    
    if verbose:
        print("energy grad: ", eg)
        print()

    # substitutions
    p_sub = {r: rp, t: tp, p: pp}
    q_sub = {r: rq, t: tq, p: pq} 

    # energy gradients, evaluated at p and q
    ep = eg.subs(p_sub)
    eq = eg.subs(q_sub)

    # Alternative way
    u = ep - eq
        
    if verbose:
        print("u: ", u)

    kern =  u.dot(u)*sp.eye(3) - u*u.T
    # print( u*u.T)
    # altern = sp.simplify(altern)
    
    if rel:
        # Relativistic part
        z = (ep+eq)/2
        cross = z.cross(u)
        kern = kern - cross*cross.T
        # altern = sp.simplify(altern)
    
    if verbose:
        # check conservation
        print()
        print("energy: ")
        print(energy)
        print()
        print("kernel: ")
        print(kern)
        print()
        print("u - relative velocity: ")
        print(u)
        print()
        print("check conservation (S*u) should be zero 0")
        print()
        print(sp.simplify(kern * u))
        print()

    return kern

# cartesian energy grad
def energy_grad_cart():
    # Symbols
    x = sp.symbols('x')
    y = sp.symbols('y')
    z = sp.symbols('z')

    # norm an unit vector
    pos = sp.Matrix([x, y, z])
    r   = sp.sqrt(pos.dot(pos))
    print(r)
    
    # energy 
    e = sp.sqrt(1+r**2)

    # energy gradient
    eg = sp.Matrix([sp.diff(e, x), sp.diff(e, y), sp.diff(e,z)])
    eg = sp.simplify(eg)

    return eg

# A test
def test():
    # energy 
    r = sp.symbols('r')
    # e = sp.sqrt(1+r**2)
    e = (r**2)/2

    # compute the kernel
    rel     = False
    verbose = True
    kern = kernel(e, verbose, rel) 

    return 0

# The main function
def main():
    test()

if __name__ == "__main__":
    main()
