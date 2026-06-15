import multiprocessing
import time
import pickle
import sympy as sp

# import pieces from operator
from kern import kernel
from landau import operator_parallel, load_quad
from sparse_rules import andrea, cai

# create iterable
def create_param_iterable(n):
    param = []

    # iterate for all test functions
    for k in range(0, n):
        for l in range(0, n):
            for m in range(-l, l+1):

                # iterate over all f(p)
                for k1 in range(0, n):
                    for l1 in range(0, n):
                        for m1 in range(-l1, l1+1):
               
                            # iterate over all dg(q)    
                            for k2 in range(0, n):
                                for l2 in range(0, n):
                                    for m2 in range(-l2, l2+1): 

                                        # create the select
                                        select = [[k,l,m], [k1,l1,m1], [k2,l2,m2]]

                                        # skip conservation laws
                                        mass = [k,l,m] == [0,0,0]
                                        px   = [k,l,m] == [0,1,-1]
                                        py   = [k,l,m] == [0,1,0]
                                        pz   = [k,l,m] == [0,1,1]
                                        e    = [k,l,m] == [1,0,0] 
                                        
                                        flag = mass or px or py or pz or e
                                        
                                        if not flag and (andrea(select) and cai(select)):
                                            param.append(select)
    print("number of coefficients to compute: ", len(param))
    return param

# parallel iterator
def parallel(sd, n):
    '''
    Arguments:
        - sd: the shared data. [quadrature, kernel]
        - n:  corresponds to max k, l
    '''
    # obtain the list of parameters
    params = create_param_iterable(n)

    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use starmap to pass the shared data to each worker
        results = pool.starmap(operator_parallel, [(select, sd) for select in params])

    # r.append(results) 
    return results

# produce collision matrix
def parallel_setup(n, energy, rel):
    # load quadrature
    quad = load_quad()
    print("quadrature size: ", len(quad))

    # produce the symbolic kernel, dependent on the energy
    verbose = False
    kern    = kernel(energy, verbose, rel)

    # define the shared data
    sd = [quad, kern] # this has to be in accordance to the unpacking at "operator_parallel"
                
    # compute it and time it
    start = time.time()
    r = parallel(sd, n)
    end = time.time()

    # Calculate elapsed time
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.6f} seconds") 
    print()
    return r

def compute_col_tensor():
    # relativistic flag
    rel     = True
    # conservative flag
    cons    = False

    # select the degrees of freedom
    n       = 3

    # where the result will be saved
    file_name = 'results/sparsity_test.pkl'
    
    '''
    Choose the energy
    '''
    # energy = sp.sqrt(1+r**2)  # relativistic
    # energy = r**3             # polynomial
    # energy = 1 + 0.43991322*r**2 - 0.0338162*r**4
    # energy = 1.0 + 0.37438846*r**2 + 0.01891801*r**4 + 0.00058631*r**6 - (6.71019908e-06)*r**8
    # energy = 1.0 + 0.35854196*r**2 - 0.01482466*r**4  + 0.00028524*r**6

    # load the energy
    if rel:
        if cons: 
            # open the numerical energy
            with open('../cheby/eh.pkl', 'rb') as f:
                energy = pickle.load(f)
        else: 
            # radial symbol
            r = sp.symbols('r')
            energy = sp.sqrt(1+r**2)  # relativistic

    else:
        # radial symbol
        r = sp.symbols('r')
        energy = (1/2)*r**2

    # compute the tensor, results will contain it
    result = parallel_setup(n, energy, rel)

    # print the result
    print(result)

    # save the result
    with open(file_name, 'wb') as file:
        pickle.dump(result, file)
        print("the result has been saved at ", file_name)

# main function
if __name__ == "__main__":
    compute_col_tensor()
    # n = 3
    # create_param_iterable(n)
