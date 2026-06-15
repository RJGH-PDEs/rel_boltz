import numpy as np
import pickle

def test():
    # open the sparse operator
    with open('../src/sparse_operators/test.pkl', 'rb') as file:
        so = pickle.load(file)

    # print(so)

    # check individual entries
    p = 18
    q = 15
    w = 24
    sparse_matrix = so[w]
    print("random entry: ", sparse_matrix[p,q])

    '''
    check the contraction
    '''
    result = np.zeros(27)

    # test it on equilibrium
    f = np.zeros(27)
    f[0] = 1 # this should be the equilibrium
    
    # apply the contraction
    landau(so, f, result)

    # print the result
    print("Landau operator on equilibrium: ")
    print(result)
    print()
    
    # change to non-equilibrium
    f[0] = 0
    f[3] = 1 # this should be the equilibrium
    print("non-equilibrium f: ")
    print(f)
    print()

    landau(so, f, result)
    print("Landau operator on non-equilibrium: ")
    print(result)
    

# the biliear Landau operator
def landau(tensor, f, result):
    '''
    computes Q(f, f) where f is given as a vector.
    the sparse operator is applied like: (f^T)sparse_op(f),
    producing a vector, whose entries are put into result vector
    
    Here, tensor should be a list of matrices, where each matrix
    is the collision operator, with the (i-th) test function fixed
    '''
    i = 0
    # compute 
    for mat in tensor:
        # compute the rayleigh form
        r = f @ (mat.dot(f))
        # store it 
        result[i] = r
        i = i + 1

# update: copies b into a
def update(a, b):
    i = 0

    for val in b:
        a[i] = val
        i = i + 1

# main function
def main():
    test()

if __name__ == "__main__":
    main()

'''
result = np.zeros(27)

for j in range(27):
    print("j ", j)
    # multiply by a vector
    vector = np.zeros(27)
    vector[j] = 1
    landau(so, vector, result)
    print(result)


# we now test it 
f = np.zeros(27)
f[20] = 1
result = np.zeros(27)
result[3] = 12

landau(so, f, result)

print()
for number in result:
    print(number)


# change f
g = np.random.rand(27)

landau(so, g, result)

print()
for number in result:
    print(number)
 
# multiply by a vector
vector = np.zeros(27)
vector[8] = 1 # <-- the equilibrium

# multiply by this matrix
product = sparse_matrix.dot(vector)
print(vector @ product)

# compute the
for i in range(27):
    # :print(i)
    sm = so[i]
    print("result: ", vector @ (sm.dot(vector)))
for j in range(27):
    print("j ", j)
    # multiply by a vector
    vector = np.zeros(27)
    vector[j] = 1

    # compute the
    for i in range(27):
        # :print(i)
        sm = so[i]
        print("result: ", vector @ (sm.dot(vector)))
'''
