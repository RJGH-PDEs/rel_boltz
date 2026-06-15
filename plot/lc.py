# import numpy as np
from test_func import test

# l and m map
def lm_index(ll, m): 
    return ll*ll + (m + ll)

# k, l, m map
def ind(k, ll, m, n):
    '''
    here, we use the convenction that 
    l is between 0 and L
    '''
    return (n*n)*k + lm_index(ll, m)

def linear_comb(coefficients, r, t, p):
    # max k, l, m
    n = 3

    # partial result
    result = 0


    for k in range(0,n):
        for l in range(0, n):
            for m in range(-l, l+1):
                # print([k, l, m], ind(k, l, m, n))
                result = result + coefficients[ind(k, l, m, n)]*test(k, l, m, r, t, p)

    return result

'''
# open the result
with open('result.pkl', 'rb') as file:
    data = pickle.load(file)

i = 0
for s in data:
    print(i, data[i])
    i = i + 1

# test it
coeff = np.zeros(27)
coeff[0] = 1

linear_comb(coeff, 1, 0, 0)
'''
