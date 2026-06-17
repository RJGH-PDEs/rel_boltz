import numpy as np
from quadrature import load_quad
from integrand import pieces, integrand

# The Boltzmann collision operator: loops over quadrature and accumulates
def operator(phi, ft, gt, quad):
    result = 0
    for pt in quad:
        w     = pt[-1]
        point = pt[:-1]
        result += w * integrand(phi, ft, gt, point)
    return result

# TODO: parallel version of the operator (see parallel.py for the Landau version)
# def operator_parallel(select, quad):
#     phi, ft, gt = pieces(select)
#     result      = operator(phi, ft, gt, quad)
#     print("select:", select, "result:", result)
#     return [select, result]

# End-to-end test
def operator_test(select):
    quad = load_quad('./quadratures/collision.pkl')
    print("quadrature points:", len(quad))

    phi, ft, gt = pieces(select)
    result      = operator(phi, ft, gt, quad)

    print("select:", select)
    print("result:", result)

def test():
    select = [[0, 0, 0], [1, 0, 0], [1, 0, 0]]
    operator_test(select)

def main():
    test()

if __name__ == "__main__":
    main()
