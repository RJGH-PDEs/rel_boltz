import numpy as np

from basis import basis, f_tilde
from postcoll import F, G, kernel, sph_to_cart

# Build the three callables for a given index selection
# select = [[k, l, m], [k1, l1, m1], [k2, l2, m2]]
#   [k,  l,  m ]: test function phi
#   [k1, l1, m1]: trial function f_tilde
#   [k2, l2, m2]: trial function g_tilde
def pieces(select):
    phi = basis(  *select[0])
    ft  = f_tilde(*select[1])
    gt  = f_tilde(*select[2])
    return phi, ft, gt

# Evaluate the integrand at a single quadrature point
# point = [rp, tp, pp, rq, tq, pq, tw, pw]  (weight excluded)
def integrand(phi, ft, gt, point):
    rp, tp, pp, rq, tq, pq, tw, pw = point

    # kernel vanishes when p || q — check first to avoid evaluating F, G
    p_vec = sph_to_cart(rp, tp, pp)
    q_vec = sph_to_cart(rq, tq, pq)
    kern  = kernel(p_vec, q_vec)

    if kern == 0.0:
        return 0.0

    # post-collisional spherical coordinates
    rp_, tp_, pp_ = F(rp, tp, pp, rq, tq, pq, tw, pw)
    rq_, tq_, pq_ = G(rp, tp, pp, rq, tq, pq, tw, pw)

    # gain and loss terms
    gain = ft(rp_, tp_, pp_) * gt(rq_, tq_, pq_)
    loss = ft(rp,  tp,  pp)  * gt(rq,  tq,  pq)

    return (gain - loss) * kern * phi(rp, tp, pp)

# A test
def test():
    select = [[0, 0, 0], [1, 0, 0], [1, 0, 0]]
    phi, ft, gt = pieces(select)

    rp, tp, pp = 2.0, np.pi/4, np.pi/3
    rq, tq, pq = 1.0, np.pi/3, np.pi/6
    tw, pw     = np.pi/5, np.pi/4

    point  = [rp, tp, pp, rq, tq, pq, tw, pw]
    result = integrand(phi, ft, gt, point)
    print("integrand value:", result)

# Equilibrium test: f_tilde = g_tilde = const (k=l=m=0) => gain - loss = 0
# regardless of the test function, for any quadrature point
def test_equilibrium():
    np.random.seed(0)
    for test_select in [[0,0,0], [1,0,0], [1,1,1]]:
        select = [test_select, [0,0,0], [0,0,0]]
        phi, ft, gt = pieces(select)

        all_ok = True
        for _ in range(10):
            rp = np.random.uniform(0.1, 3.0)
            tp = np.random.uniform(0.01, np.pi - 0.01)
            pp = np.random.uniform(0.0, 2*np.pi)
            rq = np.random.uniform(0.1, 3.0)
            tq = np.random.uniform(0.01, np.pi - 0.01)
            pq = np.random.uniform(0.0, 2*np.pi)
            tw = np.random.uniform(0.01, np.pi - 0.01)
            pw = np.random.uniform(0.0, 2*np.pi)

            point  = [rp, tp, pp, rq, tq, pq, tw, pw]
            result = integrand(phi, ft, gt, point)
            if not np.isclose(result, 0.0):
                all_ok = False

        print(f"Equilibrium test (test={test_select}): {'OK' if all_ok else 'FAIL'}")

def main():
    test()
    print()
    test_equilibrium()

if __name__ == "__main__":
    main()
