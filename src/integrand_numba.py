import numpy as np
from numba import njit
from basis_numba import gen_laguerre, assoc_legendre

# --- postcoll functions (njit) ---

@njit
def sph_to_cart(r, t, p):
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return x, y, z

@njit
def cart_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    t = np.arccos(z / r)
    p = np.arctan2(y, x)
    return r, t, p

@njit
def s_scalar(px, py, pz, qx, qy, qz):
    norm_p = np.sqrt(px**2 + py**2 + pz**2)
    norm_q = np.sqrt(qx**2 + qy**2 + qz**2)
    dot    = px*qx + py*qy + pz*qz
    return np.sqrt(max(0.0, 2 * (norm_p * norm_q - dot)))

@njit
def kernel(px, py, pz, qx, qy, qz):
    s    = s_scalar(px, py, pz, qx, qy, qz)
    norm = np.sqrt(px**2 + py**2 + pz**2) * np.sqrt(qx**2 + qy**2 + qz**2)
    return s**2 / (2 * norm)

@njit
def postcoll_F(rp, tp, pp, rq, tq, pq, tw, pw):
    px, py, pz = sph_to_cart(rp, tp, pp)
    qx, qy, qz = sph_to_cart(rq, tq, pq)
    wx, wy, wz = sph_to_cart(1.0, tw, pw)
    s  = s_scalar(px, py, pz, qx, qy, qz)
    D  = np.sqrt(px**2+py**2+pz**2) + np.sqrt(qx**2+qy**2+qz**2) + s
    sx, sy, sz = px+qx, py+qy, pz+qz
    dot = sx*wx + sy*wy + sz*wz
    fx = (sx/2) * (1 + dot/D) + (s/2) * wx
    fy = (sy/2) * (1 + dot/D) + (s/2) * wy
    fz = (sz/2) * (1 + dot/D) + (s/2) * wz
    return cart_to_sph(fx, fy, fz)

@njit
def postcoll_G(rp, tp, pp, rq, tq, pq, tw, pw):
    px, py, pz = sph_to_cart(rp, tp, pp)
    qx, qy, qz = sph_to_cart(rq, tq, pq)
    wx, wy, wz = sph_to_cart(1.0, tw, pw)
    s  = s_scalar(px, py, pz, qx, qy, qz)
    D  = np.sqrt(px**2+py**2+pz**2) + np.sqrt(qx**2+qy**2+qz**2) + s
    sx, sy, sz = px+qx, py+qy, pz+qz
    dot = sx*wx + sy*wy + sz*wz
    gx = (sx/2) * (1 - dot/D) - (s/2) * wx
    gy = (sy/2) * (1 - dot/D) - (s/2) * wy
    gz = (sz/2) * (1 - dot/D) - (s/2) * wz
    return cart_to_sph(gx, gy, gz)

# --- basis evaluation (njit) ---

@njit
def basis_eval(k, l, m, c, r, t, p):
    alpha = 2*l + 2
    lag   = gen_laguerre(k, alpha, r)
    leg   = assoc_legendre(l, m, np.cos(t))
    ang   = np.cos(m * p) if m >= 0 else np.sin(-m * p)
    return c * leg * ang * lag * r**l

@njit
def f_tilde_eval(k, l, m, mu, c, r, t, p):
    return mu * basis_eval(k, l, m, c, r, t, p)

# --- full quadrature loop (njit) ---
# quad must be a 2D numpy array of shape (N, 9): [rp,tp,pp,rq,tq,pq,tw,pw,w]

@njit
def operator_numba(k, l, m, c,
                   k1, l1, m1, mu1, c1,
                   k2, l2, m2, mu2, c2,
                   quad):
    result = 0.0
    for i in range(quad.shape[0]):
        rp, tp, pp = quad[i, 0], quad[i, 1], quad[i, 2]
        rq, tq, pq = quad[i, 3], quad[i, 4], quad[i, 5]
        tw, pw     = quad[i, 6], quad[i, 7]
        w          = quad[i, 8]

        px, py, pz = sph_to_cart(rp, tp, pp)
        qx, qy, qz = sph_to_cart(rq, tq, pq)
        kern = kernel(px, py, pz, qx, qy, qz)
        if kern == 0.0:
            continue

        rp_, tp_, pp_ = postcoll_F(rp, tp, pp, rq, tq, pq, tw, pw)
        rq_, tq_, pq_ = postcoll_G(rp, tp, pp, rq, tq, pq, tw, pw)

        gain = f_tilde_eval(k1, l1, m1, mu1, c1, rp_, tp_, pp_) * \
               f_tilde_eval(k2, l2, m2, mu2, c2, rq_, tq_, pq_)
        loss = f_tilde_eval(k1, l1, m1, mu1, c1, rp,  tp,  pp)  * \
               f_tilde_eval(k2, l2, m2, mu2, c2, rq,  tq,  pq)

        result += w * (gain - loss) * kern * basis_eval(k, l, m, c, rp, tp, pp)

    return result


def test():
    import time
    from quadrature import load_quad
    from basis_numba import spher_const, mu_const

    quad   = np.array(load_quad('./quadratures/collision.pkl'))
    select = [[1, 1, 0], [1, 0, 0], [1, 1, 0]]

    k,  l,  m  = select[0]; c   = spher_const(l,  m)
    k1, l1, m1 = select[1]; mu1 = mu_const(k1, l1); c1 = spher_const(l1, m1)
    k2, l2, m2 = select[2]; mu2 = mu_const(k2, l2); c2 = spher_const(l2, m2)

    # warmup — triggers JIT compilation
    print("compiling...")
    _ = operator_numba(k,l,m,c, k1,l1,m1,mu1,c1, k2,l2,m2,mu2,c2, quad[:100])

    # timed run
    print("running...")
    start  = time.time()
    result = operator_numba(k,l,m,c, k1,l1,m1,mu1,c1, k2,l2,m2,mu2,c2, quad)
    elapsed = time.time() - start

    print(f"select: {select}")
    print(f"result:  {result:.6f}")
    print(f"elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    test()
