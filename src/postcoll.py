import numpy as np

# Convert spherical (r, t, p) to Cartesian
def sph_to_cart(r, t, p):
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return np.array([x, y, z])

# Convert Cartesian vector to spherical (r, t, p)
def cart_to_sph(v):
    r = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    t = np.arccos(v[2] / r)
    p = np.arctan2(v[1], v[0])
    return r, t, p

# Shared scalar: s = sqrt(2(|p||q| - p.q))
def s_scalar(p_vec, q_vec):
    norm_p = np.linalg.norm(p_vec)
    norm_q = np.linalg.norm(q_vec)
    return np.sqrt(2 * (norm_p * norm_q - np.dot(p_vec, q_vec)))

# Kernel: 1 - (p.q)/(|p||q|) = s^2 / (2|p||q|)
def kernel(p_vec, q_vec):
    s    = s_scalar(p_vec, q_vec)
    norm = np.linalg.norm(p_vec) * np.linalg.norm(q_vec)
    return s**2 / (2 * norm)

# Post-collisional momentum p' = F(p, q, omega)
def F(rp, tp, pp, rq, tq, pq, tw, pw):
    p_vec = sph_to_cart(rp, tp, pp)
    q_vec = sph_to_cart(rq, tq, pq)
    w_vec = sph_to_cart(1.0, tw, pw)

    s     = s_scalar(p_vec, q_vec)
    D     = np.linalg.norm(p_vec) + np.linalg.norm(q_vec) + s
    pq_vec = p_vec + q_vec

    result = (pq_vec / 2) * (1 + np.dot(pq_vec, w_vec) / D) + (s / 2) * w_vec
    return cart_to_sph(result)

# Post-collisional momentum q' = G(p, q, omega)
def G(rp, tp, pp, rq, tq, pq, tw, pw):
    p_vec = sph_to_cart(rp, tp, pp)
    q_vec = sph_to_cart(rq, tq, pq)
    w_vec = sph_to_cart(1.0, tw, pw)

    s     = s_scalar(p_vec, q_vec)
    D     = np.linalg.norm(p_vec) + np.linalg.norm(q_vec) + s
    pq_vec = p_vec + q_vec

    result = (pq_vec / 2) * (1 - np.dot(pq_vec, w_vec) / D) - (s / 2) * w_vec
    return cart_to_sph(result)

# A test
def test():
    print(f"{'Trial':<6} {'|p prime| num':>14} {'|p prime| ana':>14} {'|q prime| num':>14} {'|q prime| ana':>14}")
    print("-" * 65)

    for i in range(10):
        # random spherical coords: r in (0,3], t in (0,pi), p in (0,2pi)
        rp = np.random.uniform(0.1, 3.0)
        tp = np.random.uniform(0.01, np.pi - 0.01)
        pp = np.random.uniform(0.0, 2*np.pi)

        rq = np.random.uniform(0.1, 3.0)
        tq = np.random.uniform(0.01, np.pi - 0.01)
        pq = np.random.uniform(0.0, 2*np.pi)

        tw = np.random.uniform(0.01, np.pi - 0.01)
        pw = np.random.uniform(0.0, 2*np.pi)

        fp = F(rp, tp, pp, rq, tq, pq, tw, pw)
        gq = G(rp, tp, pp, rq, tq, pq, tw, pw)

        p_vec = sph_to_cart(rp, tp, pp)
        q_vec = sph_to_cart(rq, tq, pq)
        w_vec = sph_to_cart(1.0, tw, pw)

        E_p_ana = (rp + rq)/2 + 0.5 * np.dot(w_vec, p_vec + q_vec)
        E_q_ana = (rp + rq)/2 - 0.5 * np.dot(w_vec, p_vec + q_vec)

        p_ok = np.isclose(fp[0], E_p_ana)
        q_ok = np.isclose(gq[0], E_q_ana)
        status = "OK" if (p_ok and q_ok) else "FAIL"
        print(f"{i+1:<6} {fp[0]:>14.8f} {E_p_ana:>14.8f} {gq[0]:>14.8f} {E_q_ana:>14.8f}  {status}")

def main():
    test()

if __name__ == "__main__":
    main()
