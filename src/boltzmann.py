import numpy as np
import multiprocessing.shared_memory as shm
from quadrature import load_quad

# Shared memory block holding the quadrature array — created once in the main
# process and attached read-only by all workers, avoiding 128x memory copies.
_quad_np   = None
_shm_block = None
_shm_name  = None
_shm_shape = None
_shm_dtype = None

def create_shared_quad(quad_path):
    """Load quadrature and copy into a shared memory block. Call from main process."""
    global _shm_block, _shm_name, _shm_shape, _shm_dtype
    quad, n_laguerre, n_lebedev = load_quad(quad_path)
    arr = np.array(quad, dtype=np.float64)
    _shm_block = shm.SharedMemory(create=True, size=arr.nbytes)
    shared = np.ndarray(arr.shape, dtype=arr.dtype, buffer=_shm_block.buf)
    shared[:] = arr
    _shm_name  = _shm_block.name
    _shm_shape = arr.shape
    _shm_dtype = arr.dtype
    return n_laguerre, n_lebedev

def free_shared_quad():
    """Release shared memory. Call from main process after pool is done."""
    global _shm_block
    if _shm_block is not None:
        _shm_block.close()
        _shm_block.unlink()
        _shm_block = None

def init_worker(shm_name, shape, dtype):
    """Attach to the shared memory block — runs once per worker process."""
    global _quad_np
    block    = shm.SharedMemory(name=shm_name)
    _quad_np = np.array(np.ndarray(shape, dtype=dtype, buffer=block.buf))
    block.close()

def operator_parallel_numba(select):
    from integrand_numba import operator_numba
    from basis_numba import spher_const, mu_const
    k,  l,  m  = select[0]; c   = spher_const(l,  m)
    k1, l1, m1 = select[1]; mu1 = mu_const(k1, l1); c1 = spher_const(l1, m1)
    k2, l2, m2 = select[2]; mu2 = mu_const(k2, l2); c2 = spher_const(l2, m2)
    result = operator_numba(k,l,m,c, k1,l1,m1,mu1,c1, k2,l2,m2,mu2,c2, _quad_np)
    return [select, result]
