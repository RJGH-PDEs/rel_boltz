import tntorch as tn
import torch
from sparse import load_operator
from functools import partial
import random

torch.set_default_dtype(torch.float64)

def target(t, p, q, tensor):  # Input arguments are vectors
    return tensor[int(t)][int(p),int(q)]
    # return np.exp(-t**2) + np.exp(-p**2)

# wrapper with extra inputs
def tar_wrap(t: torch.Tensor, p: torch.Tensor, q: torch.Tensor, tens) -> torch.Tensor:
    results = []
    for ti, pi, qi in zip(t, p, q):
        val = target(ti.item(), pi.item(), qi.item(), tens)  # convert tensor → float
        results.append(val)
    return torch.tensor(results)  # shape: [N]

# low-rank tensor train
def find_rank(N, tensor):
    # build the domain
    domain = [torch.arange(0, N) for n in range(3)]
    # print("domain: ", domain)

    # get the function
    to_sample = partial(tar_wrap, tens = tensor)

    # tensor-train factorization
    t = tn.cross(function=to_sample, domain=domain, kickrank=4, max_iter=10)
    
    # print rank
    print(t.ranks_tt)

    # check an index

    for _ in range(100):
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        k = random.randint(0, N-1)
        index = (i, j, k)
        print("index: ", index)
        print("low rank: ", t[index].item())
        print("full: ", tensor[i][j,k])

# the main function
def main():
    # number of degrees of freedom
    N = 27

    # obtain tensor
    name    = "./sparse_operators/test.pkl" 
    tensor  = load_operator(name)
    
    # tensor train
    find_rank(N, tensor)

if __name__ == "__main__":
    main()
