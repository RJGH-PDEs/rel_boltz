import pickle
import numpy as np

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def to_dict(data):
    # convert list of [select, value] to dict keyed by hashable select
    d = {}
    for entry in data:
        select, value = entry
        key = (tuple(select[0]), tuple(select[1]), tuple(select[2]))
        d[key] = value
    return d

def compare(path_old, path_new, tol=1e-6):
    old = to_dict(load(path_old))
    new = to_dict(load(path_new))

    print(f"old entries: {len(old)}")
    print(f"new entries: {len(new)}")
    print()

    # entries in both
    common = set(old.keys()) & set(new.keys())
    only_old = set(old.keys()) - set(new.keys())
    only_new = set(new.keys()) - set(old.keys())

    print(f"common entries: {len(common)}")
    print(f"only in old:    {len(only_old)}")
    print(f"only in new:    {len(only_new)}")
    print()

    # compare common entries
    mismatches = []
    for key in common:
        v_old = old[key]
        v_new = new[key]
        if not np.isclose(v_old, v_new, rtol=tol):
            mismatches.append((key, v_old, v_new))

    if mismatches:
        print(f"MISMATCHES ({len(mismatches)}):")
        for key, v_old, v_new in mismatches:
            print(f"  select={list(key)}  old={v_old:.6f}  new={v_new:.6f}  diff={abs(v_old-v_new):.2e}")
    else:
        print(f"All {len(common)} common entries match to rtol={tol}  OK")

if __name__ == "__main__":
    compare('./results/old.pkl', './results/collision_tensor.pkl')
