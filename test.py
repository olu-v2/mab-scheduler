from utils import load_swf, generate_vms
from collections import Counter

vms = generate_vms(n_vms=8)
max_vm_pes = max(v.num_pes for v in vms)
tasks = load_swf("NASA-iPSC-1993-3.1-cln.swf", max_tasks=1000, max_vm_pes=max_vm_pes)

pes_dist = Counter(t.num_pes for t in tasks)
print("num_pes distribution:", sorted(pes_dist.items()))
print("length range:", min(t.length for t in tasks), max(t.length for t in tasks))
print("VM tiers (mips, pes):", [(v.mips, v.num_pes) for v in vms])

from min_min import min_min
from heuristics import sufferage, heft_with_duplication, simulated_annealing
from utils import _reset_vms

batch = tasks[:10]
for name, fn in [("min_min", min_min), ("sufferage", sufferage),
                  ("heft_duplication", heft_with_duplication),
                  ("simulated_annealing", lambda t, v: simulated_annealing(t, v, n_steps=100, T_init=500))]:
    vms_fresh = _reset_vms(vms)
    print(name, fn(batch, vms_fresh))