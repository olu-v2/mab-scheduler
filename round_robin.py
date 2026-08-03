from typing import List, Dict
from custom_types import Task, VM

def round_robin(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Round-Robin: assign tasks cyclically to VMs by index,
    skipping VMs that cannot satisfy a task's PE requirement.
    """
    n_v = len(vms)
    assignment = {}
    idx = 0
    for task in tasks:
        attempts = 0
        while vms[idx % n_v].num_pes < task.num_pes and attempts < n_v:
            idx += 1
            attempts += 1
        if attempts == n_v:
            # No VM in the pool can satisfy this task; fall back to the
            # largest-PE VM rather than silently assigning an infeasible one.
            best_vm = max(vms, key=lambda v: v.num_pes)
            assignment[task.id] = best_vm.id
        else:
            assignment[task.id] = vms[idx % n_v].id
        idx += 1
    return assignment