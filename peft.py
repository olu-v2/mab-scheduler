from typing import List, Dict
from custom_types import Task, VM
from utils import _reset_vms, _proxy_exec_time

def peft(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Predict Earliest Finish Time (PEFT).
    Uses an Optimistic Cost Table (OCT): for each task estimates
    the best possible finish time across all VMs, then prioritises
    tasks by their OCT-based rank.
    """
    vms_copy = _reset_vms(vms)
    assignment = {}

    oct_rank = {}
    for task in tasks:
        best_times = [_proxy_exec_time(task, vm) for vm in vms_copy]
        oct_rank[task.id] = sum(best_times) / len(best_times)

    sorted_tasks = sorted(tasks, key=lambda t: oct_rank[t.id], reverse=True)
    for task in sorted_tasks:
        best_vm, best_finish = None, float('inf')
        for vm in vms_copy:
            if task.num_pes > vm.num_pes:
                    continue
            start = max(vm.available_at, task.submit_time)
            finish = start + _proxy_exec_time(task, vm)
            if finish < best_finish:
                best_vm, best_finish = vm, finish
        assignment[task.id] = best_vm.id
        best_vm.available_at = best_finish

    return assignment