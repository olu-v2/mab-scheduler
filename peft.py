from typing import List, Dict
from custom_types import Task, VM
from utils import _reset_vms

def peft(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Predict Earliest Finish Time (PEFT).
    Uses an Optimistic Cost Table (OCT): for each task estimates
    the best possible finish time across all VMs, then prioritises
    tasks by their OCT-based rank.
    """
    vms_copy = _reset_vms(vms)
    schedule = {}

    # OCT rank: average best finish time across VMs
    oct_rank = {}
    for task in tasks:
        best_times = [task.length / vm.mips for vm in vms_copy]
        oct_rank[task.id] = sum(best_times) / len(best_times)

    sorted_tasks = sorted(tasks, key=lambda t: oct_rank[t.id], reverse=True)
    for task in sorted_tasks:
        best_vm, best_start, best_finish = None, 0, float('inf')
        for vm in vms_copy:
            start  = max(vm.available_at, task.submit_time)
            finish = start + task.length / vm.mips
            if finish < best_finish:
                best_vm, best_start, best_finish = vm, start, finish
        schedule[task.id] = (best_vm.id, best_start, best_finish)
        best_vm.available_at = best_finish
    return schedule