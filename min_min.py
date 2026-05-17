from typing import List, Dict
from custom_types import Task, VM
from utils import _reset_vms

def min_min(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Min-Min: iteratively pick the task with the minimum completion
    time across all VMs and assign it.
    """
    vms_copy = _reset_vms(vms)
    schedule = {}
    remaining = list(tasks)

    while remaining:
        best_task, best_vm_idx, best_start, best_finish = None, None, 0, float('inf')
        for task in remaining:
            for vm in vms_copy:
                start  = max(vm.available_at, task.submit_time)
                finish = start + task.length / vm.mips
                if finish < best_finish:
                    best_task, best_vm_idx = task, vm
                    best_start, best_finish = start, finish

        schedule[best_task.id] = (best_vm_idx.id, best_start, best_finish)
        best_vm_idx.available_at = best_finish
        remaining.remove(best_task)

    return schedule