from typing import List, Dict
from custom_types import Task, VM
from utils import _reset_vms, _proxy_exec_time

def min_min(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Min-Min: iteratively pick the task with the minimum completion
    time across all VMs and assign it.
    """
    vms_copy = _reset_vms(vms)
    assignment = {}
    remaining = list(tasks)

    while remaining:
        best_task, best_vm, best_finish = None, None, float('inf')
        for task in remaining:
            for vm in vms_copy:
                if task.num_pes > vm.num_pes:
                    continue
                start = max(vm.available_at, task.submit_time)
                finish = start + _proxy_exec_time(task, vm)
                if finish < best_finish:
                    best_task, best_vm, best_finish = task, vm, finish

        assignment[best_task.id] = best_vm.id
        best_vm.available_at = best_finish
        remaining.remove(best_task)

    return assignment