from typing import List, Dict
from custom_types import Task, VM
from utils import _reset_vms, _proxy_exec_time

def max_min(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Max-Min: assign the task with the maximum minimum completion time.
    Balances workload by filling slower paths first.
    """
    vms_copy = _reset_vms(vms)
    assignment = {}
    remaining = list(tasks)

    while remaining:
        task_min = {}
        task_vm = {}
        for task in remaining:
            best_vm, best_finish = None, float('inf')
            for vm in vms_copy:
                if task.num_pes > vm.num_pes:
                    continue
                start = max(vm.available_at, task.submit_time)
                finish = start + _proxy_exec_time(task, vm)
                if finish < best_finish:
                    best_vm, best_finish = vm, finish
            task_min[task.id] = best_finish
            task_vm[task.id] = (best_vm, best_finish)

        chosen = max(remaining, key=lambda t: task_min[t.id])
        vm, finish = task_vm[chosen.id]
        assignment[chosen.id] = vm.id
        vm.available_at = finish
        remaining.remove(chosen)

    return assignment