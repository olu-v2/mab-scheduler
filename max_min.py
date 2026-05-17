from typing import List, Dict, Optional, Tuple
from custom_types import Task, VM
from utils import _reset_vms

def max_min(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Max-Min: assign the task with the maximum minimum completion time.
    Balances workload by filling slower paths first.
    """
    vms_copy = _reset_vms(vms)
    schedule = {}
    remaining = list(tasks)

    while remaining:
        # For each task find its minimum finish time
        task_min = {}
        task_vm  = {}
        for task in remaining:
            best_vm, best_start, best_finish = None, 0, float('inf')
            for vm in vms_copy:
                start  = max(vm.available_at, task.submit_time)
                finish = start + task.length / vm.mips
                if finish < best_finish:
                    best_vm, best_start, best_finish = vm, start, finish
            task_min[task.id] = best_finish
            task_vm[task.id]  = (best_vm, best_start, best_finish)

        # Pick task with maximum of its minimum finish times
        chosen = max(remaining, key=lambda t: task_min[t.id])
        vm, start, finish = task_vm[chosen.id]
        schedule[chosen.id] = (vm.id, start, finish)
        vm.available_at = finish
        remaining.remove(chosen)

    return schedule