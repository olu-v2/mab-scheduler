from typing import List, Dict
from custom_types import Task, VM
from utils import _reset_vms

def heft(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Heterogeneous Earliest Finish Time (HEFT).
    Sorts tasks by descending length/pes ratio (upward rank proxy),
    assigns each to the VM giving earliest finish time.
    """
    vms_copy = _reset_vms(vms)
    schedule = {}
    sorted_tasks = sorted(tasks, key=lambda t: t.length / t.num_pes, reverse=True)
    for task in sorted_tasks:
        best_vm, best_start, best_finish = None, 0, float('inf')
        for vm in vms_copy:
            start  = max(vm.available_at, task.submit_time)
            exec_t = task.length / vm.mips
            finish = start + exec_t
            if finish < best_finish:
                best_vm, best_start, best_finish = vm, start, finish
        schedule[task.id] = (best_vm.id, best_start, best_finish)
        best_vm.available_at = best_finish
    return schedule