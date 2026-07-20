from typing import List, Dict
from custom_types import Task, VM
from utils import _reset_vms, _proxy_exec_time

def heft(tasks: List[Task], vms: List[VM]) -> Dict:
    """
    Heterogeneous Earliest Finish Time (HEFT).
    Sorts tasks by descending length/pes ratio (upward rank proxy),
    assigns each to the VM giving earliest finish time.
    """
    vms_copy = _reset_vms(vms)
    assignment = {}
    sorted_tasks = sorted(tasks, key=lambda t: t.length / t.num_pes, reverse=True)
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