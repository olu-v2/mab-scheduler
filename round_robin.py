from typing import List, Dict
from custom_types import Task, VM
from utils import _reset_vms

def round_robin(tasks: List[Task], vms: List[VM]) -> Dict:
    """Round-Robin: assign tasks cyclically to VMs."""
    vms_copy = _reset_vms(vms)
    schedule = {}
    for i, task in enumerate(tasks):
        vm     = vms_copy[i % len(vms_copy)]
        start  = max(vm.available_at, task.submit_time)
        finish = start + task.length / vm.mips
        schedule[task.id] = (vm.id, start, finish)
        vm.available_at = finish
    return schedule