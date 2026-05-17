from typing import Dict
from utils import _reset_vms

def sufferage(tasks, vms) -> Dict:
    """
    Sufferage Heuristic
    -------------------
    Computes for each unassigned task:
        sufferage(t) = EFT_2nd_best(t) - EFT_best(t)

    Assigns the task with the HIGHEST sufferage to its best VM.
    A task that suffers most from missing its best VM is scheduled first.

    Complexity: O(n^2 * m)

    Reference: Maheswaran et al. (1999). Dynamic Mapping of a Class of
    Independent Tasks onto Heterogeneous Computing Systems.
    """
    vms_copy  = _reset_vms(vms)
    schedule  = {}
    remaining = list(tasks)

    while remaining:
        best_task_idx  = None
        best_vm_ref    = None
        best_sufferage = -1.0

        for i, task in enumerate(remaining):
            efts = []
            for vm in vms_copy:
                start  = max(vm.available_at, task.submit_time)
                finish = start + task.length / vm.mips
                efts.append((finish, vm))
            efts.sort(key=lambda x: x[0])

            eft1 = efts[0][0]
            eft2 = efts[1][0] if len(efts) > 1 else eft1
            suf  = eft2 - eft1

            if suf > best_sufferage:
                best_sufferage = suf
                best_task_idx  = i
                best_vm_ref    = efts[0][1]

        task   = remaining.pop(best_task_idx)
        vm     = best_vm_ref
        start  = max(vm.available_at, task.submit_time)
        finish = start + task.length / vm.mips

        schedule[task.id]  = (vm.id, start, finish)
        vm.available_at    = finish

    return schedule