from typing import List, Dict, Optional, Tuple
from utils import _reset_vms

def heft_with_duplication(tasks, vms) -> Dict:
    """
    HEFT with Insertion Scheduling
    --------------------------------
    1. Rank tasks by descending avg ETC (upward rank proxy for independent tasks).
    2. For each task, scan each VM's timeline for the EARLIEST IDLE SLOT that
       fits the task (insertion scheduling — fills gaps before appending).
    3. Assign to the VM-slot that gives the minimum Earliest Finish Time.

    The insertion step is the 'duplication-aware' enhancement over plain HEFT:
    idle slots left by previously assigned tasks are exploited.

    Reference: Topcuoglu et al. (2002) HEFT; HH-LiSch insertion variant.
    """
    vms = _reset_vms(vms)
    avg_mips = sum(vm.mips for vm in vms) / len(vms)
    ranked   = sorted(tasks, key=lambda t: t.length / avg_mips, reverse=True)

    # vm_slots[vm_id] = sorted list of (start, end) occupied intervals
    vm_slots: Dict[int, List[Tuple[float, float]]] = {vm.id: [] for vm in vms}
    vm_map   = {vm.id: vm for vm in vms}
    schedule = {}

    def _earliest_slot(vm, task, ready: float) -> Tuple[float, float]:
        """Find earliest idle gap on vm where task fits at or after ready."""
        exec_t   = task.length / vm.mips
        slots    = vm_slots[vm.id]
        prev_end = ready
        for (s, e) in slots:
            gap_start = max(prev_end, ready)
            if s - gap_start >= exec_t:
                return gap_start, gap_start + exec_t
            prev_end = e
        start = max(prev_end, ready)
        return start, start + exec_t

    for task in ranked:
        best_vm, best_st, best_ft = None, 0.0, float('inf')

        for vm in vms:
            st, ft = _earliest_slot(vm, task, task.submit_time)
            if ft < best_ft:
                best_vm, best_st, best_ft = vm, st, ft

        vm_slots[best_vm.id].append((best_st, best_ft))
        vm_slots[best_vm.id].sort(key=lambda x: x[0])
        schedule[task.id] = (best_vm.id, best_st, best_ft)

    return schedule