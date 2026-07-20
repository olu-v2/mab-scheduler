"""
Six Low-Level Heuristics (LLHs) for the MAB Hyper-Heuristic Cloud Scheduler.

Algorithms:
1. Sufferage
2. HEFT with Duplication (insertion-based)
3. WOA - Whale Optimisation Algorithm
4. ACO - Ant Colony Optimisation
5. PSO - Particle Swarm Optimisation
6. SA - Simulated Annealing

All heuristics return:
    schedule : Dict {task_id: vm_id}

Execution timing (start/finish), makespan, cost, carbon, and utilization
are NOT computed here. That responsibility is fully delegated to
CloudSim Plus via the Java bridge. Any timing computed inside this file
(e.g. _proxy_exec_time, _fitness_assignment) is a cheap internal proxy
used only to guide search/ranking decisions -- never a reported metric.
"""

import math
import random
from typing import List, Dict, Tuple

from utils import _reset_vms, _proxy_exec_time, _fitness_assignment, assignment_from_vector

def _feasible_vms_per_task(tasks, vms):
    return [[j for j, vm in enumerate(vms) if task.num_pes <= vm.num_pes]
            for task in tasks]

def _snap_to_feasible(val, feasible_list):
    idx = int(abs(round(val))) % len(feasible_list)
    return feasible_list[idx]

# =============================================================================
# 1. SUFFERAGE
# =============================================================================

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
    vms_copy = _reset_vms(vms)
    assignment: Dict[int, int] = {}
    remaining = list(tasks)

    while remaining:
        best_task_idx = None
        best_vm_ref = None
        best_sufferage = -1.0

        for i, task in enumerate(remaining):
            efts = []
            for vm in vms_copy:
                if task.num_pes > vm.num_pes:
                    continue
                start = max(vm.available_at, task.submit_time)
                finish = start + _proxy_exec_time(task, vm)
                efts.append((finish, vm))
            if not efts:
                continue
            efts.sort(key=lambda x: x[0])

            eft1 = efts[0][0]
            eft2 = efts[1][0] if len(efts) > 1 else eft1
            suf = eft2 - eft1

            if suf > best_sufferage:
                best_sufferage = suf
                best_task_idx = i
                best_vm_ref = efts[0][1]

        if best_task_idx is None:
            # No remaining task is feasible on any VM; assign the rest
            # to the largest-PE VM as a safe fallback rather than looping forever.
            fallback_vm = max(vms_copy, key=lambda v: v.num_pes)
            for task in remaining:
                assignment[task.id] = fallback_vm.id
            break

        task = remaining.pop(best_task_idx)
        vm = best_vm_ref
        start = max(vm.available_at, task.submit_time)
        finish = start + _proxy_exec_time(task, vm)

        assignment[task.id] = vm.id
        vm.available_at = finish

    return assignment

# =============================================================================
# 2. HEFT WITH DUPLICATION (insertion-based)
# =============================================================================

def heft_with_duplication(tasks, vms) -> Dict:
    """
    HEFT with Insertion Scheduling
    --------------------------------
    1. Rank tasks by descending avg ETC (upward rank proxy for independent tasks).
    2. For each task, scan each PE-feasible VM's timeline for the EARLIEST
       IDLE SLOT that fits the task (insertion scheduling -- fills gaps
       before appending).
    3. Assign to the VM-slot that gives the minimum Earliest Finish Time.

    The insertion step is the 'duplication-aware' enhancement over plain HEFT:
    idle slots left by previously assigned tasks are exploited.

    Reference: Topcuoglu et al. (2002) HEFT; HH-LiSch insertion variant.
    """
    vms = _reset_vms(vms)
    avg_mips = sum(vm.mips for vm in vms) / len(vms)
    ranked = sorted(tasks, key=lambda t: t.length / avg_mips, reverse=True)

    vm_slots: Dict[int, List[Tuple[float, float]]] = {vm.id: [] for vm in vms}
    assignment: Dict[int, int] = {}

    def _earliest_slot(vm, task, ready: float) -> Tuple[float, float]:
        """Find earliest idle gap on vm where task fits at or after ready."""
        exec_t = _proxy_exec_time(task, vm)
        slots = vm_slots[vm.id]
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
            if task.num_pes > vm.num_pes:
                continue
            st, ft = _earliest_slot(vm, task, task.submit_time)
            if ft < best_ft:
                best_vm, best_st, best_ft = vm, st, ft

        if best_vm is None:
            # No VM can satisfy this task's PE requirement; fall back
            # to the largest-PE VM rather than leaving it unassigned.
            best_vm = max(vms, key=lambda v: v.num_pes)
            best_st, best_ft = _earliest_slot(best_vm, task, task.submit_time)

        vm_slots[best_vm.id].append((best_st, best_ft))
        vm_slots[best_vm.id].sort(key=lambda x: x[0])
        assignment[task.id] = best_vm.id

    return assignment

# =============================================================================
# 6. SA - Simulated Annealing
# =============================================================================

def simulated_annealing(tasks, vms,
                         T_init: float = 1000.0,
                         T_min: float = 1.0,
                         alpha: float = 0.995,
                         n_steps: int = 500,
                         seed: int = 42) -> Dict:
    """
    Simulated Annealing (SA)
    -------------------------
    Metropolis acceptance criterion:

        P(accept worse) = exp(-delta_E / T)

    where delta_E = new_makespan - current_makespan.
    Temperature cools geometrically: T <- alpha * T.

    Neighbourhood operator: single random task reassignment, constrained
    to that task's PE-feasible VM set so no mutation can ever introduce
    an infeasible placement.
    Seeded with a greedy min-min (PE-aware) assignment to start near a
    good, feasible solution.

    Reference: Kirkpatrick et al. (1983). Optimization by simulated annealing.
    Science, 220(4598), 671-680.
    """
    random.seed(seed)
    n_t, n_v = len(tasks), len(vms)
    feasible = _feasible_vms_per_task(tasks, vms)

    # Greedy min-min seed, restricted to each task's feasible VM set.
    current = [min(feasible[i], key=lambda j: _proxy_exec_time(tasks[i], vms[j]))
               for i in range(n_t)]
    current_fit = _fitness_assignment(current, tasks, vms)
    best, best_fit = list(current), current_fit

    T = T_init
    while T > T_min:
        for _ in range(n_steps):
            neighbour = list(current)
            i = random.randint(0, n_t - 1)
            neighbour[i] = random.choice(feasible[i])

            new_fit = _fitness_assignment(neighbour, tasks, vms)
            delta = new_fit - current_fit

            if delta < 0 or random.random() < math.exp(-delta / T):
                current, current_fit = neighbour, new_fit
                if current_fit < best_fit:
                    best_fit, best = current_fit, list(current)

        T *= alpha

    return assignment_from_vector(best, tasks)

# =============================================================================
# LLH REGISTRY
# =============================================================================

LLH_POOL = {
    "sufferage": sufferage,
    "heft_duplication": heft_with_duplication,
    "simulated_annealing": simulated_annealing,
}