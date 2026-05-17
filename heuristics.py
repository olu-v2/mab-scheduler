"""
Six Low-Level Heuristics (LLHs) for the MAB Hyper-Heuristic Cloud Scheduler.

Algorithms:
  1. Sufferage
  2. HEFT with Duplication (insertion-based)
  3. WOA  — Whale Optimisation Algorithm
  4. ACO  — Ant Colony Optimisation
  5. PSO  — Particle Swarm Optimisation
  6. SA   — Simulated Annealing

All heuristics return:
    schedule : Dict  {task_id: (vm_id, start_time, finish_time)}
"""

import math
import random
from typing import List, Dict, Tuple
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _reset_vms(vms):
    """Deep copy VMs so each heuristic run starts fresh."""
    from dataclasses import fields
    return [type(vms[0])(**{f.name: (0.0 if f.name == 'available_at' else getattr(v, f.name))
                            for f in fields(vms[0])})
            for v in vms]


def _fitness_from_schedule(schedule: Dict) -> float:
    """Return makespan from a {task_id: (vm_id, start, finish)} dict."""
    return max(ft for _, _, ft in schedule.values()) if schedule else 0.0


def _assignment_to_schedule(assignment: List[int], tasks, vms) -> Dict:
    """
    Convert integer assignment vector → schedule dict.
    assignment[i] = index in vms for tasks[i].
    """
    vms_copy = _reset_vms(vms)
    vm_map   = {vm.id: vm for vm in vms_copy}
    # rebuild ordered by vm to respect available_at correctly
    vm_clock = defaultdict(float)
    schedule = {}
    for i, task in enumerate(tasks):
        vm     = vms_copy[assignment[i]]
        exec_t = task.length / vm.mips
        start  = max(vm_clock[vm.id], task.submit_time)
        finish = start + exec_t
        schedule[task.id]  = (vm.id, start, finish)
        vm_clock[vm.id]    = finish
    return schedule


def _fitness_assignment(assignment: List[int], tasks, vms) -> float:
    """Makespan of an assignment vector (fast, no schedule dict)."""
    vm_clock = defaultdict(float)
    for i, task in enumerate(tasks):
        vm     = vms[assignment[i]]
        exec_t = task.length / vm.mips
        start  = max(vm_clock[vm.id], task.submit_time)
        vm_clock[vm.id] = start + exec_t
    return max(vm_clock.values()) if vm_clock else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SUFFERAGE
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  HEFT WITH DUPLICATION  (insertion-based)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  WOA — Whale Optimisation Algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def woa(tasks, vms,
        population: int = 30,
        max_iter:   int = 100,
        seed:       int = 42) -> Dict:
    """
    Whale Optimisation Algorithm (WOA)
    ------------------------------------
    Mimics humpback whale bubble-net hunting via three mechanisms:

      p < 0.5, |A| < 1  → Shrinking encircling  (exploitation)
      p < 0.5, |A| >= 1 → Random search          (exploration)
      p >= 0.5           → Spiral update          (exploitation)

    a decreases linearly 2 → 0; controls balance of exploration/exploitation.

    Position encoding: integer vector, pos[i] = VM index for tasks[i].
    Discretisation: int(abs(round(val))) % n_vms

    Reference: Mirjalili & Lewis (2016). The Whale Optimization Algorithm.
    Advances in Engineering Software, 95, 51-67.
    """
    random.seed(seed)
    n_t, n_v, b = len(tasks), len(vms), 1.0

    pop  = [[random.randint(0, n_v - 1) for _ in range(n_t)] for _ in range(population)]
    fits = [_fitness_assignment(p, tasks, vms) for p in pop]

    best_pos = list(pop[fits.index(min(fits))])
    best_fit = min(fits)

    for t in range(max_iter):
        a = 2.0 - t * (2.0 / max_iter)

        for i in range(population):
            r1, r2 = random.random(), random.random()
            A, C, p = 2 * a * r1 - a, 2 * r2, random.random()
            new_pos = []

            if p < 0.5:
                if abs(A) < 1:
                    # Shrinking encircling (exploitation)
                    for j in range(n_t):
                        D = abs(C * best_pos[j] - pop[i][j])
                        new_pos.append(int(abs(round(best_pos[j] - A * D))) % n_v)
                else:
                    # Random search (exploration)
                    rand = random.choice(pop)
                    for j in range(n_t):
                        D = abs(C * rand[j] - pop[i][j])
                        new_pos.append(int(abs(round(rand[j] - A * D))) % n_v)
            else:
                # Spiral update (exploitation)
                l = random.uniform(-1, 1)
                for j in range(n_t):
                    D   = abs(best_pos[j] - pop[i][j])
                    val = D * math.exp(b * l) * math.cos(2 * math.pi * l) + best_pos[j]
                    new_pos.append(int(abs(round(val))) % n_v)

            new_fit = _fitness_assignment(new_pos, tasks, vms)
            pop[i], fits[i] = new_pos, new_fit
            if new_fit < best_fit:
                best_fit, best_pos = new_fit, list(new_pos)

    return _assignment_to_schedule(best_pos, tasks, vms)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ACO — Ant Colony Optimisation
# ═══════════════════════════════════════════════════════════════════════════════

def aco(tasks, vms,
        n_ants:   int   = 20,
        max_iter: int   = 100,
        alpha:    float = 1.0,
        beta:     float = 2.0,
        rho:      float = 0.1,
        Q:        float = 1.0,
        seed:     int   = 42) -> Dict:
    """
    Ant Colony Optimisation (ACO)
    ------------------------------
    Each ant constructs a solution using:

        P(task_i -> vm_j) proportional to  tau[i][j]^alpha * eta[i][j]^beta

    where:
        tau[i][j] = pheromone on (task i, vm j)
        eta[i][j] = 1 / ETC(i, j)   (prefer faster VMs)

    After each iteration:
        Evaporation : tau <- (1 - rho) * tau
        Deposit     : best ant deposits Q / makespan on its path

    Reference: Dorigo & Gambardella (1997). Also used in
    'Implementation of ML Algorithm for Task Scheduling in Cloud Computing'.
    """
    random.seed(seed)
    n_t, n_v = len(tasks), len(vms)

    tau = [[1.0] * n_v for _ in range(n_t)]
    eta = [[1.0 / max(tasks[i].length / vms[j].mips, 1e-9)
            for j in range(n_v)] for i in range(n_t)]

    best_assignment, best_makespan = None, float('inf')

    for _ in range(max_iter):
        all_assignments, all_makespans = [], []

        for _ in range(n_ants):
            assignment = []
            for i in range(n_t):
                weights = [(tau[i][j] ** alpha) * (eta[i][j] ** beta) for j in range(n_v)]
                total   = sum(weights)
                probs   = [w / total for w in weights]
                r, cumul, chosen = random.random(), 0.0, n_v - 1
                for j, p in enumerate(probs):
                    cumul += p
                    if r <= cumul:
                        chosen = j
                        break
                assignment.append(chosen)

            mks = _fitness_assignment(assignment, tasks, vms)
            all_assignments.append(assignment)
            all_makespans.append(mks)
            if mks < best_makespan:
                best_makespan, best_assignment = mks, list(assignment)

        # Evaporation
        for i in range(n_t):
            for j in range(n_v):
                tau[i][j] = max(tau[i][j] * (1 - rho), 1e-6)

        # Best-ant deposit
        bi  = all_makespans.index(min(all_makespans))
        dep = Q / max(all_makespans[bi], 1e-9)
        for i, j in enumerate(all_assignments[bi]):
            tau[i][j] += dep

    return _assignment_to_schedule(best_assignment, tasks, vms)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PSO — Particle Swarm Optimisation
# ═══════════════════════════════════════════════════════════════════════════════

def pso(tasks, vms,
        population: int   = 80,
        max_iter:   int   = 200,
        W:          float = 1.0,
        C:          float = 0.9,
        v_scale:    float = 0.5,
        seed:       int   = 42) -> Dict:
    """
    Particle Swarm Optimisation (PSO)
    -----------------------------------
    Velocity update (from CA-MLBS Algorithm 1, Adil et al. 2022):

        v(t+1) = W * v(t) * v_scale
               + C * r * (pbest - pos)
               + C * r * (gbest - pos)

    Position encoding: continuous, discretised via int(abs(round(pos))) % n_vms.

    Parameters match CA-MLBS Table 4:
        population = 80,  max_iter = 200 (900 in full paper),
        W = 1.0,  C = 0.9,  v_scale = 0.5

    Reference: Adil et al. (2022). CA-MLBS. Expert Systems, e13150.
    """
    random.seed(seed)
    n_t, n_v = len(tasks), len(vms)

    positions  = [[random.randint(0, n_v - 1)  for _ in range(n_t)] for _ in range(population)]
    velocities = [[random.uniform(-n_v, n_v)   for _ in range(n_t)] for _ in range(population)]
    pbest_pos  = [list(p) for p in positions]
    pbest_fit  = [_fitness_assignment(p, tasks, vms) for p in positions]

    gi        = pbest_fit.index(min(pbest_fit))
    gbest_pos = list(pbest_pos[gi])
    gbest_fit = pbest_fit[gi]

    for _ in range(max_iter):
        for i in range(population):
            f = _fitness_assignment(positions[i], tasks, vms)
            if f < pbest_fit[i]:
                pbest_fit[i], pbest_pos[i] = f, list(positions[i])
            if pbest_fit[i] < gbest_fit:
                gbest_fit, gbest_pos = pbest_fit[i], list(pbest_pos[i])

        for i in range(population):
            for t in range(n_t):
                r = random.random()
                velocities[i][t] = (
                    W * velocities[i][t] * v_scale
                    + C * r * (pbest_pos[i][t] - positions[i][t])
                    + C * r * (gbest_pos[t]    - positions[i][t])
                )
                positions[i][t] = int(abs(round(positions[i][t] + velocities[i][t]))) % n_v

    return _assignment_to_schedule(gbest_pos, tasks, vms)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SA — Simulated Annealing
# ═══════════════════════════════════════════════════════════════════════════════

def simulated_annealing(tasks, vms,
                        T_init:  float = 1000.0,
                        T_min:   float = 1.0,
                        alpha:   float = 0.995,
                        n_steps: int   = 500,
                        seed:    int   = 42) -> Dict:
    """
    Simulated Annealing (SA)
    -------------------------
    Metropolis acceptance criterion:

        P(accept worse) = exp(-delta_E / T)

    where delta_E = new_makespan - current_makespan.
    Temperature cools geometrically: T <- alpha * T.

    Neighbourhood operator: single random task reassignment.
    Seeded with a greedy min-min assignment to start near a good solution.

    Reference: Kirkpatrick et al. (1983). Optimization by simulated annealing.
    Science, 220(4598), 671-680.
    """
    random.seed(seed)
    n_t, n_v = len(tasks), len(vms)

    # Greedy min-min seed
    current     = [min(range(n_v), key=lambda j: tasks[i].length / vms[j].mips)
                   for i in range(n_t)]
    current_fit = _fitness_assignment(current, tasks, vms)
    best, best_fit = list(current), current_fit

    T = T_init
    while T > T_min:
        for _ in range(n_steps):
            neighbour                           = list(current)
            neighbour[random.randint(0, n_t-1)] = random.randint(0, n_v - 1)

            new_fit = _fitness_assignment(neighbour, tasks, vms)
            delta   = new_fit - current_fit

            if delta < 0 or random.random() < math.exp(-delta / T):
                current, current_fit = neighbour, new_fit
                if current_fit < best_fit:
                    best_fit, best = current_fit, list(current)

        T *= alpha

    return _assignment_to_schedule(best, tasks, vms)


# ═══════════════════════════════════════════════════════════════════════════════
# LLH REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

LLH_POOL = {
    "sufferage":           sufferage,
    "heft_duplication":    heft_with_duplication,
    "woa":                 woa,
    "aco":                 aco,
    "pso":                 pso,
    "simulated_annealing": simulated_annealing,
}
