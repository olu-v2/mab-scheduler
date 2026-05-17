import random
from typing import Dict
from utils import _reset_vms, _fitness_assignment, _assignment_to_schedule

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

    vms = _reset_vms(vms)
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