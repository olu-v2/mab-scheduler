import random
from typing import Dict
from utils import _reset_vms, _fitness_assignment, _assignment_to_schedule

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
    vms = _reset_vms(vms)
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