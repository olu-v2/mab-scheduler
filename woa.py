import random
from typing import Dict
from utils import _reset_vms, _fitness_assignment, _assignment_to_schedule

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
    vms = _reset_vms(vms)
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