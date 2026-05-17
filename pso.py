import random
from typing import Dict
from utils import _reset_vms, _fitness_assignment, _assignment_to_schedule

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
    vms = _reset_vms(vms)
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