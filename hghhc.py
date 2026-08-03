
# =============================================================================
# hghhc.py
# Henry Gas – Harris Hawks – Comprehensive Opposition (HGHHC) Algorithm
# For Cloud Task Scheduling
#
# Reference:
#   Alkaam et al. (2025). "Hybrid Henry Gas-Harris Hawks Comprehensive-
#   Opposition Algorithm for Task Scheduling in Cloud Computing."
#   IEEE Access, Vol. 13. DOI: 10.1109/ACCESS.2025.3530860
#
# Three-stage hybrid:
#   Stage 1 — Representation    : integer-encoded population (Eq. 6)
#   Stage 2 — Update            : HGSO (exploration) or HHO (exploitation)
#                                  selected per solution via fitness probability
#                                  (Eqs. 7-9)
#   Stage 3 — COBL              : worst Nw solutions replaced by their
#                                  comprehensive opposites (Eq. 10)
# =============================================================================

import math
import random
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Task:
    id: int
    length: float      # MI  (Tlen in the paper, Eq. 2)
    submit_time: float = 0.0
    num_pes: int = 1

@dataclass
class VM:
    id: int
    mips: float        # MIPSj  (Eq. 2)
    num_pes: int = 1

def build_etc(tasks: List[Task], vms: List[VM]) -> List[List[float]]:
    etc = []
    for t in tasks:
        row = []
        for v in vms:
            pes_available = max(1, v.num_pes)
            pes_needed = max(1, t.num_pes)
            parallel_efficiency = min(1.0, pes_available / pes_needed)
            row.append((t.length / v.mips) / parallel_efficiency)
        etc.append(row)
    return etc


# =============================================================================
# 3.  FITNESS FUNCTION  (Eqs. 3-5)
# =============================================================================

def makespan(assignment: List[int], etc: List[List[float]]) -> float:
    """
    MKS = max_j ( sum_i ETC[i][j] for tasks assigned to VM j )  (Eq. 3)
    """
    n_vms = max(assignment) + 1 if assignment else 1
    vm_load = [0.0] * (max(assignment) + 1)
    for i, vm_idx in enumerate(assignment):
        vm_load[vm_idx] += etc[i][vm_idx]
    return max(vm_load)


def resource_utilisation(assignment: List[int],
                         etc: List[List[float]],
                         mks: float) -> float:
    """
    RU = ( sum_j T_vmj ) / ( MKS * N_vms )  (Eq. 4)
    """
    n_vms = max(assignment) + 1 if assignment else 1
    vm_load = [0.0] * n_vms
    for i, vm_idx in enumerate(assignment):
        vm_load[vm_idx] += etc[i][vm_idx]
    total_busy = sum(vm_load)
    return total_busy / (mks * n_vms) if mks > 0 else 0.0


def fitness(assignment: List[int],
            etc: List[List[float]],
            alpha: float = 500.0) -> float:
    """
    F(v) = MKS - alpha * RU                 (Eq. 5, adapted)
    Lower is better: minimise makespan, maximise utilisation.
    alpha trades off the two objectives.
    """
    mks = makespan(assignment, etc)
    ru  = resource_utilisation(assignment, etc, mks)
    return mks - alpha * ru


# =============================================================================
# 4.  POPULATION INITIALISATION  (Eq. 6)
# =============================================================================

def init_population(n_pop, n_tasks, n_vms, rng) -> List[List[int]]:
    return [[rng.randint(0, n_vms - 1) for _ in range(n_tasks)] for _ in range(n_pop)]


# =============================================================================
# 5.  HGSO OPERATOR  (exploration)
# =============================================================================
#
# Henry Gas Solubility Optimisation (HGSO) models the solubility of gases
# in liquids. The core update moves each molecule toward the global best
# via a Henry's-law-inspired step:
#
#   X_new[j] = X[j] + K * H_cp * C * (X_best[j] - X[j])
#
# where:
#   K     ~ uniform(0,1)   random interaction coefficient
#   H_cp  ~ N(0,1) * 0.1  Henry's constant perturbation
#   C     ~ N(0,1) * 0.1  cluster interaction constant
#
# After the continuous update the position is clipped and discretised.
# =============================================================================

def hgso_update(xi, x_best, n_vms, rng) -> List[int]:
    K = rng.random()
    H_cp = rng.gauss(0, 1) * 0.1
    C = rng.gauss(0, 1) * 0.1
    new = []
    for j in range(len(xi)):
        val = xi[j] + K * H_cp * C * (x_best[j] - xi[j])
        new.append(int(abs(round(val))) % n_vms)
    return new


# =============================================================================
# 6.  HHO OPERATOR  (exploitation — 4 strategies)
# =============================================================================
#
# Harris Hawks Optimisation models hawks chasing an escaping rabbit.
# Escape energy E decreases linearly over iterations:
#   E = 2 * E0 * (1 - t / T)
#
# |E| ≥ 1  →  Exploration  (random perch / rabbit-tracking)
# |E| < 1  →  Exploitation via one of four strategies:
#   Strategy 1  (q ≥ 0.5, |E| ≥ 0.5)  : soft besiege
#   Strategy 2  (q ≥ 0.5, |E| < 0.5)  : hard besiege
#   Strategy 3  (q < 0.5, |E| ≥ 0.5)  : soft besiege + rapid Lévy dive
#   Strategy 4  (q < 0.5, |E| < 0.5)  : hard besiege + rapid Lévy dive
# =============================================================================

def _levy_flight(n, rng, beta=1.5) -> List[float]:
    sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2) / (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = [rng.gauss(0, sigma) for _ in range(n)]
    v = [rng.gauss(0, 1) for _ in range(n)]
    return [ui/(abs(vi)**(1/beta)) for ui, vi in zip(u, v)]


def hho_update(xi, x_best, population, t, max_iter, n_vms, rng) -> List[int]:
    n = len(xi)
    E0 = 2.0 * rng.random() - 1.0
    E = 2.0 * E0 * (1.0 - t/max_iter)
    J = 2.0 * (1.0 - rng.random())
    def clip(v): return int(abs(round(v))) % n_vms
    if abs(E) >= 1.0:
        q = rng.random()
        if q >= 0.5:
            r_hawk = rng.choice(population)
            new = [clip(r_hawk[j] - rng.random()*abs(r_hawk[j] - 2*rng.random()*xi[j])) for j in range(n)]
        else:
            avg = [sum(p[j] for p in population)/len(population) for j in range(n)]
            x_rand = [rng.randint(0, n_vms-1) for _ in range(n)]
            new = [clip(x_rand[j] - rng.random()*abs(x_rand[j]-avg[j])) for j in range(n)]
    else:
        q = rng.random()
        if q >= 0.5 and abs(E) >= 0.5:
            new = [clip(x_best[j] - E*abs(J*x_best[j]-xi[j])) for j in range(n)]
        elif q >= 0.5 and abs(E) < 0.5:
            new = [clip(x_best[j] - E*abs(x_best[j]-xi[j])) for j in range(n)]
        elif q < 0.5 and abs(E) >= 0.5:
            levy = _levy_flight(n, rng)
            Y = [clip(x_best[j] - E*abs(J*x_best[j]-xi[j])) for j in range(n)]
            Z = [clip(Y[j]+levy[j]) for j in range(n)]
            new = Y if sum(abs(Y[j]-x_best[j]) for j in range(n)) < sum(abs(Z[j]-x_best[j]) for j in range(n)) else Z
        else:
            avg = [sum(p[j] for p in population)/len(population) for j in range(n)]
            levy = _levy_flight(n, rng)
            Y = [clip(x_best[j] - E*abs(J*x_best[j]-avg[j])) for j in range(n)]
            Z = [clip(Y[j]+levy[j]) for j in range(n)]
            new = Y if sum(abs(Y[j]-x_best[j]) for j in range(n)) < sum(abs(Z[j]-x_best[j]) for j in range(n)) else Z
    return new


# =============================================================================
# 7.  COBL — Comprehensive Opposition-Based Learning  (Eq. 10)
# =============================================================================
#
# For the Nw worst solutions, compute the comprehensive opposite:
#   X_opp[j] = LB + UB - X[j]  =  (n_vms - 1) - X[j]
# then keep whichever of {X, X_opp} has better fitness.
# =============================================================================

def cobl_update(xi: List[int], etc: List[List[float]],
                n_vms: int, alpha: float = 500.0) -> List[int]:
    """
    Comprehensive Opposition-Based Learning step.
    Computes the opposite of xi and returns the better of the two.
    """
    lb, ub = 0, n_vms - 1
    x_opp = [lb + ub - xi[j] for j in range(len(xi))]
    x_opp = [max(0, min(n_vms - 1, v)) for v in x_opp]

    if fitness(x_opp, etc, alpha) < fitness(xi, etc, alpha):
        return x_opp
    return xi


# =============================================================================
# 8.  PROBABILITY SELECTION  (Eqs. 7-9)
# =============================================================================

def compute_probabilities(fit_vals: List[float]) -> List[float]:
    """
    Pri = F(vi) / sum_i F(vi)   (Eq. 7)
    Fitness values are negative; we shift to positive for the ratio.
    """
    shifted = [abs(f) for f in fit_vals]
    total   = sum(shifted) or 1.0
    return [s / total for s in shifted]


def random_rpr(rng, lpr=0.0, upr=1.0) -> float:
    return lpr + rng.random() * (upr - lpr)

def count_worst(n_pop, rng, c1=0.1, c2=0.2) -> int:
    r = rng.random()
    return max(1, int(n_pop * (r*(c2-c1) + c1)))


# =============================================================================
# 9.  MAIN HGHHC ALGORITHM
# =============================================================================

def hghhc(tasks: List[Task],
          vms:   List[VM],
          n_pop:    int   = 20,
          max_iter: int   = 50,
          alpha:    float = 500.0,
          lpr:      float = 0.0,
          upr:      float = 1.0,
          seed:     int   = 42,
          verbose:  bool  = False
          ) -> Dict:
    """
    Full HGHHC scheduler.

    Parameters
    ----------
    tasks    : list of Task objects
    vms      : list of VM objects
    n_pop    : population size           (default 20, paper Table 2)
    max_iter : maximum iterations        (default 50, paper Table 2)
    alpha    : MKS/RU trade-off weight   (default 500)
    lpr, upr : probability bounds        (Eq. 9, paper: 0.0 and 1.0)
    seed     : random seed for reproducibility
    verbose  : print convergence trace

    Returns
    -------
    schedule : dict  {task_id: (vm_id, start_time, finish_time)}
    info     : dict  {'best_fitness', 'best_makespan', 'best_utilisation',
                      'convergence'}
    """

    rng = random.Random(seed)

    n_tasks = len(tasks)
    n_vms   = len(vms)
    assert n_tasks > 0 and n_vms > 0, "Need at least 1 task and 1 VM."

    # ---- Build ETC matrix (Eqs. 1-2) ----------------------------------------
    etc = build_etc(tasks, vms)

    # ---- Stage 1: Initialise population (Eq. 6) -----------------------------
    pop      = init_population(n_pop, n_tasks, n_vms, rng)
    fit_vals = [fitness(p, etc, alpha) for p in pop]

    best_idx  = fit_vals.index(min(fit_vals))
    x_best    = list(pop[best_idx])
    best_fit  = fit_vals[best_idx]

    convergence = []   # track best fitness per iteration

    # ---- Main loop -----------------------------------------------------------
    for t in range(1, max_iter + 1):

        # Stage 2a: compute selection probabilities (Eq. 7)
        probs = compute_probabilities(fit_vals)

        new_pop = []
        for i in range(n_pop):
            rpr = random_rpr(rng, lpr, upr)   # Eq. 9

            if probs[i] >= rpr:
                # HHO exploitation (Eq. 8 — left branch)
                new_xi = hho_update(pop[i], x_best, pop, t, max_iter, n_vms, rng)
            else:
                # HGSO exploration (Eq. 8 — right branch)
                new_xi = hgso_update(pop[i], x_best, n_vms, rng)

            # Accept if improved
            new_fit = fitness(new_xi, etc, alpha)
            if new_fit < fit_vals[i]:
                new_pop.append(new_xi)
                fit_vals[i] = new_fit
            else:
                new_pop.append(pop[i])

        pop = new_pop

        # Stage 3: COBL on worst Nw solutions (Eq. 10)
        nw         = count_worst(n_pop, rng)
        sorted_idx = sorted(range(n_pop), key=lambda i: fit_vals[i],
                            reverse=True)   # worst = highest fitness
        for idx in sorted_idx[:nw]:
            improved = cobl_update(pop[idx], etc, n_vms, alpha)
            imp_fit  = fitness(improved, etc, alpha)
            if imp_fit < fit_vals[idx]:
                pop[idx]      = improved
                fit_vals[idx] = imp_fit

        # Update global best
        cur_best_idx = fit_vals.index(min(fit_vals))
        if fit_vals[cur_best_idx] < best_fit:
            x_best   = list(pop[cur_best_idx])
            best_fit = fit_vals[cur_best_idx]

        convergence.append(best_fit)

        if verbose and (t % max(1, max_iter // 10) == 0 or t == max_iter):
            mks = makespan(x_best, etc)
            ru  = resource_utilisation(x_best, etc, mks)
            print(f"  Iter {t:4d}/{max_iter} | "
                  f"Best fitness: {best_fit:12.4f} | "
                  f"Makespan: {mks:10.2f}s | "
                  f"Utilisation: {ru:.4f}")

    # ---- Decode best solution to schedule dict ------------------------------
    schedule = _decode(x_best, tasks, vms)

    mks  = makespan(x_best, etc)
    ru   = resource_utilisation(x_best, etc, mks)

    info = {
        'best_fitness':     round(best_fit, 6),
        'best_makespan':    round(mks, 4),
        'best_utilisation': round(ru, 6),
        'convergence':      convergence,
    }

    return schedule, info


def _decode(assignment, tasks, vms):
    vm_available = {v.id: 0.0 for v in vms}
    schedule = {}
    for k, task in enumerate(tasks):
        vm = vms[assignment[k]]
        pes_available = max(1, vm.num_pes)
        pes_needed = max(1, task.num_pes)
        parallel_efficiency = min(1.0, pes_available / pes_needed)
        start = max(vm_available[vm.id], task.submit_time)
        finish = start + (task.length / vm.mips) / parallel_efficiency
        schedule[task.id] = (vm.id, round(start, 4), round(finish, 4))
        vm_available[vm.id] = finish
    return schedule


# =============================================================================
# 10. PIR METRIC  (Eq. 11)
# =============================================================================

def pir(fitness_proposed: float, fitness_baseline: float) -> float:
    """
    Performance Improvement Rate (Eq. 11):
    PIR = (Z_d\'- Z_d) / Z_d * 100
    where Z_d = baseline fitness, Z_d\' = proposed fitness.
    Positive PIR = proposed is better (lower fitness).
    """
    if fitness_baseline == 0:
        return 0.0
    return (fitness_baseline - fitness_proposed) / abs(fitness_baseline) * 100


# =============================================================================
# 11. QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import random as _r
    _r.seed(0)

    # 20 tasks, 4 VMs — same scale as paper's smallest experiment
    tasks = [Task(id=i,
                  length=_r.uniform(100, 5000),
                  submit_time=_r.uniform(0, 200))
             for i in range(20)]

    vms = [
        VM(id=0, mips=500),
        VM(id=1, mips=1000),
        VM(id=2, mips=2000),
        VM(id=3, mips=4000),
    ]

    print("="*60)
    print("HGHHC — Henry Gas Harris Hawks Comprehensive Opposition")
    print("="*60)

    schedule, info = hghhc(
        tasks, vms,
        n_pop=20, max_iter=50,
        alpha=500.0, seed=42,
        verbose=True
    )

    print(f"\nBest makespan   : {info['best_makespan']:.2f}s")
    print(f"Best utilisation: {info['best_utilisation']:.4f}")
    print(f"Best fitness    : {info['best_fitness']:.4f}")

    print("\nSchedule (task_id → vm_id, start, finish):")
    for tid, (vm_id, st, ft) in sorted(schedule.items()):
        print(f"  Task {tid:3d} → VM {vm_id}  [{st:.1f}s – {ft:.1f}s]")
