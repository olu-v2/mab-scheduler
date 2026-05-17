import math
import random
import os
import argparse
import csv
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from custom_types import Task, VM
from utils import load_swf, generate_synthetic_tasks, generate_vms
from aco import aco
from heft_dup import heft_with_duplication
from heft import heft
from hghhc import hghhc, Task as HTask, VM as HVM
from max_min import max_min
from min_min import min_min
from peft import peft
from pso import pso
from round_robin import round_robin
from simulated_annealing import simulated_annealing
from sufferage import sufferage
from woa import woa
from heuristics import sufferage, heft_with_duplication, woa, aco, pso, simulated_annealing


def hghhc_llh(tasks, vms):
    h_tasks = [HTask(id=t.id, length=t.length, submit_time=t.submit_time) for t in tasks]
    h_vms   = [HVM(id=v.id, mips=v.mips) for v in vms]
    schedule, _ = hghhc(h_tasks, h_vms, n_pop=20, max_iter=30, verbose=False)
    return schedule




# Map heuristic names to functions
HEURISTICS = {
    'heft':        heft,
    'peft':        peft,
    'min_min':     min_min,
    'max_min':     max_min,
    'round_robin': round_robin,
    'hghhc': hghhc_llh,
    "sufferage":           sufferage,
    "heft_duplication":    heft_with_duplication,
    # "woa":                 woa,
    "aco":                 aco,
    # "pso":                 pso,
    # "simulated_annealing": simulated_annealing,
    "simulated_annealing": lambda t, v: simulated_annealing(t, v, n_steps=100, T_init=500),
    "woa":                 lambda t, v: woa(t, v, max_iter=50),
    "pso":                 lambda t, v: pso(t, v, max_iter=100),
}


# =============================================================================
# 4. METRICS
# =============================================================================

def compute_metrics(schedule: Dict, tasks: List[Task],
                    vms: List[VM]) -> Tuple[float, float, float, float]:
    """
    Returns (makespan, total_cost, total_carbon, utilization).
    """
    if not schedule:
        return float('inf'), float('inf'), float('inf'), 0.0

    vm_map = {v.id: v for v in vms}
    makespan = max(finish for (_, _, finish) in schedule.values())

    # Active VMs (those that actually ran tasks)
    vm_active_time = {}
    for task_id, (vm_id, start, finish) in schedule.items():
        exec_t = finish - start
        vm_active_time[vm_id] = vm_active_time.get(vm_id, 0) + exec_t

    total_cost   = sum(vm_map[vid].cost_per_second   * t for vid, t in vm_active_time.items())
    total_carbon = sum(vm_map[vid].carbon_per_second * t for vid, t in vm_active_time.items())

    # Utilisation: fraction of makespan VMs are busy
    total_busy = sum(vm_active_time.values())
    utilization = total_busy / (makespan * len(vm_active_time)) if makespan > 0 else 0

    return makespan, total_cost, total_carbon, utilization


def compute_reward(makespan: float, total_cost: float,
                   total_carbon: float, utilization: float,
                   w_makespan: float = 0.4, w_cost: float = 0.3,
                   w_carbon: float = 0.2, w_util: float = 0.1) -> float:
    """
    Composite normalised reward (higher = better).
    Penalises makespan, cost, carbon; rewards utilisation.
    Uses negative weighted sum scaled to [-1, 0] range per objective.
    """
    # Avoid division by zero
    ms   = makespan    if makespan    > 0 else 1
    cost = total_cost  if total_cost  > 0 else 1
    carb = total_carbon if total_carbon > 0 else 1

    score = -(w_makespan * ms / 1e6
              + w_cost   * cost
              + w_carbon * carb * 1e6
              - w_util   * utilization)
    return score


# =============================================================================
# 5. MAB ALGORITHMS
# =============================================================================

class EpsilonGreedyMAB:
    """
    ε-Greedy Multi-Armed Bandit.
    With probability ε: explore (random arm).
    Otherwise: exploit (arm with highest estimated reward).
    """
    def __init__(self, arms: List[str], epsilon: float = 0.1):
        self.arms    = arms
        self.epsilon = epsilon
        self.q  = {a: 0.0 for a in arms}   # estimated reward per arm
        self.n  = {a: 0   for a in arms}   # selection count per arm
        self.history: List[Dict] = []

    def select(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        return max(self.q, key=self.q.get)

    def update(self, arm: str, reward: float):
        self.n[arm] += 1
        # Incremental mean update
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
        self.history.append({'arm': arm, 'reward': reward,
                              'q_values': dict(self.q),
                              'counts': dict(self.n)})

    def __repr__(self):
        return f"EpsilonGreedyMAB(ε={self.epsilon}, q={self.q})"


class UCBMAB:
    """
    Upper Confidence Bound (UCB1) Multi-Armed Bandit.
    Balances exploration/exploitation via confidence intervals.
    UCB score: q_i + C * sqrt(ln(t) / n_i)
    """
    def __init__(self, arms: List[str], c: float = 2.0):
        self.arms = arms
        self.c    = c
        self.q    = {a: 0.0 for a in arms}
        self.n    = {a: 0   for a in arms}
        self.t    = 0
        self.history: List[Dict] = []

    def select(self) -> str:
        self.t += 1
        # Ensure each arm is tried at least once
        for arm in self.arms:
            if self.n[arm] == 0:
                return arm
        return max(
            self.arms,
            key=lambda a: self.q[a] + self.c * math.sqrt(math.log(self.t) / self.n[a])
        )

    def update(self, arm: str, reward: float):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
        self.history.append({'arm': arm, 'reward': reward,
                              'q_values': dict(self.q),
                              'counts': dict(self.n)})

    def __repr__(self):
        return f"UCBMAB(C={self.c}, q={self.q})"


class ThompsonSamplingMAB:
    """
    Thompson Sampling MAB using Beta distribution.
    Models each arm's reward as Bernoulli(p) with Beta(α, β) prior.
    Adapted for continuous rewards via success/failure thresholding.
    """
    def __init__(self, arms: List[str], threshold: float = -0.5):
        self.arms      = arms
        self.threshold = threshold   # reward > threshold = "success"
        self.alpha = {a: 1.0 for a in arms}  # successes + 1
        self.beta  = {a: 1.0 for a in arms}  # failures + 1
        self.history: List[Dict] = []

    def select(self) -> str:
        samples = {a: random.betavariate(self.alpha[a], self.beta[a])
                   for a in self.arms}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float):
        if reward > self.threshold:
            self.alpha[arm] += 1
        else:
            self.beta[arm]  += 1
        self.history.append({'arm': arm, 'reward': reward})

    def __repr__(self):
        return f"ThompsonSamplingMAB(α={self.alpha}, β={self.beta})"


# =============================================================================
# 6. SIMULATION ENGINE
# =============================================================================

def run_simulation(tasks: List[Task],
                   vms: List[VM],
                   mab,
                   batch_size: int = 50,
                   verbose: bool = True) -> List[Dict]:
    """
    Main simulation loop.
    Processes tasks in sliding windows (batches).
    In each round:
      1. MAB selects a heuristic
      2. Heuristic schedules the current batch
      3. Metrics are computed
      4. MAB is updated with the reward
    """
    results = []
    n_tasks  = len(tasks)
    n_rounds = math.ceil(n_tasks / batch_size)

    if verbose:
        print(f"\nRunning simulation: {n_tasks} tasks | "
              f"batch_size={batch_size} | rounds={n_rounds}")
        print(f"MAB: {type(mab).__name__} | "
              f"Heuristics: {list(HEURISTICS.keys())}\n")

    for i in range(n_rounds):
        batch = tasks[i * batch_size : (i + 1) * batch_size]
        if not batch:
            break

        chosen = mab.select()
        fn     = HEURISTICS[chosen]
        schedule = fn(batch, vms)

        makespan, cost, carbon, util = compute_metrics(schedule, batch, vms)
        reward = compute_reward(makespan, cost, carbon, util)
        mab.update(chosen, reward)

        result = {
            'round':      i + 1,
            'heuristic':  chosen,
            'batch_size': len(batch),
            'makespan':   round(makespan, 4),
            'cost':       round(cost, 6),
            'carbon':     round(carbon, 8),
            'utilization':round(util, 4),
            'reward':     round(reward, 6),
        }
        results.append(result)

        if verbose and (i % max(1, n_rounds // 10) == 0 or i == n_rounds - 1):
            print(f"  Round {i+1:4d}/{n_rounds} | "
                  f"Heuristic: {chosen:12s} | "
                  f"Makespan: {makespan:10.2f} | "
                  f"Reward: {reward:.4f}")

    return results


# =============================================================================
# 7. BASELINE COMPARISON
# =============================================================================

def run_baseline(tasks: List[Task], vms: List[VM],
                 heuristic_name: str, batch_size: int = 50) -> List[Dict]:
    """Run a single static heuristic across all batches (no MAB)."""
    results = []
    fn = HEURISTICS[heuristic_name]
    n_rounds = math.ceil(len(tasks) / batch_size)

    for i in range(n_rounds):
        batch = tasks[i * batch_size : (i + 1) * batch_size]
        if not batch:
            break
        schedule = fn(batch, vms)
        makespan, cost, carbon, util = compute_metrics(schedule, batch, vms)
        reward = compute_reward(makespan, cost, carbon, util)
        results.append({
            'round': i + 1, 'heuristic': heuristic_name,
            'makespan': round(makespan, 4), 'cost': round(cost, 6),
            'carbon': round(carbon, 8), 'utilization': round(util, 4),
            'reward': round(reward, 6),
        })
    return results


# =============================================================================
# 8. RESULTS & EXPORT
# =============================================================================

def summarise(results: List[Dict], label: str = "") -> Dict:
    if not results:
        return {}
    makespans = [r['makespan'] for r in results]
    rewards   = [r['reward']   for r in results]
    utils     = [r['utilization'] for r in results]
    costs     = [r['cost'] for r in results]
    carbons   = [r['carbon'] for r in results]

    from collections import Counter
    counts = Counter(r['heuristic'] for r in results)

    summary = {
        'label':            label,
        'rounds':           len(results),
        'avg_makespan':     round(sum(makespans) / len(makespans), 2),
        'avg_reward':       round(sum(rewards)   / len(rewards),   4),
        'avg_utilization':  round(sum(utils)     / len(utils),     4),
        'avg_cost':         round(sum(costs)     / len(costs),     6),
        'avg_carbon':       round(sum(carbons)   / len(carbons),   8),
        'heuristic_counts': dict(counts),
    }
    return summary


def save_results_csv(results: List[Dict], filepath: str):
    if not results:
        return
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


# =============================================================================
# 9. MAIN ENTRY POINT
# =============================================================================

def main(swf_path: str = None,
         n_tasks: int = 500,
         batch_size: int = 50,
         n_vms: int = 5,
         mab_type: str = None,
         ucb_c: float = 2.0,
         epsilon: float = 0.1,
         seed: int = 42):

    random.seed(seed)

    if swf_path == "nasa":
        swf_path = "NASA-iPSC-1993-3.1-cln.swf"    
    elif swf_path == "hpc2n":
        swf_path = "HPC2N-2002-2.2-cln.swf"
    else:
        swf_path = None

    # --- Load or generate tasks ---
    if swf_path and os.path.exists(swf_path):
        print(f"Loading SWF trace: {swf_path}")
        tasks = load_swf(swf_path, max_tasks=n_tasks)
        print(f"Loaded {len(tasks)} valid tasks.")
    else:
        print(f"No SWF file found. Generating {n_tasks} synthetic tasks.")
        tasks = generate_synthetic_tasks(n_tasks, seed=seed)

    vms = generate_vms(n_vms)

    # --- Initialise MAB ---
    arms = list(HEURISTICS.keys())
    if mab_type == 'ucb':
        mab = UCBMAB(arms=arms, c=ucb_c)
    elif mab_type == 'epsilon':
        mab = EpsilonGreedyMAB(arms=arms, epsilon=epsilon)
    elif mab_type == 'thompson':
        mab = ThompsonSamplingMAB(arms=arms)
    else:
        raise ValueError(f"Unknown mab_type: {mab_type}")

    # --- Run MAB simulation ---
    start_time = time.time()
    mab_results = run_simulation(tasks, vms, mab,
                                 batch_size=batch_size, verbose=True)
    elapsed = time.time() - start_time

    # --- Run baselines ---
    print("\nRunning baselines...")
    baseline_results = {}
    for name in HEURISTICS:
        baseline_results[name] = run_baseline(tasks, vms, name, batch_size)

    # --- Summarise ---
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    mab_summary = summarise(mab_results, label=f"MAB-{mab_type.upper()}")
    print(f"\n[MAB-{mab_type.upper()}]")
    for k, v in mab_summary.items():
        print(f"  {k}: {v}")

    baseline_summaries = {}
    for name, res in baseline_results.items():
        s = summarise(res, label=name)
        baseline_summaries[name] = s
        print(f"\n[Baseline: {name}]")
        for k, v in s.items():
            print(f"  {k}: {v}")

    print(f"\nTotal simulation time: {elapsed:.2f}s")
    print(f"Final MAB Q-values: {mab.q}")
    print(f"Final selection counts: {mab.n}")

    # --- Save results ---
    save_results_csv(mab_results, 'output/mab_results.csv')
    for name, res in baseline_results.items():
        save_results_csv(res, f'output/baseline_{name}.csv')

    print("\nResults saved to output/ directory.")
    return mab_results, baseline_results, mab_summary, baseline_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking multiple task scheduling algotrithms")
    parser.add_argument("--ds", default="nasa")
    parser.add_argument("--mab", default='mab')
    args = parser.parse_args()
    main(
        swf_path=args.ds,
        n_tasks=10000,
        batch_size=100,
        n_vms=5,
        mab_type=args.mab,
        ucb_c=2.0,
        seed=42
    )
