import sys
"""
MAB Hyper-Heuristic Cloud Task Scheduler.

Delegates all execution timing, makespan, cost, carbon, and utilization
computation to CloudSim Plus via a Java subprocess bridge. Python is
responsible only for: (1) choosing a low-level heuristic via a
multi-armed bandit, (2) having that heuristic produce a task->VM
assignment, and (3) sending that assignment to CloudSim Plus for
authoritative simulation.
"""

import math
import random
import os
import argparse
import csv
import time
import subprocess
import json
import shutil
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import Counter

from custom_types import Task, VM
from utils import _reset_vms, get_workload_fingerprint, load_swf, generate_synthetic_tasks, generate_vms
from heft import heft
from hghhc import hghhc, Task as HTask, VM as HVM
from max_min import max_min
from min_min import min_min
from peft import peft
from round_robin import round_robin
from heuristics import sufferage, heft_with_duplication, simulated_annealing

def _normalize_batch(batch: List[Task]) -> List[Task]:
    if not batch:
        return batch
    t0 = min(t.submit_time for t in batch)
    return [Task(id=t.id, submit_time=t.submit_time - t0,
                 length=t.length, num_pes=t.num_pes) for t in batch]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class CloudSimDaemon:
    def __init__(self, jar_path: str):
        self.proc = subprocess.Popen(
            ["java", "-jar", jar_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    
    def evaluate(self, payload: dict) -> dict:
        # Send payload as a single JSON line
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

        # Wait for and read the single JSON line response
        response_line = self.proc.stdout.readline()
        if not response_line:
            raise RuntimeError("CloudSim JVM died unexpectedly.")
        
        result = json.loads(response_line)
        if "error" in result:
            raise RuntimeError(f"CloudSim Error: {result['error']}")
        
        return result
    
    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# =============================================================================
# OPTION A: THEORETICAL LOWER-BOUND REWARD CALCULATOR
# =============================================================================

class RewardCalculator:
    """Calculates MAB scheduling rewards using Option A: Theoretical Lower-Bound Ratio Scaling.
    
    This replaces local batch dynamic min-max scaling with an absolute physical lower bound
    (T_min), preventing suboptimal heuristics (e.g., PEFT/Max-Min) from receiving artificial
    high rewards (1.1000) when optimal heuristics (HGHHC) haven't been sampled yet.
    """

    def __init__(self, rank_bonus: float = 0.1, eps: float = 1e-8):
        """
        Args:
            rank_bonus: Rank bonus added to reward if heuristic is top-performing in a round.
            eps: Epsilon factor to prevent division by zero.
        """
        self.rank_bonus = rank_bonus
        self.eps = eps

    def calculate_theoretical_lower_bound(
        self, batch_tasks: List[Any], vms: List[Any]
    ) -> float:
        """Computes T_min (theoretical minimum makespan) for a given task batch and VM cluster.
        
        Accounts for total workload volume (Capacity Bound) and individual task bottlenecks
        (Bottleneck Bound), taking multi-core (num_pes) tasks into consideration.
        """
        if not batch_tasks or not vms:
            return self.eps

        total_work = 0.0
        max_single_task_time = 0.0

        # Extract total processing capacity of VM cluster
        if hasattr(vms[0], 'mips') and hasattr(vms[0], 'num_pes'):
            total_cluster_capacity = sum(vm.mips * vm.num_pes for vm in vms)
            max_vm_speed = max(vm.mips for vm in vms)
        elif hasattr(vms[0], 'speed') and hasattr(vms[0], 'num_pes'):
            total_cluster_capacity = sum(vm.speed * vm.num_pes for vm in vms)
            max_vm_speed = max(vm.speed for vm in vms)
        elif hasattr(vms[0], 'speed'):
            total_cluster_capacity = sum(vm.speed for vm in vms)
            max_vm_speed = max(vm.speed for vm in vms)
        else:
            total_cluster_capacity = float(len(vms))
            max_vm_speed = 1.0

        for task in batch_tasks:
            length = getattr(task, 'length', getattr(task, 'run_time', getattr(task, 'duration', 0.0)))
            pes = getattr(task, 'num_pes', getattr(task, 'pes', 1))

            task_work = length * pes
            total_work += task_work

            single_task_time = length / max(max_vm_speed, self.eps)
            if single_task_time > max_single_task_time:
                max_single_task_time = single_task_time

        # 1. Capacity Bound: Ideal work distribution across total VM capacity
        capacity_bound = total_work / max(total_cluster_capacity, self.eps)

        # 2. Bottleneck Bound: Hard limit set by longest un-splittable task
        bottleneck_bound = max_single_task_time

        # Theoretical Lower Bound T_min = max(Capacity Bound, Bottleneck Bound)
        t_min = max(capacity_bound, bottleneck_bound)
        return t_min

    def compute_reward(
        self,
        batch_tasks: List[Any],
        actual_makespan: float,
        vms: List[Any],
        is_best_in_round: bool = False
    ) -> float:
        """Computes Option A ratio reward bounded strictly to (0.0, 1.0 + rank_bonus].
        
        Formula:
            Reward = (T_min / Actual_Makespan) + (rank_bonus if best else 0.0)
        """
        t_min = self.calculate_theoretical_lower_bound(batch_tasks, vms)

        clamped_makespan = max(actual_makespan, t_min)
        ratio_reward = t_min / (clamped_makespan + self.eps)

        base_reward = max(0.0, min(1.0, ratio_reward))

        if is_best_in_round:
            return base_reward + self.rank_bonus

        return base_reward


# =============================================================================
# 1. HGHHC ADAPTER
# =============================================================================

def hghhc_llh(tasks, vms):
    h_tasks = [HTask(id=t.id, length=t.length, submit_time=t.submit_time) for t in tasks]
    h_vms = [HVM(id=v.id, mips=v.mips, num_pes=v.num_pes) for v in vms]
    schedule, _ = hghhc(h_tasks, h_vms, n_pop=20, max_iter=30, verbose=False)
    return {tid: vm_id for tid, (vm_id, _, _) in schedule.items()}


# =============================================================================
# 2. HEURISTIC REGISTRY
# =============================================================================

HEURISTICS = {
    "heft": heft,
    "peft": peft,
    "min_min": min_min,
    "max_min": max_min,
    "round_robin": round_robin,
    "hghhc": hghhc_llh,
    "sufferage": sufferage,
    "heft_duplication": heft_with_duplication,
    "simulated_annealing": lambda t, v: simulated_annealing(t, v, n_steps=100, T_init=500),
}


# =============================================================================
# 3. METRICS VIA CLOUDSIM PLUS
# =============================================================================

JAR_PATH = os.path.expanduser("~/Documents/mab-vm-pool/target/mab-vm-pool-1.0-SNAPSHOT.jar")

def compute_metrics_via_cloudsim(schedule: Dict, tasks: List[Task],
                                  vms: List[VM], daemon: CloudSimDaemon) -> Tuple[float, float, float, float]:
    if not schedule:
        return float('inf'), float('inf'), float('inf'), 0.0

    payload = {
        "tasks": [
            {"id": t.id, "length": t.length, "pesNumber": t.num_pes, "submitTime": t.submit_time}
            for t in tasks
        ],
        "vms": [
            {
                "id": v.id, "mips": v.mips, "pesNumber": v.num_pes,
                "ram": 2048, "bw": 1000, "size": 10000,
                "costPerSecond": v.cost_per_second,
                "carbonPerSecond": v.carbon_per_second
            }
            for v in vms
        ],
        "assignment": {
            str(task_id): (v[0] if isinstance(v, (tuple, list)) else v)
            for task_id, v in schedule.items()
        }
    }

    result = daemon.evaluate(payload)
    return result["makespan"], result["totalCost"], result["totalCarbon"], result["utilization"]

def compute_relative_reward(base_metrics: Tuple[float, float, float, float],
                            chosen_metrics: Tuple[float, float, float, float],
                            w_makespan: float = 0.6, w_cost: float = 0.2,
                            w_carbon: float = 0.1, w_util: float = 0.1) -> float:
    b_ms, b_cost, b_carb, b_util = base_metrics
    c_ms, c_cost, c_carb, c_util = chosen_metrics
    
    b_ms = b_ms if b_ms > 0 else 1.0
    b_cost = b_cost if b_cost > 0 else 1.0
    b_carb = b_carb if b_carb > 0 else 1.0
    b_util = b_util if b_util > 0 else 0.01

    r_ms = (b_ms - c_ms) / b_ms
    r_cost = (b_cost - c_cost) / b_cost
    r_carb = (b_carb - c_carb) / b_carb
    r_util = (c_util - b_util) / b_util
    
    reward = (w_makespan * r_ms) + (w_cost * r_cost) + (w_carbon * r_carb) + (w_util * r_util)
    return math.tanh(reward)

def get_dynamic_hyperparameters(cov: float):
    if cov > 1.5:
        ucb_c = 0.04
        rank_bonus = 0.1
        mode_label = "EXPLORATION-HEAVY (Volatile Workload)"
    else:
        ucb_c = 0.01
        rank_bonus = 0.3
        mode_label = "EXPLOITATION-HEAVY (Stable Workload)"
        
    return ucb_c, rank_bonus, mode_label


# =============================================================================
# 4. MAB ALGORITHMS
# =============================================================================

class EpsilonGreedyMAB:
    def __init__(self, arms: List[str], epsilon: float = 0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.q: Dict[str, float] = {a: 0.0 for a in arms}
        self.n: Dict[str, int] = {a: 0 for a in arms}
        self.total_pulls: int = 0
        self.history: List[Dict] = []

    def select(self) -> str:
        # =========================================================
        # 1. WARM-UP PHASE: Pull each arm once sequentially
        # =========================================================
        if self.total_pulls < len(self.arms):
            # Select the arm at index 'total_pulls'
            return self.arms[self.total_pulls]

        # =========================================================
        # 2. STANDARD EPSILON-GREEDY LOGIC (After warm-up completes)
        # =========================================================
        if random.random() < self.epsilon:
            # Explore: Choose a random arm uniformly
            return random.choice(self.arms)
        else:
            # Exploit: Choose the arm with the highest estimated Q-value
            max_q = max(self.q.values())
            # Break ties randomly among arms that share the maximum Q-value
            best_arms = [arm for arm, q in self.q.items() if q == max_q]
            return random.choice(best_arms)

    def update(self, arm: str, reward: float):
        self.n[arm] += 1
        self.total_pulls += 1
        
        # Incremental moving average update rule: Q(a) = Q(a) + (reward - Q(a)) / N(a)
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
        
        self.history.append({'arm': arm, 'reward': reward})

    def __repr__(self):
        return f"EpsilonGreedyMAB(epsilon={self.epsilon}, q={self.q})"


class UCBMAB:
    def __init__(self, arms: List[str], c: float = 2.0):
        self.arms = arms
        self.c = c
        self.q = {a: 0.0 for a in arms}
        self.n = {a: 0 for a in arms}
        self.t = 0
        self.history: List[Dict] = []

    def select(self) -> str:
        self.t += 1
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
    

class DiscountedUCBMAB:
    def __init__(self, arms: List[str], c: float = 2.0, gamma: float = 0.99):
        self.arms = arms
        self.c = c
        self.gamma = gamma
        self.sum_r = {a: 0.0 for a in arms}
        self.n = {a: 0.0 for a in arms}
        self.q = {a: 0.0 for a in arms}
        self.t = 0
        self.history: List[Dict] = []

    def select(self) -> str:
        self.t += 1
        for arm in self.arms:
            if self.n[arm] == 0:
                return arm
        return max(
            self.arms,
            key=lambda a: self.q[a] + self.c * math.sqrt(math.log(self.t) / self.n[a])
        )

    def update(self, arm: str, reward: float):
        for a in self.arms:
            self.n[a] *= self.gamma
            self.sum_r[a] *= self.gamma
        self.n[arm] += 1
        self.sum_r[arm] += reward
        self.q[arm] = self.sum_r[arm] / self.n[arm]
        self.history.append({'arm': arm, 'reward': reward,
                              'q_values': dict(self.q), 'counts': dict(self.n)})

# =====================================================================
# Min-Max Scaled Beta Thompson Sampling
# =====================================================================
class ScaledThompsonSamplingMAB:
    def __init__(self, arms: List[str], n_arms: int = None, **kwargs):
        self.arms = arms
        # Initialize self.n as a dictionary mapping arm string keys to selection counts
        self.n: Dict[str, int] = {a: 0 for a in arms}
        
        # Initialize alpha and beta prior weights
        self.alpha: Dict[str, float] = {a: 1.0 for a in arms}
        self.beta: Dict[str, float] = {a: 1.0 for a in arms}
        self.total_pulls: int = 0
        
        # Min-max tracking for continuous reward normalization
        self.min_reward: float = float('inf')
        self.max_reward: float = float('-inf')

    @property
    def q(self) -> Dict[str, float]:
        return {
            arm: self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
            for arm in self.arms
        }

    def select(self) -> str:
        # Warm-up phase: Pull each arm once sequentially
        if self.total_pulls < len(self.arms):
            return self.arms[self.total_pulls]

        # Draw a sample from Beta(alpha, beta) for each arm
        samples = {
            arm: np.random.beta(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        # Pick the arm with the highest sample draw
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float):
        self.n[arm] += 1
        self.total_pulls += 1
        
        # Update dynamic min and max rewards observed so far
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        
        # Normalize reward to [0, 1] range
        reward_range = self.max_reward - self.min_reward
        if reward_range > 1e-8:
            r_norm = (reward - self.min_reward) / reward_range
        else:
            r_norm = 0.5  # Default neutral scaling if all rewards are identical

        # Update Beta parameters with normalized reward
        self.alpha[arm] += r_norm
        self.beta[arm] += (1.0 - r_norm)

MAB_CONFIGS = {
    'ucb': {
        'class': UCBMAB,
        'kwargs': lambda c, **kwargs: {'c': c}
    },
    'discounted_ucb': {
        'class': DiscountedUCBMAB,
        'kwargs': lambda c, **kwargs: {'c': c, 'gamma': 0.99}
    },
    'epsilon': {
        'class': EpsilonGreedyMAB,
        'kwargs': lambda c, **kwargs: {'epsilon': kwargs.get('epsilon', 0.1)}
    },
    'thompson': {
        'class': ScaledThompsonSamplingMAB,
        'kwargs': lambda c, **kwargs: {'rank_bonus': kwargs.get('rank_bonus', 0.1)}
    }
}

def create_mab(mab_type: str, arms: List[str], adaptive_ucb_c: float, epsilon: float = 0.1, rank_bonus: float = 0.1):
    if mab_type not in MAB_CONFIGS:
        raise ValueError(f"Unknown MAB type '{mab_type}'. Available: {list(MAB_CONFIGS.keys())}")
    
    cfg = MAB_CONFIGS[mab_type]
    mab_cls = cfg['class']
    extra_kwargs = cfg['kwargs'](c=adaptive_ucb_c, epsilon=epsilon, rank_bonus=rank_bonus)
    
    return mab_cls(arms=arms, **extra_kwargs)


# =============================================================================
# 5. SIMULATION ENGINE (WIRED WITH OPTION A REWARD CALCULATOR)
# =============================================================================

def run_simulation(tasks: List[Task],
                   vms: List[VM],
                   mab,
                   daemon,
                   batch_size: int = 50,
                   rank_bonus: float = 0.2,
                   verbose: bool = True) -> List[Dict]:
    """
    Main simulation loop.
    Processes tasks in sliding windows (batches).
    In each round:
      1. MAB selects a heuristic
      2. Heuristic schedules current batch
      3. CloudSim Plus computes authoritative metrics
      4. Option A Reward Calculator evaluates physical performance against theoretical limit
      5. MAB is updated with the reward
    """
    results = []
    n_tasks = len(tasks)
    n_rounds = math.ceil(n_tasks / batch_size)

    # Instantiate Option A Reward Calculator
    reward_calculator = RewardCalculator(rank_bonus=rank_bonus)

    # Performance cache used strictly to determine if current run is best seen in history
    last_performance = {h: float('inf') for h in HEURISTICS.keys()}

    if verbose:
        print(f"\nRunning simulation: {n_tasks} tasks | "
              f"batch_size={batch_size} | rounds={n_rounds} | n_vms={len(vms)}")
        print(f"MAB: {type(mab).__name__} | "
              f"Heuristics: {list(HEURISTICS.keys())}\n")

    for i in range(n_rounds):
        batch = tasks[i * batch_size: (i + 1) * batch_size]
        if not batch:
            break

        batch = _normalize_batch(batch)

        chosen = mab.select()
        fn = HEURISTICS[chosen]
        vms_fresh = _reset_vms(vms)
        schedule = fn(batch, vms_fresh)

        makespan, cost, carbon, util = compute_metrics_via_cloudsim(schedule, batch, vms_fresh, daemon)

        # 1. Determine if this heuristic achieved best performance relative to historical cache
        is_best = makespan <= min(last_performance.values())

        # 2. Compute Option A Reward (Ratio of Theoretical Minimum to Actual Makespan)
        reward = reward_calculator.compute_reward(
            batch_tasks=batch,
            actual_makespan=makespan,
            vms=vms_fresh,
            is_best_in_round=is_best
        )

        # 3. Update local performance tracker
        last_performance[chosen] = makespan

        # 4. Update MAB Agent
        mab.update(chosen, reward)

        result = {
            'round': i + 1,
            'heuristic': chosen,
            'batch_size': len(batch),
            'makespan': round(makespan, 4),
            'cost': round(cost, 6),
            'carbon': round(carbon, 8),
            'utilization': round(util, 4),
            'reward': round(reward, 6),
        }
        results.append(result)

        if verbose and (i % max(1, n_rounds // 10) == 0 or i == n_rounds - 1):
            print(f"  Round {i+1:4d}/{n_rounds} | "
                  f"Heuristic: {chosen:18s} | "
                  f"Makespan: {makespan:10.2f} | "
                  f"Reward: {reward:.4f}")

    return results


# =============================================================================
# 6. BASELINE COMPARISON
# =============================================================================

def run_baseline(tasks: List[Task], vms: List[VM],
                  heuristic_name: str, daemon, batch_size: int = 50) -> List[Dict]:
    results = []
    fn = HEURISTICS[heuristic_name]
    n_rounds = math.ceil(len(tasks) / batch_size)

    for i in range(n_rounds):
        batch = tasks[i * batch_size: (i + 1) * batch_size]
        if not batch:
            break
        batch = _normalize_batch(batch)

        vms_base = _reset_vms(vms)
        base_schedule = HEURISTICS["round_robin"](batch, vms_base)
        base_metrics = compute_metrics_via_cloudsim(base_schedule, batch, vms_base, daemon)

        vms_fresh = _reset_vms(vms)
        schedule = fn(batch, vms_fresh)
        chosen_metrics = compute_metrics_via_cloudsim(schedule, batch, vms_fresh, daemon)
        
        reward = compute_relative_reward(base_metrics, chosen_metrics)
        results.append({
            'round': i + 1, 'heuristic': heuristic_name,
            'makespan': chosen_metrics[0], 'cost': chosen_metrics[1],
            'carbon': chosen_metrics[2], 'utilization': chosen_metrics[3],
            'reward': round(reward, 6),
        })
    return results


# =============================================================================
# 7. RESULTS & EXPORT
# =============================================================================

def summarise(results: List[Dict], label: str = "") -> Dict:
    if not results:
        return {}
    makespans = [r['makespan'] for r in results]
    rewards = [r['reward'] for r in results]
    utils_ = [r['utilization'] for r in results]
    costs = [r['cost'] for r in results]
    carbons = [r['carbon'] for r in results]

    counts = Counter(r['heuristic'] for r in results)

    summary = {
        'label': label,
        'rounds': len(results),
        'avg_makespan': round(sum(makespans) / len(makespans), 2),
        'avg_reward': round(sum(rewards) / len(rewards), 4),
        'avg_utilization': round(sum(utils_) / len(utils_), 4),
        'avg_cost': round(sum(costs) / len(costs), 6),
        'avg_carbon': round(sum(carbons) / len(carbons), 8),
        'heuristic_counts': dict(counts),
    }
    return summary


def save_results_csv(results: List[Dict], filepath: str):
    if not results:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


# =============================================================================
# 8. MAIN ENTRY POINT
# =============================================================================

def main(swf_path: str = None,
         n_tasks: int = 500,
         batch_size: int = 50,
         n_vms: int = 8,
         mab_type: str = 'ucb',
         ucb_c: float = 2.0,
         epsilon: float = 0.1,
         seed: int = 42):

    set_seed(seed)

    if swf_path == "nasa":
        dataset_label = "nasa"
        swf_path = "NASA-iPSC-1993-3.1-cln.swf"
    elif swf_path == "hpc2n":
        dataset_label = "hpc2n"
        swf_path = "HPC2N-2002-2.2-cln.swf"
    else:
        dataset_label = "synthetic"
        swf_path = None

    output_dir = f"output/{dataset_label}/{mab_type}"    
    os.makedirs(output_dir, exist_ok=True)

    if swf_path and os.path.exists(swf_path):
        print(f"Loading SWF trace: {swf_path}")
        vms = generate_vms(n_vms=8)
        max_vm_pes = max(v.num_pes for v in vms)
        tasks = load_swf(swf_path, max_tasks=n_tasks, max_vm_pes=max_vm_pes)
        cov = get_workload_fingerprint(tasks)
        
        # Dynamic Hyperparameter Adaptation
        ucb_c, rank_bonus, mode_label = get_dynamic_hyperparameters(cov)
        
        print(f"Max task length: {max(t.length for t in tasks):.1f} | Max num_pes: {max(t.num_pes for t in tasks)}")
        print(f"Loaded {len(tasks)} valid tasks.")
        print(f"Workload CoV: {cov:.4f}")
        print(f"Adaptive Mode: {mode_label}")
        print(f"Active Parameters -> ucb_c: {ucb_c} | rank_bonus: {rank_bonus}")
        print(f"--------------------")
    else:
        print(f"No SWF file found. Generating {n_tasks} synthetic tasks.")
        tasks = generate_synthetic_tasks(n_tasks, seed=seed)
        cov = get_workload_fingerprint(tasks)
        ucb_c, rank_bonus, mode_label = get_dynamic_hyperparameters(cov)

    vms = generate_vms(n_vms)

    # Instantiate MAB using mab_type parameter
    mab = create_mab(mab_type=mab_type, arms=list(HEURISTICS.keys()), adaptive_ucb_c=ucb_c, epsilon=epsilon, rank_bonus=rank_bonus)

    daemon = CloudSimDaemon(JAR_PATH)

    try:
        start_time = time.time()
        mab_results = run_simulation(
            tasks, vms, mab, daemon, 
            batch_size=batch_size, 
            rank_bonus=rank_bonus, 
            verbose=True
        )
        elapsed = time.time() - start_time

        print("\nRunning baselines...")
        baseline_results = {}
        for name in HEURISTICS:
            baseline_results[name] = run_baseline(tasks, vms, name, daemon, batch_size)
    finally:
        daemon.close()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

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

    save_results_csv(mab_results, f'{output_dir}/mab_results.csv')
    for name, res in baseline_results.items():
        save_results_csv(res, f'{output_dir}/baseline_{name}.csv')

    print(f"\nResults saved to {output_dir}/ directory.")
    return mab_results, baseline_results, mab_summary, baseline_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking multiple task scheduling algorithms")
    parser.add_argument("--ds", default="nasa")
    parser.add_argument("--mab", default='ucb', choices=list(MAB_CONFIGS.keys()))
    parser.add_argument("--ucb-c", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--n-tasks", type=int, default=3000)
    args = parser.parse_args()
    main(
        swf_path=args.ds,
        n_tasks=args.n_tasks,
        batch_size=args.batch_size,
        n_vms=4,
        mab_type=args.mab,
        ucb_c=args.ucb_c,
        seed=42
    )