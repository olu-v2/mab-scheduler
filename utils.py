import random
from collections import defaultdict
from typing import List, Dict
from custom_types import Task, VM
import numpy as np

REFERENCE_MIPS = 1000  # baseline used to convert SWF run_time (s) into MI

def get_workload_fingerprint(tasks):
    lengths = [t.length for t in tasks]
    mean = np.mean(lengths)
    std = np.std(lengths)
    # Coefficient of Variation
    cov = std / mean if mean > 0 else 0
    return cov  # e.g., < 1.0 is stable, > 2.0 is highly volatile

def load_swf(filepath: str, max_tasks: int = None, max_vm_pes: int = 16) -> List[Task]:
    tasks = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                job_id = int(fields[0])
                submit_time = float(fields[1])
                run_time = float(fields[3])
                num_procs = int(fields[4])

                if run_time <= 0 or num_procs <= 0:
                    continue

                clamped_pes = min(max(1, num_procs), max_vm_pes)

                tasks.append(Task(
                    id=job_id,
                    submit_time=submit_time,
                    length=run_time * REFERENCE_MIPS,
                    num_pes=clamped_pes
                ))
            except (ValueError, IndexError):
                continue

            if max_tasks and len(tasks) >= max_tasks:
                break

    return tasks


def generate_synthetic_tasks(n: int, seed: int = 42) -> List[Task]:
    """Generate synthetic tasks for testing without a real SWF file."""
    random.seed(seed)
    tasks = []
    for i in range(n):
        tasks.append(Task(
            id=i,
            submit_time=random.uniform(0, 1000),
            length=random.uniform(100, 5000),
            num_pes=random.choice([1, 2, 4, 8])
        ))
    return tasks


def generate_vms(n_vms: int = 8) -> List[VM]:
    """
    Heterogeneous VM pool with a genuine speed/parallelism trade-off.
    Unlike the previous config (where bigger VMs were always faster
    per-PE too), high-PE tiers here trade away per-PE speed for raw
    parallel capacity -- mirroring real clusters where many-core nodes
    often clock lower per-core than few-core "fast" nodes.
    """
    configs = [
        (2_000_000, 1,  0.0116 / 3600, 0.0003 / 3600),
        (3_000_000, 2,  0.0464 / 3600, 0.0006 / 3600),
        (4_800_000, 4,  0.1664 / 3600, 0.0012 / 3600),
        (7_200_000, 8,  0.3400 / 3600, 0.0024 / 3600),
        (9_600_000, 16, 0.6800 / 3600, 0.0048 / 3600), # 16 PE
        (9_600_000, 16, 0.6800 / 3600, 0.0048 / 3600), # 16 PE
        (9_600_000, 16, 0.6800 / 3600, 0.0048 / 3600), # 16 PE
        (7_200_000, 8,  0.3400 / 3600, 0.0024 / 3600),
    ]
    
    # Ensure at least one 16-PE node if requested n_vms > 0
    selected_configs = []
    if n_vms > 0:
        # Guarantee inclusion of a 16-PE node (index 4)
        selected_configs.append(configs[4])
        # Fill remaining with a mix (randomly or otherwise)
        remaining_slots = n_vms - 1
        # Pick from the rest
        others = [configs[i] for i in range(len(configs)) if i != 4]
        selected_configs.extend(others[:remaining_slots])
    
    vms = []
    for i, (mips, pes, cost, carbon) in enumerate(selected_configs):
        vms.append(VM(id=i, mips=mips, num_pes=pes, cost_per_second=cost, carbon_per_second=carbon))
    return vms

def _reset_vms(vms: List[VM]) -> List[VM]:
    """Deep copy VMs so each heuristic run starts fresh."""
    return [VM(id=v.id, mips=v.mips, num_pes=v.num_pes,
            cost_per_second=v.cost_per_second,
            carbon_per_second=v.carbon_per_second, available_at=0.0) for v in vms]


def _proxy_exec_time(task: Task, vm: VM) -> float:
    """
    CHEAP PROXY ONLY — used exclusively for guiding iterative search
    heuristics (ACO, PSO, simulated annealing) toward promising
    assignments. This is NOT the source of truth for reported metrics;
    CloudSim Plus computes the authoritative execution time, makespan,
    cost, carbon, and utilization once an assignment is finalized.
    """
    pes_available = max(1, vm.num_pes)
    pes_needed = max(1, task.num_pes)
    parallel_efficiency = min(1.0, pes_available / pes_needed)
    return (task.length / vm.mips) / parallel_efficiency


def _fitness_assignment(assignment, tasks, vms) -> float:
    vm_clock = defaultdict(float)
    for i, task in enumerate(tasks):
        vm = vms[assignment[i]]
        if task.num_pes > vm.num_pes:
            return float('inf')
        exec_t = task.length / vm.mips
        start = max(vm_clock[vm.id], task.submit_time)
        vm_clock[vm.id] = start + exec_t
    return max(vm_clock.values()) if vm_clock else 0.0


def assignment_from_vector(assignment: List[int], tasks: List[Task]) -> Dict[int, int]:
    """
    Convert an integer assignment vector into the {task_id: vm_id}
    mapping expected by the CloudSim Plus bridge. Replaces the old
    _assignment_to_schedule, which computed start/finish in Python —
    that responsibility now belongs entirely to CloudSim Plus.
    """
    return {task.id: assignment[i] for i, task in enumerate(tasks)}