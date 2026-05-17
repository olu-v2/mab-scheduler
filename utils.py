import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
from custom_types import Task, VM

def load_swf(filepath: str, max_tasks: int = None) -> List[Task]:
    """
    Parse a Standard Workload Format (SWF) file.
    SWF columns (0-indexed):
      0: job_id, 1: submit_time, 3: run_time, 4: num_procs
    Lines starting with ';' are comments.
    """
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
                job_id      = int(fields[0])
                submit_time = float(fields[1])
                run_time    = float(fields[3])
                num_procs   = int(fields[4])

                # Skip jobs with invalid data
                if run_time <= 0 or num_procs <= 0:
                    continue

                tasks.append(Task(
                    id=job_id,
                    submit_time=submit_time,
                    length=run_time * max(1, num_procs),  # MI estimate
                    num_pes=max(1, num_procs)
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

def generate_vms(n_vms: int = 5) -> List[VM]:
    """
    Generate a heterogeneous VM pool.
    Modelled loosely on AWS EC2 instance types (us-east-1).
    """
    configs = [
        # (mips, pes, cost/s,  carbon/s)
        (500,   1,  0.0116/3600, 0.0003/3600),  # t3.small
        (1000,  2,  0.0464/3600, 0.0006/3600),  # t3.large
        (2000,  4,  0.1664/3600, 0.0012/3600),  # c6i.xlarge
        (4000,  8,  0.3400/3600, 0.0024/3600),  # c6i.2xlarge
        (8000, 16,  0.6800/3600, 0.0048/3600),  # c6i.4xlarge
    ]
    vms = []
    for i in range(min(n_vms, len(configs))):
        mips, pes, cost, carbon = configs[i]
        vms.append(VM(id=i, mips=mips, num_pes=pes,
                      cost_per_second=cost, carbon_per_second=carbon))
    return vms

def _reset_vms(vms: List[VM]) -> List[VM]:
    """Deep copy VMs so each heuristic run starts fresh."""
    return [VM(id=v.id, mips=v.mips, num_pes=v.num_pes,
               cost_per_second=v.cost_per_second,
               carbon_per_second=v.carbon_per_second,
               available_at=0.0) for v in vms]

def _fitness_assignment(assignment: List[int], tasks, vms) -> float:
    """Makespan of an assignment vector (fast, no schedule dict)."""
    vm_clock = defaultdict(float)
    for i, task in enumerate(tasks):
        vm     = vms[assignment[i]]
        exec_t = task.length / vm.mips
        start  = max(vm_clock[vm.id], task.submit_time)
        vm_clock[vm.id] = start + exec_t
    return max(vm_clock.values()) if vm_clock else 0.0

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