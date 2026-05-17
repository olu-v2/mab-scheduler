from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class Task:
    id: int
    submit_time: float
    length: float        # MI (millions of instructions)
    num_pes: int         # number of processors required
    deadline: Optional[float] = None

@dataclass
class VM:
    id: int
    mips: float          # processing speed (MIPS)
    num_pes: int
    cost_per_second: float
    carbon_per_second: float
    available_at: float = 0.0

@dataclass
class ScheduleResult:
    heuristic: str
    makespan: float
    total_cost: float
    total_carbon: float
    utilization: float
    reward: float
    schedule: Dict       # task_id -> (vm_id, start, finish)