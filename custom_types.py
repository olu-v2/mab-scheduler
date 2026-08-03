from dataclasses import dataclass
from typing import Optional


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