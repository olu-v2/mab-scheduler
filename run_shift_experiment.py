import math
import random
import numpy as np
import pandas as pd
import os

# Import setup utilities and helpers directly from your scheduler.py
from scheduler import (
    create_mab,
    CloudSimDaemon,
    _normalize_batch,
    _reset_vms,
    compute_metrics_via_cloudsim,
    RewardCalculator,
    HEURISTICS,
)

from utils import (
    load_swf,
    generate_vms,
)

SWF_PATH = "HPC2N-2002-2.2-cln.swf"
JAR_PATH = os.path.expanduser("~/Documents/mab-vm-pool/target/mab-vm-pool-1.0-SNAPSHOT.jar")
BATCH_SIZE = 150
N_VMS = 4
SHIFT_ROUND = 50
MIPS_DROP_FACTOR = 0.1  # 90% capacity drop


def run_simulation_with_shift(
    tasks,
    vms,
    mab,
    daemon,
    batch_size: int = 150,
    shift_round: int = 50,
    rank_bonus: float = 0.1,
    verbose: bool = True,
):
    results = []
    n_tasks = len(tasks)
    n_rounds = math.ceil(n_tasks / batch_size)

    reward_calculator = RewardCalculator(rank_bonus=rank_bonus)
    last_performance = {h: float("inf") for h in HEURISTICS.keys()}

    if verbose:
        print(
            f"\nRunning Shift Experiment: {n_tasks} tasks | "
            f"batch_size={batch_size} | rounds={n_rounds} | n_vms={len(vms)}"
        )
        print(
            f"MAB: {type(mab).__name__} | Capacity Drop at Round {shift_round}\n"
        )

    for i in range(n_rounds):
        round_num = i + 1

        # --- NON-STATIONARY SHIFT INJECTION AT TARGET ROUND ---
        if round_num == shift_round:
            for vm in vms:
                vm.mips = vm.mips * MIPS_DROP_FACTOR
            if verbose:
                print(
                    f"\n⚡ [WORKLOAD SHIFT INJECTED] Round {round_num}: VM capacity dropped by {int((1 - MIPS_DROP_FACTOR) * 100)}%!\n"
                )
        # ------------------------------------------------------

        batch = tasks[i * batch_size : (i + 1) * batch_size]
        if not batch:
            break

        batch = _normalize_batch(batch)

        chosen = mab.select()
        fn = HEURISTICS[chosen]
        vms_fresh = _reset_vms(vms)
        schedule = fn(batch, vms_fresh)

        makespan, cost, carbon, util = compute_metrics_via_cloudsim(
            schedule, batch, vms_fresh, daemon
        )

        is_best = makespan <= min(last_performance.values())

        reward = reward_calculator.compute_reward(
            batch_tasks=batch,
            actual_makespan=makespan,
            vms=vms_fresh,
            is_best_in_round=is_best,
        )

        last_performance[chosen] = makespan
        mab.update(chosen, reward)

        results.append(
            {
                "round": round_num,
                "heuristic": chosen,
                "makespan": round(makespan, 4),
                "reward": round(reward, 6),
            }
        )

        if verbose and (
            round_num % 10 == 0 or round_num == 1 or round_num == shift_round
        ):
            print(
                f"  Round {round_num:4d}/{n_rounds} | "
                f"Heuristic: {chosen:18s} | "
                f"Makespan: {makespan:10.2f} | "
                f"Reward: {reward:.4f}"
            )

    return results


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def main():
    print("Initializing environment...")
    tasks = load_swf(SWF_PATH, max_tasks=15000)
    daemon = CloudSimDaemon(JAR_PATH)

    # Compare Discounted UCB vs Standard UCB on the shift
    algorithms_to_showcase = ["discounted_ucb", "ucb"]

    for mab_name in algorithms_to_showcase:
        set_seed(42)  # Reset seed for an exact side-by-side comparison
        vms = generate_vms(n_vms=N_VMS)
        mab = create_mab(
            mab_type=mab_name,
            arms=list(HEURISTICS.keys()),
            adaptive_ucb_c=0.04,
        )

        print(
            "\n============================================================"
        )
        print(f"SHOWCASE SHIFT DEMO: {mab_name.upper()}")
        print(
            "============================================================"
        )

        run_simulation_with_shift(
            tasks=tasks,
            vms=vms,
            mab=mab,
            daemon=daemon,
            batch_size=BATCH_SIZE,
            shift_round=SHIFT_ROUND,
            verbose=True,
        )


if __name__ == "__main__":
    main()