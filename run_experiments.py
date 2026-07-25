import random
import numpy as np
import pandas as pd
import os
from typing import List, Dict

# Import your setup functions and classes from scheduler.py
from scheduler import (
    create_mab,
    run_simulation,
    CloudSimDaemon, # Or whatever daemon launcher/reference you use
)
from utils import (
    load_swf,
    generate_vms,
)

JAR_PATH = os.path.expanduser("~/Documents/mab-vm-pool/target/mab-vm-pool-1.0-SNAPSHOT.jar")
SEEDS = [42, 101, 2024, 7, 888, 99, 123, 456, 789, 1337]  # 10 seeds
MAB_TYPES = ['ucb', 'discounted_ucb', 'epsilon', 'thompson']

SWF_PATH = "NASA-iPSC-1993-3.1-cln.swf"
BATCH_SIZE = 150
N_VMS = 4

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def run_seed_experiment(mab_type: str, seed: int, tasks, vms, daemon) -> Dict[str, float]:
    """Runs a single seed pass for a given MAB type and returns aggregated run metrics."""
    set_seed(seed)
    
    # 1. Instantiate fresh MAB for this seed
    mab = create_mab(
        mab_type=mab_type,
        arms=['heft', 'peft', 'min_min', 'max_min', 'round_robin', 'hghhc', 'sufferage', 'heft_duplication', 'simulated_annealing'],
        adaptive_ucb_c=0.04,
        epsilon=0.1
    )
    
    # 2. Run simulation loop (verbose=False to avoid massive log dumps)
    round_results = run_simulation(
        tasks=tasks,
        vms=vms,
        mab=mab,
        daemon=daemon,
        batch_size=BATCH_SIZE,
        rank_bonus=0.1,
        verbose=False
    )
    
    # 3. Calculate average metrics across all rounds in this run
    avg_makespan = np.mean([r['makespan'] for r in round_results])
    avg_cost = np.mean([r['cost'] for r in round_results])
    avg_reward = np.mean([r['reward'] for r in round_results])
    
    return {
        'avg_makespan': avg_makespan,
        'avg_cost': avg_cost,
        'avg_reward': avg_reward
    }

def main():
    print("Initializing environment tasks, VMs, and CloudSim daemon...")
    # Load trace and VMs once to share across experiments
    tasks = load_swf(SWF_PATH, max_tasks=15000)
    vms = generate_vms(n_vms=N_VMS)
    daemon = CloudSimDaemon(JAR_PATH)

    summary_rows = []

    print(f"\n=== Starting Multi-Seed Evaluation ({len(SEEDS)} seeds per MAB) ===")

    for mab_type in MAB_TYPES:
        print(f"Testing {mab_type}...", end="", flush=True)
        seed_runs = []
        
        for seed in SEEDS:
            run_metrics = run_seed_experiment(mab_type, seed, tasks, vms, daemon)
            seed_runs.append(run_metrics)
            print(".", end="", flush=True)
        print(" Done!")

        # Compute mean and std dev across all seeds
        makespans = [r['avg_makespan'] for r in seed_runs]
        costs = [r['avg_cost'] for r in seed_runs]
        rewards = [r['avg_reward'] for r in seed_runs]

        summary_rows.append({
            "MAB Algorithm": mab_type,
            "Makespan (Mean ± Std)": f"{np.mean(makespans)/1e6:.2f}M ± {np.std(makespans)/1e6:.2f}M",
            "Cost ($)": f"${np.mean(costs):.2f} ± ${np.std(costs):.2f}",
            "Reward": f"{np.mean(rewards):.4f} ± {np.std(rewards):.4f}"
        })

    # Print nicely formatted summary table
    df = pd.DataFrame(summary_rows)
    print("\n============================================================")
    print("FINAL MULTI-SEED STATISTICAL RESULTS")
    print("============================================================")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()