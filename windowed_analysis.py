import sys
import pandas as pd

CURRENT_HEURISTICS = ["heft", "peft", "min_min", "max_min", "round_robin",
                      "hghhc", "sufferage", "heft_duplication", "simulated_annealing"]

dataset = sys.argv[1] if len(sys.argv) > 1 else "nasa"
output_dir = f"output/{dataset}"

mab = pd.read_csv(f"{output_dir}/mab_results.csv")

baselines = {}
for name in CURRENT_HEURISTICS:
    baselines[name] = pd.read_csv(f"{output_dir}/baseline_{name}.csv")

max_round = mab["round"].max()
windows = [(1, max_round), (max_round-199, max_round), (max_round-99, max_round), (max_round-49, max_round)]

print(f"{'Window':<12}{'MAB avg_reward':>18}{'MAB avg_makespan':>20}")
for start, end in windows:
    sub = mab[(mab["round"] >= start) & (mab["round"] <= end)]
    print(f"{start}-{end:<8}{sub['reward'].mean():>18.2f}{sub['makespan'].mean():>20.2f}")

print()
print(f"{'Heuristic':<22}{'Full avg_reward':>18}{'Last-100 avg_reward':>22}{'Full avg_makespan':>20}{'Last-100 avg_makespan':>22}")
for name, df in baselines.items():
    full_r = df["reward"].mean()
    last100_r = df[df["round"] > df["round"].max() - 100]["reward"].mean()
    full_m = df["makespan"].mean()
    last100_m = df[df["round"] > df["round"].max() - 100]["makespan"].mean()
    print(f"{name:<22}{full_r:>18.2f}{last100_r:>22.2f}{full_m:>20.2f}{last100_m:>22.2f}")

mab_last100 = mab[mab["round"] > mab["round"].max() - 100]
print()
print(f"MAB last-100 avg_reward:   {mab_last100['reward'].mean():.2f}")
print(f"MAB last-100 avg_makespan: {mab_last100['makespan'].mean():.2f}")

results_summary = []
for name, df in baselines.items():
    last100 = df[df["round"] > df["round"].max() - 100]
    results_summary.append({
        "label": name,
        "full_avg_reward": df["reward"].mean(),
        "last100_avg_reward": last100["reward"].mean(),
        "full_avg_makespan": df["makespan"].mean(),
        "last100_avg_makespan": last100["makespan"].mean(),
    })
results_summary.append({
    "label": "MAB-UCB",
    "full_avg_reward": mab["reward"].mean(),
    "last100_avg_reward": mab_last100["reward"].mean(),
    "full_avg_makespan": mab["makespan"].mean(),
    "last100_avg_makespan": mab_last100["makespan"].mean(),
})
pd.DataFrame(results_summary).to_csv(f"{output_dir}/windowed_comparison.csv", index=False)
print(f"\nSaved {output_dir}/windowed_comparison.csv")