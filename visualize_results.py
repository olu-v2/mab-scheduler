
"""
Visualization script for MAB scheduler results (Matplotlib version -
no Chrome/Kaleido dependency).
Reads output/mab_results.csv, output/baseline_*.csv, output/windowed_comparison.csv
Run this AFTER scheduler.py and windowed_analysis.py have produced fresh output/ files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


dataset = sys.argv[1] if len(sys.argv) > 1 else "nasa"
output_dir = f"output/{dataset}"

os.makedirs(F"charts/{dataset}", exist_ok=True)

CURRENT_HEURISTICS = ["heft", "peft", "min_min", "max_min", "round_robin", "hghhc", "sufferage", "heft_duplication", "simulated_annealing"]

mab = pd.read_csv(f"{output_dir}/mab_results.csv")

baselines = {}
for name in CURRENT_HEURISTICS:
    path = f"{output_dir}/baseline_{name}.csv"
    if os.path.exists(path):
        baselines[name] = pd.read_csv(path)

windowed = None
if os.path.exists("output/windowed_comparison.csv"):
    windowed = pd.read_csv("output/windowed_comparison.csv")
    windowed = windowed[windowed["label"].isin(CURRENT_HEURISTICS + ["MAB-UCB"])]

# -----------------------------------------------------------------------
# CHART 1: Makespan over rounds - MAB vs all baselines (line chart)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(mab["round"], mab["makespan"], label="MAB-UCB", linewidth=2.5, color="black")
for name, df in baselines.items():
    ax.plot(df["round"], df["makespan"], label=name, linewidth=1, linestyle="--", alpha=0.6)
ax.set_title("Makespan per round: MAB-UCB vs static heuristics")
ax.set_xlabel("Round")
ax.set_ylabel("Makespan (s)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(f"charts/{dataset}/makespan_over_rounds.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# CHART 2: Final avg reward comparison (bar chart)
# -----------------------------------------------------------------------
avg_rewards = {"MAB-UCB": mab["reward"].mean()}
for name, df in baselines.items():
    avg_rewards[name] = df["reward"].mean()

df_rewards = pd.Series(avg_rewards).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
df_rewards.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Average reward by heuristic (higher is better)")
ax.set_xlabel("Heuristic")
ax.set_ylabel("Avg reward")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"charts/{dataset}/avg_reward_comparison.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# CHART 3: Heuristic selection counts (pie chart) - MAB arm distribution
# -----------------------------------------------------------------------
selection_counts = mab["heuristic"].value_counts()

fig, ax = plt.subplots(figsize=(8, 8))
selection_counts.plot(kind="pie", ax=ax, autopct="%1.1f%%", textprops={"fontsize": 9})
ax.set_title("MAB-UCB heuristic selection distribution")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"charts/{dataset}/mab_selection_distribution.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# CHART 4: Full-run vs Last-100-round avg reward (grouped bar) - windowed
# -----------------------------------------------------------------------
if windowed is not None:
    plot_df = windowed.set_index("label")[["full_avg_reward", "last100_avg_reward"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Reward: full run vs last-100-round window")
    ax.set_xlabel("Heuristic")
    ax.set_ylabel("Avg reward")
    ax.legend(["Full run", "Last 100 rounds"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"charts/{dataset}/windowed_reward_comparison.png", dpi=150)
    plt.close()

# -----------------------------------------------------------------------
# CHART 5: Cost vs Carbon scatter (bubble sized by makespan) - baselines + MAB
# -----------------------------------------------------------------------
scatter_rows = []
scatter_rows.append({"label": "MAB-UCB", "avg_cost": mab["cost"].mean(),
                     "avg_carbon": mab["carbon"].mean(), "avg_makespan": mab["makespan"].mean()})
for name, df in baselines.items():
    scatter_rows.append({"label": name, "avg_cost": df["cost"].mean(),
                          "avg_carbon": df["carbon"].mean(), "avg_makespan": df["makespan"].mean()})
df_scatter = pd.DataFrame(scatter_rows)

fig, ax = plt.subplots(figsize=(10, 7))
sizes = (df_scatter["avg_makespan"] / df_scatter["avg_makespan"].max()) * 2000 + 100
ax.scatter(df_scatter["avg_cost"], df_scatter["avg_carbon"], s=sizes, alpha=0.5, color="steelblue")

used_positions = []
for i, row in df_scatter.iterrows():
    dx, dy = 8, 8
    for ux, uy in used_positions:
        if abs(row["avg_cost"] - ux) < 15 and abs(row["avg_carbon"] - uy) < 0.05:
            dy += 14
    used_positions.append((row["avg_cost"], row["avg_carbon"]))
    ax.annotate(row["label"], (row["avg_cost"], row["avg_carbon"]),
                textcoords="offset points", xytext=(dx, dy), fontsize=8,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

ax.set_title("Cost vs carbon footprint by heuristic (bubble = makespan)")
ax.set_xlabel("Avg cost ($)")
ax.set_ylabel("Avg carbon (g)")
plt.tight_layout()
plt.savefig(f"charts/{dataset}/cost_vs_carbon.png", dpi=150)
plt.close()

print("All charts saved to charts/ directory.")
