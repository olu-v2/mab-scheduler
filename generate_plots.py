import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

WORKLOADS = ["hpc2n", "nasa"]
MAB_ALGORITHMS = ["ucb", "discounted_ucb", "epsilon", "thompson"]

# -------------------------------------------------------------------
# 1. Plot Cumulative Makespan Trajectory
# -------------------------------------------------------------------
def plot_cumulative_makespan(workload: str, mab_alg: str, base_dir: str, plot_out_dir: str):
    mab_csv = os.path.join(base_dir, "mab_results.csv")
    if not os.path.exists(mab_csv):
        print(f"  [Skipped Plot 1] {mab_csv} not found.")
        return

    df_mab = pd.read_csv(mab_csv)
    
    plt.figure(figsize=(10, 5))
    
    # Plot MAB cumulative makespan
    cum_mab = df_mab["makespan"].cumsum() / 1e6
    plt.plot(df_mab["round"], cum_mab, label=f"MAB ({mab_alg.upper()})", linewidth=3, color="blue")

    # Overlay select baseline CSVs
    baseline_files = glob.glob(os.path.join(base_dir, "baseline_*.csv"))
    highlight_baselines = ["heft", "peft", "hghhc", "min_min"]

    for b_file in baseline_files:
        b_name = os.path.basename(b_file).replace("baseline_", "").replace(".csv", "")
        if b_name in highlight_baselines:
            df_b = pd.read_csv(b_file)
            cum_b = df_b["makespan"].cumsum() / 1e6
            plt.plot(df_b["round"], cum_b, label=f"Baseline: {b_name}", linestyle="--", alpha=0.7)

    plt.title(f"Cumulative Makespan Trajectory ({workload.upper()} - {mab_alg.upper()})", fontsize=14, fontweight="bold")
    plt.xlabel("Simulation Round")
    plt.ylabel("Cumulative Makespan ($10^6$ seconds)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    # Nomenclature: [trace]_[mab_algo]_[plot]
    filename = f"{workload}_{mab_alg}_cumulative_makespan.png"
    out_file = os.path.join(plot_out_dir, filename)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {filename}")

# -------------------------------------------------------------------
# 2. Plot Arm Selection Evolution (Stacked Area Chart)
# -------------------------------------------------------------------
def plot_arm_selection_evolution(workload: str, mab_alg: str, base_dir: str, plot_out_dir: str):
    mab_csv = os.path.join(base_dir, "mab_results.csv")
    if not os.path.exists(mab_csv):
        return

    df_mab = pd.read_csv(mab_csv)
    
    heuristics = df_mab["heuristic"].unique()
    window = 10
    
    history_df = pd.DataFrame({'round': df_mab['round']})
    for h in heuristics:
        history_df[h] = (df_mab['heuristic'] == h).astype(int)
        history_df[h] = history_df[h].rolling(window=window, min_periods=1).mean()

    # Normalize across rows
    norm_counts = history_df[heuristics].div(history_df[heuristics].sum(axis=1), axis=0)

    plt.figure(figsize=(12, 6))
    plt.stackplot(df_mab["round"], norm_counts.T, labels=heuristics, alpha=0.85, cmap="tab10")
    
    plt.title(f"Heuristic Selection Evolution Over Time ({workload.upper()} - {mab_alg.upper()})", fontsize=14, fontweight="bold")
    plt.xlabel("Round")
    plt.ylabel("Selection Ratio (10-Round Window)")
    plt.xlim(1, len(df_mab))
    plt.ylim(0, 1)
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Heuristics")
    plt.tight_layout()
    
    # Nomenclature: [trace]_[mab_algo]_[plot]
    filename = f"{workload}_{mab_alg}_arm_selection_evolution.png"
    out_file = os.path.join(plot_out_dir, filename)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {filename}")

# -------------------------------------------------------------------
# 3. Multi-Metric Radar Chart
# -------------------------------------------------------------------
def plot_multi_metric_radar(workload: str, mab_alg: str, base_dir: str, plot_out_dir: str):
    mab_csv = os.path.join(base_dir, "mab_results.csv")
    if not os.path.exists(mab_csv):
        return

    metrics = ["makespan", "cost", "carbon", "utilization"]
    data = {}

    df_mab = pd.read_csv(mab_csv)
    data["MAB"] = [df_mab[m].mean() for m in metrics]

    for b_file in glob.glob(os.path.join(base_dir, "baseline_*.csv")):
        b_name = os.path.basename(b_file).replace("baseline_", "").replace(".csv", "")
        if b_name in ["hghhc", "peft", "heft"]:
            df_b = pd.read_csv(b_file)
            data[b_name] = [df_b[m].mean() for m in metrics]

    df_radar = pd.DataFrame(data, index=["Makespan", "Cost", "Carbon", "Utilization"])

    # Normalize metrics to [0, 1] relative scale for fair comparison
    df_norm = df_radar.copy()
    for col in df_norm.index:
        max_val = df_norm.loc[col].max()
        if max_val > 0:
            df_norm.loc[col] = df_norm.loc[col] / max_val

    labels = df_norm.index.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for column in df_norm.columns:
        values = df_norm[column].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=column)
        ax.fill(angles, values, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels, size=11)
    
    plt.title(f"Normalized Multi-Objective Trade-Off ({workload.upper()} - {mab_alg.upper()})", fontsize=13, fontweight="bold", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    
    # Nomenclature: [trace]_[mab_algo]_[plot]
    filename = f"{workload}_{mab_alg}_radar_multimetric.png"
    out_file = os.path.join(plot_out_dir, filename)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {filename}")


if __name__ == "__main__":
    print("Starting automated plot generation across all workloads and MAB algorithms...\n")
    
    for workload in WORKLOADS:
        for mab_alg in MAB_ALGORITHMS:
            base_dir = f"output/{workload}/{mab_alg}"
            plot_out_dir = f"output/plots/{workload}/{mab_alg}"

            if not os.path.exists(base_dir):
                print(f"Directory missing: {base_dir} — skipping.")
                continue

            os.makedirs(plot_out_dir, exist_ok=True)
            print(f"Processing: Workload='{workload.upper()}' | Algorithm='{mab_alg.upper()}'")
            
            plot_cumulative_makespan(workload, mab_alg, base_dir, plot_out_dir)
            plot_arm_selection_evolution(workload, mab_alg, base_dir, plot_out_dir)
            plot_multi_metric_radar(workload, mab_alg, base_dir, plot_out_dir)
            print("-" * 60)

    print("\nAll batch visualizations completed!")