import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set publication style aesthetics
sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

BASE_OUTPUT_DIR = "output"
PLOT_OUT_DIR = "output/plots"


def generate_selection_heatmaps():
    # Find all mab_results.csv files matching output/[workload]/[mab_algorithm]/mab_results.csv
    search_pattern = os.path.join(BASE_OUTPUT_DIR, "*", "*", "mab_results.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"❌ No 'mab_results.csv' files found under pattern: {search_pattern}")
        print("Ensure your CSV results are stored in output/[workload]/[mab_algorithm]/mab_results.csv")
        return

    records = []

    print(f"Found {len(csv_files)} MAB result CSV files. Aggregating data...")

    for filepath in csv_files:
        # Normalize path for cross-platform OS separator compatibility
        norm_path = os.path.normpath(filepath)
        parts = norm_path.split(os.sep)

        # Expected path structure: ['output', '[workload]', '[mab_algorithm]', 'mab_results.csv']
        if len(parts) >= 4:
            workload = parts[-3]
            mab_alg = parts[-2]
        else:
            continue

        try:
            df = pd.read_csv(filepath)
            if 'heuristic' not in df.columns:
                print(f"⚠️ Warning: 'heuristic' column missing in {filepath}. Skipping.")
                continue

            # Count occurrences of each heuristic
            counts = df['heuristic'].value_counts().to_dict()
            for heuristic, count in counts.items():
                records.append({
                    'workload': workload,
                    'mab_algorithm': mab_alg,
                    'heuristic': heuristic,
                    'count': count
                })
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")

    if not records:
        print("❌ No selection data could be extracted.")
        return

    df_all = pd.DataFrame(records)
    os.makedirs(PLOT_OUT_DIR, exist_ok=True)

    # Process and plot heatmap per workload
    workloads = df_all['workload'].unique()

    for wl in workloads:
        df_wl = df_all[df_all['workload'] == wl]

        # Pivot table: Rows = MAB Algorithms, Columns = Heuristics, Values = Selection Count
        pivot_df = df_wl.pivot_table(
            index='mab_algorithm',
            columns='heuristic',
            values='count',
            aggfunc='sum',
            fill_value=0
        )

        # Normalize across rows (Convert raw counts to row percentages 0% - 100%)
        pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

        # Create Seaborn Heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(
            pivot_pct,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            cbar_kws={'label': 'Selection Frequency (%)'},
            linewidths=0.8,
            linecolor='white',
            ax=ax
        )

        # Format Titles and Labels
        ax.set_title(
            f"Cross-Algorithm Heuristic Selection Profile ({wl.upper()})",
            fontsize=14,
            fontweight="bold",
            pad=15
        )
        ax.set_xlabel("Low-Level Heuristics", fontsize=12, fontweight="bold")
        ax.set_ylabel("MAB Algorithm", fontsize=12, fontweight="bold")
        
        plt.xticks(rotation=35, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        out_file = os.path.join(PLOT_OUT_DIR, f"heatmap_selection_{wl}.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

        print(f"✅ Saved heatmap for workload '{wl}' to: {out_file}")


if __name__ == "__main__":
    generate_selection_heatmaps()