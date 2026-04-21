import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

sns.set_theme(style="whitegrid", font_scale=1.2)

def plot_kmeans():
    dfs = []
    for f in ['kmeans_n10k.csv', 'kmeans_n50k.csv', 'kmeans_n200k.csv', 'kmeans_n500k.csv']:
        path = f"experiments/results/{f}"
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    
    if not dfs:
        print("No kmeans data found.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Plot 1: K-Means Runtime Scaling (Log-Log)
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x='n_samples', y='median_s', hue='impl', marker='o', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("K-Means Runtime Scaling (Lower is Better)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Median Runtime (seconds)")
    plt.tight_layout()
    plt.savefig("experiments/results/kmeans_scaling_runtime.png", dpi=300)
    
    # Plot 2: K-Means Memory Peak
    plt.figure(figsize=(10, 6))
    # Excluding loops because it's too slow and irrelevant for large scale memory
    df_mem = df[df['impl'] != 'loops']
    sns.barplot(data=df_mem, x='n_samples', y='peak_mb', hue='impl')
    plt.title("K-Means Peak Memory Allocation (Lower is Better)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Peak Memory (MB)")
    plt.tight_layout()
    plt.savefig("experiments/results/kmeans_memory_peak.png", dpi=300)

def plot_permtest():
    dfs = []
    for f in ['perm_n2k.csv', 'perm_n10k.csv', 'perm_n20k.csv']:
        path = f"experiments/results/{f}"
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
            
    if not dfs:
        print("No permtest data found.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Plot 3: Permutation Test Runtime Scaling
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='median_s', hue='impl', marker='s', linewidth=2, markersize=8)
    plt.title("Permutation Test Runtime Scaling")
    plt.xlabel("Total Sample Size (N)")
    plt.ylabel("Median Runtime (seconds)")
    plt.tight_layout()
    plt.savefig("experiments/results/permtest_scaling_runtime.png", dpi=300)
    
    # Plot 4: Permutation Test JAX Compilation/Overhead
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='n', y='median_s', hue='impl')
    plt.title("Permutation Test Bar Comparison")
    plt.xlabel("Total Sample Size (N)")
    plt.ylabel("Median Runtime (seconds)")
    plt.tight_layout()
    plt.savefig("experiments/results/permtest_bar_runtime.png", dpi=300)

if __name__ == "__main__":
    plot_kmeans()
    plot_permtest()
    print("Generated all high-quality charts.")
