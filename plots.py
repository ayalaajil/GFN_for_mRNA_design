import matplotlib.pyplot as plt
import torch
import seaborn as sns
from Levenshtein import distance as levenshtein_distance
import ternary
import numpy as np
import pandas as pd

from evaluate import is_pareto_efficient_3d


def plot_training_curves(
    loss_history, reward_components, out_path="training_curves.png"
):

    reward_tensor = torch.tensor(reward_components)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Training Curves", fontsize=18)

    # Plot Loss
    axes[0].plot(loss_history, linewidth=2)
    axes[0].set_title("Loss", fontsize=14)
    axes[0].set_xlabel("Iteration", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)

    # Plot Reward Components
    labels = ["GC", "MFE", "CAI"]
    colors = ["green", "blue", "orange"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        print(reward_tensor[:, i])
        axes[i + 1].plot(reward_tensor[:, i], label=label, color=color, linewidth=2)
        axes[i + 1].set_title(f"{label} Evolution", fontsize=14)
        axes[i + 1].set_xlabel("Iteration", fontsize=12)
        axes[i + 1].legend()

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_sweep_results(csv_path="sweep_results.csv"):

    df = pd.read_csv(csv_path)

    metrics = ["MFE", "CAI", "GC"]
    titles = ["MFE comparison", "CAI comparison", "GC comparison"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):

        ax = axes[i]
        sns.boxplot(x="Config", y=metric, data=df, ax=ax, palette="colorblind")
        sns.stripplot(
            x="Config", y=metric, data=df, ax=ax, color="black", alpha=0.4, jitter=True
        )
        ax.set_title(titles[i])
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("sweep_results_boxplots.png", dpi=300)
    plt.close()


def plot_metric_histograms(
    gc_list, mfe_list, cai_list, out_path="metric_distributions.png"
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [gc_list, mfe_list, cai_list]
    titles = ["GC Content", "MFE", "CAI"]
    colors = ["green", "blue", "orange"]

    for i in range(3):
        axes[i].hist(metrics[i], bins=20, color=colors[i])
        axes[i].set_title(f"{titles[i]} Distribution", fontsize=14)
        axes[i].set_xlabel(titles[i])
        axes[i].set_ylabel("Count")
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pareto_front(gc_list, mfe_list, cai_list, out_path="pareto_scatter.png"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gc_list, mfe_list, cai_list, alpha=0.6)
    ax.set_xlabel("GC Content")
    ax.set_ylabel("MFE")
    ax.set_zlabel("CAI")
    ax.set_title("Pareto Front")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cai_vs_mfe(cai_list, mfe_list, out_path="cai_vs_mfe.png"):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=mfe_list, y=cai_list, color="blue", alpha=0.7)
    plt.xlabel("MFE (kcal/mol)")
    plt.ylabel("CAI")
    plt.title("CAI vs MFE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_gc_vs_mfe(gc_list, mfe_list, out_path="gc_vs_mfe.png"):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=mfe_list, y=gc_list, color="green", alpha=0.7)
    plt.xlabel("MFE (kcal/mol)")
    plt.ylabel("GC Content")
    plt.title("GC Content vs MFE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_diversity(sequences, out_path="edit_distance_distribution.png"):
    distances = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            d = levenshtein_distance(sequences[i], sequences[j])
            distances.append(d)

    plt.figure(figsize=(7, 5))
    sns.histplot(distances, bins=20, kde=True)
    plt.xlabel("Levenshtein Distance")
    plt.ylabel("Frequency")
    plt.title("Edit Distance Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return distances


def plot_of_weights_over_iterations(
    sampled_weights, out_path="plot_of_weights_over_iterations.png"
):

    plt.figure(figsize=(10, 4))
    plt.plot(sampled_weights[:, 0], label="w_gc")
    plt.plot(sampled_weights[:, 1], label="w_mfe")
    plt.plot(sampled_weights[:, 2], label="w_cai")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.title("Sampled Weights during Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_ternary_plot_of_weights(
    sampled_weights, out_path="sampled_weights_ternary.png"
):

    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    tax.gridlines(multiple=0.1, color="gray")
    tax.scatter(sampled_weights, marker="o", color="blue", alpha=0.5)
    tax.set_title("Distribution of Sampled Weights")
    tax.ticks(axis="lbr", multiple=0.1)
    figure.tight_layout()
    figure.savefig(out_path, dpi=300)

    tax.show()


def plot_pairwise_scatter(results, x_key, y_key):
    """
    Scatter-plot results[i].metrics[x_key] vs. results[i].metrics[y_key]
    for each config in results.
    """

    default_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    default_colors = plt.cm.tab10.colors
    markers = default_markers
    colors = default_colors

    plt.figure(figsize=(6, 6))

    for i, res in enumerate(results):
        x = res["metrics"][x_key]
        y = res["metrics"][y_key]
        m = markers[i % len(markers)]
        c = colors[i % len(colors)]
        plt.scatter(
            x,
            y,
            marker=m,
            color=c,
            alpha=0.4,
            label=res["name"],
            edgecolors="none",
            s=30,
        )

    plt.legend(loc="best", fontsize="small", framealpha=0.8)
    plt.title(f"{y_key.upper()} vs {x_key.upper()} by Config")
    plt.xlabel(x_key.upper())
    plt.ylabel(y_key.upper())
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{y_key.upper()}_vs_{x_key.upper()}_by_config.png", dpi=300)
    plt.show()

def plot_pairwise_scatter_with_pareto(results, x_key="CAI", y_key="MFE"):
    """
    Scatter plot of y_key vs x_key (e.g., MFE vs CAI) by config,
    with Pareto front computed using all 3 objectives: CAI (maximize), MFE (minimize), GC (maximize).
    """

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = plt.cm.tab10.colors

    all_points_2d = []
    all_points_3d = []

    plt.figure(figsize=(7, 6))

    for i, res in enumerate(results):
        xs = np.array(res["metrics"][x_key])
        ys = np.array(res["metrics"][y_key])
        gcs = np.array(res["metrics"]["GC"])
        m = markers[i % len(markers)]
        c = colors[i % len(colors)]

        all_points_2d.extend(zip(xs, ys))
        all_points_3d.extend(zip(xs, ys, gcs))

        plt.scatter(
            xs,
            ys,
            marker=m,
            color=c,
            alpha=0.4,
            label=res["name"],
            edgecolors="none",
            s=40,
        )

    # Compute Pareto front
    cost_array = np.array([[-x, y, -gc] for x, y, gc in all_points_3d])
    pareto_mask = is_pareto_efficient_3d(cost_array)
    pareto_points = np.array(all_points_2d)[pareto_mask]

    pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]

    plt.plot(
        pareto_sorted[:, 0],
        pareto_sorted[:, 1],
        color="black",
        linestyle="--",
        label="Pareto Front",
        linewidth=1.5,
    )
    plt.scatter(
        pareto_sorted[:, 0],
        pareto_sorted[:, 1],
        color="black",
        edgecolor="k",
        marker="X",
        s=60,
        label="Pareto Points",
    )

    plt.gca().invert_yaxis()  # since lower MFE is better
    plt.xlabel(x_key.upper())
    plt.ylabel(y_key.upper())
    plt.title(
        f"{y_key.upper()} vs {x_key.upper()} by Config\nwith Pareto Front (incl. GC)"
    )
    plt.legend(loc="best", fontsize="small", framealpha=0.8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{y_key.upper()}_vs_{x_key.upper()}_with_Pareto.png", dpi=300)
    plt.show()


def plot_task_distribution_analysis(task_distribution_history, curriculum_tasks, output_dir="."):
    """Create task distribution analysis plots for curriculum learning"""

    if not task_distribution_history:
        print("No task distribution history available for plotting")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Task Distribution Analysis", fontsize=16, fontweight='bold')

    # Extract data
    steps = [record['step'] for record in task_distribution_history]
    task_indices = [record['task_idx'] for record in task_distribution_history]
    distributions = [record['distribution'] for record in task_distribution_history]

    # 1. Task selection over time
    axes[0, 0].plot(steps, task_indices, marker='o', markersize=2, alpha=0.7)
    axes[0, 0].set_title('Task Selection Over Time')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Task Index')
    axes[0, 0].set_yticks(range(len(curriculum_tasks)))
    axes[0, 0].grid(True, alpha=0.3)

    # Add task labels
    task_labels = []
    for task in curriculum_tasks:
        if isinstance(task, (list, tuple)) and len(task) == 2:
            task_labels.append(f"[{task[0]}-{task[1]}]")
        else:
            task_labels.append(f"Len {task}")
    axes[0, 0].set_yticklabels(task_labels)

    # 2. Task distribution evolution
    task_dist_matrix = np.array(distributions).T  # Shape: (n_tasks, n_steps)

    for i in range(len(curriculum_tasks)):
        axes[0, 1].plot(steps, task_dist_matrix[i], label=task_labels[i], linewidth=2)

    axes[0, 1].set_title('Task Distribution Evolution')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Task selection frequency (histogram)
    task_counts = np.bincount(task_indices, minlength=len(curriculum_tasks))
    bars = axes[1, 0].bar(range(len(curriculum_tasks)), task_counts, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Task Selection Frequency')
    axes[1, 0].set_xlabel('Tasks')
    axes[1, 0].set_ylabel('Selection Count')
    axes[1, 0].set_xticks(range(len(curriculum_tasks)))
    axes[1, 0].set_xticklabels(task_labels, rotation=45, ha='right')

    # Add count labels on bars
    for bar, count in zip(bars, task_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(task_counts)*0.01,
                       str(count), ha='center', va='bottom', fontsize=8)

    # 4. Distribution entropy over time
    from scipy.stats import entropy
    entropies = [entropy(dist, base=2) for dist in distributions]

    axes[1, 1].plot(steps, entropies, linewidth=2, color='purple')
    axes[1, 1].set_title('Distribution Entropy Over Time')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Entropy (bits)')
    axes[1, 1].grid(True, alpha=0.3)

    # Add moving average for entropy
    if len(entropies) > 10:
        window_size = min(10, len(entropies) // 5)
        moving_avg_entropy = pd.Series(entropies).rolling(window=window_size).mean()
        axes[1, 1].plot(steps, moving_avg_entropy, color='red', linewidth=2,
                       label=f'Moving Avg (window={window_size})')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/task_distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Task distribution analysis plot saved to {output_dir}/task_distribution_analysis.png")
