import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
from Levenshtein import distance as levenshtein_distance
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import wandb

def plot_training_curves(loss_history, reward_components, out_path="training_curves.png"):

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
    colors = ['green', 'blue', 'orange']

    for i, (label, color) in enumerate(zip(labels, colors)):
        axes[i + 1].plot(reward_tensor[:, i], label=label, color=color, linewidth=2)
        axes[i + 1].set_title(f"{label} Evolution", fontsize=14)
        axes[i + 1].set_xlabel("Iteration", fontsize=12)
        axes[i + 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.savefig(out_path, dpi=300) 
    plt.close()


def plot_metric_histograms(gc_list, mfe_list, cai_list, out_path="metric_distributions.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [gc_list, mfe_list, cai_list]
    titles = ['GC Content', 'MFE', 'CAI']
    colors = ['green', 'blue', 'orange']

    for i in range(3):
        axes[i].hist(metrics[i], bins=20, color=colors[i])
        axes[i].set_title(f'{titles[i]} Distribution', fontsize=14)
        axes[i].set_xlabel(titles[i])
        axes[i].set_ylabel("Count")
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    

def plot_pareto_front(gc_list, mfe_list, cai_list, out_path="pareto_scatter.png"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gc_list, mfe_list, cai_list, alpha=0.6)
    ax.set_xlabel('GC Content')
    ax.set_ylabel('MFE')
    ax.set_zlabel('CAI')
    ax.set_title("Pareto Front")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cai_vs_mfe(cai_list, mfe_list, out_path="cai_vs_mfe.png"):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=mfe_list, y=cai_list, color='blue', alpha=0.7)
    plt.xlabel("MFE (kcal/mol)")
    plt.ylabel("CAI")
    plt.title("CAI vs MFE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_gc_vs_mfe(gc_list, mfe_list, out_path="gc_vs_mfe.png"):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=mfe_list, y=gc_list, color='green', alpha=0.7)
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