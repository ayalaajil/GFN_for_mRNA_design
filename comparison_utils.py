"""
Comparison and analysis functions for GFlowNet experiments.
Includes diversity metrics, quality metrics, and comprehensive comparison tables.
"""

import os
import numpy as np
from datetime import datetime
from prettytable import PrettyTable
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt

from utils import (
    compute_gc_content_vectorized,
    compute_cai,
    compute_mfe_energy,
    codon_gc_counts,
    decode_sequence,
    tokenize_sequence_to_tensor,
)

def compute_sequence_diversity(sequences):
    """Compute diversity metrics for a set of sequences."""
    if len(sequences) < 2:
        return {
            'mean_edit_distance': 0.0,
            'std_edit_distance': 0.0,
            'min_edit_distance': 0.0,
            'max_edit_distance': 0.0,
            'unique_sequences': len(sequences),
            'total_sequences': len(sequences),
            'uniqueness_ratio': 1.0
        }

    # Calculate pairwise edit distances
    distances = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            d = levenshtein_distance(sequences[i], sequences[j])
            distances.append(d)

    # Count unique sequences
    unique_sequences = len(set(sequences))

    return {
        'mean_edit_distance': np.mean(distances),
        'std_edit_distance': np.std(distances),
        'min_edit_distance': np.min(distances),
        'max_edit_distance': np.max(distances),
        'unique_sequences': unique_sequences,
        'total_sequences': len(sequences),
        'uniqueness_ratio': unique_sequences / len(sequences)
    }


def compute_quality_metrics(samples, gc_list, mfe_list, cai_list):
    """Compute quality metrics for generated sequences."""

    # Basic statistics
    rewards = [reward for reward, _ in samples.values()]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Pareto efficiency
    objectives = np.column_stack([gc_list, mfe_list, cai_list])
    pareto_mask = is_pareto_efficient_3d(objectives)
    pareto_efficiency = np.mean(pareto_mask)

    # Objective statistics
    gc_stats = {
        'mean': np.mean(gc_list),
        'std': np.std(gc_list),
        'min': np.min(gc_list),
        'max': np.max(gc_list)
    }

    mfe_stats = {
        'mean': np.mean(mfe_list),
        'std': np.std(mfe_list),
        'min': np.min(mfe_list),
        'max': np.max(mfe_list)
    }

    cai_stats = {
        'mean': np.mean(cai_list),
        'std': np.std(cai_list),
        'min': np.min(cai_list),
        'max': np.max(cai_list)
    }

    return {
        'reward_stats': {'mean': mean_reward, 'std': std_reward},
        'pareto_efficiency': pareto_efficiency,
        'gc_stats': gc_stats,
        'mfe_stats': mfe_stats,
        'cai_stats': cai_stats,
        'total_samples': len(samples)
    }


def is_pareto_efficient_3d(costs):
    """Determine Pareto-efficient points for 3 objectives."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)

    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Remove dominated points
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] > c, axis=1
            ) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True

    return is_efficient


def create_comprehensive_comparison_table(
    samples, natural_sequence, output_dir, timestamp,
    top_n=50, include_best_by_objective=True
):
    """Create a comprehensive comparison table with natural sequence."""

    # Sort samples by reward
    sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

    # Get top sequences
    top_sequences = [seq for seq, _ in sorted_samples[:top_n]]

    # Add best-by-objective sequences if requested
    if include_best_by_objective:
        best_gc = max(samples.items(), key=lambda x: x[1][1][0])  # GC content
        best_mfe = min(samples.items(), key=lambda x: x[1][1][1])  # MFE
        best_cai = max(samples.items(), key=lambda x: x[1][1][2])  # CAI

        additional_seqs = [best_gc[0], best_mfe[0], best_cai[0]]
        for seq in additional_seqs:
            if seq not in top_sequences:
                top_sequences.append(seq)

    # Create comparison table
    table = PrettyTable()
    table.field_names = [
        "Seq", "GC %", "MFE", "CAI", "Reward",
        "Levenshtein", "Identity %", "Length Diff"
    ]

    # Add generated sequences
    for i, seq in enumerate(top_sequences):
        # Get metrics from samples
        if seq in samples:
            reward, (gc, mfe, cai) = samples[seq]
        else:
            # Calculate if not in samples (for best-by-objective)
            seq_tensor = tokenize_sequence_to_tensor(seq)
            gc = compute_gc_content_vectorized(seq_tensor, codon_gc_counts).item()
            mfe = compute_mfe_energy(seq_tensor).item()
            cai = compute_cai(seq_tensor).item()
            reward = 0.0  # Placeholder

        # Calculate comparison metrics
        lev_dist = levenshtein_distance(seq, natural_sequence)
        identity = compute_identity(seq, natural_sequence)
        length_diff = len(seq) - len(natural_sequence)

        label = f"Gen {i+1}" if i < top_n else f"Best {['GC', 'MFE', 'CAI'][i-top_n]}"

        table.add_row([
            label,
            f"{gc:.2f}",
            f"{mfe:.2f}",
            f"{cai:.2f}",
            f"{reward:.2f}",
            lev_dist,
            f"{identity:.2f}",
            length_diff
        ])

    # Add natural sequence
    natural_tensor = tokenize_sequence_to_tensor(natural_sequence)
    nat_gc = compute_gc_content_vectorized(natural_tensor, codon_gc_counts).item()
    nat_mfe = compute_mfe_energy(natural_tensor).item()
    nat_cai = compute_cai(natural_tensor).item()

    table.add_row([
        "Natural",
        f"{nat_gc:.2f}",
        f"{nat_mfe:.2f}",
        f"{nat_cai:.2f}",
        "N/A",
        0,
        "100.00",
        0
    ])

    # Save table
    table_path = os.path.join(output_dir, f"comprehensive_comparison_{timestamp}.txt")
    with open(table_path, "w") as f:
        f.write("Comprehensive Sequence Comparison Table\n")
        f.write("=" * 50 + "\n\n")
        f.write(table.get_string())

    return table_path


def create_metrics_summary(
    samples, gc_list, mfe_list, cai_list, output_dir, timestamp
):
    """Create a comprehensive metrics summary."""

    # Compute diversity metrics
    sequences = list(samples.keys())
    diversity_metrics = compute_sequence_diversity(sequences)

    # Compute quality metrics
    quality_metrics = compute_quality_metrics(samples, gc_list, mfe_list, cai_list)

    # Create summary table
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value", "Description"]

    # Add diversity metrics
    summary_table.add_row(["Mean Edit Distance", f"{diversity_metrics['mean_edit_distance']:.2f}",
                          "Average Levenshtein distance between sequences"])
    summary_table.add_row(["Edit Distance Std", f"{diversity_metrics['std_edit_distance']:.2f}",
                          "Standard deviation of edit distances"])
    summary_table.add_row(["Unique Sequences", f"{diversity_metrics['unique_sequences']}",
                          "Number of unique sequences generated"])
    summary_table.add_row(["Uniqueness Ratio", f"{diversity_metrics['uniqueness_ratio']:.3f}",
                          "Ratio of unique to total sequences"])

    # Add quality metrics
    summary_table.add_row(["Mean Reward", f"{quality_metrics['reward_stats']['mean']:.3f}",
                          "Average reward across all sequences"])
    summary_table.add_row(["Reward Std", f"{quality_metrics['reward_stats']['std']:.3f}",
                          "Standard deviation of rewards"])
    summary_table.add_row(["Pareto Efficiency", f"{quality_metrics['pareto_efficiency']:.3f}",
                          "Fraction of sequences on Pareto front"])

    # Add objective statistics
    summary_table.add_row(["GC Mean", f"{quality_metrics['gc_stats']['mean']:.3f}",
                          "Average GC content"])
    summary_table.add_row(["MFE Mean", f"{quality_metrics['mfe_stats']['mean']:.3f}",
                          "Average MFE energy"])
    summary_table.add_row(["CAI Mean", f"{quality_metrics['cai_stats']['mean']:.3f}",
                          "Average CAI score"])

    # Save summary
    summary_path = os.path.join(output_dir, f"metrics_summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write("Experiment Metrics Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(summary_table.get_string())

        # Add detailed statistics
        f.write("\n\nDetailed Statistics\n")
        f.write("-" * 20 + "\n\n")

        f.write("GC Content Statistics:\n")
        f.write(f"  Mean: {quality_metrics['gc_stats']['mean']:.3f}\n")
        f.write(f"  Std:  {quality_metrics['gc_stats']['std']:.3f}\n")
        f.write(f"  Min:  {quality_metrics['gc_stats']['min']:.3f}\n")
        f.write(f"  Max:  {quality_metrics['gc_stats']['max']:.3f}\n\n")

        f.write("MFE Energy Statistics:\n")
        f.write(f"  Mean: {quality_metrics['mfe_stats']['mean']:.3f}\n")
        f.write(f"  Std:  {quality_metrics['mfe_stats']['std']:.3f}\n")
        f.write(f"  Min:  {quality_metrics['mfe_stats']['min']:.3f}\n")
        f.write(f"  Max:  {quality_metrics['mfe_stats']['max']:.3f}\n\n")

        f.write("CAI Score Statistics:\n")
        f.write(f"  Mean: {quality_metrics['cai_stats']['mean']:.3f}\n")
        f.write(f"  Std:  {quality_metrics['cai_stats']['std']:.3f}\n")
        f.write(f"  Min:  {quality_metrics['cai_stats']['min']:.3f}\n")
        f.write(f"  Max:  {quality_metrics['cai_stats']['max']:.3f}\n\n")

    return summary_path, diversity_metrics, quality_metrics


def plot_enhanced_diversity_analysis(sequences, output_dir, timestamp):
    """Create enhanced diversity analysis plots."""

    # Calculate edit distances
    distances = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            d = levenshtein_distance(sequences[i], sequences[j])
            distances.append(d)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Edit distance distribution
    axes[0, 0].hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Levenshtein Distance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Edit Distance Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Sequence length distribution
    lengths = [len(seq) for seq in sequences]
    axes[0, 1].hist(lengths, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sequence Length Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. GC content distribution
    gc_contents = []
    for seq in sequences:
        seq_tensor = tokenize_sequence_to_tensor(seq)
        gc = compute_gc_content_vectorized(seq_tensor, codon_gc_counts).item()
        gc_contents.append(gc)

    axes[1, 0].hist(gc_contents, bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('GC Content')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('GC Content Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. MFE energy distribution
    mfe_energies = []
    for seq in sequences:
        seq_tensor = tokenize_sequence_to_tensor(seq)
        mfe = compute_mfe_energy(seq_tensor).item()
        mfe_energies.append(mfe)

    axes[1, 1].hist(mfe_energies, bins=15, alpha=0.7, color='pink', edgecolor='black')
    axes[1, 1].set_xlabel('MFE Energy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('MFE Energy Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f"enhanced_diversity_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def run_comprehensive_analysis(
    samples, gc_list, mfe_list, cai_list, natural_sequence,
    output_dir, timestamp, top_n=50
):
    """Run comprehensive analysis and generate all comparison files."""

    print(f"[INFO] Running comprehensive analysis...")

    # Create comprehensive comparison table
    table_path = create_comprehensive_comparison_table(
        samples, natural_sequence, output_dir, timestamp, top_n
    )
    print(f"[INFO] Comparison table saved to: {table_path}")

    # Create metrics summary
    summary_path, diversity_metrics, quality_metrics = create_metrics_summary(
        samples, gc_list, mfe_list, cai_list, output_dir, timestamp
    )
    print(f"[INFO] Metrics summary saved to: {summary_path}")

    # Create enhanced diversity plots
    sequences = list(samples.keys())
    plot_path = plot_enhanced_diversity_analysis(sequences, output_dir, timestamp)
    print(f"[INFO] Diversity analysis plot saved to: {plot_path}")

    return {
        'table_path': table_path,
        'summary_path': summary_path,
        'plot_path': plot_path,
        'diversity_metrics': diversity_metrics,
        'quality_metrics': quality_metrics
    }


def compute_identity(seq1, seq2):
    """Percentage of identical characters in the same positions."""
    length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return (matches / length) * 100


def analyze_sequence_properties(
    seqs_tensor, natural_tensor, labels=None, out_dir="sequence_analysis", run_name=None
):

    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"run_{timestamp}"
    out_path = os.path.join(out_dir, f"{run_name}_comparison_table.txt")

    table = PrettyTable()
    table.field_names = ["Seq", "GC %", "MFE", "CAI", "Levenshtein", "Identity %"]

    decoded_nat = decode_sequence(natural_tensor)

    for i, s in enumerate(seqs_tensor):

        gc = compute_gc_content_vectorized(s, codon_gc_counts).item()
        mfe = compute_mfe_energy(s).item()
        cai = compute_cai(s).item()

        decoded_s = decode_sequence(s)
        lev = levenshtein_distance(decoded_s, decoded_nat)

        identity = compute_identity(decoded_s, decoded_nat)

        label = labels[i] if labels and i < len(labels) else f"Gen {i+1}"
        table.add_row(
            [label, f"{gc:.2f}", f"{mfe:.2f}", f"{cai:.2f}", lev, f"{identity:.2f}"]
        )

    # Natural sequence comparison
    gc = compute_gc_content_vectorized(natural_tensor, codon_gc_counts).item()
    mfe = compute_mfe_energy(natural_tensor).item()
    cai = compute_cai(natural_tensor).item()

    table.add_row(["Natural", f"{gc:.2f}", f"{mfe:.2f}", f"{cai:.2f}", 0, "100.00"])

    with open(out_path, "w") as f:
        f.write(table.get_string())

    print(f"[INFO] Sequence analysis saved to {out_path}")

