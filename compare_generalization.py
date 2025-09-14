#!/usr/bin/env python3
"""
Simple script to compare generalization test results between different models.

This script helps you analyze and compare how different models perform
across extreme weight configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from pathlib import Path
import glob


def load_generalization_results(results_dir: str) -> pd.DataFrame:

    csv_files = glob.glob(f"{results_dir}/**/generalization_results_*.csv", recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"No generalization result CSV files found in {results_dir}")

    all_results = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract model info from path
        path_parts = Path(csv_file).parts
        model_name = "unknown"
        for part in path_parts:
            if "conditional" in part.lower():
                model_name = "conditional"
                break
            elif "unconditional" in part.lower():
                model_name = "unconditional"
                break

        df['model_type'] = model_name
        df['model_path'] = str(Path(csv_file).parent)
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def create_comparison_plots(df: pd.DataFrame, output_dir: str = "comparison_results"):

    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Overall performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Model Generalization Comparison", fontsize=16)

    # Average metrics by model
    model_metrics = df.groupby('model_type').agg({
        'gc_mean': 'mean',
        'mfe_mean': 'mean',
        'cai_mean': 'mean',
        'reward_mean': 'mean'
    }).reset_index()

    # GC Content comparison
    axes[0, 0].bar(model_metrics['model_type'], model_metrics['gc_mean'], alpha=0.7)
    axes[0, 0].set_title("Average GC Content")
    axes[0, 0].set_ylabel("GC Content")

    # MFE comparison
    axes[0, 1].bar(model_metrics['model_type'], model_metrics['mfe_mean'], alpha=0.7)
    axes[0, 1].set_title("Average MFE")
    axes[0, 1].set_ylabel("MFE")

    # CAI comparison
    axes[1, 0].bar(model_metrics['model_type'], model_metrics['cai_mean'], alpha=0.7)
    axes[1, 0].set_title("Average CAI")
    axes[1, 0].set_ylabel("CAI")

    # Reward comparison
    axes[1, 1].bar(model_metrics['model_type'], model_metrics['reward_mean'], alpha=0.7)
    axes[1, 1].set_title("Average Reward")
    axes[1, 1].set_ylabel("Reward")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance by configuration
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("Performance by Weight Configuration", fontsize=16)

    # GC Content by configuration
    pivot_gc = df.pivot_table(values='gc_mean', index='config_name', columns='model_type', aggfunc='mean')
    sns.heatmap(pivot_gc, ax=axes[0, 0], annot=True, cmap='viridis', fmt='.3f')
    axes[0, 0].set_title("GC Content by Configuration")

    # MFE by configuration
    pivot_mfe = df.pivot_table(values='mfe_mean', index='config_name', columns='model_type', aggfunc='mean')
    sns.heatmap(pivot_mfe, ax=axes[0, 1], annot=True, cmap='viridis', fmt='.3f')
    axes[0, 1].set_title("MFE by Configuration")

    # CAI by configuration
    pivot_cai = df.pivot_table(values='cai_mean', index='config_name', columns='model_type', aggfunc='mean')
    sns.heatmap(pivot_cai, ax=axes[1, 0], annot=True, cmap='viridis', fmt='.3f')
    axes[1, 0].set_title("CAI by Configuration")

    # Reward by configuration
    pivot_reward = df.pivot_table(values='reward_mean', index='config_name', columns='model_type', aggfunc='mean')
    sns.heatmap(pivot_reward, ax=axes[1, 1], annot=True, cmap='viridis', fmt='.3f')
    axes[1, 1].set_title("Reward by Configuration")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/configuration_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Scatter plots for detailed comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Detailed Model Comparison", fontsize=16)

    # GC vs MFE
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]
        axes[0, 0].scatter(model_data['gc_mean'], model_data['mfe_mean'],
                          label=model_type, alpha=0.7, s=100)
    axes[0, 0].set_xlabel("GC Content")
    axes[0, 0].set_ylabel("MFE")
    axes[0, 0].set_title("GC vs MFE")
    axes[0, 0].legend()

    # GC vs CAI
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]
        axes[0, 1].scatter(model_data['gc_mean'], model_data['cai_mean'],
                          label=model_type, alpha=0.7, s=100)
    axes[0, 1].set_xlabel("GC Content")
    axes[0, 1].set_ylabel("CAI")
    axes[0, 1].set_title("GC vs CAI")
    axes[0, 1].legend()

    # MFE vs CAI
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]
        axes[1, 0].scatter(model_data['mfe_mean'], model_data['cai_mean'],
                          label=model_type, alpha=0.7, s=100)
    axes[1, 0].set_xlabel("MFE")
    axes[1, 0].set_ylabel("CAI")
    axes[1, 0].set_title("MFE vs CAI")
    axes[1, 0].legend()

    # Reward vs Weight Balance
    df['weight_balance'] = df[['weight_gc', 'weight_mfe', 'weight_cai']].std(axis=1)
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]
        axes[1, 1].scatter(model_data['weight_balance'], model_data['reward_mean'],
                          label=model_type, alpha=0.7, s=100)
    axes[1, 1].set_xlabel("Weight Balance (std)")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].set_title("Reward vs Weight Balance")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plots saved to: {output_dir}")


def create_summary_table(df: pd.DataFrame, output_dir: str = "comparison_results"):

    os.makedirs(output_dir, exist_ok=True)

    # Create summary statistics
    summary = df.groupby('model_type').agg({
        'gc_mean': ['mean', 'std'],
        'mfe_mean': ['mean', 'std'],
        'cai_mean': ['mean', 'std'],
        'reward_mean': ['mean', 'std'],
        'n_samples': 'sum',
        'config_name': 'count'
    }).round(3)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    # Save summary
    summary_file = f"{output_dir}/model_comparison_summary.csv"
    summary.to_csv(summary_file, index=False)

    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare generalization test results between models")
    parser.add_argument("--results_dir", type=str, default="outputs",
                       help="Directory containing generalization results")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                       help="Output directory for comparison results")

    args = parser.parse_args()

    print("Loading generalization results...")
    try:
        df = load_generalization_results(args.results_dir)
        print(f"Loaded results for {len(df)} configurations across {df['model_type'].nunique()} models")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Creating comparison plots...")
    create_comparison_plots(df, args.output_dir)

    print("Creating summary table...")
    create_summary_table(df, args.output_dir)

    print(f"\nComparison complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
