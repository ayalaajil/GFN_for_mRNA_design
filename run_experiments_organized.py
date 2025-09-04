#!/usr/bin/env python3
"""
Organized experiment runner for conditional vs unconditional GFlowNet experiments
on different protein sizes (small, medium, large).

This script helps organize and run multiple experiments with proper output directory structure.
"""

import os
import subprocess
import argparse
from datetime import datetime

def run_experiment(experiment_type, protein_size, config_file, additional_args=None):
    """
    Run a single experiment with proper organization.

    Args:
        experiment_type: 'conditional' or 'unconditional'
        protein_size: 'small', 'medium', or 'large'
        config_file: path to config file
        additional_args: list of additional command line arguments
    """

        # Base command - use different scripts for conditional vs unconditional
    if experiment_type == "conditional":
        cmd = ["python", "main_conditional.py", "--config_path", config_file]
    else:
        cmd = ["python", "main.py", "--config_path", config_file]

    # Add additional arguments
    if additional_args:
        cmd.extend(additional_args)

    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_type}_{protein_size}_{timestamp}"
    cmd.extend(["--run_name", run_name])

    print(f"\n{'='*60}")
    print(f"Running {experiment_type.upper()} experiment on {protein_size.upper()} protein")
    print(f"Config: {config_file}")
    print(f"Run name: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n {experiment_type.upper()} {protein_size.upper()} experiment completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n {experiment_type.upper()} {protein_size.upper()} experiment failed with return code {e.returncode}")
        return False

def main():

    parser = argparse.ArgumentParser(description="Run organized GFlowNet experiments")
    parser.add_argument("--experiment_type", choices=["conditional", "unconditional", "both"],
                       default="both", help="Type of experiment to run")
    parser.add_argument("--protein_size", choices=["small", "medium", "large", "all"],
                       default="all", help="Protein size to test")
    parser.add_argument("--config_dir", default=".", help="Directory containing config files")
    parser.add_argument("--additional_args", nargs="*", help="Additional arguments to pass to main.py")

    args = parser.parse_args()

    config_files = {
        "small": "config_small.yaml",
        "medium": "config_medium.yaml",
        "large": "config_large.yaml"
    }

    experiment_types = ["conditional", "unconditional"] if args.experiment_type == "both" else [args.experiment_type]
    protein_sizes = ["small", "medium", "large"] if args.protein_size == "all" else [args.protein_size]

    results = {}

    print(f"Starting organized experiments...")
    print(f"Experiment types: {experiment_types}")
    print(f"Protein sizes: {protein_sizes}")
    print(f"Config directory: {args.config_dir}")

    # Run all combinations
    for exp_type in experiment_types:
        results[exp_type] = {}
        for protein_size in protein_sizes:
            config_file = os.path.join(args.config_dir, config_files[protein_size])

            if not os.path.exists(config_file):
                print(f"Config file {config_file} not found, skipping...")
                results[exp_type][protein_size] = False
                continue

            success = run_experiment(exp_type, protein_size, config_file, args.additional_args)
            results[exp_type][protein_size] = success


    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for exp_type in experiment_types:
        print(f"\n{exp_type.upper()} Experiments:")
        for protein_size in protein_sizes:
            status = "SUCCESS" if results[exp_type][protein_size] else "FAILED"
            print(f"  {protein_size.upper()}: {status}")

    print(f"\n Results are organized in: outputs/{{experiment_type}}/{{protein_size}}/{{run_name}}/")
    print(f"Each directory contains:")
    print(f"  - experiment_summary.txt (experiment details)")
    print(f"  - trained_gflownet_*.pth (model weights)")
    print(f"  - generated_sequences_*.txt (generated sequences)")
    print(f"  - *.png (plots and visualizations)")

if __name__ == "__main__":
    main()
