#!/usr/bin/env python3
"""
Simple script to run the three curriculum learning configurations for comparison.
"""

import subprocess
import sys
import os
from curriculum_configs import get_curriculum_configs

def run_single_experiment(config_name, config, n_iterations=20, eval_every=5, train_steps_per_task=100):
    """Run a single curriculum learning experiment"""

    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {config['name']}")
    print(f"{'='*80}")
    print(f"Description: {config['description']}")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")

    cmd = [
        sys.executable, "curriculum_main_complete.py",
        "--run_name", f"{config_name}_research",
        "--wandb_project", "mRNA_GFN_Curriculum_Research",
        "--n_iterations", str(n_iterations),
        "--eval_every", str(eval_every),
        "--train_steps_per_task", str(train_steps_per_task),
        "--curriculum_tasks", "[25,40]", "[45,60]", "[65,80]", "[85,120]", "[125,180]", "[185,250]",
        "--lpe", config['lpe'],
        "--acp", config['acp'],
        "--a2d", config['a2d'],
    ]


    if 'lpe_alpha' in config:
        cmd.extend(["--lpe_alpha", str(config['lpe_alpha'])])
    if 'lpe_K' in config:
        cmd.extend(["--lpe_K", str(config['lpe_K'])])
    if 'a2d_eps' in config:
        cmd.extend(["--a2d_eps", str(config['a2d_eps'])])
    if 'a2d_tau' in config:
        cmd.extend(["--a2d_tau", str(config['a2d_tau'])])
    if 'acp_MR_K' in config:
        cmd.extend(["--acp_MR_K", str(config['acp_MR_K'])])
    if 'acp_MR_power' in config:
        cmd.extend(["--acp_MR_power", str(config['acp_MR_power'])])
    if 'acp_MR_pot_prop' in config:
        cmd.extend(["--acp_MR_pot_prop", str(config['acp_MR_pot_prop'])])
    if 'acp_MR_att_pred' in config:
        cmd.extend(["--acp_MR_att_pred", str(config['acp_MR_att_pred'])])
    if 'acp_MR_att_succ' in config:
        cmd.extend(["--acp_MR_att_succ", str(config['acp_MR_att_succ'])])

    print(f"Command: {' '.join(cmd)}")

    try:
        print("Starting experiment...")
        result = subprocess.run(cmd, check=True)
        print(f"Experiment '{config_name}' completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment '{config_name}' failed with error: {e}")
        return False

def main():
    """Run all three curriculum learning experiments"""

    print("Curriculum Learning Research Experiments for mRNA Design")
    print("=" * 70)


    configs = get_curriculum_configs()

    n_iterations = 20
    eval_every = 5
    train_steps_per_task = 100

    print(f"Experiment parameters:")
    print(f"  - Iterations: {n_iterations}")
    print(f"  - Evaluation every: {eval_every} steps")
    print(f"  - Training steps per task: {train_steps_per_task}")
    print(f"  - Total training steps: {n_iterations * train_steps_per_task}")


    results = {}

    for config_name, config in configs.items():
        print(f"\n Starting {config['name']}...")
        success = run_single_experiment(config_name, config, n_iterations, eval_every, train_steps_per_task)
        results[config_name] = success

        if success:
            print(f"{config['name']} completed successfully!")
        else:
            print(f"{config['name']} failed!")

    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    for config_name, success in results.items():
        status = "SUCCESS" if success else " FAILED"
        print(f"{config_name}: {status}")

    successful_experiments = sum(results.values())
    total_experiments = len(results)

    print(f"\nOverall: {successful_experiments}/{total_experiments} experiments completed successfully")

    if successful_experiments == total_experiments:
        print("All experiments completed successfully!")
        print("\nNext steps:")
        print("1. Check Weights & Biases for detailed results")
        print("2. Analyze the curriculum learning effectiveness")
        print("3. Compare performance across configurations")
        print("4. Use results for your research paper")
    else:
        print("Some experiments failed. Check the error messages above.")

if __name__ == "__main__":
    main()
