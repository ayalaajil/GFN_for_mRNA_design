import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import os
from datetime import datetime
import logging

from evaluate import evaluate


def get_extreme_weight_configs() -> Dict[str, List[float]]:

    return {
        # Single objective extremes
        "GC_only": [1.0, 0.0, 0.0],
        "MFE_only": [0.0, 1.0, 0.0],
        "CAI_only": [0.0, 0.0, 1.0],

        # Two objective combinations
        "GC_MFE": [0.5, 0.5, 0.0],
        "GC_CAI": [0.5, 0.0, 0.5],
        "MFE_CAI": [0.0, 0.5, 0.5],

        # Balanced configurations
        "Balanced": [0.33, 0.33, 0.34],
        "Slightly_GC": [0.5, 0.25, 0.25],
        "Slightly_MFE": [0.25, 0.5, 0.25],
        "Slightly_CAI": [0.25, 0.25, 0.5],

        # Extreme unbalanced
        "Very_GC": [0.8, 0.1, 0.1],
        "Very_MFE": [0.1, 0.8, 0.1],
        "Very_CAI": [0.1, 0.1, 0.8],
    }


def test_model_generalization(env, sampler, device, model_type="unconditional",
                            n_samples=50, weight_configs=None) -> Dict[str, Any]:
    """
    Test model generalization across different weight configurations.
    """
    if weight_configs is None:
        weight_configs = get_extreme_weight_configs()

    logging.info(f"Testing generalization with {len(weight_configs)} configurations")

    results = {}

    for config_name, weights in weight_configs.items():
        logging.info(f"Testing {config_name}: {weights}")

        try:
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

            if model_type == "conditional":
                samples, gc_list, mfe_list, cai_list = evaluate(
                    env, sampler, weight_tensor, n_samples, conditional=True
                )
            else:
                samples, gc_list, mfe_list, cai_list = evaluate(
                    env, sampler, weight_tensor, n_samples, conditional=False
                )

            stats = {
                'config_name': config_name,
                'weights': weights,
                'n_samples': len(samples),
                'n_unique': len(set(samples.keys())),
                'gc_mean': float(np.mean(gc_list)),
                'gc_std': float(np.std(gc_list)),
                'mfe_mean': float(np.mean(mfe_list)),
                'mfe_std': float(np.std(mfe_list)),
                'cai_mean': float(np.mean(cai_list)),
                'cai_std': float(np.std(cai_list)),
                'reward_mean': float(np.mean([r[0] for r in samples.values()])),
                'reward_std': float(np.std([r[0] for r in samples.values()])),
            }

            results[config_name] = {
                'samples': samples,
                'gc_list': gc_list,
                'mfe_list': mfe_list,
                'cai_list': cai_list,
                'stats': stats
            }

        except Exception as e:
            logging.error(f"Error testing {config_name}: {str(e)}")
            continue

    return results


def create_generalization_summary(results: Dict[str, Any], output_dir: str) -> str:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    summary_file = f"{output_dir}/generalization_summary_{timestamp}.txt"

    with open(summary_file, 'w') as f:
        f.write("Generalization Test Summary\n")
        f.write("=" * 40 + "\n\n")

        all_configs = list(results.keys())
        f.write(f"Configurations tested: {len(all_configs)}\n")
        f.write(f"Total samples: {sum(r['stats']['n_samples'] for r in results.values())}\n\n")

        f.write("Results by Configuration:\n")
        f.write("-" * 30 + "\n")

        for config_name, result in results.items():
            stats = result['stats']
            f.write(f"\n{config_name}:\n")
            f.write(f"  Weights: GC={stats['weights'][0]:.2f}, MFE={stats['weights'][1]:.2f}, CAI={stats['weights'][2]:.2f}\n")
            f.write(f"  Samples: {stats['n_samples']} total, {stats['n_unique']} unique\n")
            f.write(f"  GC: {stats['gc_mean']:.3f} ± {stats['gc_std']:.3f}\n")
            f.write(f"  MFE: {stats['mfe_mean']:.3f} ± {stats['mfe_std']:.3f}\n")
            f.write(f"  CAI: {stats['cai_mean']:.3f} ± {stats['cai_std']:.3f}\n")
            f.write(f"  Reward: {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}\n")

        # Summary statistics
        f.write(f"\nSummary Statistics:\n")
        f.write("-" * 20 + "\n")

        gc_means = [r['stats']['gc_mean'] for r in results.values()]
        mfe_means = [r['stats']['mfe_mean'] for r in results.values()]
        cai_means = [r['stats']['cai_mean'] for r in results.values()]
        reward_means = [r['stats']['reward_mean'] for r in results.values()]

        f.write(f"GC Content - Mean: {np.mean(gc_means):.3f}, Std: {np.std(gc_means):.3f}\n")
        f.write(f"MFE - Mean: {np.mean(mfe_means):.3f}, Std: {np.std(mfe_means):.3f}\n")
        f.write(f"CAI - Mean: {np.mean(cai_means):.3f}, Std: {np.std(cai_means):.3f}\n")
        f.write(f"Reward - Mean: {np.mean(reward_means):.3f}, Std: {np.std(reward_means):.3f}\n")

    logging.info(f"Generalization summary saved to: {summary_file}")
    return summary_file

def save_generalization_results(results: Dict[str, Any], output_dir: str) -> str:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_file = f"{output_dir}/generalization_results_{timestamp}.csv"

    # Prepare data for CSV
    rows = []
    for config_name, result in results.items():
        stats = result['stats']
        rows.append({
            'config_name': config_name,
            'weight_gc': stats['weights'][0],
            'weight_mfe': stats['weights'][1],
            'weight_cai': stats['weights'][2],
            'n_samples': stats['n_samples'],
            'n_unique': stats['n_unique'],
            'gc_mean': stats['gc_mean'],
            'gc_std': stats['gc_std'],
            'mfe_mean': stats['mfe_mean'],
            'mfe_std': stats['mfe_std'],
            'cai_mean': stats['cai_mean'],
            'cai_std': stats['cai_std'],
            'reward_mean': stats['reward_mean'],
            'reward_std': stats['reward_std'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)

    logging.info(f"Generalization results saved to: {csv_file}")
    return csv_file


def run_simple_generalization_tests(env, sampler, device, model_type="unconditional",
                                  n_samples=50, output_dir="generalization_results"):

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run tests
    results = test_model_generalization(env, sampler, device, model_type, n_samples)

    # Save results
    summary_file = create_generalization_summary(results, output_dir)
    csv_file = save_generalization_results(results, output_dir)

    logging.info(f"Generalization tests completed!")
    logging.info(f"Results saved to: {output_dir}")

    return results
