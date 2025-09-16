#!/usr/bin/env python3
"""
Evaluation Script for Short Sequence Specialist Conditional GFlowNet

This script evaluates a trained Conditional GFlowNet specialist on unseen protein sequences
to test its generalization capabilities beyond the training set.
"""

import sys
import os
import time
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

import torch
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from reward import compute_simple_reward
from DeepArchi import *
from comparison import analyze_sequence_properties
from enhanced_comparison import run_comprehensive_analysis
from utils import *
from plots import *
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor2
from evaluate import *

from gfn.gflownet import TBGFlowNet, SubTBGFlowNet
from gfn.estimators import (
    ConditionalDiscretePolicyEstimator,
    ConditionalScalarEstimator,
    ScalarEstimator,
)
from gfn.utils.modules import MLP
from ENN_ENH import MLP_ENN
from gfn.samplers import Sampler


class ShortSpecialistEvaluator:
    """
    Evaluator class for testing a trained Conditional GFlowNet specialist on unseen proteins.
    """

    def __init__(self, model_path: str, args, config):
        self.model_path = model_path
        self.args = args
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load model information
        self.model_info = self._load_model_info()

        # Create output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.output_dir = Path(f"outputs/short_specialist_evaluation/Evaluation_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        logging.info(f"Short Specialist Evaluator initialized")
        logging.info(f"Model path: {model_path}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Output directory: {self.output_dir}")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "evaluation.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _load_model_info(self) -> Dict[str, Any]:
        """Load model information from saved file"""
        try:
            with open(self.model_path, 'r') as f:
                model_info = json.load(f)
            return model_info
        except Exception as e:
            raise ValueError(f"Could not load model info from {self.model_path}: {e}")

    def load_test_sequences(self, test_type: str = "unseen_short",
                          num_sequences: int = 20) -> List[str]:
        """
        Load test sequences for evaluation.

        Args:
            test_type: Type of test sequences ("unseen_short", "medium", "long", "mixed")
            num_sequences: Number of sequences to load

        Returns:
            List of test protein sequences
        """
        logging.info(f"Loading {num_sequences} {test_type} test sequences")

        # Get training sequences to avoid overlap
        training_sequences = set(self.model_info['training_metadata']['sequences'])

        # Load from datasets
        dataset_files = [
            'training_dataset_very_short.csv',
            'training_dataset_short.csv',
            'training_dataset_medium.csv',
            'training_dataset_long.csv',
            'training_dataset_very_long.csv'
        ]

        all_sequences = []

        for dataset_file in dataset_files:
            try:
                df = pd.read_csv(dataset_file)
                if 'protein_sequence' in df.columns:
                    valid_sequences = df[
                        (df['protein_sequence'].str.len() >= 10) &
                        (df['protein_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY\*]+$'))
                    ]['protein_sequence'].tolist()

                    # Filter out training sequences
                    unseen_sequences = [seq for seq in valid_sequences if seq not in training_sequences]
                    all_sequences.extend(unseen_sequences)

            except FileNotFoundError:
                logging.warning(f"Dataset file {dataset_file} not found, skipping...")
                continue
            except Exception as e:
                logging.warning(f"Error loading {dataset_file}: {e}")
                continue

        # Remove duplicates and shuffle
        unique_sequences = list(set(all_sequences))
        np.random.shuffle(unique_sequences)

        # Filter by test type
        if test_type == "unseen_short":
            # Same length range as training
            min_length = self.model_info['training_metadata']['min_length']
            max_length = self.model_info['training_metadata']['max_length']
            filtered_sequences = [seq for seq in unique_sequences
                                if min_length <= len(seq) <= max_length]
        elif test_type == "medium":
            filtered_sequences = [seq for seq in unique_sequences if 50 <= len(seq) <= 100]
        elif test_type == "long":
            filtered_sequences = [seq for seq in unique_sequences if len(seq) > 100]
        elif test_type == "mixed":
            # Mix of different lengths
            filtered_sequences = unique_sequences
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        # Select the requested number of sequences
        selected_sequences = filtered_sequences[:num_sequences]

        logging.info(f"Selected {len(selected_sequences)} {test_type} sequences for evaluation")
        if selected_sequences:
            lengths = [len(seq) for seq in selected_sequences]
            logging.info(f"Length range: {min(lengths)}-{max(lengths)} AA")
            logging.info(f"Average length: {np.mean(lengths):.1f} AA")

        return selected_sequences

    def build_model_for_sequence(self, protein_seq: str) -> Tuple[Any, Any, Any]:
        """
        Build conditional GFlowNet model for a specific protein sequence.
        This recreates the model architecture from the training configuration.

        Args:
            protein_seq: Target protein sequence

        Returns:
            Tuple of (gflownet, pf_estimator, pb_estimator)
        """
        logging.info(f"Building model for sequence of length {len(protein_seq)}")

        # Create environment
        env = CodonDesignEnv(protein_seq=protein_seq, device=self.device)

        # Create preprocessor
        preprocessor = CodonSequencePreprocessor2(
            len(protein_seq) + 50,
            embedding_dim=self.model_info['model_architecture']['embedding_dim'],
            device=self.device
        )

        # Build conditional GFlowNet using saved architecture
        gflownet, pf_estimator, pb_estimator = self._build_subTB_gflownet(
            env, preprocessor, self.model_info['model_architecture']
        )

        # Move to device
        gflownet = gflownet.to(self.device)
        pf_estimator = pf_estimator.to(self.device)
        pb_estimator = pb_estimator.to(self.device)

        return gflownet, pf_estimator, pb_estimator

    def _build_subTB_gflownet(self, env, preprocessor, model_arch):
        """Build SubTB GFlowNet with conditional estimators using saved architecture"""

        # Forward policy estimator
        pf_module = MLP_ENN(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=model_arch['hidden_dim'],
            n_hidden_layers=model_arch['n_hidden_layers'],
            conditioning_dim=3  # For GC, MFE, CAI weights
        )
        pf_estimator = ConditionalDiscretePolicyEstimator(
            module=pf_module, preprocessor=preprocessor
        )

        # Backward policy estimator
        pb_module = MLP_ENN(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=model_arch['hidden_dim'],
            n_hidden_layers=model_arch['n_hidden_layers'],
            conditioning_dim=3
        )
        pb_estimator = ConditionalDiscretePolicyEstimator(
            module=pb_module, preprocessor=preprocessor
        )

        # LogZ estimator
        logZ_module = MLP_ENN(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=model_arch['hidden_dim'],
            n_hidden_layers=model_arch['n_hidden_layers'],
            conditioning_dim=3
        )
        logZ_estimator = ConditionalScalarEstimator(
            module=logZ_module, preprocessor=preprocessor
        )

        # Create SubTB GFlowNet
        gflownet = SubTBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            logZ=logZ_estimator,
            lamda=model_arch['subTB_lambda']
        )

        return gflownet, pf_estimator, pb_estimator

    def evaluate_on_sequence(self, protein_seq: str, sequence_idx: int,
                           num_samples: int = 100) -> Dict[str, Any]:
        """
        Evaluate the specialist model on a single protein sequence.

        Args:
            protein_seq: Protein sequence to evaluate on
            sequence_idx: Index of the sequence in test set
            num_samples: Number of sequences to generate for evaluation

        Returns:
            Evaluation results dictionary
        """
        logging.info(f"Evaluating on sequence {sequence_idx + 1}")
        logging.info(f"Sequence length: {len(protein_seq)} AA")

        # Build model
        gflownet, pf_estimator, pb_estimator = self.build_model_for_sequence(protein_seq)

        # Create environment and sampler
        env = CodonDesignEnv(protein_seq=protein_seq, device=self.device)
        sampler = Sampler(estimator=pf_estimator)

        # Generate sequences with different weight combinations
        weight_combinations = [
            [1.0, 0.0, 0.0],  # GC-focused
            [0.0, 1.0, 0.0],  # MFE-focused
            [0.0, 0.0, 1.0],  # CAI-focused
            [0.33, 0.33, 0.34],  # Balanced
        ]

        all_results = []

        for weight_idx, weights in enumerate(weight_combinations):
            logging.info(f"Testing weight combination {weight_idx + 1}: GC={weights[0]:.2f}, MFE={weights[1]:.2f}, CAI={weights[2]:.2f}")

            # Set weights
            env.set_weights(weights)

            # Build conditioning tensor
            weights_tensor = torch.tensor(weights, dtype=torch.get_default_dtype(), device=self.device)
            conditioning = weights_tensor.unsqueeze(0).expand(num_samples, *weights_tensor.shape)

            # Generate sequences
            generated_sequences = []
            rewards = []
            reward_components = []

            for sample_idx in tqdm(range(num_samples), desc=f"Weight {weight_idx + 1}"):
                # Sample trajectory
                trajectory = gflownet.sample_trajectories(
                    env,
                    n=1,
                    conditioning=conditioning[sample_idx:sample_idx+1],
                    save_logprobs=False,
                    save_estimator_outputs=False,
                    epsilon=self.args.epsilon,
                )

                # Get final state
                final_state = trajectory.terminating_states.tensor[0].to(self.device)

                # Compute reward
                reward, components = compute_simple_reward(
                    final_state, env.codon_gc_counts, env.weights
                )

                # Convert to sequence
                sequence = "".join([env.idx_to_codon[i.item()] for i in final_state])

                generated_sequences.append(sequence)
                rewards.append(reward)
                reward_components.append(components)

            # Analyze results for this weight combination
            weight_results = {
                'weights': weights,
                'generated_sequences': generated_sequences,
                'rewards': rewards,
                'reward_components': reward_components,
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'avg_gc': np.mean([c[0] for c in reward_components]),
                'avg_mfe': np.mean([c[1] for c in reward_components]),
                'avg_cai': np.mean([c[2] for c in reward_components]),
                'unique_sequences': len(set(generated_sequences)),
                'sequence_diversity': len(set(generated_sequences)) / num_samples
            }

            all_results.append(weight_results)

        # Compile overall results
        results = {
            'sequence': protein_seq,
            'sequence_idx': sequence_idx,
            'sequence_length': len(protein_seq),
            'num_samples_per_weight': num_samples,
            'weight_results': all_results,
            'overall_avg_reward': np.mean([wr['avg_reward'] for wr in all_results]),
            'overall_sequence_diversity': np.mean([wr['sequence_diversity'] for wr in all_results]),
            'total_unique_sequences': sum([wr['unique_sequences'] for wr in all_results])
        }

        logging.info(f"Evaluation completed for sequence {sequence_idx + 1}")
        logging.info(f"Overall average reward: {results['overall_avg_reward']:.4f}")
        logging.info(f"Overall sequence diversity: {results['overall_sequence_diversity']:.4f}")
        logging.info(f"Total unique sequences: {results['total_unique_sequences']}")

        return results

    def evaluate_specialist(self, test_type: str = "unseen_short",
                          num_sequences: int = 20,
                          num_samples: int = 100) -> Dict[str, Any]:
        """
        Evaluate the specialist model on a collection of test sequences.

        Args:
            test_type: Type of test sequences
            num_sequences: Number of sequences to evaluate on
            num_samples: Number of sequences to generate per weight combination

        Returns:
            Complete evaluation results
        """
        logging.info(f"Starting specialist evaluation on {test_type} sequences")

        # Load test sequences
        test_sequences = self.load_test_sequences(test_type, num_sequences)

        if not test_sequences:
            raise ValueError(f"No test sequences found for type: {test_type}")

        # Initialize Weights & Biases
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    **vars(self.args),
                    'test_type': test_type,
                    'num_test_sequences': len(test_sequences),
                    'num_samples_per_weight': num_samples,
                    'model_info': self.model_info
                },
                name=f"short_specialist_eval_{test_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )

        # Evaluate on each sequence
        all_results = []
        start_time = time.time()

        for i, protein_seq in enumerate(test_sequences):
            sequence_start_time = time.time()

            # Evaluate on this sequence
            results = self.evaluate_on_sequence(protein_seq, i, num_samples)
            all_results.append(results)

            sequence_time = time.time() - sequence_start_time
            logging.info(f"Sequence {i + 1} evaluation time: {sequence_time:.2f} seconds")

            # Save intermediate results
            self._save_intermediate_results(all_results, i)

        total_evaluation_time = time.time() - start_time

        # Compile final results
        final_results = {
            'model_info': self.model_info,
            'evaluation_metadata': {
                'test_type': test_type,
                'num_test_sequences': len(test_sequences),
                'num_samples_per_weight': num_samples,
                'test_sequences': test_sequences,
                'test_sequence_lengths': [len(seq) for seq in test_sequences]
            },
            'sequence_results': all_results,
            'total_evaluation_time': total_evaluation_time,
            'avg_evaluation_time_per_sequence': total_evaluation_time / len(test_sequences),
            'overall_statistics': self._compute_overall_statistics(all_results)
        }

        logging.info("Specialist evaluation completed!")
        logging.info(f"Total evaluation time: {total_evaluation_time:.2f} seconds")
        logging.info(f"Average time per sequence: {final_results['avg_evaluation_time_per_sequence']:.2f} seconds")

        # Save final results
        self._save_final_results(final_results)

        return final_results

    def _compute_overall_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute overall statistics from all sequence results"""

        all_rewards = []
        all_diversities = []
        all_unique_counts = []

        for result in results:
            all_rewards.append(result['overall_avg_reward'])
            all_diversities.append(result['overall_sequence_diversity'])
            all_unique_counts.append(result['total_unique_sequences'])

        return {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'avg_diversity': np.mean(all_diversities),
            'std_diversity': np.std(all_diversities),
            'avg_unique_sequences': np.mean(all_unique_counts),
            'total_unique_sequences': sum(all_unique_counts)
        }

    def _save_intermediate_results(self, results: List[Dict], sequence_idx: int):
        """Save intermediate evaluation results"""
        intermediate_file = self.output_dir / f"intermediate_evaluation_seq_{sequence_idx + 1}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            # Convert numpy arrays in weight_results
            if 'weight_results' in serializable_result:
                for wr in serializable_result['weight_results']:
                    if 'rewards' in wr:
                        wr['rewards'] = [float(x) for x in wr['rewards']]
                    if 'reward_components' in wr:
                        wr['reward_components'] = [[float(c) for c in comp] for comp in wr['reward_components']]
            serializable_results.append(serializable_result)

        with open(intermediate_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def _save_final_results(self, results: Dict[str, Any]):
        """Save final evaluation results"""

        # Save results as JSON
        results_file = self.output_dir / "evaluation_results.json"

        # Make results JSON serializable
        serializable_results = results.copy()

        # Convert numpy arrays to lists
        if 'overall_statistics' in serializable_results:
            stats = serializable_results['overall_statistics']
            for key, value in stats.items():
                if isinstance(value, np.ndarray):
                    stats[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    stats[key] = float(value)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Create summary report
        self._create_summary_report(results)

        logging.info(f"Results saved to {self.output_dir}")

    def _create_summary_report(self, results: Dict[str, Any]):
        """Create a human-readable summary report"""
        report_file = self.output_dir / "evaluation_summary.txt"

        with open(report_file, 'w') as f:
            f.write("Short Sequence Specialist Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Test Type: {results['evaluation_metadata']['test_type']}\n")
            f.write(f"Number of test sequences: {results['evaluation_metadata']['num_test_sequences']}\n")
            f.write(f"Samples per weight combination: {results['evaluation_metadata']['num_samples_per_weight']}\n")
            f.write(f"Total evaluation time: {results['total_evaluation_time']:.2f} seconds\n")
            f.write(f"Average time per sequence: {results['avg_evaluation_time_per_sequence']:.2f} seconds\n\n")

            stats = results['overall_statistics']
            f.write("Overall Performance Statistics:\n")
            f.write(f"  Average Reward: {stats['avg_reward']:.4f} ± {stats['std_reward']:.4f}\n")
            f.write(f"  Reward Range: [{stats['min_reward']:.4f}, {stats['max_reward']:.4f}]\n")
            f.write(f"  Average Diversity: {stats['avg_diversity']:.4f} ± {stats['std_diversity']:.4f}\n")
            f.write(f"  Average Unique Sequences per Test: {stats['avg_unique_sequences']:.1f}\n")
            f.write(f"  Total Unique Sequences Generated: {stats['total_unique_sequences']}\n\n")

            f.write("Model Training Information:\n")
            training_meta = self.model_info['training_metadata']
            f.write(f"  Training Sequences: {training_meta['num_sequences']}\n")
            f.write(f"  Training Length Range: {training_meta['min_length']}-{training_meta['max_length']} AA\n")
            f.write(f"  Average Training Length: {training_meta['avg_length']:.1f} AA\n\n")

            f.write("Test Sequences:\n")
            for i, seq in enumerate(results['evaluation_metadata']['test_sequences']):
                f.write(f"  {i + 1:2d}. {seq} (length: {len(seq)})\n")


def main():
    """Main function to run the short sequence specialist evaluation"""

    parser = argparse.ArgumentParser(description="Evaluate Short Sequence Specialist Conditional GFN")

    # Model and evaluation parameters
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model file")
    parser.add_argument("--test_type", type=str, default="unseen_short",
                       choices=["unseen_short", "medium", "long", "mixed"],
                       help="Type of test sequences")
    parser.add_argument("--num_sequences", type=int, default=20,
                       help="Number of test sequences")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples per weight combination")

    # Sampling parameters
    parser.add_argument("--epsilon", type=float, default=0.25, help="Epsilon for sampling")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="short-specialist-evaluation",
                       help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    # Create a simple config object
    class Config:
        def __init__(self):
            self.wandb_project = args.wandb_project
            self.run_name = args.run_name

    config = Config()

    # Initialize evaluator
    evaluator = ShortSpecialistEvaluator(args.model_path, args, config)

    # Run evaluation
    results = evaluator.evaluate_specialist(
        test_type=args.test_type,
        num_sequences=args.num_sequences,
        num_samples=args.num_samples
    )

    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {evaluator.output_dir}")
    print(f"Overall average reward: {results['overall_statistics']['avg_reward']:.4f}")
    print(f"Overall sequence diversity: {results['overall_statistics']['avg_diversity']:.4f}")
    print(f"Total unique sequences generated: {results['overall_statistics']['total_unique_sequences']}")


if __name__ == "__main__":
    main()
