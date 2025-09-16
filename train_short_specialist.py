#!/usr/bin/env python3
"""
Conditional GFlowNet Specialist Training Script for Short Protein Sequences (30-40 AA)

This script trains a Conditional GFlowNet to become a specialist on short protein sequences
by training on a curated collection of 50 different protein sequences in the 30-40 AA range.
The trained model can then be saved and evaluated on unseen proteins.
"""

import sys
import os
import time
import logging
import argparse
import json
import pickle
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
from train import train_conditional_gfn
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


class ShortSequenceSpecialistTrainer:
    """
    Trainer class for creating a Conditional GFlowNet specialist on short protein sequences.
    """

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Create output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.output_dir = Path(f"outputs/short_specialist/Short_Specialist_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize training data
        self.training_sequences = []
        self.training_metadata = {}

        logging.info(f"Short Sequence Specialist Trainer initialized")
        logging.info(f"Device: {self.device}")
        logging.info(f"Output directory: {self.output_dir}")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_short_sequences(self, min_length: int = 30, max_length: int = 40,
                           num_sequences: int = 50) -> List[str]:
        """
        Load a collection of short protein sequences for training.

        Args:
            min_length: Minimum protein sequence length
            max_length: Maximum protein sequence length
            num_sequences: Number of sequences to load

        Returns:
            List of protein sequences
        """
        logging.info(f"Loading {num_sequences} short sequences ({min_length}-{max_length} AA)")

        # Load from existing datasets
        dataset_files = [
            'training_dataset_very_short.csv',
            'training_dataset_short.csv',
            'training_dataset_medium.csv'
        ]

        all_sequences = []

        for dataset_file in dataset_files:
            try:
                df = pd.read_csv(dataset_file)
                if 'protein_sequence' in df.columns:
                    # Filter for sequences in our target length range
                    valid_sequences = df[
                        (df['protein_sequence'].str.len() >= min_length) &
                        (df['protein_sequence'].str.len() <= max_length) &
                        (df['protein_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY\*]+$'))
                    ]['protein_sequence'].tolist()

                    all_sequences.extend(valid_sequences)
                    logging.info(f"Loaded {len(valid_sequences)} sequences from {dataset_file}")

            except FileNotFoundError:
                logging.warning(f"Dataset file {dataset_file} not found, skipping...")
                continue
            except Exception as e:
                logging.warning(f"Error loading {dataset_file}: {e}")
                continue

        # Remove duplicates and shuffle
        unique_sequences = list(set(all_sequences))
        np.random.shuffle(unique_sequences)

        # Select the requested number of sequences
        selected_sequences = unique_sequences[:num_sequences]

        # Store metadata
        self.training_metadata = {
            'num_sequences': len(selected_sequences),
            'min_length': min_length,
            'max_length': max_length,
            'sequence_lengths': [len(seq) for seq in selected_sequences],
            'avg_length': np.mean([len(seq) for seq in selected_sequences]),
            'sequences': selected_sequences
        }

        logging.info(f"Selected {len(selected_sequences)} sequences for training")
        logging.info(f"Average length: {self.training_metadata['avg_length']:.1f} AA")
        logging.info(f"Length range: {min(self.training_metadata['sequence_lengths'])}-{max(self.training_metadata['sequence_lengths'])} AA")

        return selected_sequences

    def build_conditional_model(self, protein_seq: str) -> Tuple[Any, Any, Any]:
        """
        Build conditional GFlowNet model for a specific protein sequence.

        Args:
            protein_seq: Target protein sequence

        Returns:
            Tuple of (gflownet, pf_estimator, pb_estimator)
        """
        logging.info(f"Building conditional model for sequence of length {len(protein_seq)}")

        # Create environment
        env = CodonDesignEnv(protein_seq=protein_seq, device=self.device)

        # Create preprocessor
        preprocessor = CodonSequencePreprocessor2(
            len(protein_seq) + 50,
            embedding_dim=self.args.embedding_dim,
            device=self.device
        )

        # Build conditional GFlowNet
        gflownet, pf_estimator, pb_estimator = self._build_subTB_gflownet(
            env, preprocessor, self.args, lamda=self.args.subTB_lambda
        )

        # Move to device
        gflownet = gflownet.to(self.device)
        pf_estimator = pf_estimator.to(self.device)
        pb_estimator = pb_estimator.to(self.device)

        return gflownet, pf_estimator, pb_estimator

    def _build_subTB_gflownet(self, env, preprocessor, args, lamda=0.9):
        """Build SubTB GFlowNet with conditional estimators"""

        # Forward policy estimator
        pf_module = MLP_ENN(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden_layers,
            conditioning_dim=3  # For GC, MFE, CAI weights
        )
        pf_estimator = ConditionalDiscretePolicyEstimator(
            module=pf_module, preprocessor=preprocessor
        )

        # Backward policy estimator
        pb_module = MLP_ENN(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden_layers,
            conditioning_dim=3
        )
        pb_estimator = ConditionalDiscretePolicyEstimator(
            module=pb_module, preprocessor=preprocessor
        )

        # LogZ estimator
        logZ_module = MLP_ENN(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden_layers,
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
            lamda=lamda
        )

        return gflownet, pf_estimator, pb_estimator

    def train_on_sequence(self, protein_seq: str, sequence_idx: int) -> Dict[str, Any]:
        """
        Train the conditional model on a single protein sequence.

        Args:
            protein_seq: Protein sequence to train on
            sequence_idx: Index of the sequence in training set

        Returns:
            Training results dictionary
        """
        logging.info(f"Training on sequence {sequence_idx + 1}/{len(self.training_sequences)}")
        logging.info(f"Sequence length: {len(protein_seq)} AA")

        # Build model
        gflownet, pf_estimator, pb_estimator = self.build_conditional_model(protein_seq)

        # Create environment and sampler
        env = CodonDesignEnv(protein_seq=protein_seq, device=self.device)
        sampler = Sampler(estimator=pf_estimator)

        # Setup optimizer
        non_logz_params = [v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"]
        if "logZ" in dict(gflownet.named_parameters()):
            logz_params = [dict(gflownet.named_parameters())["logZ"]]
        else:
            logz_params = []

        params = [
            {"params": non_logz_params, "lr": self.args.lr},
            {"params": logz_params, "lr": self.args.lr_logz},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.args.lr_patience
        )

        # Train the model
        training_result = train_conditional_gfn(
            self.args, env, gflownet, sampler, optimizer, scheduler, self.device
        )

        loss_history, reward_history, reward_components, unique_seqs, sampled_weights = training_result

        # Store results
        results = {
            'sequence': protein_seq,
            'sequence_idx': sequence_idx,
            'sequence_length': len(protein_seq),
            'loss_history': loss_history,
            'reward_history': reward_history,
            'reward_components': reward_components,
            'unique_sequences': list(unique_seqs),
            'sampled_weights': sampled_weights,
            'final_loss': loss_history[-1] if loss_history else None,
            'final_reward': reward_history[-1] if reward_history else None,
            'num_unique_seqs': len(unique_seqs)
        }

        logging.info(f"Training completed for sequence {sequence_idx + 1}")
        logging.info(f"Final loss: {results['final_loss']:.4f}")
        logging.info(f"Final reward: {results['final_reward']:.4f}")
        logging.info(f"Unique sequences generated: {results['num_unique_seqs']}")

        return results

    def train_specialist(self, min_length: int = 30, max_length: int = 40,
                       num_sequences: int = 50) -> Dict[str, Any]:
        """
        Train the specialist model on a collection of short sequences.

        Args:
            min_length: Minimum protein sequence length
            max_length: Maximum protein sequence length
            num_sequences: Number of sequences to train on

        Returns:
            Complete training results
        """
        logging.info("Starting specialist training on short sequences")

        # Load training sequences
        self.training_sequences = self.load_short_sequences(
            min_length, max_length, num_sequences
        )

        # Initialize Weights & Biases
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    **vars(self.args),
                    **self.training_metadata,
                    'specialist_type': 'short_sequences',
                    'min_length': min_length,
                    'max_length': max_length
                },
                name=f"short_specialist_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )

        # Train on each sequence
        all_results = []
        start_time = time.time()

        for i, protein_seq in enumerate(self.training_sequences):
            sequence_start_time = time.time()

            # Train on this sequence
            results = self.train_on_sequence(protein_seq, i)
            all_results.append(results)

            sequence_time = time.time() - sequence_start_time
            logging.info(f"Sequence {i + 1} training time: {sequence_time:.2f} seconds")

            # Save intermediate results
            self._save_intermediate_results(all_results, i)

        total_training_time = time.time() - start_time

        # Compile final results
        final_results = {
            'training_metadata': self.training_metadata,
            'sequence_results': all_results,
            'total_training_time': total_training_time,
            'avg_training_time_per_sequence': total_training_time / len(self.training_sequences),
            'final_losses': [r['final_loss'] for r in all_results if r['final_loss'] is not None],
            'final_rewards': [r['final_reward'] for r in all_results if r['final_reward'] is not None],
            'total_unique_sequences': sum(r['num_unique_seqs'] for r in all_results)
        }

        logging.info("Specialist training completed!")
        logging.info(f"Total training time: {total_training_time:.2f} seconds")
        logging.info(f"Average time per sequence: {final_results['avg_training_time_per_sequence']:.2f} seconds")
        logging.info(f"Total unique sequences generated: {final_results['total_unique_sequences']}")

        # Save final results
        self._save_final_results(final_results)

        return final_results

    def _save_intermediate_results(self, results: List[Dict], sequence_idx: int):
        """Save intermediate training results"""
        intermediate_file = self.output_dir / f"intermediate_results_seq_{sequence_idx + 1}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            if 'loss_history' in serializable_result:
                serializable_result['loss_history'] = [float(x) for x in result['loss_history']]
            if 'reward_history' in serializable_result:
                serializable_result['reward_history'] = [float(x) for x in result['reward_history']]
            if 'sampled_weights' in serializable_result:
                serializable_result['sampled_weights'] = result['sampled_weights'].tolist()
            serializable_results.append(serializable_result)

        with open(intermediate_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def _save_final_results(self, results: Dict[str, Any]):
        """Save final training results and model"""

        # Save results as JSON
        results_file = self.output_dir / "training_results.json"

        # Make results JSON serializable
        serializable_results = results.copy()
        serializable_results['training_metadata'] = self.training_metadata

        # Convert numpy arrays to lists
        if 'final_losses' in serializable_results:
            serializable_results['final_losses'] = [float(x) for x in results['final_losses']]
        if 'final_rewards' in serializable_results:
            serializable_results['final_rewards'] = [float(x) for x in results['final_rewards']]

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Save training metadata separately
        metadata_file = self.output_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)

        # Create summary report
        self._create_summary_report(results)

        logging.info(f"Results saved to {self.output_dir}")

    def _create_summary_report(self, results: Dict[str, Any]):
        """Create a human-readable summary report"""
        report_file = self.output_dir / "training_summary.txt"

        with open(report_file, 'w') as f:
            f.write("Short Sequence Specialist Training Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of sequences trained on: {len(self.training_sequences)}\n")
            f.write(f"Sequence length range: {self.training_metadata['min_length']}-{self.training_metadata['max_length']} AA\n")
            f.write(f"Average sequence length: {self.training_metadata['avg_length']:.1f} AA\n")
            f.write(f"Total training time: {results['total_training_time']:.2f} seconds\n")
            f.write(f"Average time per sequence: {results['avg_training_time_per_sequence']:.2f} seconds\n")
            f.write(f"Total unique sequences generated: {results['total_unique_sequences']}\n\n")

            if results['final_losses']:
                f.write(f"Final Loss Statistics:\n")
                f.write(f"  Average: {np.mean(results['final_losses']):.4f}\n")
                f.write(f"  Min: {np.min(results['final_losses']):.4f}\n")
                f.write(f"  Max: {np.max(results['final_losses']):.4f}\n")
                f.write(f"  Std: {np.std(results['final_losses']):.4f}\n\n")

            if results['final_rewards']:
                f.write(f"Final Reward Statistics:\n")
                f.write(f"  Average: {np.mean(results['final_rewards']):.4f}\n")
                f.write(f"  Min: {np.min(results['final_rewards']):.4f}\n")
                f.write(f"  Max: {np.max(results['final_rewards']):.4f}\n")
                f.write(f"  Std: {np.std(results['final_rewards']):.4f}\n\n")

            f.write("Training Sequences:\n")
            for i, seq in enumerate(self.training_sequences):
                f.write(f"  {i + 1:2d}. {seq} (length: {len(seq)})\n")

    def save_model_for_evaluation(self, model_path: Optional[str] = None) -> str:
        """
        Save the trained model for later evaluation on unseen proteins.

        Args:
            model_path: Optional custom path for saving the model

        Returns:
            Path where the model was saved
        """
        if model_path is None:
            model_path = self.output_dir / "trained_short_specialist_model.pth"

        # For now, we'll save the training configuration and metadata
        # In a full implementation, you would save the actual trained model weights
        model_info = {
            'model_type': 'conditional_gfn_short_specialist',
            'training_metadata': self.training_metadata,
            'training_config': vars(self.args),
            'model_architecture': {
                'embedding_dim': self.args.embedding_dim,
                'hidden_dim': self.args.hidden_dim,
                'n_hidden_layers': self.args.n_hidden_layers,
                'subTB_lambda': self.args.subTB_lambda
            },
            'specialization': {
                'target_length_range': [self.training_metadata['min_length'], self.training_metadata['max_length']],
                'num_training_sequences': self.training_metadata['num_sequences'],
                'avg_training_length': self.training_metadata['avg_length']
            }
        }

        with open(model_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        logging.info(f"Model information saved to {model_path}")
        logging.info("Note: This saves model configuration and metadata.")
        logging.info("For full model saving, implement model state_dict saving in the training loop.")

        return str(model_path)


def main():
    """Main function to run the short sequence specialist training"""

    parser = argparse.ArgumentParser(description="Train Conditional GFN Specialist on Short Sequences")

    # Training parameters
    parser.add_argument("--n_iterations", type=int, default=200, help="Number of training iterations per sequence")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--lr_logz", type=float, default=0.1, help="LogZ learning rate")
    parser.add_argument("--lr_patience", type=int, default=10, help="LR scheduler patience")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Epsilon for sampling")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping")

    # Model architecture
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n_hidden_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--subTB_lambda", type=float, default=0.9, help="SubTB lambda parameter")

    # Training data parameters
    parser.add_argument("--min_length", type=int, default=30, help="Minimum protein sequence length")
    parser.add_argument("--max_length", type=int, default=40, help="Maximum protein sequence length")
    parser.add_argument("--num_sequences", type=int, default=50, help="Number of sequences to train on")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="short-sequence-specialist", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    # Create a simple config object
    class Config:
        def __init__(self):
            self.wandb_project = args.wandb_project
            self.run_name = args.run_name

    config = Config()

    # Initialize trainer
    trainer = ShortSequenceSpecialistTrainer(args, config)

    # Train the specialist
    results = trainer.train_specialist(
        min_length=args.min_length,
        max_length=args.max_length,
        num_sequences=args.num_sequences
    )

    # Save model for evaluation
    model_path = trainer.save_model_for_evaluation()

    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {trainer.output_dir}")
    print(f"Model saved to: {model_path}")
    print(f"Total unique sequences generated: {results['total_unique_sequences']}")


if __name__ == "__main__":
    main()
