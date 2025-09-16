#!/usr/bin/env python3
"""
Fixed Range GFlowNet Training Script

This script trains a Conditional GFlowNet on a fixed range of protein sequences
(e.g., 30-40 AA) similar to curriculum_main.py but without curriculum learning.
Instead, it samples proteins from the specified range and trains the GFN.
"""

import sys
import os
import argparse
from datetime import datetime
import time
sys.path.insert(0, os.path.dirname(__file__))
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import scipy
from collections import deque
import wandb
from utils import set_seed
from reward import compute_simple_reward
from env import CodonDesignEnv
from main_conditional import build_subTB_gflownet, load_config
from preprocessor import CodonSequencePreprocessor2
from gfn.samplers import Sampler
from plots import plot_training_time_analysis, plot_task_distribution_analysis


class ProteinDatasetLoader:
    """Loads and manages protein sequences from datasets for fixed range training"""

    def __init__(self):
        self.protein_cache = {}
        self.length_index = {}
        self._load_datasets()

    def _load_datasets(self):
        """Load protein sequences from available training datasets"""
        import pandas as pd

        dataset_files = [
            'training_dataset_very_short.csv',
            'training_dataset_short.csv',
            'training_dataset_medium.csv',
            'training_dataset_long.csv',
            'training_dataset_very_long.csv',
        ]

        all_proteins = []

        for dataset_file in dataset_files:
            try:
                df = pd.read_csv(dataset_file)
                if 'protein_sequence' in df.columns and 'protein_length' in df.columns:

                    valid_proteins = df[
                        (df['protein_sequence'].str.len() >= 10) &
                        (df['protein_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY\*]+$'))
                    ]

                    for _, row in valid_proteins.iterrows():
                        protein_seq = row['protein_sequence']
                        length = len(protein_seq)

                        if length not in self.length_index:
                            self.length_index[length] = []
                        self.length_index[length].append(protein_seq)
                        all_proteins.append(protein_seq)

                    print(f"Loaded {len(valid_proteins)} proteins from {dataset_file}")

            except FileNotFoundError:
                print(f"Warning: {dataset_file} not found, skipping...")
                continue
            except Exception as e:
                print(f"Warning: Error loading {dataset_file}: {e}")
                continue

        self.protein_cache = {seq: len(seq) for seq in all_proteins}

        print(f"Total proteins loaded: {len(all_proteins)}")
        print(f"Length range: {min(self.length_index.keys()) if self.length_index else 0} - {max(self.length_index.keys()) if self.length_index else 0}")

    def _is_valid_protein_sequence(self, protein_seq):
        """Check if protein sequence is valid for the environment"""
        # Check if all amino acids are in the codon table
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")
        return all(aa in valid_aas for aa in protein_seq) and len(protein_seq) > 0

    def sample_protein_by_length_range(self, min_length, max_length):
        """Sample a real protein sequence within the specified length range (inclusive)"""

        # Find all proteins within the length range
        valid_proteins = []
        for length in range(min_length, max_length + 1):
            if length in self.length_index:
                proteins_at_length = [p for p in self.length_index[length] if self._is_valid_protein_sequence(p)]
                valid_proteins.extend(proteins_at_length)

        if valid_proteins:
            return np.random.choice(valid_proteins)

        # If no proteins in range, find the closest available length
        available_lengths = sorted(self.length_index.keys())
        if not available_lengths:
            raise ValueError("No protein data available in the dataset")

        # Find closest length to the range
        closest_length = None
        min_distance = float('inf')

        for length in available_lengths:
            if length < min_length:
                distance = min_length - length
            elif length > max_length:
                distance = length - max_length
            else:
                distance = 0  # Within range

            if distance < min_distance:
                min_distance = distance
                closest_length = length

        if closest_length is not None:
            proteins_at_closest = [p for p in self.length_index[closest_length] if self._is_valid_protein_sequence(p)]
            if proteins_at_closest:
                print(f"Warning: No proteins in range [{min_length}, {max_length}], using closest length {closest_length}")
                return np.random.choice(proteins_at_closest)

        raise ValueError(f"No valid proteins found in range [{min_length}, {max_length}] or nearby lengths")

    def get_available_lengths(self):
        """Get list of available protein lengths in the dataset"""
        return sorted(self.length_index.keys())

    def get_length_distribution(self):
        """Get distribution of protein lengths"""
        return {length: len(proteins) for length, proteins in self.length_index.items()}

    def get_proteins_in_range(self, min_length, max_length):
        """Get all proteins within the specified range"""
        valid_proteins = []
        for length in range(min_length, max_length + 1):
            if length in self.length_index:
                proteins_at_length = [p for p in self.length_index[length] if self._is_valid_protein_sequence(p)]
                valid_proteins.extend(proteins_at_length)
        return valid_proteins


def train_fixed_range(args, config, min_length, max_length):
    """Main training function for fixed range training"""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    protein_loader = ProteinDatasetLoader()

    # Get all proteins in the specified range
    proteins_in_range = protein_loader.get_proteins_in_range(min_length, max_length)

    if not proteins_in_range:
        raise ValueError(f"No proteins found in range [{min_length}, {max_length}]")

    print(f"Found {len(proteins_in_range)} proteins in range [{min_length}, {max_length}]")

    # Create preprocessor and model
    preprocessor = CodonSequencePreprocessor2(250, embedding_dim=args.embedding_dim, device=device)
    dummy_env = CodonDesignEnv("A*", device=device)
    gflownet, pf_estimator, pb_estimator = build_subTB_gflownet(dummy_env, preprocessor, args, lamda=args.subTB_lambda)

    gflownet = gflownet.to(device)
    pf_estimator = pf_estimator.to(device)
    pb_estimator = pb_estimator.to(device)

    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    sampler = Sampler(estimator=gflownet.pf)

    total_steps = args.n_iterations
    eval_every = args.eval_every
    train_steps_per_protein = args.train_steps_per_protein

    global_step = 0

    HISTORY_LEN = 200
    EMA_ALPHA = 0.05

    training_stats = {
        'total_loss': 0.0,
        'valid_steps': 0,
        'skipped_steps': 0,
        'current_lr': args.lr,
        'recent_losses': deque(maxlen=HISTORY_LEN),
        'global_ema': None,
        'protein_counts': {},
        'best_performance': 0.0
    }

    run_name = args.run_name or f"fixed_range_{min_length}_{max_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                **vars(config),
                'min_length': min_length,
                'max_length': max_length,
                'total_proteins_in_range': len(proteins_in_range),
                'total_steps': total_steps,
                'eval_every': eval_every,
                'train_steps_per_protein': train_steps_per_protein,
                'device': str(device),
                'protein_loader_stats': protein_loader.get_length_distribution()
            },
            tags=['fixed_range_training', 'mRNA_design', 'GFlowNet']
        )

    print("=" * 80)
    print("FIXED RANGE TRAINING STARTED")
    print("=" * 80)
    print(f"Protein length range: [{min_length}, {max_length}]")
    print(f"Total proteins in range: {len(proteins_in_range)}")
    print(f"Total iterations: {total_steps}")
    print(f"Evaluation every: {eval_every} steps")
    print(f"Training steps per protein: {train_steps_per_protein}")
    print(f"Device: {device}")
    print(f"Initial learning rate: {args.lr}")
    print("=" * 80)

    # Global time tracking
    time_start = time.time()
    training_phase_times = {
        'initialization': 0.0,
        'training': 0.0,
        'evaluation': 0.0,
        'total': 0.0
    }
    time_per_iteration = []
    protein_sequence_history = []

    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    while global_step < total_steps:

        iteration_start_time = time.time()

        # Sample a protein from the fixed range
        protein_seq = protein_loader.sample_protein_by_length_range(min_length, max_length)

        # Track protein usage
        if protein_seq not in training_stats['protein_counts']:
            training_stats['protein_counts'][protein_seq] = 0
        training_stats['protein_counts'][protein_seq] += 1

        task_env = CodonDesignEnv(protein_seq=protein_seq, device=device)

        # Record protein sequence history
        protein_sequence_history.append({
            'step': global_step,
            'protein_seq': protein_seq,
            'protein_length': len(protein_seq)
        })

        print(f"\n{'='*60}")
        print(f"STEP {global_step + 1}/{total_steps} | PROTEIN LENGTH: {len(protein_seq)}")
        print(f"{'='*60}")
        print(f"Protein sequence: {protein_seq}")
        print(f"Current learning rate: {training_stats['current_lr']:.3f}")

        if args.wandb_project:
            wandb.log({
                'training/protein_length': len(protein_seq),
                'training/protein_sequence': protein_seq,
            })

        protein_losses = []
        protein_start_time = time.time()

        for step_in_protein in (pbar := tqdm(range(train_steps_per_protein), dynamic_ncols=True)):

            try:
                print(f"Training GFlowNet on protein at step {step_in_protein + 1}/{train_steps_per_protein}....")

                # Sample weights
                weights = np.random.dirichlet([1, 1, 1])
                task_env.set_weights(weights)
                weights_tensor = torch.tensor(weights, dtype=torch.get_default_dtype(), device=device)
                conditioning = weights_tensor.unsqueeze(0).expand(args.batch_size, *weights_tensor.shape)

                # Sample trajectories with conditioning
                trajectories = gflownet.sample_trajectories(
                    task_env,
                    n=args.batch_size,
                    conditioning=conditioning,
                    save_logprobs=True,
                    save_estimator_outputs=True,
                    epsilon=args.epsilon,
                )

                optimizer.zero_grad()
                loss = gflownet.loss_from_trajectories(
                    task_env, trajectories, recalculate_all_logprobs=False
                )
                loss.backward()
                optimizer.step()

                scheduler.step(loss)
                protein_losses.append(loss.item())

                training_stats['total_loss'] += loss.item()
                training_stats['valid_steps'] += 1
                training_stats['current_lr'] = optimizer.param_groups[0]['lr']

                training_stats['recent_losses'].append(loss.item())

                prev = training_stats.get('global_ema')
                training_stats['global_ema'] = (
                    loss.item() if prev is None else EMA_ALPHA * loss.item() + (1 - EMA_ALPHA) * prev
                )

                print(f" Step {step_in_protein + 1}/{train_steps_per_protein}: Loss = {loss.item():.6f}")

            except Exception as e:
                print(f"Error during training step {global_step}: {e}")
                training_stats['skipped_steps'] += 1
                continue

            # Track protein completion time
            protein_end_time = time.time()
            protein_duration = protein_end_time - protein_start_time
            training_phase_times['training'] += protein_duration

            if args.wandb_project:
                wandb.log({
                    'training/global_ema_loss': training_stats['global_ema'],
                    'training/step': global_step,
                    'training/elapsed_time': time.time() - time_start,
                    'training/loss': loss.item(),
                    'training/learning_rate': optimizer.param_groups[0]['lr']
                })

        if global_step % eval_every == 0:
            eval_start_time = time.time()
            eval_log_data = {'global_step': global_step}

            print(f"\n EVALUATION at step {global_step+1}")
            print("-" * 40)

            # Evaluate on a sample protein from the range
            eval_protein = protein_loader.sample_protein_by_length_range(min_length, max_length)
            eval_env = CodonDesignEnv(protein_seq=eval_protein, device=device)

            performance = evaluate_protein_performance(gflownet, eval_env, sampler, n_samples=16)

            if performance > training_stats['best_performance']:
                training_stats['best_performance'] = performance

            eval_log_data['eval/performance'] = performance
            eval_log_data['eval/best_performance'] = training_stats['best_performance']

            print(f"Evaluation protein: {eval_protein} (length: {len(eval_protein)})")
            print(f"Performance = {performance:.4f} (Best: {training_stats['best_performance']:.4f})")

            avg_protein_loss = sum(protein_losses) / len(protein_losses) if protein_losses else 0.0
            eval_log_data['eval/protein_avg_loss'] = avg_protein_loss
            eval_log_data['training/learning_rate'] = optimizer.param_groups[0]['lr']

            # Track evaluation time
            eval_end_time = time.time()
            eval_duration = eval_end_time - eval_start_time
            training_phase_times['evaluation'] += eval_duration
            eval_log_data['eval/evaluation_duration'] = eval_duration

            if args.wandb_project:
                wandb.log(eval_log_data)

        if protein_losses:
            avg_loss = sum(protein_losses) / len(protein_losses)
            print(f"Protein training completed - Avg loss: {avg_loss:.6f}")
        else:
            print(f"Protein training completed - No valid steps")

        # Track iteration time
        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        time_per_iteration.append(iteration_duration)

        global_step += 1

    total_time = time.time() - time_start
    training_phase_times['total'] = total_time
    training_phase_times['initialization'] = time_start - time_start

    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Training phase breakdown:")
    print(f"  - Training: {training_phase_times['training']:.2f}s ({training_phase_times['training']/total_time*100:.1f}%)")
    print(f"  - Evaluation: {training_phase_times['evaluation']:.2f}s ({training_phase_times['evaluation']/total_time*100:.1f}%)")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED - FINAL SUMMARY")
    print("=" * 80)
    print(f"Total steps completed: {global_step}/{total_steps}")
    print(f"Protein length range: [{min_length}, {max_length}]")
    print(f"Total proteins used: {len(training_stats['protein_counts'])}")
    print(f"Best performance: {training_stats['best_performance']:.4f}")
    print(f"Average loss: {training_stats['total_loss']/training_stats['valid_steps']:.6f}")
    print("=" * 80)

    # Create output directory
    output_dir = f"outputs/fixed_range/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating training analysis plots...")
    plot_training_time_analysis(protein_sequence_history, training_phase_times, {}, time_per_iteration, output_dir)

    if args.wandb_project:
        final_metrics = {
            'final/total_steps': global_step,
            'final/success_rate': training_stats['valid_steps']/(training_stats['valid_steps'] + training_stats['skipped_steps']) if (training_stats['valid_steps'] + training_stats['skipped_steps']) > 0 else 0,
            'final/learning_rate': training_stats['current_lr'],
            'final/overall_avg_loss': training_stats['total_loss']/training_stats['valid_steps'] if training_stats['valid_steps'] > 0 else 0,
            'final/best_performance': training_stats['best_performance'],
            'final/total_training_time': total_time,
            'final/training_phase_time': training_phase_times['training'],
            'final/evaluation_phase_time': training_phase_times['evaluation'],
            'final/avg_time_per_iteration': np.mean(time_per_iteration) if time_per_iteration else 0,
            'final/std_time_per_iteration': np.std(time_per_iteration) if time_per_iteration else 0,
            'final/unique_proteins_used': len(training_stats['protein_counts'])
        }

        wandb.log({**final_metrics,
        'final/training_time_analysis': wandb.Image(f"{output_dir}/training_time_analysis.png"),
        })
        wandb.summary['best_performance'] = training_stats['best_performance']
        wandb.finish()

    print("Saving final fixed-range-trained model...")
    model_save_path = os.path.join(output_dir, "final_fixed_range_gflownet.pth")

    torch.save({
        'model_state_dict': gflownet.state_dict(),
        'args': args,
        'min_length': min_length,
        'max_length': max_length,
        'training_stats': training_stats,
        'protein_counts': training_stats['protein_counts']
    }, model_save_path)

    print(f"Model saved to {model_save_path}")


def evaluate_protein_performance(gflownet, env, sampler, n_samples=32):
    """Evaluate GFlowNet performance on a specific protein"""

    with torch.no_grad():

        total_reward = 0
        device = env.device

        weights = torch.tensor([0.3, 0.3, 0.4], device=device)
        conditioning = weights.unsqueeze(0).expand(n_samples, *weights.shape)

        trajectories = sampler.sample_trajectories(
            env,
            n=n_samples,
            conditioning=conditioning
        )

        final_states = trajectories.terminating_states.tensor

        rewards = []
        components_list = []
        valid_samples = 0

        for state in final_states:
            reward, components = compute_simple_reward(state, env.codon_gc_counts, weights)
            total_reward += reward
            rewards.append(reward)
            components_list.append(components)
            valid_samples += 1

        avg_reward = total_reward / valid_samples if valid_samples > 0 else 0.0
        avg_gc = sum(components[0] for components in components_list) / valid_samples if valid_samples > 0 else 0.0
        avg_mfe = sum(components[1] for components in components_list) / valid_samples if valid_samples > 0 else 0.0
        avg_cai = sum(components[2] for components in components_list) / valid_samples if valid_samples > 0 else 0.0

        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            std_reward = (sum((r - avg_reward)**2 for r in rewards) / len(rewards))**0.5
            print(f" Reward stats: avg={avg_reward:.4f}, min={min_reward:.4f}, max={max_reward:.4f}, std={std_reward:.4f}")
            print(f" GC stats: avg={avg_gc:.4f}")
            print(f" MFE stats: avg={avg_mfe:.4f}")
            print(f" CAI stats: avg={avg_cai:.4f}")
            print(f" Valid samples: {valid_samples}/{n_samples} ({valid_samples/n_samples*100:.1f}%)")

        return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument('--min_length', type=int, default=25, help='Minimum protein length')
    parser.add_argument('--max_length', type=int, default=180, help='Maximum protein length')

    parser.add_argument('--n_iterations', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--train_steps_per_protein', type=int, default=200)

    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=4)
    parser.add_argument('--subTB_lambda', type=float, default=0.9)
    parser.add_argument('--arch', type=str, default='Transformer')
    parser.add_argument('--tied', action='store_true')

    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--lr_logz', type=float, default=1e-1)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--subTB_weighting', type=str, default="geometric_within")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--top_n', type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epsilon', type=float, default=0.25)

    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--save_progress', action='store_true', default=True, help='Save training progress to files')

    parser.add_argument("--config_path", type=str, default="config.yaml")

    parser.add_argument('--wandb_project', type=str, default='mRNA_GFN_Experiments_FixedRange')

    args = parser.parse_args()
    config = load_config(args.config_path)
    set_seed(args.seed)

    print(f"Training on protein length range: [{args.min_length}, {args.max_length}]")
    train_fixed_range(args, config, args.min_length, args.max_length)


if __name__ == "__main__":
    main()
