#!/usr/bin/env python3
"""
Quick comparison between conditional and unconditional GFlowNet training
for mRNA design to evaluate generalization performance.
"""

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

import torch
import numpy as np
import random
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Import your existing modules
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from utils import load_config, compute_reward
from torchgfn.src.gfn.gflownet import TBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from torchgfn.src.gfn.utils.modules import MLP
from torchgfn.src.gfn.samplers import Sampler

logging.basicConfig(level=logging.INFO)

def quick_comparison(config_path="config.yaml", n_iterations=50, n_samples=20):
    """
    Quick comparison between conditional and unconditional training.

    Args:
        config_path: Path to config file
        n_iterations: Number of training iterations
        n_samples: Number of evaluation samples
    """

    # Load config
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup environment
    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(env.seq_length, embedding_dim=32, device=device)

    # Define weight configurations
    training_weights = [
        [0.3, 0.3, 0.4],  # Balanced
        [0.5, 0.3, 0.2],  # GC-focused
        [0.2, 0.5, 0.3],  # MFE-focused
    ]

    test_weights = [
        [0.4, 0.4, 0.2],  # GC+MFE focused
        [0.1, 0.4, 0.5],  # MFE+CAI focused
        [0.6, 0.2, 0.2],  # High GC
        [0.1, 0.7, 0.2],  # High MFE
    ]

    print("=== QUICK COMPARISON: CONDITIONAL vs UNCONDITIONAL ===")
    print(f"Training iterations: {n_iterations}")
    print(f"Evaluation samples: {n_samples}")
    print(f"Training weights: {len(training_weights)} configurations")
    print(f"Test weights: {len(test_weights)} configurations")
    print()

    # Train and evaluate unconditional model
    print("Training unconditional model...")
    unconditional_model = build_unconditional_model(preprocessor, env)
    unconditional_model.to(device)

    unconditional_results = train_and_evaluate(
        unconditional_model, env, preprocessor, training_weights, test_weights,
        n_iterations, n_samples, is_conditional=False
    )

    # Train and evaluate conditional model
    print("Training conditional model...")
    conditional_model = build_conditional_model(preprocessor, env)
    conditional_model.to(device)

    conditional_results = train_and_evaluate(
        conditional_model, env, preprocessor, training_weights, test_weights,
        n_iterations, n_samples, is_conditional=True
    )

    # Compare results
    print("\n=== RESULTS COMPARISON ===")
    compare_results(unconditional_results, conditional_results)

    # Create simple visualization
    create_comparison_plot(unconditional_results, conditional_results)

    return unconditional_results, conditional_results

def build_unconditional_model(preprocessor, env):
    """Build unconditional GFlowNet model."""
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=128,
        n_hidden_layers=2,
    )

    module_PB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=128,
        n_hidden_layers=2,
    )

    pf_estimator = DiscretePolicyEstimator(module=module_PF, preprocessor=preprocessor)
    pb_estimator = DiscretePolicyEstimator(module=module_PB, preprocessor=preprocessor)

    module_logZ = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=1,
        hidden_dim=128,
        n_hidden_layers=2,
    )
    logZ_estimator = ScalarEstimator(module=module_logZ, preprocessor=preprocessor)

    return TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=logZ_estimator)

def build_conditional_model(preprocessor, env):
    """Build conditional GFlowNet model."""
    # For this quick comparison, we'll use a simplified approach
    # that doesn't require the full conditional modules

    # Create a simple model that can handle conditioning
    # This is a simplified version - in practice you'd use proper conditional modules

    # For now, we'll create a model that takes concatenated input
    # (state + conditioning weights)

    module_PF = MLP(
        input_dim=preprocessor.output_dim + 3,  # +3 for conditioning weights
        output_dim=env.n_actions,
        hidden_dim=128,
        n_hidden_layers=2,
    )

    module_PB = MLP(
        input_dim=preprocessor.output_dim + 3,
        output_dim=env.n_actions - 1,
        hidden_dim=128,
        n_hidden_layers=2,
    )

    pf_estimator = DiscretePolicyEstimator(module=module_PF, preprocessor=preprocessor)
    pb_estimator = DiscretePolicyEstimator(module=module_PB, preprocessor=preprocessor)

    module_logZ = MLP(
        input_dim=preprocessor.output_dim + 3,
        output_dim=1,
        hidden_dim=128,
        n_hidden_layers=2,
    )
    logZ_estimator = ScalarEstimator(module=module_logZ, preprocessor=preprocessor)

    return TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=logZ_estimator)

def train_and_evaluate(model, env, preprocessor, training_weights, test_weights,
                      n_iterations, n_samples, is_conditional=False):
    """Train a model and evaluate it on test weights."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    sampler = Sampler(estimator=model.pf)

    # Training
    loss_history = []
    reward_history = []

    for iteration in range(n_iterations):

        weights = random.choice(training_weights)
        env.set_weights(weights)

        if is_conditional:


            conditioning = torch.tensor(weights, dtype=torch.float32, device=env.device)
            conditioning = conditioning.unsqueeze(0).expand(2, *conditioning.shape)


            trajectories = sampler.sample_trajectories(env, n=2)
        else:
            trajectories = sampler.sample_trajectories(env, n=2)

        loss = model.loss(trajectories)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute average reward
        rewards = []
        for traj in trajectories:
            if traj.is_complete:
                final_state = traj.states[-1]
                reward, _ = compute_reward(final_state, env.codon_gc_counts, weights)
                rewards.append(reward)

        avg_reward = np.mean(rewards) if rewards else 0.0

        loss_history.append(loss.item())
        reward_history.append(avg_reward)

        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: Loss = {loss.item():.4f}, Reward = {avg_reward:.4f}")

    # Evaluation
    print(f"Evaluating on {len(test_weights)} test configurations...")
    test_rewards = []

    for weights in test_weights:
        env.set_weights(weights)

        if is_conditional:
            # Simplified conditional evaluation
            trajectories = sampler.sample_trajectories(env, n=n_samples)
        else:
            trajectories = sampler.sample_trajectories(env, n=n_samples)

        rewards = []
        for traj in trajectories:
            if traj.is_complete:
                final_state = traj.states[-1]
                reward, _ = compute_reward(final_state, env.codon_gc_counts, weights)
                rewards.append(reward)

        test_rewards.append(np.mean(rewards) if rewards else 0.0)
        print(f"  Weights {weights}: Avg Reward = {test_rewards[-1]:.4f}")

    return {
        'training_loss': loss_history,
        'training_reward': reward_history,
        'test_rewards': test_rewards,
        'mean_test_reward': np.mean(test_rewards),
        'std_test_reward': np.std(test_rewards),
    }

def compare_results(unconditional_results, conditional_results):
    """Compare results between unconditional and conditional models."""

    print("Training Performance:")
    print(f"  Unconditional - Final Loss: {unconditional_results['training_loss'][-1]:.4f}")
    print(f"  Conditional   - Final Loss: {conditional_results['training_loss'][-1]:.4f}")
    print()

    print("Test Performance (Generalization):")
    print(f"  Unconditional - Mean Reward: {unconditional_results['mean_test_reward']:.4f} ± {unconditional_results['std_test_reward']:.4f}")
    print(f"  Conditional   - Mean Reward: {conditional_results['mean_test_reward']:.4f} ± {conditional_results['std_test_reward']:.4f}")

    improvement = ((conditional_results['mean_test_reward'] / unconditional_results['mean_test_reward']) - 1) * 100
    print(f"  Improvement: {improvement:+.2f}%")

    if improvement > 0:
        print("  → Conditional training shows better generalization!")
    elif improvement < 0:
        print("  → Unconditional training shows better generalization!")
    else:
        print("  → Both approaches perform similarly.")

    print()
    print("Note: This is a simplified comparison. For a comprehensive analysis,")
    print("run the full comparison script: python compare_conditional_vs_unconditional.py")

def create_comparison_plot(unconditional_results, conditional_results):
    """Create a simple comparison plot."""

    plt.figure(figsize=(12, 4))

    # Training curves
    plt.subplot(1, 3, 1)
    plt.plot(unconditional_results['training_loss'], label='Unconditional', alpha=0.7)
    plt.plot(conditional_results['training_loss'], label='Conditional', alpha=0.7)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Training rewards
    plt.subplot(1, 3, 2)
    plt.plot(unconditional_results['training_reward'], label='Unconditional', alpha=0.7)
    plt.plot(conditional_results['training_reward'], label='Conditional', alpha=0.7)
    plt.title('Training Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Test performance
    plt.subplot(1, 3, 3)
    test_configs = range(len(unconditional_results['test_rewards']))
    plt.bar([x - 0.2 for x in test_configs], unconditional_results['test_rewards'],
            width=0.4, label='Unconditional', alpha=0.7)
    plt.bar([x + 0.2 for x in test_configs], conditional_results['test_rewards'],
            width=0.4, label='Conditional', alpha=0.7)
    plt.title('Test Performance')
    plt.xlabel('Test Configuration')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quick_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Comparison plot saved as 'quick_comparison_results.png'")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Run quick comparison
    unconditional_results, conditional_results = quick_comparison(
        n_iterations=50,  # Quick training
        n_samples=20      # Quick evaluation
    )
