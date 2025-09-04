#!/usr/bin/env python3
"""
Comprehensive comparison between conditional and unconditional GFlowNet training
for mRNA design to evaluate generalization performance.
"""

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import argparse
import wandb
from typing import Dict, List, Tuple, Any
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import your existing modules
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import *
from evaluate import evaluate, evaluate_conditional
from utils import *
from plots import *
from comparison import analyze_sequence_properties
from torchgfn.src.gfn.gflownet import TBGFlowNet, SubTBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from torchgfn.src.gfn.utils.modules import MLP
from torchgfn.src.gfn.samplers import Sampler

from gfn.estimators import ConditionalDiscretePolicyEstimator, ConditionalScalarEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class GeneralizationEvaluator:
    """Evaluates generalization performance of conditional vs unconditional models."""

    def __init__(self, config_path: str, device: str = "cuda"):

        self.config = load_config(config_path)
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.results = {}

        # Define training weight configurations
        self.training_weights = [
            [0.3, 0.3, 0.4],  # Balanced
            [0.5, 0.3, 0.2],  # GC-focused
            [0.2, 0.5, 0.3],  # MFE-focused
            [0.2, 0.3, 0.5],  # CAI-focused
        ]

        # Define test weight configurations (unseen during training)
        self.test_weights = [
            [0.4, 0.4, 0.2],  # GC+MFE focused
            [0.1, 0.4, 0.5],  # MFE+CAI focused
            [0.6, 0.2, 0.2],  # High GC
            [0.1, 0.7, 0.2],  # High MFE
            [0.1, 0.2, 0.7],  # High CAI
            [0.33, 0.33, 0.34],  # Nearly equal
            [0.7, 0.1, 0.2],  # Very high GC
            [0.1, 0.1, 0.8],  # Very high CAI
        ]

        # Extreme test cases for generalization
        self.extreme_test_weights = [
            [0.9, 0.05, 0.05],  # Extreme GC
            [0.05, 0.9, 0.05],  # Extreme MFE
            [0.05, 0.05, 0.9],  # Extreme CAI
            [0.0, 0.5, 0.5],    # Zero GC
            [0.5, 0.0, 0.5],    # Zero MFE
            [0.5, 0.5, 0.0],    # Zero CAI
        ]

    def setup_environment(self):
        """Setup the environment and preprocessor."""

        logging.info("Setting up environment...")

        self.env = CodonDesignEnv(protein_seq=self.config.protein_seq, device=self.device)
        self.preprocessor = CodonSequencePreprocessor(
            self.env.seq_length, embedding_dim=32, device=self.device
        )
        logging.info(f"Environment setup complete. Sequence length: {len(self.config.protein_seq)}")

    def build_unconditional_model(self, args):
        """Build unconditional GFlowNet model."""

        logging.info("Building unconditional model...")

        module_PF = MLP(
        input_dim=self.preprocessor.output_dim,
        output_dim=self.env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        )

        module_PB = MLP(
            input_dim=self.preprocessor.output_dim,
            output_dim=self.env.n_actions - 1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=module_PF.trunk if args.tied else None,
        )

        pf_estimator = DiscretePolicyEstimator(
            module_PF, self.env.n_actions, preprocessor=self.preprocessor, is_backward=False
        )

        pb_estimator = DiscretePolicyEstimator(
            module_PB, self.env.n_actions, preprocessor=self.preprocessor, is_backward=True
        )

        module_logF = MLP(
            input_dim=self.preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=module_PF.trunk if args.tied else None,
        )

        logF_estimator = ScalarEstimator(module=module_logF, preprocessor=self.preprocessor)

        gflownet = SubTBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            logF=logF_estimator,
            weighting=args.subTB_weighting,
            lamda=args.subTB_lambda,
        )
        return gflownet


    def build_conditional_model(self, args):
        """Build conditional GFlowNet model."""

        logging.info("Building conditional model...")

        CONCAT_SIZE = 16
        module_PF = MLP(
            input_dim=self.preprocessor.output_dim,
            output_dim=CONCAT_SIZE,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )
        module_PB = MLP(
            input_dim=self.preprocessor.output_dim,
            output_dim=CONCAT_SIZE,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=module_PF.trunk if args.tied else None,
        )
        module_cond = MLP(
            input_dim=3,
            output_dim=CONCAT_SIZE,
            hidden_dim=args.hidden_dim,
        )
        module_final_PF = MLP(
            input_dim=CONCAT_SIZE * 2,
            output_dim=self.env.n_actions,
        )
        module_final_PB = MLP(
            input_dim=CONCAT_SIZE * 2,
            output_dim=self.env.n_actions - 1,
            trunk=module_final_PF.trunk,
        )
        pf_estimator = ConditionalDiscretePolicyEstimator(
            module_PF,
            module_cond,
            module_final_PF,
            self.env.n_actions,
            preprocessor=self.preprocessor,
            is_backward=False,
        )
        pb_estimator = ConditionalDiscretePolicyEstimator(
            module_PB,
            module_cond,
            module_final_PB,
            self.env.n_actions,
            preprocessor=self.preprocessor,
            is_backward=True,
        )
        module_state_logF = MLP(
            input_dim=self.preprocessor.output_dim,
            output_dim=CONCAT_SIZE,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=1,
        )
        module_conditioning_logF = MLP(
            input_dim=3,
            output_dim=CONCAT_SIZE,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=1,
        )
        module_final_logF = MLP(
            input_dim=CONCAT_SIZE * 2,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=1,
        )
        logF_estimator = ConditionalScalarEstimator(
            module_state_logF,
            module_conditioning_logF,
            module_final_logF,
            preprocessor=self.preprocessor,
        )
        gflownet = SubTBGFlowNet(
            logF=logF_estimator,
            pf=pf_estimator,
            pb=pb_estimator,
            lamda=args.subTB_lambda,
        )
        return gflownet


    def train_model(self, args, gflownet, weights_list: List[List[float]], is_conditional: bool = False):
        """Train a model on multiple weight configurations."""

        logging.info(f"Training {'conditional' if is_conditional else 'unconditional'} model...")

        sampler = Sampler(estimator=gflownet.pf)
        gflownet = gflownet.to(self.env.device)

        non_logz_params = [
            v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
        ]
        if "logZ" in dict(gflownet.named_parameters()):
            logz_params = [dict(gflownet.named_parameters())["logZ"]]
        else:
            logz_params = []

        params = [
            {"params": non_logz_params, "lr": args.lr},
            {"params": logz_params, "lr": args.lr_logz},
        ]
        optimizer = torch.optim.Adam(params)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)

        loss_history = []
        reward_history = []
        reward_components_history = []
        unique_sequences = set()

        weights = np.array(random.choice(weights_list))  # fix the weights for the unconditional training

        for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):

            iter_start_time = time.time()

            if is_conditional:

                # Sample a random weight configuration per iteration
                weights = np.random.dirichlet([1, 1, 1])
                self.env.set_weights(weights.tolist())

                conditioning = (torch.tensor(weights, dtype=torch.get_default_dtype(), device=self.env.device))
                conditioning = conditioning.unsqueeze(0).expand(args.batch_size, *conditioning.shape)

                trajectories = gflownet.sample_trajectories(
                    self.env,
                    n=args.batch_size,
                    conditioning=conditioning,
                    save_logprobs=True,
                    save_estimator_outputs=True,
                    epsilon=args.epsilon,
                )

            else:

                self.env.set_weights(weights.tolist())

                trajectories = sampler.sample_trajectories(
                    self.env,
                    n=args.batch_size,
                    save_logprobs=True,
                    epsilon=args.epsilon
                )

            optimizer.zero_grad()
            loss = gflownet.loss_from_trajectories(
                self.env, trajectories, recalculate_all_logprobs=False
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            iter_time = time.time() - iter_start_time

            final_states = trajectories.terminating_states.tensor.to(self.device)
            rewards, components = [], []

            for state in final_states:

                state = state.to(self.device)
                r, c = compute_reward(state, self.env.codon_gc_counts, self.env.weights)
                rewards.append(r)
                seq = "".join([self.env.idx_to_codon[i.item()] for i in state])
                unique_sequences.add(seq)

                rewards.append(r)
                components.append(c)

            avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0

            reward_history.append(avg_reward)
            reward_components_history.extend(components)
            components_tensor = torch.tensor(components)
            avg_gc, avg_mfe, avg_cai = components_tensor.mean(dim=0).tolist()


            loss_history.append(float(loss.item()))

            if (it % max(1, args.n_iterations // 10)) == 0:
                logging.info(f"Iter {it}: loss={loss.item():.4f} avg_reward={avg_reward:.4f}")

            wandb.log(
            {
                "iteration": it,
                "loss": loss.item(),
                "avg_reward": avg_reward,
                "avg_gc": avg_gc,
                "avg_mfe": avg_mfe,
                "avg_cai": avg_cai,
                "w_gc": self.env.weights[0],
                "w_mfe": self.env.weights[1],
                "w_cai": self.env.weights[2],
                "iter_time": iter_time,
            }
             )

        return loss_history, reward_history, unique_sequences


    def evaluate_model(self, gflownet, weights_list: List[List[float]],
                      n_samples: int = 50, is_conditional: bool = False) -> Dict:
        """Evaluate a trained model on multiple weight configurations."""
        logging.info(f"Evaluating {'conditional' if is_conditional else 'unconditional'} model...")

        sampler = Sampler(estimator=gflownet.pf)
        results = {
            'weights': [],
            'mean_rewards': [],
            'std_rewards': [],
            'mean_gc': [],
            'mean_mfe': [],
            'mean_cai': [],
            'pareto_efficiency': [],
            'diversity': [],
        }

        for weights in weights_list:
            
            self.env.set_weights(weights)

            if is_conditional:
                samples, gc_list, mfe_list, cai_list = evaluate_conditional(
                    self.env, sampler, weights, n_samples
                )
            else:
                samples, gc_list, mfe_list, cai_list = evaluate(
                    self.env, sampler, weights, n_samples
                )

            # Compute metrics
            rewards = [reward for reward, _ in samples.values()]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            mean_gc = np.mean(gc_list)
            mean_mfe = np.mean(mfe_list)
            mean_cai = np.mean(cai_list)

            # Pareto efficiency
            objectives = np.column_stack([gc_list, mfe_list, cai_list])
            pareto_mask = self.is_pareto_efficient_3d(objectives)
            pareto_efficiency = np.mean(pareto_mask)

            # Diversity (using edit distance)
            sequences = list(samples.keys())
            diversity = self.compute_sequence_diversity(sequences)

            # Store results
            results['weights'].append(weights)
            results['mean_rewards'].append(mean_reward)
            results['std_rewards'].append(std_reward)
            results['mean_gc'].append(mean_gc)
            results['mean_mfe'].append(mean_mfe)
            results['mean_cai'].append(mean_cai)
            results['pareto_efficiency'].append(pareto_efficiency)
            results['diversity'].append(diversity)

            logging.info(f"Weights {weights}: Reward={mean_reward:.4f}±{std_reward:.4f}, "
                        f"GC={mean_gc:.3f}, MFE={mean_mfe:.3f}, CAI={mean_cai:.3f}, "
                        f"Pareto={pareto_efficiency:.3f}, Diversity={diversity:.3f}")

        return results

    def is_pareto_efficient_3d(self, costs):
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

    def compute_sequence_diversity(self, sequences: List[str]) -> float:
        """Compute average edit distance between sequences."""
        if len(sequences) < 2:
            return 0.0

        from Levenshtein import distance as levenshtein_distance

        total_distance = 0
        count = 0

        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                total_distance += levenshtein_distance(sequences[i], sequences[j])
                count += 1

        return total_distance / count if count > 0 else 0.0

    def run_comparison(self, args, n_iterations: int = 100, n_samples: int = 50,
                      hidden_dim: int = 256, n_hidden: int = 2):
        """Run the complete comparison between conditional and unconditional models."""
        logging.info("Starting comprehensive comparison...")

        self.setup_environment()

        # Train and evaluate unconditional model
        logging.info("=== UNCONDITIONAL MODEL ===")
        unconditional_model = self.build_unconditional_model(args)
        unconditional_model.to(self.device)

        # Train on training weights
        unconditional_loss, unconditional_reward, unconditional_unique_sequences = self.train_model(args,
            unconditional_model, self.training_weights)

        # Evaluate on test weights
        unconditional_test_results = self.evaluate_model(
            unconditional_model, self.test_weights, n_samples, is_conditional=False
        )

        # Evaluate on extreme test weights
        unconditional_extreme_results = self.evaluate_model(
            unconditional_model, self.extreme_test_weights, n_samples, is_conditional=False
        )

        # Train and evaluate conditional model
        logging.info("=== CONDITIONAL MODEL ===")
        conditional_model = self.build_conditional_model(args)
        conditional_model.to(self.device)

        # Train on training weights
        conditional_loss, conditional_reward, conditional_unique_sequences = self.train_model(args,
            conditional_model, self.training_weights, is_conditional=True
        )

        # Evaluate on test weights
        conditional_test_results = self.evaluate_model(
            conditional_model, self.test_weights, n_samples, is_conditional=True
        )

        # Evaluate on extreme test weights
        conditional_extreme_results = self.evaluate_model(
            conditional_model, self.extreme_test_weights, n_samples, is_conditional=True
        )

        self.results = {
            'unconditional': {
                'training_loss': unconditional_loss,
                'training_reward': unconditional_reward,
                'test_results': unconditional_test_results,
                'extreme_results': unconditional_extreme_results,
            },
            'conditional': {
                'training_loss': conditional_loss,
                'training_reward': conditional_reward,
                'test_results': conditional_test_results,
                'extreme_results': conditional_extreme_results,
            }
        }

        return self.results

    def analyze_generalization(self) -> Dict[str, Any]:
        """Analyze generalization performance."""
        logging.info("Analyzing generalization performance...")

        analysis = {}

        # Compare test performance
        unc_test = self.results['unconditional']['test_results']
        cond_test = self.results['conditional']['test_results']

        # Statistical tests
        analysis['reward_comparison'] = {
            'unconditional_mean': np.mean(unc_test['mean_rewards']),
            'conditional_mean': np.mean(cond_test['mean_rewards']),
            'unconditional_std': np.std(unc_test['mean_rewards']),
            'conditional_std': np.std(cond_test['mean_rewards']),
            't_statistic': stats.ttest_ind(unc_test['mean_rewards'], cond_test['mean_rewards'])[0],
            'p_value': stats.ttest_ind(unc_test['mean_rewards'], cond_test['mean_rewards'])[1],
        }

        analysis['pareto_comparison'] = {
            'unconditional_mean': np.mean(unc_test['pareto_efficiency']),
            'conditional_mean': np.mean(cond_test['pareto_efficiency']),
            't_statistic': stats.ttest_ind(unc_test['pareto_efficiency'], cond_test['pareto_efficiency'])[0],
            'p_value': stats.ttest_ind(unc_test['pareto_efficiency'], cond_test['pareto_efficiency'])[1],
        }

        analysis['diversity_comparison'] = {
            'unconditional_mean': np.mean(unc_test['diversity']),
            'conditional_mean': np.mean(cond_test['diversity']),
            't_statistic': stats.ttest_ind(unc_test['diversity'], cond_test['diversity'])[0],
            'p_value': stats.ttest_ind(unc_test['diversity'], cond_test['diversity'])[1],
        }

        # Extreme case analysis
        unc_extreme = self.results['unconditional']['extreme_results']
        cond_extreme = self.results['conditional']['extreme_results']

        analysis['extreme_case_analysis'] = {
            'unconditional_mean_reward': np.mean(unc_extreme['mean_rewards']),
            'conditional_mean_reward': np.mean(cond_extreme['mean_rewards']),
            'improvement_ratio': np.mean(cond_extreme['mean_rewards']) / np.mean(unc_extreme['mean_rewards']),
        }

        # Consistency analysis (lower std = better generalization)
        analysis['consistency_analysis'] = {
            'unconditional_reward_std': np.mean(unc_test['std_rewards']),
            'conditional_reward_std': np.mean(cond_test['std_rewards']),
            'consistency_improvement': np.mean(unc_test['std_rewards']) / np.mean(cond_test['std_rewards']),
        }

        return analysis

    def create_visualizations(self, save_dir: str = "generalization_analysis"):
        """Create comprehensive visualizations."""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Training curves comparison
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.results['unconditional']['training_loss'], label='Unconditional', alpha=0.7)
        plt.plot(self.results['conditional']['training_loss'], label='Conditional', alpha=0.7)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(self.results['unconditional']['training_reward'], label='Unconditional', alpha=0.7)
        plt.plot(self.results['conditional']['training_reward'], label='Conditional', alpha=0.7)
        plt.title('Training Reward')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        unc_rewards = self.results['unconditional']['test_results']['mean_rewards']
        cond_rewards = self.results['conditional']['test_results']['mean_rewards']
        plt.boxplot([unc_rewards, cond_rewards], patch_artist=True)
        plt.xticks([1, 2], ['Unconditional', 'Conditional'])
        plt.title('Test Reward Distribution')
        plt.ylabel('Mean Reward')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Performance across different weight configurations
        test_weights = self.results['unconditional']['test_results']['weights']
        unc_rewards = self.results['unconditional']['test_results']['mean_rewards']
        cond_rewards = self.results['conditional']['test_results']['mean_rewards']

        plt.figure(figsize=(12, 8))

        x = range(len(test_weights))
        width = 0.35

        plt.bar([i - width/2 for i in x], unc_rewards, width, label='Unconditional', alpha=0.7)
        plt.bar([i + width/2 for i in x], cond_rewards, width, label='Conditional', alpha=0.7)

        plt.xlabel('Test Weight Configuration')
        plt.ylabel('Mean Reward')
        plt.title('Performance Across Different Weight Configurations')
        plt.xticks(x, [f"W{i+1}" for i in range(len(test_weights))])
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/weight_configuration_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Pareto efficiency comparison
        plt.figure(figsize=(10, 6))

        unc_pareto = self.results['unconditional']['test_results']['pareto_efficiency']
        cond_pareto = self.results['conditional']['test_results']['pareto_efficiency']

        plt.scatter(unc_pareto, cond_pareto, alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal performance')
        plt.xlabel('Unconditional Pareto Efficiency')
        plt.ylabel('Conditional Pareto Efficiency')
        plt.title('Pareto Efficiency Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/pareto_efficiency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Diversity comparison
        plt.figure(figsize=(10, 6))

        unc_diversity = self.results['unconditional']['test_results']['diversity']
        cond_diversity = self.results['conditional']['test_results']['diversity']

        plt.scatter(unc_diversity, cond_diversity, alpha=0.7)
        plt.plot([0, max(max(unc_diversity), max(cond_diversity))],
                [0, max(max(unc_diversity), max(cond_diversity))], 'r--', alpha=0.5, label='Equal diversity')
        plt.xlabel('Unconditional Diversity')
        plt.ylabel('Conditional Diversity')
        plt.title('Sequence Diversity Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/diversity_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, save_dir: str = "generalization_analysis"):
        """Save all results to files."""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_for_save = {}
        for model_type, data in self.results.items():
            results_for_save[model_type] = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    results_for_save[model_type][key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            results_for_save[model_type][key][subkey] = subvalue.tolist()
                        else:
                            results_for_save[model_type][key][subkey] = subvalue
                elif isinstance(value, np.ndarray):
                    results_for_save[model_type][key] = value.tolist()
                else:
                    results_for_save[model_type][key] = value

        with open(f"{save_dir}/raw_results_{timestamp}.json", 'w') as f:
            json.dump(results_for_save, f, indent=2)

        analysis = self.analyze_generalization()
        with open(f"{save_dir}/analysis_{timestamp}.json", 'w') as f:
            json.dump(analysis, f, indent=2)

        self.create_summary_report(save_dir, timestamp, analysis)

    def create_summary_report(self, save_dir: str, timestamp: str, analysis: Dict):
        """Create a human-readable summary report."""
        report_path = f"{save_dir}/summary_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("GENERALIZATION ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("EXPERIMENT SETUP\n")
            f.write("-" * 20 + "\n")
            f.write(f"Training weight configurations: {len(self.training_weights)}\n")
            f.write(f"Test weight configurations: {len(self.test_weights)}\n")
            f.write(f"Extreme test configurations: {len(self.extreme_test_weights)}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write("PERFORMANCE COMPARISON\n")
            f.write("-" * 25 + "\n")

            reward_comp = analysis['reward_comparison']
            f.write(f"Average Test Reward:\n")
            f.write(f"  Unconditional: {reward_comp['unconditional_mean']:.4f} ± {reward_comp['unconditional_std']:.4f}\n")
            f.write(f"  Conditional:   {reward_comp['conditional_mean']:.4f} ± {reward_comp['conditional_std']:.4f}\n")
            f.write(f"  Improvement:   {((reward_comp['conditional_mean'] / reward_comp['unconditional_mean']) - 1) * 100:.2f}%\n")
            f.write(f"  P-value:       {reward_comp['p_value']:.4f}\n")
            f.write(f"  Significant:   {'Yes' if reward_comp['p_value'] < 0.05 else 'No'}\n\n")

            # Pareto efficiency
            pareto_comp = analysis['pareto_comparison']
            f.write(f"Pareto Efficiency:\n")
            f.write(f"  Unconditional: {pareto_comp['unconditional_mean']:.4f}\n")
            f.write(f"  Conditional:   {pareto_comp['conditional_mean']:.4f}\n")
            f.write(f"  P-value:       {pareto_comp['p_value']:.4f}\n\n")

            # Diversity
            diversity_comp = analysis['diversity_comparison']
            f.write(f"Sequence Diversity:\n")
            f.write(f"  Unconditional: {diversity_comp['unconditional_mean']:.4f}\n")
            f.write(f"  Conditional:   {diversity_comp['conditional_mean']:.4f}\n")
            f.write(f"  P-value:       {diversity_comp['p_value']:.4f}\n\n")

            # Extreme cases
            extreme_comp = analysis['extreme_case_analysis']
            f.write(f"Extreme Case Performance:\n")
            f.write(f"  Unconditional: {extreme_comp['unconditional_mean_reward']:.4f}\n")
            f.write(f"  Conditional:   {extreme_comp['conditional_mean_reward']:.4f}\n")
            f.write(f"  Improvement:   {((extreme_comp['improvement_ratio']) - 1) * 100:.2f}%\n\n")

            # Consistency
            consistency_comp = analysis['consistency_analysis']
            f.write(f"Consistency (Lower is better):\n")
            f.write(f"  Unconditional: {consistency_comp['unconditional_reward_std']:.4f}\n")
            f.write(f"  Conditional:   {consistency_comp['conditional_reward_std']:.4f}\n")
            f.write(f"  Improvement:   {((consistency_comp['consistency_improvement']) - 1) * 100:.2f}%\n\n")

            f.write("CONCLUSION\n")
            f.write("-" * 10 + "\n")

            # Determine if conditional is better
            reward_better = reward_comp['conditional_mean'] > reward_comp['unconditional_mean']
            pareto_better = pareto_comp['conditional_mean'] > pareto_comp['unconditional_mean']
            diversity_better = diversity_comp['conditional_mean'] > diversity_comp['unconditional_mean']
            extreme_better = extreme_comp['improvement_ratio'] > 1.0
            consistency_better = consistency_comp['consistency_improvement'] > 1.0

            better_count = sum([reward_better, pareto_better, diversity_better, extreme_better, consistency_better])

            f.write(f"Conditional model performs better on {better_count}/5 metrics.\n")

            if better_count >= 3:
                f.write("CONCLUSION: Conditional training appears to help with generalization.\n")
            elif better_count >= 2:
                f.write("CONCLUSION: Conditional training shows some benefits for generalization.\n")
            else:
                f.write("CONCLUSION: Conditional training does not show clear benefits for generalization.\n")

            if reward_comp['p_value'] < 0.05:
                f.write("The improvement in reward is statistically significant.\n")
            else:
                f.write("The improvement in reward is not statistically significant.\n")

        logging.info(f"Summary report saved to {report_path}")


def main():

    parser = argparse.ArgumentParser(description="Compare conditional vs unconditional GFlowNet training")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to config file")

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr_logz', type=float, default=1e-1)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--subTB_lambda', type=float, default=0.8)
    parser.add_argument("--subTB_weighting",type=str,default="geometric_within",help="weighting scheme for SubTB")
    parser.add_argument('--tied', action='store_true')

    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_hidden', type=int, default=2)

    parser.add_argument("--n_iterations", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of evaluation samples")

    parser.add_argument("--save_dir", type=str, default="generalization_analysis", help="Directory to save results")
    parser.add_argument("--wandb_project", type=str, default="Comparison_Cond_UnCond_GFN")
    parser.add_argument("--wandb_run_name", type=str, default='')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.wandb_project:

        print('Wandb initialization...')
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"generalization_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )

    # Run comparison
    evaluator = GeneralizationEvaluator(args.config_path)
    results = evaluator.run_comparison(args,
        n_iterations=args.n_iterations,
        n_samples=args.n_samples,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden
    )

    # Analyze and save results
    analysis = evaluator.analyze_generalization()
    evaluator.create_visualizations(args.save_dir)
    evaluator.save_results(args.save_dir)

    # Log to WandB if enabled
    if args.wandb_project:
        # Log key metrics
        reward_comp = analysis['reward_comparison']
        wandb.log({
            "unconditional_mean_reward": reward_comp['unconditional_mean'],
            "conditional_mean_reward": reward_comp['conditional_mean'],
            "reward_improvement_pct": ((reward_comp['conditional_mean'] / reward_comp['unconditional_mean']) - 1) * 100,
            "reward_p_value": reward_comp['p_value'],
            "reward_significant": reward_comp['p_value'] < 0.05,
        })

        # Log plots
        wandb.log({
            "training_comparison": wandb.Image(f"{args.save_dir}/training_comparison.png"),
            "weight_configuration_performance": wandb.Image(f"{args.save_dir}/weight_configuration_performance.png"),
            "pareto_efficiency_comparison": wandb.Image(f"{args.save_dir}/pareto_efficiency_comparison.png"),
            "diversity_comparison": wandb.Image(f"{args.save_dir}/diversity_comparison.png"),
        })

    logging.info("Comparison completed successfully!")
    logging.info(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
