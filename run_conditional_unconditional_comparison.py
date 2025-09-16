#!/usr/bin/env python3
"""
Comprehensive experiment to run both conditional and unconditional models
"""

import sys
import os
import argparse
import logging
import subprocess
from datetime import datetime
import torch
import numpy as np
import wandb
from tqdm import tqdm
import json
import pandas as pd

# Adjust this path if your project structure is different
sys.path.insert(0, os.path.dirname(__file__))

# Import necessary components
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor2
from main_conditional import build_subTB_gflownet, load_config
from reward import compute_simple_reward
from utils import *
from plots import *
from enhanced_comparison import run_comprehensive_analysis
from gfn.samplers import Sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class ConditionalUnconditionalComparison:
    """Comprehensive comparison between conditional and unconditional models"""

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")

        # Use the same protein sequences as in curriculum evaluation
        self.evaluation_proteins = {
            "short_seen": "MINTQDSSILPLSNCPQLQCCRHIVPGPLWCS*",  # Length 32
            "medium_unseen": "MKLVRFLMKLSHETVTIELKNGTQVHGTITGVDVSMNTHLKAVKMTLKNREPVQLETLSIRGNNIRYFILPDSLPLDTLLVDVEPKVKSKKREAVAGRGRGRGRGRGRGRGRGRGGPRR*",  # Length 120
            "long_seen": "MGASARLLRAVIMGAPGSGKGTVSSRITTHFELKHLSSGDLLRDNMLRGTEIGVLAKAFIDQGKLIPDDVMTRLALHELKNLTQYSWLLDGFPRTLPQAEALDRAYQIDTVINLNVPFEVIKQRLTARWIHPASGRVYNIEFNPPKTVGIDDLTGEPLIQREDDKPETVIKRLKAYEDQTKPVLEYYQKKGVLETFSGTETNKIWPYVYAFLQTKVPQRSQKASVTP*"  # Length 228
        }

        self.natural_mRNA_sequences = {
            "short_seen": "AUGAUAAACACCCAGGACAGUAGUAUUUUGCCUUUGAGUAACUGUCCCCAGCUCCAGUGCUGCAGGCACAUUGUUCCAGGGCCUCUGUGGUGCUCCUAA",
            "medium_unseen": "AUGAAGCUCGUGAGAUUUUUGAUGAAAUUGAGUCAUGAAACUGUAACCAUUGAAUUGAAGAACGGAACACAGGUCCAUGGAACAAUCACAGGUGUGGAUGUCAGCAUGAAUACACAUCUUAAAGCUGUGAAAAUGACCCUGAAGAACAGAGAACCUGUACAGCUGGAAACGCUGAGUAUUCGAGGAAAUAACAUUCGGUAUUUUAUUCUACCAGACAGUUUACCUCUGGAUACACUACUUGUGGAUGUUGAACCUAAGGUGAAAUCUAAGAAAAGGGAAGCUGUUGCAGGAAGAGGCAGAGGAAGAGGAAGAGGAAGAGGACGUGGCCGUGGCAGAGGAAGAGGGGGUCCUAGGCGAUAA",
            "long_seen": "AUGGGGGCGUCCGCGCGGCUGCUGCGAGCGGUGAUCAUGGGGGCCCCGGGCUCGGGCAAGGGCACCGUGUCGUCGCGCAUCACUACACACUUCGAGCUGAAGCACCUCUCCAGCGGGGACCUGCUCCGGGACAACAUGCUGCGGGGCACAGAAAUUGGCGUGUUAGCCAAGGCUUUCAUUGACCAAGGGAAACUCAUCCCAGAUGAUGUCAUGACUCGGCUGGCCCUUCAUGAGCUGAAAAAUCUCACCCAGUAUAGCUGGCUGUUGGAUGGUUUUCCAAGGACACUUCCACAGGCAGAAGCCCUAGAUAGAGCUUAUCAGAUCGACACAGUGAUUAACCUGAAUGUGCCCUUUGAGGUCAUUAAACAACGCCUUACUGCUCGCUGGAUUCAUCCCGCCAGUGGCCGAGUCUAUAACAUUGAAUUCAACCCUCCCAAAACUGUGGGCAUUGAUGACCUGACUGGGGAGCCUCUCAUUCAGCGUGAGGAUGAUAAACCAGAGACGGUUAUCAAGAGACUAAAGGCUUAUGAAGACCAAACAAAGCCAGUCCUGGAAUAUUACCAGAAAAAAGGGGUGCUGGAAACAUUCUCCGGAACAGAAACCAACAAGAUUUGGCCCUAUGUAUAUGCUUUCCUACAAACUAAAGUUCCACAAAGAAGCCAGAAAGCUUCAGUUACUCCAUGA",
        }

        self.results = {
            'conditional': {},
            'unconditional': {}
        }

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.output_dir = f"outputs/conditional_unconditional_comparison_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(f"Initialized comparison with device: {self.device}")
        logging.info(f"Output directory: {self.output_dir}")

    def train_conditional_model(self, protein_seq, task_name):
        """Train conditional model on a specific protein sequence"""
        logging.info(f"Training conditional model on {task_name} (length: {len(protein_seq)})")

        env = CodonDesignEnv(protein_seq=protein_seq, device=self.device)
        preprocessor = CodonSequencePreprocessor2(len(protein_seq) + 50, embedding_dim=self.args.embedding_dim, device=self.device)

        gflownet, pf_estimator, pb_estimator = build_subTB_gflownet(env, preprocessor, self.args, lamda=self.args.subTB_lambda)
        gflownet = gflownet.to(self.device)
        pf_estimator = pf_estimator.to(self.device)
        pb_estimator = pb_estimator.to(self.device)

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

        sampler = Sampler(estimator=pf_estimator)

        loss_history = []
        reward_history = []

        logging.info(f"Starting conditional training for {self.args.n_iterations} iterations...")

        for iteration in tqdm(range(self.args.n_iterations), desc=f"Conditional training {task_name}"):
            try:
                weights = np.random.dirichlet([1, 1, 1])
                env.set_weights(weights)
                weights_tensor = torch.tensor(weights, dtype=torch.get_default_dtype(), device=self.device)
                conditioning = weights_tensor.unsqueeze(0).expand(self.args.batch_size, *weights_tensor.shape)

                trajectories = gflownet.sample_trajectories(
                    env,
                    n=self.args.batch_size,
                    conditioning=conditioning,
                    save_logprobs=True,
                    save_estimator_outputs=True,
                    epsilon=self.args.epsilon,
                )

                optimizer.zero_grad()
                loss = gflownet.loss_from_trajectories(
                    env, trajectories, recalculate_all_logprobs=False
                )
                loss.backward()

                if self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(gflownet.parameters(), self.args.clip_grad_norm)

                optimizer.step()
                scheduler.step(loss)

                loss_history.append(loss.item())

                if iteration % self.args.eval_every == 0:
                    eval_reward = self.evaluate_model(gflownet, env, sampler, weights_tensor)
                    reward_history.append(eval_reward)

                    logging.info(f"Iteration {iteration}: Loss = {loss.item():.6f}, Reward = {eval_reward:.4f}")

            except Exception as e:
                logging.error(f"Error in conditional training iteration {iteration}: {e}")
                continue

        final_weights = torch.tensor([0.3, 0.3, 0.4], device=self.device)
        final_reward = self.evaluate_model(gflownet, env, sampler, final_weights)

        logging.info(f"Conditional training completed. Final reward: {final_reward:.4f}")

        return gflownet, loss_history, reward_history, final_reward

    def train_unconditional_model(self, protein_seq, task_name):
        """Train unconditional model on a specific protein sequence"""
        logging.info(f"Training unconditional model on {task_name} (length: {len(protein_seq)})")

        env = CodonDesignEnv(protein_seq=protein_seq, device=self.device)
        preprocessor = CodonSequencePreprocessor2(len(protein_seq) + 50, embedding_dim=self.args.embedding_dim, device=self.device)

        gflownet, pf_estimator, pb_estimator = build_subTB_gflownet(env, preprocessor, self.args, lamda=self.args.subTB_lambda)
        gflownet = gflownet.to(self.device)
        pf_estimator = pf_estimator.to(self.device)
        pb_estimator = pb_estimator.to(self.device)

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

        sampler = Sampler(estimator=pf_estimator)

        loss_history = []
        reward_history = []

        fixed_weights = torch.tensor([0.3, 0.3, 0.4], device=self.device)
        env.set_weights(fixed_weights)

        logging.info(f"Starting unconditional training for {self.args.n_iterations} iterations...")

        for iteration in tqdm(range(self.args.n_iterations), desc=f"Unconditional training {task_name}"):
            try:
                conditioning = fixed_weights.unsqueeze(0).expand(self.args.batch_size, *fixed_weights.shape)

                trajectories = gflownet.sample_trajectories(
                    env,
                    n=self.args.batch_size,
                    conditioning=conditioning,
                    save_logprobs=True,
                    save_estimator_outputs=True,
                    epsilon=self.args.epsilon,
                )

                optimizer.zero_grad()
                loss = gflownet.loss_from_trajectories(
                    env, trajectories, recalculate_all_logprobs=False
                )
                loss.backward()

                if self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(gflownet.parameters(), self.args.clip_grad_norm)

                optimizer.step()
                scheduler.step(loss)

                loss_history.append(loss.item())

                if iteration % self.args.eval_every == 0:
                    eval_reward = self.evaluate_model(gflownet, env, sampler, fixed_weights)
                    reward_history.append(eval_reward)

                    logging.info(f"Iteration {iteration}: Loss = {loss.item():.6f}, Reward = {eval_reward:.4f}")

            except Exception as e:
                logging.error(f"Error in unconditional training iteration {iteration}: {e}")
                continue

        final_reward = self.evaluate_model(gflownet, env, sampler, fixed_weights)

        logging.info(f"Unconditional training completed. Final reward: {final_reward:.4f}")

        return gflownet, loss_history, reward_history, final_reward

    def evaluate_model(self, gflownet, env, sampler, weights, n_samples=100):
        """Evaluate model performance"""
        with torch.no_grad():
            conditioning = weights.unsqueeze(0).expand(n_samples, *weights.shape)
            trajectories = sampler.sample_trajectories(env, n=n_samples, conditioning=conditioning)
            final_states = trajectories.terminating_states.tensor

            total_reward = 0
            valid_samples = 0

            for state in final_states:
                reward, components = compute_simple_reward(state, env.codon_gc_counts, weights)
                total_reward += reward
                valid_samples += 1

            avg_reward = total_reward / valid_samples if valid_samples > 0 else 0.0
            return avg_reward

    def comprehensive_evaluation(self, gflownet, protein_seq, task_name, model_type):
        """Run comprehensive evaluation similar to curriculum evaluation"""
        logging.info(f"Running comprehensive evaluation for {model_type} model on {task_name}")

        env = CodonDesignEnv(protein_seq=protein_seq, device=self.device)
        preprocessor = CodonSequencePreprocessor2(len(protein_seq) + 50, embedding_dim=self.args.embedding_dim, device=self.device)
        sampler = Sampler(estimator=gflownet.pf)

        weights = torch.tensor([0.3, 0.3, 0.4], device=self.device)
        conditioning = weights.unsqueeze(0).expand(self.args.n_samples, *weights.shape)

        with torch.no_grad():
            trajectories = sampler.sample_trajectories(env, n=self.args.n_samples, conditioning=conditioning)
            final_states = trajectories.terminating_states.tensor

        samples = {}
        gc_list, mfe_list, cai_list = [], [], []

        for state in tqdm(final_states, desc=f"Analyzing {model_type} sequences"):
            seq_str = "".join([env.idx_to_codon[idx.item()] for idx in state if idx != -1])
            if not seq_str:
                continue

            reward, components = compute_simple_reward(state, env.codon_gc_counts, weights)
            samples[seq_str] = (reward, components)
            gc_list.append(components[0])
            mfe_list.append(components[1])
            cai_list.append(components[2])

        task_output_dir = os.path.join(self.output_dir, f"{model_type}_{task_name}")
        os.makedirs(task_output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        analysis_results = run_comprehensive_analysis(
            samples,
            gc_list,
            mfe_list,
            cai_list,
            self.natural_mRNA_sequences[task_name],
            task_output_dir,
            timestamp,
            top_n=self.args.top_n
        )

        return {
            "samples": samples,
            "gc_list": gc_list,
            "mfe_list": mfe_list,
            "cai_list": cai_list,
            "protein_length": len(protein_seq),
            "analysis_results": analysis_results
        }

    def run_comparison(self):
        """Run the complete comparison experiment"""
        logging.info("Starting Conditional vs Unconditional comparison experiment")

        if self.args.wandb_project:
            wandb.init(
                project=self.args.wandb_project,
                config={**vars(self.args), **vars(self.config)},
                name=f"cond_uncond_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}",
                group="Conditional_Unconditional_Comparison",
                tags=['comparison', 'conditional', 'unconditional']
            )

        for task_name, protein_seq in self.evaluation_proteins.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"PROCESSING TASK: {task_name}")
            logging.info(f"Protein length: {len(protein_seq)}")
            logging.info(f"{'='*60}")

            logging.info("Training conditional model...")
            cond_gflownet, cond_losses, cond_rewards, cond_final_reward = self.train_conditional_model(protein_seq, task_name)

            logging.info("Training unconditional model...")
            uncond_gflownet, uncond_losses, uncond_rewards, uncond_final_reward = self.train_unconditional_model(protein_seq, task_name)

            logging.info("Running comprehensive evaluation...")
            cond_results = self.comprehensive_evaluation(cond_gflownet, protein_seq, task_name, "conditional")
            uncond_results = self.comprehensive_evaluation(uncond_gflownet, protein_seq, task_name, "unconditional")

            self.results['conditional'][task_name] = {
                'loss_history': cond_losses,
                'reward_history': cond_rewards,
                'final_reward': cond_final_reward,
                'evaluation': cond_results
            }

            self.results['unconditional'][task_name] = {
                'loss_history': uncond_losses,
                'reward_history': uncond_rewards,
                'final_reward': uncond_final_reward,
                'evaluation': uncond_results
            }

            if self.args.wandb_project:
                self.log_task_results(task_name, cond_results, uncond_results)

        self.generate_comparison_report()

        if self.args.wandb_project:
            wandb.finish()

        logging.info("Comparison experiment completed!")

    def log_task_results(self, task_name, cond_results, uncond_results):
        """Log results for a specific task to WandB"""

        table = wandb.Table(columns=["Model", "Task", "Final_Reward", "GC_Mean", "MFE_Mean", "CAI_Mean", "Unique_Sequences"])

        cond_gc_mean = np.mean(cond_results['gc_list']) if cond_results['gc_list'] else 0
        cond_mfe_mean = np.mean(cond_results['mfe_list']) if cond_results['mfe_list'] else 0
        cond_cai_mean = np.mean(cond_results['cai_list']) if cond_results['cai_list'] else 0
        cond_unique = len(cond_results['samples'])

        table.add_data("Conditional", task_name,
                      self.results['conditional'][task_name]['final_reward'],
                      cond_gc_mean, cond_mfe_mean, cond_cai_mean, cond_unique)

        uncond_gc_mean = np.mean(uncond_results['gc_list']) if uncond_results['gc_list'] else 0
        uncond_mfe_mean = np.mean(uncond_results['mfe_list']) if uncond_results['mfe_list'] else 0
        uncond_cai_mean = np.mean(uncond_results['cai_list']) if uncond_results['cai_list'] else 0
        uncond_unique = len(uncond_results['samples'])

        table.add_data("Unconditional", task_name,
                      self.results['unconditional'][task_name]['final_reward'],
                      uncond_gc_mean, uncond_mfe_mean, uncond_cai_mean, uncond_unique)

        wandb.log({
            f"{task_name}/conditional_final_reward": self.results['conditional'][task_name]['final_reward'],
            f"{task_name}/unconditional_final_reward": self.results['unconditional'][task_name]['final_reward'],
            f"{task_name}/conditional_gc_mean": cond_gc_mean,
            f"{task_name}/unconditional_gc_mean": uncond_gc_mean,
            f"{task_name}/conditional_mfe_mean": cond_mfe_mean,
            f"{task_name}/unconditional_mfe_mean": uncond_mfe_mean,
            f"{task_name}/conditional_cai_mean": cond_cai_mean,
            f"{task_name}/unconditional_cai_mean": uncond_cai_mean,
            f"{task_name}/conditional_unique_sequences": cond_unique,
            f"{task_name}/unconditional_unique_sequences": uncond_unique,
            f"{task_name}/comparison_table": table,
            f"{task_name}/conditional_pareto_plot": wandb.Image(cond_results['analysis_results']['plot_path']),
            f"{task_name}/unconditional_pareto_plot": wandb.Image(uncond_results['analysis_results']['plot_path'])
        })

    def generate_comparison_report(self):
        """Generate a comprehensive comparison report"""
        logging.info("Generating comparison report...")

        report_path = os.path.join(self.output_dir, "comparison_report.txt")

        with open(report_path, 'w') as f:
            f.write("CONDITIONAL vs UNCONDITIONAL MODEL COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Architecture: {self.args.arch}\n")
            f.write(f"Embedding Dim: {self.args.embedding_dim}\n")
            f.write(f"Hidden Dim: {self.args.hidden_dim}\n")
            f.write(f"Hidden Layers: {self.args.n_hidden}\n")
            f.write(f"Learning Rate: {self.args.lr}\n")
            f.write(f"LogZ Learning Rate: {self.args.lr_logz}\n")
            f.write(f"SubTB Lambda: {self.args.subTB_lambda}\n")
            f.write(f"Training Iterations: {self.args.n_iterations}\n")
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Evaluation Samples: {self.args.n_samples}\n\n")

            for task_name in self.evaluation_proteins.keys():
                f.write(f"\nTASK: {task_name.upper()}\n")
                f.write("-" * 40 + "\n")

                cond_data = self.results['conditional'][task_name]
                uncond_data = self.results['unconditional'][task_name]

                f.write(f"Protein Length: {cond_data['evaluation']['protein_length']}\n\n")

                f.write("CONDITIONAL MODEL:\n")
                f.write(f"  Final Reward: {cond_data['final_reward']:.4f}\n")
                f.write(f"  Final Loss: {cond_data['loss_history'][-1]:.6f}\n")
                f.write(f"  Unique Sequences: {len(cond_data['evaluation']['samples'])}\n")

                if cond_data['evaluation']['gc_list']:
                    f.write(f"  GC Content - Mean: {np.mean(cond_data['evaluation']['gc_list']):.4f}, Std: {np.std(cond_data['evaluation']['gc_list']):.4f}\n")
                if cond_data['evaluation']['mfe_list']:
                    f.write(f"  MFE - Mean: {np.mean(cond_data['evaluation']['mfe_list']):.4f}, Std: {np.std(cond_data['evaluation']['mfe_list']):.4f}\n")
                if cond_data['evaluation']['cai_list']:
                    f.write(f"  CAI - Mean: {np.mean(cond_data['evaluation']['cai_list']):.4f}, Std: {np.std(cond_data['evaluation']['cai_list']):.4f}\n")

                f.write("\nUNCONDITIONAL MODEL:\n")
                f.write(f"  Final Reward: {uncond_data['final_reward']:.4f}\n")
                f.write(f"  Final Loss: {uncond_data['loss_history'][-1]:.6f}\n")
                f.write(f"  Unique Sequences: {len(uncond_data['evaluation']['samples'])}\n")

                if uncond_data['evaluation']['gc_list']:
                    f.write(f"  GC Content - Mean: {np.mean(uncond_data['evaluation']['gc_list']):.4f}, Std: {np.std(uncond_data['evaluation']['gc_list']):.4f}\n")
                if uncond_data['evaluation']['mfe_list']:
                    f.write(f"  MFE - Mean: {np.mean(uncond_data['evaluation']['mfe_list']):.4f}, Std: {np.std(uncond_data['evaluation']['mfe_list']):.4f}\n")
                if uncond_data['evaluation']['cai_list']:
                    f.write(f"  CAI - Mean: {np.mean(uncond_data['evaluation']['cai_list']):.4f}, Std: {np.std(uncond_data['evaluation']['cai_list']):.4f}\n")
                reward_diff = cond_data['final_reward'] - uncond_data['final_reward']
                f.write(f"\nCOMPARISON:\n")
                f.write(f"  Reward Difference (Cond - Uncond): {reward_diff:.4f}\n")
                f.write(f"  Conditional Better: {'Yes' if reward_diff > 0 else 'No'}\n")
                f.write(f"  Improvement: {abs(reward_diff):.4f} ({abs(reward_diff)/uncond_data['final_reward']*100:.2f}%)\n")


        json_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(json_path, 'w') as f:

            json_results = {}
            for model_type in self.results:
                json_results[model_type] = {}
                for task_name in self.results[model_type]:
                    task_data = self.results[model_type][task_name].copy()

                    if 'gc_list' in task_data['evaluation']:
                        task_data['evaluation']['gc_list'] = [float(x) for x in task_data['evaluation']['gc_list']]
                    if 'mfe_list' in task_data['evaluation']:
                        task_data['evaluation']['mfe_list'] = [float(x) for x in task_data['evaluation']['mfe_list']]
                    if 'cai_list' in task_data['evaluation']:
                        task_data['evaluation']['cai_list'] = [float(x) for x in task_data['evaluation']['cai_list']]
                    task_data['loss_history'] = [float(x) for x in task_data['loss_history']]
                    task_data['reward_history'] = [float(x) for x in task_data['reward_history']]
                    json_results[model_type][task_name] = task_data

            json.dump(json_results, f, indent=2)

        logging.info(f"Comparison report saved to: {report_path}")
        logging.info(f"Detailed results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare conditional vs unconditional models with identical architecture")

    # Model architecture arguments (matching curriculum learning)
    parser.add_argument('--arch', type=str, default='Transformer', help="Model architecture")
    parser.add_argument('--embedding_dim', type=int, default=32, help="Embedding dimension")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension")
    parser.add_argument('--n_hidden', type=int, default=4, help="Number of hidden layers")
    parser.add_argument('--subTB_lambda', type=float, default=0.9, help="SubTB lambda parameter")
    parser.add_argument('--tied', action='store_true', help="Use tied parameters")

    # Training arguments (matching curriculum learning)
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--lr_logz', type=float, default=1e-1, help="LogZ learning rate")
    parser.add_argument('--n_iterations', type=int, default=300, help="Number of training iterations")
    parser.add_argument('--eval_every', type=int, default=10, help="Evaluate every N iterations")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--epsilon', type=float, default=0.25, help="Epsilon for exploration")
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument('--lr_patience', type=int, default=10, help="LR scheduler patience")

    # Evaluation arguments
    parser.add_argument('--n_samples', type=int, default=200, help="Number of samples for evaluation")
    parser.add_argument('--top_n', type=int, default=50, help="Top N sequences for analysis")

    # System arguments
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--config_path', type=str, default="config.yaml", help="Config file path")
    parser.add_argument('--wandb_project', type=str, default='BASELINE_Experiments', help="WandB project name")
    parser.add_argument('--run_name', type=str, default='Conditional_Unconditional_Comparison', help="Run name")

    args = parser.parse_args()
    config = load_config(args.config_path)
    set_seed(args.seed)

    comparison = ConditionalUnconditionalComparison(args, config)
    comparison.run_comparison()


if __name__ == "__main__":
    main()
