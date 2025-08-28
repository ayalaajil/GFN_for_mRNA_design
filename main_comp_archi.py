import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import train
from evaluate import evaluate
from plots import *
from datetime import datetime
from utils import *
import logging
import torch
import argparse
import numpy as np
import time
import wandb
from comparison import analyze_sequence_properties
from torchgfn.src.gfn.gflownet import TBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator
from torchgfn.src.gfn.utils.modules import MLP
from torchgfn.src.gfn.samplers import Sampler

# Import architecture testing framework
from architecture_test import (
    create_architecture, run_architecture_experiment, run_architecture_sweep,
    save_architecture_results, ARCHITECTURE_CONFIGS, count_parameters
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main_single_architecture(args, config):
    """Original main function with architecture selection"""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    if config.wandb_project:
        logging.info("Initializing Weights & Biases...")
        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=config.run_name if config.run_name else None,
        )

    start_time = time.time()
    logging.info("Creating environment...")

    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(
        env.seq_length, embedding_dim=args.embedding_dim, device=device
    )

    logging.info(f"Protein sequence length: {len(config.protein_seq)}")
    logging.info("Building GFlowNet model...")

    # Create architectures based on command line arguments
    if hasattr(args, 'architecture_pf') and args.architecture_pf:
        logging.info(f"Using custom architecture for PF: {args.architecture_pf}")
        module_PF = create_architecture(
            args.architecture_pf,
            preprocessor.output_dim,
            env.n_actions
        )
    else:
        # Default architecture
        module_PF = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )

    if hasattr(args, 'architecture_pb') and args.architecture_pb:
        logging.info(f"Using custom architecture for PB: {args.architecture_pb}")
        module_PB = create_architecture(
            args.architecture_pb,
            preprocessor.output_dim,
            env.n_actions - 1
        )
    else:
        # Default architecture
        module_PB = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=module_PF.trunk if args.tied else None,
        )

    # Log parameter counts
    pf_params = count_parameters(module_PF)
    pb_params = count_parameters(module_PB)
    logging.info(f"PF parameters: {pf_params:,}")
    logging.info(f"PB parameters: {pb_params:,}")

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)
    sampler = Sampler(estimator=pf_estimator)
    gflownet = gflownet.to(env.device)

    total_params = count_parameters(gflownet)
    logging.info(f"Total GFlowNet parameters: {total_params:,}")

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group(
        {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    logging.info("Starting training loop...")
    start_time = time.time()

    loss_history, reward_history, reward_components, unique_seqs = train(
        args, env, gflownet, sampler, optimizer, scheduler, device
    )

    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds.")

    plot_training_curves(loss_history, reward_components)

    if config.wandb_project:
        logging.info("Logging training summary to WandB...")
        wandb.log(
            {
                "final_loss": loss_history[-1],
                "total_training_time": total_time,
                "final_unique_sequences": len(unique_seqs),
                "training_curves": wandb.Image("training_curves.png"),
                "pf_parameters": pf_params,
                "pb_parameters": pb_params,
                "total_parameters": total_params,
            }
        )

    logging.info("Evaluating final model on sampled sequences...")
    start_inference_time = time.time()

    with torch.no_grad():
        samples, gc_list, mfe_list, cai_list = evaluate(
            env,
            sampler,
            weights=torch.tensor([0.3, 0.3, 0.4]),
            n_samples=args.n_samples,
        )

    inference_time = time.time() - start_inference_time
    avg_time_per_seq = inference_time / args.n_samples

    logging.info(f"Inference (sampling) completed in {inference_time:.2f} seconds.")
    logging.info(f"Average time per generated sequence is {avg_time_per_seq:.2f} seconds.")

    # Calculate final means
    eval_mean_gc = float(torch.tensor(gc_list).mean())
    eval_mean_mfe = float(torch.tensor(mfe_list).mean())
    eval_mean_cai = float(torch.tensor(cai_list).mean())

    logging.info("Saving trained model and metrics...")
    torch.save(
        {
            "model_state": gflownet.state_dict(),
            "logZ": gflownet.logZ,
            "training_history": {"loss": loss_history, "reward": reward_history},
            "architecture_info": {
                "pf_architecture": getattr(args, 'architecture_pf', 'standard_mlp'),
                "pb_architecture": getattr(args, 'architecture_pb', 'standard_mlp'),
                "pf_parameters": pf_params,
                "pb_parameters": pb_params,
                "total_parameters": total_params,
            }
        },
        "trained_gflownet.pth",
    )

    logging.info("Plotting final metric histograms and Pareto front...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    plot_metric_histograms(
        gc_list, mfe_list, cai_list,
        out_path=f"outputs_{config.type}/metric_distributions_{timestamp}.png"
    )
    plot_pareto_front(
        gc_list, mfe_list, cai_list,
        out_path=f"outputs_{config.type}/pareto_scatter_{timestamp}.png"
    )
    plot_cai_vs_mfe(
        cai_list, mfe_list,
        out_path=f"outputs_{config.type}/cai_vs_mfe_{timestamp}.png"
    )
    plot_gc_vs_mfe(
        gc_list, mfe_list,
        out_path=f"outputs_{config.type}/gc_vs_mfe_{timestamp}.png"
    )

    filename = f"outputs_{config.type}/generated_sequences_{timestamp}.txt"
    logging.info(f"Saving generated sequences to {filename}")
    sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

    # Extract best-by-objective sequences
    best_gc = max(samples.items(), key=lambda x: x[1][1][0])
    best_mfe = min(samples.items(), key=lambda x: x[1][1][1])
    best_cai = max(samples.items(), key=lambda x: x[1][1][2])

    with open(filename, "w") as f:
        for i, (seq, reward) in enumerate(sorted_samples):
            f.write(
                f"Sequence {i+1}: {seq}, "
                f"Reward: {reward[0]:.2f}, "
                f"GC Content: {reward[1][0]:.2f}, "
                f"MFE: {reward[1][1]:.2f}, "
                f"CAI: {reward[1][2]:.2f}\n"
            )

    # Analysis and logging
    top_n = 50
    sequences = [seq for seq, _ in sorted_samples[:top_n]]
    distances = analyze_diversity(sequences)

    generated_sequences_tensor = [tokenize_sequence_to_tensor(seq) for seq in sequences]
    additional_best_seqs = [best_gc[0], best_mfe[0], best_cai[0]]

    for s in additional_best_seqs:
        generated_sequences_tensor.append(tokenize_sequence_to_tensor(s))

    natural_tensor = tokenize_sequence_to_tensor(config.natural_mRNA_seq)
    sequence_labels = [f"Gen {i+1}" for i in range(top_n)] + [
        "Best GC", "Best MFE", "Best CAI",
    ]

    analyze_sequence_properties(
        generated_sequences_tensor, natural_tensor, labels=sequence_labels
    )

    Eval_avg_reward = sum(
        w * r for w, r in zip(env.weights, [eval_mean_gc, -eval_mean_mfe, eval_mean_cai])
    )

    if config.wandb_project:
        logging.info("Logging evaluation metrics to WandB...")

        # Create wandb table for top sequences
        table = wandb.Table(
            columns=["Index", "Sequence", "Reward", "GC Content", "MFE", "CAI", "Label"]
        )

        for i, (seq, reward) in enumerate(sorted_samples[:5]):
            table.add_data(
                i + 1,
                seq,
                reward[0],
                reward[1][0],
                reward[1][1],
                reward[1][2],
                "Pareto Optimal",
            )

        table.add_data(31, best_gc[0], best_gc[1][0], best_gc[1][1][0], best_gc[1][1][1], best_gc[1][1][2], "Best GC")
        table.add_data(32, best_mfe[0], best_mfe[1][0], best_mfe[1][1][0], best_mfe[1][1][1], best_mfe[1][1][2], "Best MFE")
        table.add_data(33, best_cai[0], best_cai[1][0], best_cai[1][1][0], best_cai[1][1][1], best_cai[1][1][2], "Best CAI")

        wandb.log({
            "Top_Sequences": table,
            "Pareto Plot": wandb.Image(f"outputs_{config.type}/pareto_scatter_{timestamp}.png"),
            "Training_time": total_time,
            "Inference_time": inference_time,
            "Avg_time_per_sequence": avg_time_per_seq,
            "Reward Metric distributions": wandb.Image(f"outputs_{config.type}/metric_distributions_{timestamp}.png"),
            "edit_distance_distribution": wandb.Image("edit_distance_distribution.png"),
            "CAI vs MFE": wandb.Image(f"outputs_{config.type}/cai_vs_mfe_{timestamp}.png"),
            "GC vs MFE": wandb.Image(f"outputs_{config.type}/gc_vs_mfe_{timestamp}.png"),
            "mean_edit_distance": np.mean(distances),
            "std_edit_distance": np.std(distances),
            "Eval_mean_gc": eval_mean_gc,
            "Eval_mean_mfe": eval_mean_mfe,
            "Eval_mean_cai": eval_mean_cai,
            "Eval_avg_reward": Eval_avg_reward,
            "architecture_pf": getattr(args, 'architecture_pf', 'standard_mlp'),
            "architecture_pb": getattr(args, 'architecture_pb', 'standard_mlp'),
            "pf_parameters": pf_params,
            "pb_parameters": pb_params,
            "total_parameters": total_params,
        })

    wandb.summary["final_loss"] = loss_history[-1]
    wandb.summary["unique_sequences"] = len(unique_seqs)

    return Eval_avg_reward

def main_architecture_sweep(args, config):
    """Run architecture comparison sweep"""

    logging.info("Starting architecture comparison sweep...")

    # Define which architectures to test
    if hasattr(args, 'test_architectures') and args.test_architectures:
        architectures_to_test = args.test_architectures.split(',')
    else:
        # Default set of architectures to compare
        architectures_to_test = [
            "standard_mlp",
            "deep_mlp",
            "wide_mlp",
            "residual_mlp",
            "attention_mlp",
            "dropout_mlp",
            "small_mlp"
        ]

    logging.info(f"Testing architectures: {architectures_to_test}")

    # Run the sweep
    results = run_architecture_sweep(args, config, architectures_to_test)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_architecture_results(results)

    if config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=f"architecture_sweep_{timestamp}",
            config=vars(args)
        )

        comparison_table = wandb.Table(
            columns=["Architecture", "Description", "Parameters", "Final_Reward",
                    "Final_Loss", "Unique_Seqs", "Mean_GC", "Mean_MFE", "Mean_CAI"]
        )

        for result in sorted(results, key=lambda x: x['eval_avg_reward'], reverse=True):
            comparison_table.add_data(
                result['architecture_pf'],
                result['pf_description'],
                result['total_parameters'],
                result['eval_avg_reward'],
                result['final_loss'],
                result['unique_sequences'],
                result['eval_mean_gc'],
                result['eval_mean_mfe'],
                result['eval_mean_cai']
            )

        wandb.log({"Architecture_Comparison": comparison_table})

        # Log best performing architecture
        best_result = max(results, key=lambda x: x['eval_avg_reward'])
        wandb.summary.update({
            "best_architecture": best_result['architecture_pf'],
            "best_reward": best_result['eval_avg_reward'],
            "best_parameters": best_result['total_parameters']
        })

        wandb.finish()

    return results

def list_available_architectures():
    """Print available architectures and their descriptions"""
    print("\nAvailable Architectures:")
    print("=" * 50)
    for name, config in ARCHITECTURE_CONFIGS.items():
        print(f"{name:20s}: {config['description']}")
        print(f"{'':20s}  Parameters: {config['params']}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the estimators' modules")
    parser.add_argument("--lr_logz", type=float, default=1e-1, help="Learning rate for the logZ parameter")
    parser.add_argument("--n_iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon for the sampler")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of codon embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the networks")
    parser.add_argument("--n_hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--tied", action="store_true", help="Whether to tie the parameters of PF and PB")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--lr_patience", type=int, default=10, help="Patience for learning rate scheduler")
    parser.add_argument("--wandb_project", type=str, default="mRNA_design", help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default="", help="Name for the wandb run")
    parser.add_argument("--config_path", type=str, default="config.yaml")

    # Architecture testing arguments
    parser.add_argument("--mode", type=str, choices=["single", "sweep", "list"],
                       default="single", help="Mode: single architecture, sweep, or list available")
    parser.add_argument("--architecture_pf", type=str, default=None,
                       help="Architecture name for policy forward network")
    parser.add_argument("--architecture_pb", type=str, default=None,
                       help="Architecture name for policy backward network (defaults to same as PF)")
    parser.add_argument("--test_architectures", type=str, default=None,
                       help="Comma-separated list of architectures to test in sweep mode")

    args = parser.parse_args()
    config = load_config(args.config_path)

    if args.mode == "list":
        list_available_architectures()
    elif args.mode == "single":
        main_single_architecture(args, config)
    elif args.mode == "sweep":
        main_architecture_sweep(args, config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


# Example usage commands:
"""
# Run with standard MLP (default)
python main.py --mode single

# Run with residual MLP for both PF and PB
python main.py --mode single --architecture_pf residual_mlp

# Run with different architectures for PF and PB
python main.py --mode single --architecture_pf attention_mlp --architecture_pb residual_mlp

# Run architecture sweep with default set
python main.py --mode sweep

# Run architecture sweep with specific architectures
python main.py --mode sweep --test_architectures "standard_mlp,residual_mlp,attention_mlp"

# List available architectures
python main.py --mode list
"""

