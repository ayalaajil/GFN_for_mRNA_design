import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

import itertools
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import *
from evaluate import evaluate
from plots import *
from datetime import datetime
from transformers.optimization import get_linear_schedule_with_warmup

from utils import *
from simple_reward_function import compute_simple_reward
from DeepArchi import *
import logging
import torch
import argparse
import numpy as np
import time
import wandb
from comparison import analyze_sequence_properties
from enhanced_comparison import run_comprehensive_analysis
from torchgfn.src.gfn.gflownet import TBGFlowNet, SubTBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator, Estimator, ScalarEstimator
from torchgfn.src.gfn.utils.modules import MLP
from torchgfn.src.gfn.samplers import Sampler

from ENN_ENH import MLP_ENN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def set_up_logF_estimator(
    args, preprocessor, pf_module
):
    """Returns a LogStateFlowEstimator."""

    module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=(
                pf_module.trunk
                if args.tied and isinstance(pf_module.trunk, torch.nn.Module)
                else None
            ),
        )

    return ScalarEstimator(module=module, preprocessor=preprocessor)


def main(args, config):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    if config.wandb_project:

        logging.info("Initializing Weights & Biases...")
        wandb.init(project= config.wandb_project,config={**vars(config), **vars(args)},name= config.run_name)

    start_time = time.time()
    logging.info("Creating environment...")

    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(
        env.seq_length, embedding_dim=args.embedding_dim, device=device
    )

    logging.info(f"Protein sequence length: {len(config.protein_seq)}")
    logging.info("Building GFlowNet model...")


    arch = getattr(config, 'arch', 'MLP')

    if arch == 'MLP_EHH':

            module_PF = MLP_ENN(
                    input_dim=preprocessor.output_dim,
                    output_dim=env.n_actions,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                )

            module_PB = MLP_ENN(
                    input_dim=preprocessor.output_dim,
                    output_dim=env.n_actions - 1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    trunk=module_PF.trunk if args.tied else None,
                )

    if arch == 'Transformer':

            module_PF = TransformerModel(
                    input_dim=preprocessor.output_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=env.n_actions,
                    n_layers=args.n_hidden,
                    n_head= 8
                    )

            module_PB = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk= None,
            )

    else :
            module_PF = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )

            module_PB = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk=module_PF.trunk if args.tied else None,
            )

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )

    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    logF_estimator = set_up_logF_estimator(args,preprocessor,pf_estimator)

    gflownet =  SubTBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    weighting=args.subTB_weighting,
                    lamda=args.subTB_lambda,)


    sampler = Sampler(estimator=pf_estimator)
    gflownet = gflownet.to(env.device)

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

    loss_history = []
    reward_history = []

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

    # plot_of_weights_over_iterations(sampled_weights)
    # plot_ternary_plot_of_weights(sampled_weights)

    if args.wandb_project:

        logging.info("Logging training summary to WandB...")
        wandb.log(
            {
                "final_loss": loss_history[-1],
                "total_training_time": total_time,
                "final_unique_sequences": len(unique_seqs),
                "training_curves": wandb.Image("training_curves.png"),
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
    logging.info(
        f"Average time per generated sequence is {avg_time_per_seq:.2f} seconds."
    )

    ################ Final means over the evaluation samples ###############

    eval_mean_gc = float(torch.tensor(gc_list).mean())
    eval_mean_mfe = float(torch.tensor(mfe_list).mean())
    eval_mean_cai = float(torch.tensor(cai_list).mean())

    logging.info("Saving trained model and metrics...")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Create organized output directory structure
    # Format: outputs/{experiment_type}/{protein_size}/{run_name}_{timestamp}/
    experiment_type = "conditional" if hasattr(args, 'conditional') and args.conditional else "unconditional"
    protein_size = getattr(config, 'type', 'unknown')
    output_dir = f"outputs/{experiment_type}/{protein_size}/{args.run_name}_{timestamp}"

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create experiment summary file
    summary_file = f"{output_dir}/experiment_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Experiment Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Experiment Type: {experiment_type}\n")
        f.write(f"Protein Size: {protein_size}\n")
        f.write(f"Run Name: {args.run_name}\n")
        f.write(f"Architecture: {getattr(config, 'arch', 'MLP')}\n")
        f.write(f"Protein Sequence Length: {len(config.protein_seq)}\n")
        f.write(f"Training Iterations: {args.n_iterations}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Hidden Dimension: {args.hidden_dim}\n")
        f.write(f"Number of Hidden Layers: {args.n_hidden}\n")
        f.write(f"SubTB Lambda: {args.subTB_lambda}\n")
        f.write(f"Epsilon: {args.epsilon}\n")
        f.write(f"WandB Project: {config.wandb_project}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nProtein Sequence: {config.protein_seq}\n")
        f.write(f"Natural mRNA Sequence: {config.natural_mRNA_seq}\n")
        f.write(f"\nGenerated Files:\n")
        f.write(f"- experiment_summary.txt (this file)\n")
        f.write(f"- trained_gflownet_{args.run_name}_{timestamp}.pth (model weights)\n")
        f.write(f"- generated_sequences_{timestamp}.txt (generated sequences)\n")
        f.write(f"- metric_distributions_{timestamp}.png (metric histograms)\n")
        f.write(f"- pareto_scatter_{timestamp}.png (Pareto front)\n")
        f.write(f"- cai_vs_mfe_{timestamp}.png (CAI vs MFE plot)\n")
        f.write(f"- gc_vs_mfe_{timestamp}.png (GC vs MFE plot)\n")
        f.write(f"- comprehensive_comparison_{timestamp}.txt (comparison table)\n")
        f.write(f"- metrics_summary_{timestamp}.txt (detailed metrics)\n")
        f.write(f"- enhanced_diversity_analysis_{timestamp}.png (diversity plots)\n")

    model_filename = f"{output_dir}/trained_gflownet_{args.run_name}_{timestamp}.pth"

    torch.save(
        {
            "model_state": gflownet.state_dict(),
            "training_history": {"loss": loss_history, "reward": reward_history},
        },
        model_filename,
    )

    logging.info("Plotting final metric histograms and Pareto front...")

    plot_metric_histograms(gc_list, mfe_list, cai_list, out_path=f"{output_dir}/metric_distributions_{timestamp}.png")
    plot_pareto_front(gc_list, mfe_list, cai_list, out_path=f"{output_dir}/pareto_scatter_{timestamp}.png")
    plot_cai_vs_mfe(cai_list, mfe_list, out_path = f"{output_dir}/cai_vs_mfe_{timestamp}.png")
    plot_gc_vs_mfe(gc_list, mfe_list, out_path= f"{output_dir}/gc_vs_mfe_{timestamp}.png")

    filename = f"{output_dir}/generated_sequences_{timestamp}.txt"

    logging.info(f"Saving generated sequences to {filename}")
    sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

    ############################# Extract Best-by-Objective Sequences ##########################

    best_gc = max(samples.items(), key=lambda x: x[1][1][0])  # GC content
    best_mfe = min(samples.items(), key=lambda x: x[1][1][1])  # MFE
    best_cai = max(samples.items(), key=lambda x: x[1][1][2])  # CAI


    with open(filename, "w") as f:
        for i, (seq, reward) in enumerate(sorted_samples):
            f.write(
                f"Sequence {i+1}: {seq}, "
                f"Reward: {reward[0]:.2f}, "
                f"GC Content: {reward[1][0]:.2f}, "
                f"MFE: {reward[1][1]:.2f}, "
                f"CAI: {reward[1][2]:.2f}\n"
            )

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

    table.add_data(
        31,
        best_gc[0],
        best_gc[1][0],
        best_gc[1][1][0],
        best_gc[1][1][1],
        best_gc[1][1][2],
        "Best GC",
    )
    table.add_data(
        32,
        best_mfe[0],
        best_mfe[1][0],
        best_mfe[1][1][0],
        best_mfe[1][1][1],
        best_mfe[1][1][2],
        "Best MFE",
    )
    table.add_data(
        33,
        best_cai[0],
        best_cai[1][0],
        best_cai[1][1][0],
        best_cai[1][1][1],
        best_cai[1][1][2],
        "Best CAI",
    )

    wandb.log({"Top_Sequences": table})

    top_n = 50
    sequences = [seq for seq, _ in sorted_samples[:top_n]]
    distances = analyze_diversity(sequences)

    generated_sequences_tensor = [tokenize_sequence_to_tensor(seq) for seq in sequences]
    additional_best_seqs = [best_gc[0], best_mfe[0], best_cai[0]]

    for s in additional_best_seqs:
        generated_sequences_tensor.append(tokenize_sequence_to_tensor(s))

    natural_tensor = tokenize_sequence_to_tensor(config.natural_mRNA_seq)
    sequence_labels = [f"Gen {i+1}" for i in range(top_n)] + [
        "Best GC",
        "Best MFE",
        "Best CAI",
    ]

    analyze_sequence_properties(generated_sequences_tensor, natural_tensor, labels=sequence_labels)

    # Run comprehensive analysis with enhanced metrics
    logging.info("Running comprehensive analysis with diversity and quality metrics...")
    analysis_results = run_comprehensive_analysis(
        samples, gc_list, mfe_list, cai_list, config.natural_mRNA_seq,
        output_dir, timestamp, top_n=args.top_n
    )

    # Compute the reward for each sample and then average them
    Eval_avg_reward = 0.0
    if len(samples) > 0:
        total_reward = 0.0
        for seq in samples.keys():

            seq_indices = torch.tensor([env.codon_to_idx[codon] for codon in seq], device=env.device)
            reward, _ = compute_simple_reward(
                seq_indices,
                env.codon_gc_counts,
                env.weights,
                protein_seq=env.protein_seq
            )
            total_reward += reward
        Eval_avg_reward = total_reward / len(samples)

    if args.wandb_project:

        logging.info("Logging evaluation metrics to WandB...")
        wandb.log(
            {
                "Pareto Plot": wandb.Image(f"{output_dir}/pareto_scatter_{timestamp}.png"),
                "Training_time": total_time,
                "Inference_time": inference_time,
                "Avg_time_per_sequence": avg_time_per_seq,
                "Reward Metric distributions": wandb.Image(f"{output_dir}/metric_distributions_{timestamp}.png"),
                "edit_distance_distribution": wandb.Image(
                    f"{output_dir}/edit_distance_distribution_{timestamp}.png"
                ),
                "CAI vs MFE": wandb.Image( f"{output_dir}/cai_vs_mfe_{timestamp}.png"),
                "GC vs MFE": wandb.Image(f"{output_dir}/gc_vs_mfe_{timestamp}.png"),
                "mean_edit_distance": np.mean(distances),
                "std_edit_distance": np.std(distances),
                "Eval_mean_gc": eval_mean_gc,
                "Eval_mean_mfe": eval_mean_mfe,
                "Eval_mean_cai": eval_mean_cai,
                "Eval_avg_reward": Eval_avg_reward,
                
                # Enhanced diversity and quality metrics
                "diversity_mean_edit_distance": analysis_results['diversity_metrics']['mean_edit_distance'],
                "diversity_std_edit_distance": analysis_results['diversity_metrics']['std_edit_distance'],
                "diversity_unique_sequences": analysis_results['diversity_metrics']['unique_sequences'],
                "diversity_uniqueness_ratio": analysis_results['diversity_metrics']['uniqueness_ratio'],
                "quality_pareto_efficiency": analysis_results['quality_metrics']['pareto_efficiency'],
                "quality_reward_mean": analysis_results['quality_metrics']['reward_stats']['mean'],
                "quality_reward_std": analysis_results['quality_metrics']['reward_stats']['std'],
                "Comprehensive Comparison Table": wandb.Image(analysis_results['table_path']),
                "Metrics Summary": wandb.Image(analysis_results['summary_path']),
                "Enhanced Diversity Analysis": wandb.Image(analysis_results['plot_path']),
            }
        )

    wandb.summary["final_loss"] = loss_history[-1]
    wandb.summary["unique_sequences"] = len(unique_seqs)
    wandb.finish()

    return Eval_avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # training-related
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr_logz', type=float, default=1e-1)

    parser.add_argument('--n_iterations', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--top_n', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--subTB_lambda', type=float, default=0.9)
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within", help="weighting scheme for SubTB")

    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_hidden', type=int, default=2)

    parser.add_argument('--tied', action='store_true')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_patience', type=int, default=10)

    parser.add_argument('--wandb_project', type=str, default='mRNA_design')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--conditional', action='store_true', help='Run conditional GFlowNet experiment')
    parser.add_argument("--config_path", type=str, default="config.yaml")

    args = parser.parse_args()
    set_seed(args.seed)
    config = load_config(args.config_path)

    main(args, config)















# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")

#     parser.add_argument("--subTB_weighting",type=str,default="geometric_within",help="weighting scheme for SubTB")
#     parser.add_argument("--subTB_lambda", type=float, default=0.8, help="Lambda parameter for SubTB")

#     parser.add_argument("--lr", type=float,default=0.005, help="Learning rate for the estimators' modules",)
#     parser.add_argument("--lr_logz",type=float,default=1e-1,help="Learning rate for the logZ parameter",)

#     parser.add_argument("--n_iterations", type=int, default=500, help="Number of iterations")
#     parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate")

#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
#     parser.add_argument("--epsilon", type=float, default=0.25, help="Epsilon for the sampler")

#     parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of codon embeddings")
#     parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of the networks")
#     parser.add_argument("--n_hidden", type=int, default=3, help="Number of hidden layers")

#     parser.add_argument("--tied", action="store_true", help="Whether to tie the parameters of PF and PB")
#     parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
#     parser.add_argument("--lr_patience", type=int, default=10, help="Patience for learning rate scheduler",)

#     parser.add_argument("--wandb_project", type=str, default="mRNA_design", help="Weights & Biases project name")
#     parser.add_argument("--run_name", type=str, default="", help="Name for the wandb run")

#     parser.add_argument("--config_path", type=str, default="config.yaml")

#     args = parser.parse_args()
#     config = load_config(args.config_path)

#     main(args, config)







# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--config_path", type=str, default="config.yaml")
#     parser.add_argument("--wandb_project", type=str, default="med_seq_ablation_study", help="Weights & Biases project name")
#     parser.add_argument("--run_name", type=str, default="", help="Name for the wandb run")
#     args = parser.parse_args()


#     config = load_config(args.config_path)

#     hyperparams = {
#         #"arch": ["MLP", "MLP_EHH", "Transformer"],
#         "arch": ["Transformer"],
#         "subTB_lambda": [0.8, 0.9, 0.99],
#         "batch_size": [16, 32, 64],
#         "lr": [0.0005, 0.001, 0.005],
#         "hidden_dim": [64, 128, 256]
#     }

#     param_grid = list(itertools.product(
#         hyperparams["arch"],
#         hyperparams["subTB_lambda"],
#         hyperparams["batch_size"],
#         hyperparams["lr"],
#         hyperparams["hidden_dim"]
#     ))

#     for arch, lm, bs, lr, hd in param_grid:

#         run_name = (
#             f"{arch}_"
#             f"lambda{lm}_"
#             f"bs{bs}_"
#             f"lr{lr}_"
#             f"hd{hd}"
#         )

#         print(f"\n=== Running: {run_name} ===")

#         class DummyArgs: pass
#         run_args = DummyArgs()
#         run_args.__dict__.update({
#             "batch_size": bs,
#             "lr": lr,
#             "lr_logz": 0.1,
#             "embedding_dim": 64,
#             "hidden_dim": hd,
#             "n_hidden": 3,
#             "n_iterations": 500,
#             "n_samples": 100,
#             "subTB_weighting": "geometric_within",
#             "subTB_lambda": lm,
#             "epsilon": 0.25,
#             "tied": False,
#             "clip_grad_norm": 1.0,
#             "lr_patience": 10,

#             # wandb logging
#             "wandb_project": args.wandb_project or config.wandb_project,
#             "run_name": run_name,

#         })

#         main(run_args, config)




















