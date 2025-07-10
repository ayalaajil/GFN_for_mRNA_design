from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import train
from evaluate import evaluate, sweep_weight_configs
from plots import *
from datetime import datetime
from utils import load_config, tokenize_sequence_to_tensor
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main(args):

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) 
    logging.info(f"Using device: {device}")

    if args.wandb_project:

        logging.info("Initializing Weights & Biases...")

        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=config.run_name if config.run_name else None
        )

    # training time
    start_time = time.time()

    # 1. Create the environment.
    logging.info("Creating environment...")

    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)   
    preprocessor = CodonSequencePreprocessor(env.seq_length, embedding_dim=args.embedding_dim, device=device)   

    logging.info("Building GFlowNet model...")
    # Build the GFlowNet.
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

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)

    # Feed pf to the sampler.
    sampler = Sampler(estimator=pf_estimator)
    gflownet = gflownet.to(env.device)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    loss_history = []
    reward_history = []
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience
    )

    logging.info("Starting training loop...")

    start_time = time.time()
    
    loss_history, reward_history, reward_components, unique_seqs, sampled_weights = train(
        args, config, env, gflownet, sampler, optimizer, scheduler, device
    )

    total_time = time.time() - start_time

    logging.info(f"Training completed in {total_time:.2f} seconds.")

    plot_training_curves(loss_history, reward_components)
    plot_of_weights_over_iterations(sampled_weights)
    plot_ternary_plot_of_weights(sampled_weights)

    if args.wandb_project:

        logging.info("Logging training summary to WandB...")
        wandb.log({
            "final_loss": loss_history[-1],
            "total_training_time": total_time,
            "final_unique_sequences": len(unique_seqs),
            "training_curves": wandb.Image("training_curves.png")
        })

    logging.info("Evaluating final model on sampled sequences...")

    start_inference_time = time.time()


    # Sample final sequences
    with torch.no_grad():

        samples, gc_list, mfe_list, cai_list = evaluate(env, sampler, weights=torch.tensor([0.3, 0.3, 0.4]), n_samples= args.n_samples)

    inference_time = time.time() - start_inference_time
    avg_time_per_seq = inference_time / args.n_samples

    logging.info(f"Inference (sampling) completed in {inference_time:.2f} seconds.")
    logging.info(f"Average time per generated sequence is {avg_time_per_seq:.2f} seconds.")


    # weight_configs = {
    # "GC_only": [1.0, 0.0, 0.0],
    # "MFE_only": [0.0, 1.0, 0.0],
    # "CAI_only": [0.0, 0.0, 1.0],
    # "MFE+CAI": [0.0, 0.5, 0.5],
    # }

    # all_results = sweep_weight_configs(env, sampler, weight_configs, n_samples=50)

    # plot_pairwise_scatter(all_results, 'CAI', 'MFE')
    # plot_pairwise_scatter(all_results, 'CAI', 'GC')

    # plot_pairwise_scatter_with_pareto(all_results)


######################################## Save model ########################################


    logging.info("Saving trained model and metrics...")
    torch.save({
            'model_state': gflownet.state_dict(),
            'logZ': gflownet.logZ,
            'training_history': {
                'loss': loss_history,
                'reward': reward_history
            }
        }, "trained_gflownet.pth")

    logging.info("Plotting final metric histograms and Pareto front...")

    plot_metric_histograms(gc_list, mfe_list, cai_list, out_path="metric_distributions.png")
    plot_pareto_front(gc_list, mfe_list, cai_list, out_path="pareto_scatter.png")
    plot_cai_vs_mfe(cai_list, mfe_list, out_path="cai_vs_mfe.png")
    plot_gc_vs_mfe(gc_list, mfe_list, out_path="gc_vs_mfe.png")


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"outputs/generated_sequences_{timestamp}.txt"

    logging.info(f"Saving generated sequences to {filename}")

    sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)


############################# Extract Best-by-Objective Sequences ##########################


    # Get best sequences for each reward component
    best_gc = max(samples.items(), key=lambda x: x[1][1][0])   # GC content
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

    table = wandb.Table(columns=["Index", "Sequence", "Reward", "GC Content", "MFE", "CAI", "Label"])


    for i, (seq, reward) in enumerate(sorted_samples[:30]):
        table.add_data(i + 1, seq, reward[0].item(), reward[1][0], reward[1][1], reward[1][2], "Pareto Optimal")

    # Add best-by-objective to the table
    table.add_data(31, best_gc[0], best_gc[1][0].item(), best_gc[1][1][0], best_gc[1][1][1], best_gc[1][1][2], "Best GC")
    table.add_data(32, best_mfe[0], best_mfe[1][0].item(), best_mfe[1][1][0], best_mfe[1][1][1], best_mfe[1][1][2], "Best MFE")
    table.add_data(33, best_cai[0], best_cai[1][0].item(), best_cai[1][1][0], best_cai[1][1][1], best_cai[1][1][2], "Best CAI")

    wandb.log({"Top_Sequences": table})


    # Extract top N sequences
    top_n = 20    
    sequences = [seq for seq, _ in sorted_samples[:top_n]]

    # Diversity analysis
    distances = analyze_diversity(sequences) 

    generated_sequences_tensor = [
        tokenize_sequence_to_tensor(seq)
        for seq in sequences
    ]

    # best-by-objective sequences
    additional_best_seqs = [best_gc[0], best_mfe[0], best_cai[0]]

    for s in additional_best_seqs:
        generated_sequences_tensor.append(tokenize_sequence_to_tensor(s))

    natural_tensor = tokenize_sequence_to_tensor(config.natural_mRNA_seq)
    sequence_labels = [f"Gen {i+1}" for i in range(top_n)] + ["Best GC", "Best MFE", "Best CAI"]

    analyze_sequence_properties(
        generated_sequences_tensor,
        natural_tensor,
        labels=sequence_labels
    )


    # Wandb logging
    if args.wandb_project:

        logging.info("Logging evaluation metrics to WandB...")
        wandb.log({
                "Pareto Plot": wandb.Image('pareto_scatter.png'),
                "Training_time" : total_time,
                "Inference_time": inference_time,
                "Avg_time_per_sequence": avg_time_per_seq,
                "Reward Metric distributions": wandb.Image('metric_distributions.png'),
                "edit_distance_distribution": wandb.Image("edit_distance_distribution.png"),
                "CAI vs MFE": wandb.Image("cai_vs_mfe.png"),
                "GC vs MFE": wandb.Image("gc_vs_mfe.png"),
                "mean_edit_distance": np.mean(distances),
                "std_edit_distance": np.std(distances),

            })

    wandb.run.summary["final_loss"] = loss_history[-1]
    wandb.run.summary["inference_time"] = inference_time
    wandb.run.summary["Avg_time_per_sequence"] = avg_time_per_seq
    wandb.run.summary["total_training_time"] = total_time
    wandb.run.summary["unique_sequences"] = len(unique_seqs)
    wandb.run.summary["mean_edit_distance"] = np.mean(distances)
    wandb.run.summary["std_edit_distance"] = np.std(distances)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--lr",type=float,default=1e-3,help="Learning rate for the estimators' modules")
    parser.add_argument("--lr_logz",type=float,default=1e-1,help="Learning rate for the logZ parameter")

    parser.add_argument("--n_iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to generate")

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the sampler")
    
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of codon embeddings")
    
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the networks")
    parser.add_argument("--n_hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--tied", action="store_true", help="Whether to tie the parameters of PF and PB")
    
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--lr_patience", type=int, default=10, help="Patience for learning rate scheduler")
    
    parser.add_argument("--wandb_project", type=str, default="mRNA_design", help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default="", help="Name for the wandb run")

    parser.add_argument('--config_path', type=str, default="config.yaml")

    args = parser.parse_args()
    config = load_config(args.config_path)

    main(args)

