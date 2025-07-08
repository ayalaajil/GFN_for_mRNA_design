from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import train
from evaluate import evaluate
from plots import plot_training_curves, analyze_diversity, plot_metric_histograms, plot_pareto_front, plot_cai_vs_mfe, plot_gc_vs_mfe
from datetime import datetime
from utils import load_config, tokenize_sequence_to_tensor
import logging
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import Levenshtein
import seaborn as sns
import wandb
from comparison import analyze_sequence_properties
from torchgfn.src.gfn.gflownet import TBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator
from torchgfn.src.gfn.utils.modules import MLP, DiscreteUniform, Tabular
from torchgfn.src.gfn.samplers import Sampler

import os
import pandas as pd
import pygmo as pg


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

    # Track training time
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
    
    loss_history, reward_history, reward_components, unique_seqs = train(
        args, config, env, gflownet, sampler, optimizer, scheduler, device
    )

    total_time = time.time() - start_time

    logging.info(f"Training completed in {total_time:.2f} seconds.")

    plot_training_curves(loss_history, reward_components)

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

    n_samples= 100

    # Sample final sequences
    with torch.no_grad():

        samples, gc_list, mfe_list, cai_list = evaluate(env, sampler, weights=torch.tensor([0.3, 0.3, 0.4]), n_samples= n_samples)

    inference_time = time.time() - start_inference_time
    avg_time_per_seq = inference_time / n_samples

    logging.info(f"Inference (sampling) completed in {inference_time:.2f} seconds.")
    logging.info(f"Average time per generated sequence is {avg_time_per_seq:.2f} seconds.")


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

    ######################### Extract Best-by-Objective Sequences ##############################

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

    # Diversity analysis of generated sequences 
    distances = analyze_diversity(sequences) 

    generated_sequences_tensor = [
        tokenize_sequence_to_tensor(seq)
        for seq in sequences
    ]

    # Add best-by-objective sequences to generated tensor list for comparison
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
    wandb.run.summary["total_training_time"] = total_time
    wandb.run.summary["unique_sequences"] = len(unique_seqs)
    wandb.run.summary["mean_edit_distance"] = np.mean(distances)
    wandb.run.summary["std_edit_distance"] = np.std(distances)


    # Systematic weight-sweep over [w_gc, w_mfe, w_cai] on a regular grid.
    # For each (w_gc, w_mfe, w_cai), samples sequences via evaluate(), computes per-objective means and hypervolume, then plots the hypervolume heatmap.
        

    OUTPUT_DIR = "weight_sweep_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)



    # --- GRID SWEEP ---
    grid = np.linspace(0, 1, GRID_SIZE)
    records = []

    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0:
                continue
            w3 = 1.0 - (w1 + w2)
            weights = torch.tensor([w1, w2, w3], dtype=torch.float32)

            # sample
            samples, gc_list, mfe_list, cai_list = evaluate(
                env, sampler, weights=weights, n_samples=N_SAMPLES
            )

            # compute means
            mean_gc  = float(np.mean(gc_list))
            mean_mfe = float(np.mean(mfe_list))
            mean_cai = float(np.mean(cai_list))

            # prepare points for hypervolume (flip MFE to max)
            points = np.vstack([
                gc_list,
                -np.array(mfe_list),
                cai_list
            ]).T

            # compute hypervolume
            hv_calc = pg.hypervolume(points)
            hv = hv_calc.compute(REF_POINT)

            records.append({
                "w_gc": w1, "w_mfe": w2, "w_cai": w3,
                "mean_gc": mean_gc, "mean_mfe": mean_mfe,
                "mean_cai": mean_cai, "hypervolume": hv
            })
            print(f"w=({w1:.2f},{w2:.2f},{w3:.2f}) -> HV={hv:.4f}")


    df = pd.DataFrame(records)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"weight_sweep_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


    # --- PLOT HEATMAP of Hypervolume ---
    # pivot to matrix form over w1 (rows) × w2 (cols)
    pivot = df.pivot(index="w_gc", columns="w_mfe", values="hypervolume")

    plt.figure(figsize=(6,5))

    plt.imshow(
        pivot.values,
        origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()],
        aspect="auto"
    )

    plt.colorbar(label="Hypervolume")
    plt.xlabel("w_mfe")
    plt.ylabel("w_gc")
    plt.title("Hypervolume over weight grid (w_cai=1−w_gc−w_mfe)")

    heatmap_path = os.path.join(OUTPUT_DIR, f"hypervolume_heatmap_{ts}.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")
    plt.show()






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr",type=float,default=1e-3,help="Learning rate for the estimators' modules")
    parser.add_argument("--lr_logz",type=float,default=1e-1,help="Learning rate for the logZ parameter")

    parser.add_argument("--n_iterations", type=int, default=200, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the sampler")
    
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of codon embeddings")
    
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the networks")
    parser.add_argument("-- ", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--tied", action="store_true", help="Whether to tie the parameters of PF and PB")
    
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--lr_patience", type=int, default=10, help="Patience for learning rate scheduler")
    
    parser.add_argument("--wandb_project", type=str, default="mRNA_design", help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default="", help="Name for the wandb run")

    parser.add_argument('--config_path', type=str, default="config.yaml")

    args = parser.parse_args()
    config = load_config(args.config_path)

    # logging.info(f"Loaded config from {args.config_path}")
    # logging.info(f"Training config: {config}")
    # logging.info(f"Run arguments: {args}")

    main(args)

