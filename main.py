from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import train
from evaluate import evaluate
from plots import plot_training_curves
from utils import load_config

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import Levenshtein
import seaborn as sns
import wandb

from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.modules import MLP
from gfn.samplers import Sampler


def main(args):

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) if config.device == "auto" else torch.device(config.device)
    
    if args.wandb_project:

        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=config.run_name if config.run_name else None
        )

    # Track training time
    start_time = time.time()

    # 1. Create the environment.
    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)   
    preprocessor = CodonSequencePreprocessor(env.seq_length, embedding_dim=args.embedding_dim, device=device)   

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
    visited_terminating_states = env.states_from_batch_shape((0,))

    loss_history = []
    reward_history = []
    rewards_components = []

    unique_sequences = set()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience
    )

    
    loss_history, reward_history, reward_components, unique_seqs = train(
        args, config, env, gflownet, sampler, optimizer, scheduler
    )

    plot_training_curves(loss_history, reward_components)

    # Sample final sequences
    with torch.no_grad():

        samples, gc_list, mfe_list, cai_list = evaluate(env, sampler, weights=torch.tensor([0.3, 0.3, 0.4]))

        # save
        torch.save({
            'model_state': gflownet.state_dict(),
            'logZ': gflownet.logZ,
            'training_history': {
                'loss': loss_history,
                'reward': reward_history
            }
        }, "trained_gflownet.pth")

        # Plot histograms
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 3, 1)
        plt.hist(gc_list, bins=20, color='green')
        plt.title('GC Content Distribution')

        plt.subplot(1, 3, 2)
        plt.hist(mfe_list, bins=20, color='blue')
        plt.title('MFE Distribution')

        plt.subplot(1, 3, 3)
        plt.hist(cai_list, bins=20, color='orange')
        plt.title('CAI Distribution')

        plt.tight_layout()
        plt.savefig("metric_distributions.png")

        # Plot the Pareto front
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(gc_list, mfe_list, cai_list, alpha=0.6)
        ax.set_xlabel('GC Content')
        ax.set_ylabel('MFE')
        ax.set_zlabel('CAI')
        plt.savefig("pareto_scatter.png")

        # Sort sequences by reward
        sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

        with open("generated_sequences.txt", "w") as f:
            for i, (seq, reward) in enumerate(sorted_samples):
                f.write(f"Sequence {i+1}: {seq}, Reward: {reward[0]:.2f}, GC Content: {reward[1][0]:.2f}, MFE: {reward[1][1]:.2f}, CAI: {reward[1][2]:.2f} \n")


#################################################################################################################################################################################
# Diversity analysis of generated sequences 
#################################################################################################################################################################################
        

        # Extract top N sequences
        top_n = 30 
        sequences = [seq for seq, _ in sorted_samples[:top_n]]

        distances = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                d = Levenshtein.distance(sequences[i], sequences[j])
                distances.append(d)

        # Plot histogram of distances
        plt.figure(figsize=(6, 4))
        sns.histplot(distances, bins=20, kde=True)
        plt.xlabel("Levenshtein Distance")
        plt.ylabel("Frequency")
        plt.title(f"Edit Distance Distribution (Top {top_n} sequences)")
        plt.tight_layout()
        plt.savefig("edit_distance_distribution.png")
        plt.close()

        # Wandb logging
        if args.wandb_project:
            wandb.log({
                "edit_distance_distribution": wandb.Image("edit_distance_distribution.png"),
                "mean_edit_distance": np.mean(distances),
                "std_edit_distance": np.std(distances),
            })


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_logz",
        type=float,
        default=1e-1,
        help="Learning rate for the logZ parameter",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon for the sampler"
    )
    
    # New arguments for model architecture
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of codon embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the networks")
    parser.add_argument("--n_hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--tied", action="store_true", help="Whether to tie the parameters of PF and PB")
    
    # Training optimization arguments
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--lr_patience", type=int, default=10, help="Patience for learning rate scheduler")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="mRNA_design", help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default="", help="Name for the wandb run")

    # config 
    parser.add_argument('--config_path', type=str, default="config.yaml")

    args = parser.parse_args()
    config = load_config(args.config_path)

    main(args)

