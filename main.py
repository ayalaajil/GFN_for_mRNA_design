from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import train
from evaluate import evaluate
from plots import plot_training_curves, analyze_diversity, plot_metric_histograms, plot_pareto_front
from datetime import datetime
from utils import load_config

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import Levenshtein
import seaborn as sns
import wandb

from torchgfn.src.gfn.gflownet import TBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator
from torchgfn.src.gfn.utils.modules import MLP, DiscreteUniform, Tabular
from torchgfn.src.gfn.samplers import Sampler


def main(args):

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) 
    
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
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience
    )

    
    loss_history, reward_history, reward_components, unique_seqs = train(
        args, config, env, gflownet, sampler, optimizer, scheduler, device
    )

    total_time = time.time() - start_time

    plot_training_curves(loss_history, reward_components)

    if args.wandb_project:
        wandb.log({
            "final_loss": loss_history[-1],
            "total_training_time": total_time,
            "final_unique_sequences": len(unique_seqs),
            "training_curves": wandb.Image("training_curves.png")
        })

    # Sample final sequences
    with torch.no_grad():

        samples, gc_list, mfe_list, cai_list = evaluate(env, sampler, weights=torch.tensor([0.3, 0.3, 0.4]), n_samples= 100)

        # save
        torch.save({
            'model_state': gflownet.state_dict(),
            'logZ': gflownet.logZ,
            'training_history': {
                'loss': loss_history,
                'reward': reward_history
            }
        }, "trained_gflownet.pth")

        plot_metric_histograms(gc_list, mfe_list, cai_list, out_path="metric_distributions.png")
        plot_pareto_front(gc_list, mfe_list, cai_list, out_path="pareto_scatter.png")


        # Sort sequences by reward
        sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"outputs/generated_sequences_{timestamp}.txt"

        with open(filename, "w") as f:
            for i, (seq, reward) in enumerate(sorted_samples):
                f.write(
                    f"Sequence {i+1}: {seq}, "
                    f"Reward: {reward[0]:.2f}, "
                    f"GC Content: {reward[1][0]:.2f}, "
                    f"MFE: {reward[1][1]:.2f}, "
                    f"CAI: {reward[1][2]:.2f}\n"
                )

        # Extract top N sequences
        top_n = 30 
        sequences = [seq for seq, _ in sorted_samples[:top_n]]

        # Diversity analysis of generated sequences 
        distances = analyze_diversity(sequences) 

        # Wandb logging
        if args.wandb_project:
            wandb.log({
                "Pareto Plot": wandb.Image('pareto_scatter.png'),
                "Reward Metric distributions": wandb.Image('metric_distributions.png'),
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

