import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

from datetime import datetime
import logging
import torch
import argparse
import numpy as np
import time
import wandb

from comparison import analyze_sequence_properties
from utils import *
from plots import *
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import *
from evaluate import *

from gfn.gflownet import TBGFlowNet

from gfn.estimators import (
    ConditionalDiscretePolicyEstimator,
    ConditionalScalarEstimator,
    ScalarEstimator,
)
from gfn.utils.modules import MLP
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States

# Create a wrapper class to make ConditionalScalarEstimator compatible with TBGFlowNet
class ConditionalLogZWrapper(ScalarEstimator):
    def __init__(self, conditional_estimator, env):
        super().__init__(
            conditional_estimator.module, conditional_estimator.preprocessor
        )
        self.conditional_estimator = conditional_estimator
        self.state_shape = env.state_shape
        self.device = env.device
        self.States = env.States  # Store the environment's States class

    def forward(self, conditioning):

        # Create dummy states for the conditional estimator
        # The conditional estimator expects states, but we only have conditioning
        # We'll create dummy states with the same batch shape as conditioning

        # Handle different conditioning tensor shapes
        if conditioning.ndim == 1:
            # Single condition tensor of shape (cond_dim,)
            batch_shape = (1,)
            conditioning = conditioning.unsqueeze(0)  # Make it (1, cond_dim)
        elif conditioning.ndim == 2:
            # Batch of conditions of shape (batch_size, cond_dim)
            batch_shape = (conditioning.shape[0],)
        elif conditioning.ndim == 3:
            # The conditioning tensor has been expanded by the sampler to (max_length, n_trajectories, cond_dim)
            # For logZ calculation, we want the batch size to be n_trajectories
            # We can take any time step since the conditioning should be the same across time
            batch_shape = (conditioning.shape[1],)  # n_trajectories
            conditioning = conditioning[0, :, :]  # Take first time step: (n_trajectories, cond_dim)
        else:
            raise ValueError(f"Unexpected conditioning tensor shape: {conditioning.shape}")

        # Create dummy states manually instead of using env.reset()
        dummy_states_tensor = torch.full(
            (batch_shape[0],) + self.state_shape, 
            fill_value=-1, 
            dtype=torch.long, 
            device=self.device
        )
        
        dummy_states = self.States(dummy_states_tensor)
        
        result = self.conditional_estimator(dummy_states, conditioning)
        return result


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def build_tb_gflownet(env, pf_estimator, pb_estimator, preprocessor, cond_dim: int = 3) -> TBGFlowNet:

    module_logZ_state = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=16,
        hidden_dim=16,
        n_hidden_layers=2,
    )

    module_logZ_cond = MLP(
        input_dim=cond_dim,
        output_dim=16,
        hidden_dim=16,
        n_hidden_layers=2,
    )

    module_logZ_final = MLP(
        input_dim=32,  # 16 + 16
        output_dim=1,
        hidden_dim=16,
        n_hidden_layers=2,
    )

def build_tb_gflownet(env, pf_estimator, pb_estimator, preprocessor, cond_dim: int = 3) -> TBGFlowNet:

    module_logZ_state = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=16,
        hidden_dim=16,
        n_hidden_layers=2,
    )

    module_logZ_cond = MLP(
        input_dim=cond_dim,
        output_dim=16,
        hidden_dim=16,
        n_hidden_layers=2,
    )

    module_logZ_final = MLP(
        input_dim=32,  # 16 + 16
        output_dim=1,
        hidden_dim=16,
        n_hidden_layers=2,
    )



    conditional_logZ = ConditionalScalarEstimator(
        module_logZ_state,
        module_logZ_cond,
        module_logZ_final,
        preprocessor=preprocessor,
    )
    logZ_estimator = ConditionalLogZWrapper(conditional_logZ, env)
    gflownet = TBGFlowNet(logZ=logZ_estimator, pf=pf_estimator, pb=pb_estimator)
    return gflownet


def main(args, config):

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

    CONCAT_SIZE = 16
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
    )
    module_PB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        n_hidden_layers=args.n_hidden,
        trunk=module_PF.trunk if args.tied else None,
    )

    # Encoder for the Conditioning information.
    module_cond = MLP(
        input_dim=3,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )

    # Modules post-concatenation.
    module_final_PF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions,
    )

    module_final_PB = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions - 1,
        trunk=module_final_PF.trunk,
    )

    pf_estimator = ConditionalDiscretePolicyEstimator(
        module_PF,
        module_cond,
        module_final_PF,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=False,
    )
    pb_estimator = ConditionalDiscretePolicyEstimator(
        module_PB,
        module_cond,
        module_final_PB,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=True,
    )

    gflownet = build_tb_gflownet(env, pf_estimator, pb_estimator, preprocessor, cond_dim= 3)

    sampler = Sampler(estimator=pf_estimator)
    gflownet = gflownet.to(env.device)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group(
        {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
    )

    loss_history = []
    reward_history = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    logging.info("Starting training loop...")
    start_time = time.time()

    loss_history, reward_history, reward_components, unique_seqs, sampled_weights = train_conditional_gfn(
        args, env, gflownet, sampler, optimizer, scheduler, device
    )

    total_time = time.time() - start_time

    logging.info(f"Training completed in {total_time:.2f} seconds.")
    plot_training_curves(loss_history, reward_components)

    plot_of_weights_over_iterations(sampled_weights)
    plot_ternary_plot_of_weights(sampled_weights)

    if config.wandb_project:

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

        samples, gc_list, mfe_list, cai_list = evaluate_conditional(
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

    # logging.info("Saving trained model and metrics...")
    # torch.save(
    #     {
    #         "model_state": gflownet.state_dict(),
    #         "logZ": gflownet.logZ,
    #         "training_history": {"loss": loss_history, "reward": reward_history},
    #         "config": vars(config),  # Save config for recreating environment
    #         "args": vars(args),      # Save args for recreating environment
    #     },
    #     "trained_gflownet.pth",
    # )

    logging.info("Plotting final metric histograms and Pareto front...")

    plot_metric_histograms(
        gc_list, mfe_list, cai_list, out_path="metric_distributions.png"
    )
    plot_pareto_front(gc_list, mfe_list, cai_list, out_path="pareto_scatter.png")
    plot_cai_vs_mfe(cai_list, mfe_list, out_path="cai_vs_mfe.png")
    plot_gc_vs_mfe(gc_list, mfe_list, out_path="gc_vs_mfe.png")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"outputs_condi/generated_sequences_{timestamp}.txt"

    logging.info(f"Saving generated sequences to {filename}")
    sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

    ############################# Extract Best-by-Objective Sequences ##########################

    # Get best sequences for each reward component
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

    # top_n = 50
    sequences = [seq for seq, _ in sorted_samples[:args.top_n]]
    distances = analyze_diversity(sequences)

    generated_sequences_tensor = [tokenize_sequence_to_tensor(seq) for seq in sequences]

    # best-by-objective sequences
    additional_best_seqs = [best_gc[0], best_mfe[0], best_cai[0]]

    for s in additional_best_seqs:
        generated_sequences_tensor.append(tokenize_sequence_to_tensor(s))

    natural_tensor = tokenize_sequence_to_tensor(config.natural_mRNA_seq)
    sequence_labels = [f"Gen {i+1}" for i in range(args.top_n)] + [
        "Best GC",
        "Best MFE",
        "Best CAI",
    ]

    analyze_sequence_properties(
        generated_sequences_tensor, natural_tensor, labels=sequence_labels
    )

    Eval_avg_reward = sum(
        w * r
        for w, r in zip(env.weights, [eval_mean_gc, -eval_mean_mfe, eval_mean_cai])
    )

    if config.wandb_project:

        logging.info("Logging evaluation metrics to WandB...")
        wandb.log(
            {
                "Pareto Plot": wandb.Image("pareto_scatter.png"),
                "Training_time": total_time,
                "Inference_time": inference_time,
                "Avg_time_per_sequence": avg_time_per_seq,
                "Reward Metric distributions": wandb.Image("metric_distributions.png"),
                "edit_distance_distribution": wandb.Image(
                    "edit_distance_distribution.png"
                ),
                "CAI vs MFE": wandb.Image("cai_vs_mfe.png"),
                "GC vs MFE": wandb.Image("gc_vs_mfe.png"),
                "mean_edit_distance": np.mean(distances),
                "std_edit_distance": np.std(distances),
                "Eval_mean_gc": eval_mean_gc,
                "Eval_mean_mfe": eval_mean_mfe,
                "Eval_mean_cai": eval_mean_cai,
                "Eval_avg_reward": Eval_avg_reward,
            }
        )

    wandb.summary["final_loss"] = loss_history[-1]
    wandb.summary["unique_sequences"] = len(unique_seqs)

    return Eval_avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
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
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--top_n", type=int, default=50, help="Top K sequences"
    )

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon for the sampler"
    )

    parser.add_argument(
        "--embedding_dim", type=int, default=32, help="Dimension of codon embeddings"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension of the networks"
    )
    parser.add_argument(
        "--n_hidden", type=int, default=2, help="Number of hidden layers"
    )

    parser.add_argument(
        "--tied", action="store_true", help="Whether to tie the parameters of PF and PB"
    )
    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="Patience for learning rate scheduler",
    )

    parser.add_argument("--wandb_project", type=str, default="mRNA_design", help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default="", help="Name for the wandb run")

    parser.add_argument("--config_path", type=str, default="config.yaml")

    args = parser.parse_args()
    config = load_config(args.config_path)

    main(args, config)




























############################### Comparison between configs ##############################

# weight_configs = {
# "GC_only": [1.0, 0.0, 0.0],
# "MFE_only": [0.0, 1.0, 0.0],
# "CAI_only": [0.0, 0.0, 1.0],
# "MFE+CAI": [0.0, 0.5, 0.5],
# "MFE+CAI+GC":[0.3, 0.3, 0.4]
# }
# all_results = sweep_weight_configs(env, sampler, weight_configs, n_samples=50)
# plot_sweep_results(csv_path="sweep_results.csv")

#############################################################################################
