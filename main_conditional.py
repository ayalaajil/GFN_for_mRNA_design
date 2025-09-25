"""
Conditional GFlowNet training script for mRNA design.
"""
import sys
import os
import time
import logging
import argparse
from datetime import datetime
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

import torch
import numpy as np
import wandb
from reward import compute_simple_reward
from Architecture import *
from comparison_utils import run_comprehensive_analysis, analyze_sequence_properties
from utils import *
from plots import *
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import *
from evaluate import *

from gfn.gflownet import TBGFlowNet, SubTBGFlowNet
from gfn.estimators import (
    ConditionalDiscretePolicyEstimator,
    ConditionalScalarEstimator,
    ScalarEstimator,
)
from gfn.utils.modules import MLP
from gfn.samplers import Sampler


# ----------------------------- Model helper wrappers ----------------------------
class ConditionalLogZWrapper(ScalarEstimator):
    """Wrapper that turns a ConditionalScalarEstimator into a ScalarEstimator-like object.
    The purpose is to allow using conditional logZ estimators with existing TBGFlowNet
    which expects a ScalarEstimator-like interface for `logZ`.
    """

    def __init__(self, conditional_estimator, env):
        super().__init__(conditional_estimator.module, conditional_estimator.preprocessor)
        self.conditional_estimator = conditional_estimator
        self.state_shape = env.state_shape
        self.device = env.device
        self.States = env.States

    def forward(self, conditioning):
        if conditioning.ndim == 1:
            conditioning = conditioning.unsqueeze(0)
            batch_shape = (1,)
        elif conditioning.ndim == 2:
            batch_shape = (conditioning.shape[0],)
        elif conditioning.ndim == 3:
            batch_shape = (conditioning.shape[1],)
            conditioning = conditioning[0, :, :]
        else:
            raise ValueError(f"Unexpected conditioning tensor shape: {conditioning.shape}")

        dummy_states_tensor = torch.full(
            (batch_shape[0],) + self.state_shape,
            fill_value=-1,
            dtype=torch.long,
            device=self.device,
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

    module_logZ_state = MLP(input_dim=preprocessor.output_dim, output_dim=16, hidden_dim=16, n_hidden_layers=2)
    module_logZ_cond = MLP(input_dim=cond_dim, output_dim=16, hidden_dim=16, n_hidden_layers=2)
    module_logZ_final = MLP(input_dim=32, output_dim=1, hidden_dim=16, n_hidden_layers=2)

    conditional_logZ = ConditionalScalarEstimator(
        module_logZ_state, module_logZ_cond, module_logZ_final, preprocessor=preprocessor
    )
    logZ_estimator = ConditionalLogZWrapper(conditional_logZ, env)
    gflownet = TBGFlowNet(logZ=logZ_estimator, pf=pf_estimator, pb=pb_estimator)
    return gflownet


def build_conditional_pf_pb(env, preprocessor, args) -> Tuple[ConditionalDiscretePolicyEstimator, ConditionalDiscretePolicyEstimator]:

    CONCAT_SIZE = 64

    if args.arch == 'Transformer':

            module_PF = TransformerModel(
                    input_dim=preprocessor.output_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=CONCAT_SIZE,
                    n_layers=args.n_hidden,
                    n_head= 8
                    )

            module_PB = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=CONCAT_SIZE,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk= None,
            )

    else :
            module_PF = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=CONCAT_SIZE,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )

            module_PB = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=CONCAT_SIZE,
                hidden_dim=args.hidden_dim,
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

    return pf_estimator, pb_estimator


def build_conditional_logF_scalar_estimator(env, preprocessor) -> ConditionalScalarEstimator:
    """Build conditional log flow estimator.
    Args:
        env: The environment
        preprocessor: The preprocessor for the environment
    Returns:
        A conditional scalar estimator for log flow
    """

    CONCAT_SIZE = 64

    module_state_logF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        n_hidden_layers=1,
    )
    module_conditioning_logF = MLP(
        input_dim= 3,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        n_hidden_layers=1,
    )
    module_final_logF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=1,
        hidden_dim=256,
        n_hidden_layers=1,
    )
    logF_estimator = ConditionalScalarEstimator(
        module_state_logF,
        module_conditioning_logF,
        module_final_logF,
        preprocessor=preprocessor,
    )

    return logF_estimator

def build_subTB_gflownet(env, preprocessor, args, lamda=0.9):

    pf_estimator, pb_estimator = build_conditional_pf_pb(env, preprocessor, args)
    logF_estimator = build_conditional_logF_scalar_estimator(env, preprocessor)
    gflownet = SubTBGFlowNet(logF=logF_estimator, pf=pf_estimator, pb=pb_estimator, lamda=lamda)
    return gflownet, pf_estimator, pb_estimator


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
    preprocessor = CodonSequencePreprocessor(env.seq_length , embedding_dim=args.embedding_dim, device=device)

    logging.info(f"Protein sequence length: {len(config.protein_seq)}")
    logging.info("Building GFlowNet model...")

    gflownet, pf_estimator, pb_estimator = build_subTB_gflownet(env, preprocessor, args, lamda=args.subTB_lambda)

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)

    logging.info("Starting training loop...")
    start_time = time.time()

    training_result = train_conditional_gfn(args, env, gflownet, sampler, optimizer, scheduler, device)

    loss_history, reward_history, reward_components, unique_seqs, sampled_weights = training_result

    training_time = time.time() - start_time

    logging.info(f"Training completed in {training_time:.2f} seconds.")
    plot_training_curves(loss_history, reward_components)

    if config.wandb_project:

        logging.info("Logging training summary to WandB...")
        wandb.log(
            {
                "final_loss": loss_history[-1],
                "total_training_time": training_time,
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
            weights=torch.tensor([0.3, 0.3, 0.4], device=device),
            n_samples=args.n_samples,
            conditional=True,
            )

    inference_time = time.time() - start_inference_time
    avg_time_per_seq = inference_time / args.n_samples

    logging.info(f"Inference (sampling) completed in {inference_time:.2f} seconds.")
    logging.info(f"Average time per generated sequence is {avg_time_per_seq:.2f} seconds.")

    ################ Final means over the evaluation samples ###############

    eval_mean_gc = float(torch.tensor(gc_list).mean())
    eval_mean_mfe = float(torch.tensor(mfe_list).mean())
    eval_mean_cai = float(torch.tensor(cai_list).mean())

    logging.info("Saving trained model and metrics...")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    experiment_type = "conditional"
    protein_size = getattr(config, 'type', 'unknown')
    run_name = args.run_name if args.run_name else config.run_name
    output_dir = f"outputs/{experiment_type}/{protein_size}/{run_name}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    create_experiment_summary(args, config, output_dir, timestamp, experiment_type, protein_size, device)

    model_filename = f"{output_dir}/trained_gflownet_{run_name}_{timestamp}.pth"
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

    filename = f"{output_dir}/generated_sequences_{timestamp}.txt"

    logging.info(f"Saving generated sequences to {filename}")
    sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

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

    table = wandb.Table(columns=["Index", "Sequence", "Reward", "GC Content", "MFE", "CAI", "Label"])

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

    sequences = [seq for seq, _ in sorted_samples[:args.top_n]]
    distances = analyze_diversity(sequences, out_path=f"{output_dir}/edit_distance_distribution_{timestamp}.png")
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

            seq_indices = tokenize_sequence_to_tensor(seq).to(env.device)
            reward, _ = compute_simple_reward(
                seq_indices,
                env.codon_gc_counts,
                env.weights
            )
            total_reward += reward
        Eval_avg_reward = total_reward / len(samples)


    if config.wandb_project:

        logging.info("Logging evaluation metrics to WandB...")
        wandb.log(
            {
                "Pareto Plot": wandb.Image(f"{output_dir}/pareto_scatter_{timestamp}.png"),
                "Training_time": training_time,
                "Inference_time": inference_time,
                "Avg_time_per_sequence": avg_time_per_seq,
                "Reward Metric distributions": wandb.Image(f"{output_dir}/metric_distributions_{timestamp}.png"),
                "edit_distance_distribution": wandb.Image(f"{output_dir}/edit_distance_distribution_{timestamp}.png"),
                "mean_edit_distance": np.mean(distances),
                "std_edit_distance": np.std(distances),
                "Eval_mean_gc": eval_mean_gc,
                "Eval_mean_mfe": eval_mean_mfe,
                "Eval_mean_cai": eval_mean_cai,
                "Eval_avg_reward": Eval_avg_reward
            }
        )

    wandb.summary["final_loss"] = loss_history[-1]
    wandb.summary["unique_sequences"] = len(unique_seqs)

    return Eval_avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument('--arch', type=str, default='Transformer')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr_logz', type=float, default=0.01)

    parser.add_argument('--n_iterations', type=int, default=200)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--top_n', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--subTB_lambda', type=float, default=0.9)

    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=4)

    parser.add_argument('--tied', action='store_true')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--run_name', type=str, default='', help='Name for this experiment run')

    parser.add_argument("--config_path", type=str, default="config.yaml")

    args = parser.parse_args()
    set_seed(args.seed)
    config = load_config(args.config_path)

    main(args, config)
