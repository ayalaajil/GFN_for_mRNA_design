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
from DeepArchi import *
import logging
import torch
import argparse
import numpy as np
import time
import wandb
from comparison import analyze_sequence_properties
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

    if args.wandb_project:

        logging.info("Initializing Weights & Biases...")
        wandb.init(project= args.wandb_project,config={**vars(config), **vars(args)},name= args.run_name)

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

    model_filename = f"trained_gflownet_{args.run_name}_{timestamp}.pth"
    torch.save(
        {
            "model_state": gflownet.state_dict(),
            "training_history": {"loss": loss_history, "reward": reward_history},
        },
        model_filename,
    )

    logging.info("Plotting final metric histograms and Pareto front...")

    plot_metric_histograms(gc_list, mfe_list, cai_list, out_path=f"outputs_{config.type}/metric_distributions_{timestamp}.png")
    plot_pareto_front(gc_list, mfe_list, cai_list, out_path=f"outputs_{config.type}/pareto_scatter_{timestamp}.png")
    plot_cai_vs_mfe(cai_list, mfe_list, out_path = f"outputs_{config.type}/cai_vs_mfe_{timestamp}.png")
    plot_gc_vs_mfe(gc_list, mfe_list, out_path= f"outputs_{config.type}/gc_vs_mfe_{timestamp}.png")

    filename = f"outputs_{config.type}/generated_sequences_{timestamp}.txt"

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

    Eval_avg_reward = sum(
        w * r
        for w, r in zip(env.weights, [eval_mean_gc, -eval_mean_mfe, eval_mean_cai])
    )

    if args.wandb_project:

        logging.info("Logging evaluation metrics to WandB...")
        wandb.log(
            {
                "Pareto Plot": wandb.Image(f"outputs_{config.type}/pareto_scatter_{timestamp}.png"),
                "Training_time": total_time,
                "Inference_time": inference_time,
                "Avg_time_per_sequence": avg_time_per_seq,
                "Reward Metric distributions": wandb.Image(f"outputs_{config.type}/metric_distributions_{timestamp}.png"),
                "edit_distance_distribution": wandb.Image(
                    "edit_distance_distribution.png"
                ),
                "CAI vs MFE": wandb.Image( f"outputs_{config.type}/cai_vs_mfe_{timestamp}.png"),
                "GC vs MFE": wandb.Image(f"outputs_{config.type}/gc_vs_mfe_{timestamp}.png"),
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

    wandb.finish()

    return Eval_avg_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--wandb_project", type=str, default="med_seq_ablation_study", help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default="", help="Name for the wandb run")
    args = parser.parse_args()


    config = load_config(args.config_path)

    hyperparams = {
        #"arch": ["MLP", "MLP_EHH", "Transformer"],
        "arch": ["Transformer"],
        "subTB_lambda": [0.8, 0.9, 0.99],
        "batch_size": [16, 32, 64],
        "lr": [0.0005, 0.001, 0.005],
        "hidden_dim": [64, 128, 256]
    }

    param_grid = list(itertools.product(
        hyperparams["arch"],
        hyperparams["subTB_lambda"],
        hyperparams["batch_size"],
        hyperparams["lr"],
        hyperparams["hidden_dim"]
    ))

    for arch, lm, bs, lr, hd in param_grid:

        run_name = (
            f"{arch}_"
            f"lambda{lm}_"
            f"bs{bs}_"
            f"lr{lr}_"
            f"hd{hd}"
        )

        print(f"\n=== Running: {run_name} ===")

        class DummyArgs: pass
        run_args = DummyArgs()
        run_args.__dict__.update({
            "batch_size": bs,
            "lr": lr,
            "lr_logz": 0.1,
            "embedding_dim": 64,
            "hidden_dim": hd,
            "n_hidden": 3,
            "n_iterations": 500,
            "n_samples": 100,
            "subTB_weighting": "geometric_within",
            "subTB_lambda": lm,
            "epsilon": 0.25,
            "tied": False,
            "clip_grad_norm": 1.0,
            "lr_patience": 10,

            # wandb logging
            "wandb_project": args.wandb_project or config.wandb_project,
            "run_name": run_name,

        })

        main(run_args, config)















