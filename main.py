import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import *
from evaluate import evaluate
from plots import *
from utils import *
from reward import compute_simple_reward
from DeepArchi import *
from datetime import datetime
import logging
import torch
import argparse
import numpy as np
import time
import wandb
from comparison import analyze_sequence_properties
from enhanced_comparison import run_comprehensive_analysis
from simple_generalization import run_simple_generalization_tests
from torchgfn.src.gfn.gflownet import SubTBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator, ScalarEstimator
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
    experiment_type = "unconditional"
    protein_size = getattr(config, 'type', 'unknown')
    run_name = args.run_name if args.run_name else config.run_name
    output_dir = f"outputs/{experiment_type}/{protein_size}/{run_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    plot_training_curves(loss_history, reward_components, out_path=f"{output_dir}/training_curves_{timestamp}.png")

    # Create experiment summary file
    create_experiment_summary(args, config, output_dir, timestamp, experiment_type, protein_size, device)

    model_filename = f"{output_dir}/trained_gflownet_{run_name}_{timestamp}.pth"


    torch.save({"model_state": gflownet.state_dict(),"training_history": {"loss": loss_history, "reward": reward_history},},model_filename,)

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
    sequences = [seq for seq, _ in sorted_samples[:args.top_n]]
    distances = analyze_diversity(sequences, out_path=f"{output_dir}/edit_distance_distribution_{timestamp}.png")

    generated_sequences_tensor = [tokenize_sequence_to_tensor(seq) for seq in sequences]
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


    logging.info("Running comprehensive analysis with diversity and quality metrics...")
    analysis_results = run_comprehensive_analysis(
        samples, gc_list, mfe_list, cai_list, config.natural_mRNA_seq,
        output_dir, timestamp, top_n=args.top_n
    )

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
                "Enhanced Diversity Analysis": wandb.Image(analysis_results['plot_path']),
            }
        )

    wandb.summary["final_loss"] = loss_history[-1]
    wandb.summary["unique_sequences"] = len(unique_seqs)

    if args.run_generalization_tests:

        logging.info("Running generalization tests...")
        gen_results = run_simple_generalization_tests(
            env, sampler, device,
            model_type="unconditional",
            n_samples=args.generalization_n_samples,
            output_dir=f"{output_dir}/generalization_tests"
        )

        if config.wandb_project and gen_results:
            gc_means = [r['stats']['gc_mean'] for r in gen_results.values()]
            mfe_means = [r['stats']['mfe_mean'] for r in gen_results.values()]
            cai_means = [r['stats']['cai_mean'] for r in gen_results.values()]
            reward_means = [r['stats']['reward_mean'] for r in gen_results.values()]

            wandb.log({
                "gen_avg_gc": np.mean(gc_means),
                "gen_avg_mfe": np.mean(mfe_means),
                "gen_avg_cai": np.mean(cai_means),
                "gen_avg_reward": np.mean(reward_means),
                "gen_n_configs": len(gen_results),
            })

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
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--subTB_lambda', type=float, default=0.9)
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within", help="weighting scheme for SubTB")

    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=4)

    parser.add_argument('--tied', action='store_true')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_patience', type=int, default=10)

    parser.add_argument('--wandb_project', type=str, default='mRNA_design')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--conditional', action='store_true', help='Run conditional GFlowNet experiment')
    parser.add_argument("--config_path", type=str, default="config.yaml")

    parser.add_argument("--run_generalization_tests", action="store_true", help="Run generalization tests")
    parser.add_argument("--generalization_n_samples", type=int, default=30, help="Number of samples to generate for generalization tests")

    args = parser.parse_args()
    set_seed(args.seed)
    config = load_config(args.config_path)

    main(args, config)