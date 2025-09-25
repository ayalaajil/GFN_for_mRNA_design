"""
Evaluate a trained Curriculum Learning GFlowNet model on a set of protein sequences.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import torch
import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor2
from main_conditional import build_subTB_gflownet
from reward import compute_simple_reward
from utils import *
from plots import *
from comparison_utils import run_comprehensive_analysis
from gfn.samplers import Sampler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_curriculum_model(model_path: str, env: CodonDesignEnv, preprocessor, args) -> torch.nn.Module:
    """
    Loads a GFlowNet model saved from the curriculum trainer.
    """
    logging.info(f"Loading model from {model_path}...")
    gflownet, _, _ = build_subTB_gflownet(env, preprocessor, args, lamda=args.subTB_lambda)
    checkpoint = torch.load(model_path, map_location=env.device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        gflownet.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        gflownet.load_state_dict(checkpoint['model_state'])
    else:
        raise KeyError("Could not find a valid model state dictionary in the checkpoint file.")

    gflownet.to(env.device)
    gflownet.eval()
    logging.info("Model loaded successfully and set to evaluation mode.")
    return gflownet

def evaluate_model_on_task(
    gflownet: torch.nn.Module,
    protein_seq: str,
    weights: torch.Tensor,
    args,
    device: torch.device,
) -> dict:
    """
    Evaluates the loaded GFlowNet model on a single protein sequence task.
    This function generates samples, analyzes them, and returns a dictionary of results.
    """
    logging.info(f"Evaluating on protein of length {len(protein_seq)}...")

    env = CodonDesignEnv(protein_seq=protein_seq, device=device)
    sampler = Sampler(estimator=gflownet.pf)
    conditioning = weights.unsqueeze(0).expand(args.n_samples, *weights.shape)

    with torch.no_grad():
        trajectories = sampler.sample_trajectories(
            env, n=args.n_samples, conditioning=conditioning
        )
        final_states = trajectories.terminating_states.tensor

    samples = {}
    gc_list, mfe_list, cai_list = [], [], []

    logging.info(f"Processing {len(final_states)} generated sequences...")

    for state in tqdm(final_states, desc="Analyzing sequences"):
        seq_str = "".join([env.idx_to_codon[idx.item()] for idx in state if idx != -1])
        if not seq_str:
            continue

        reward, components = compute_simple_reward(state, env.codon_gc_counts, weights)

        samples[seq_str] = (reward, components)
        gc_list.append(components[0])
        mfe_list.append(components[1])
        cai_list.append(components[2])

    logging.info(f"Successfully processed {len(samples)} unique sequences.")

    return {
        "samples": samples,
        "gc_list": gc_list,
        "mfe_list": mfe_list,
        "cai_list": cai_list,
        "protein_length": len(protein_seq),
    }

def main(args):

    initial_time = datetime.now()
    device = torch.device("cuda") if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")

    # --- Setup: Load a dummy environment to build the model ---
    # The protein sequence doesn't matter here, it's just to initialize the model structure.
    dummy_protein = "A" * 20
    dummy_env = CodonDesignEnv(protein_seq=dummy_protein, device=device)
    dummy_preprocessor = CodonSequencePreprocessor2(250, embedding_dim=args.embedding_dim, device=device)

    gflownet = load_curriculum_model(args.model_path, dummy_env, dummy_preprocessor, args)

    if args.protein_sequences:
        evaluation_proteins = {}
        for i in range(len(args.protein_sequences)):
            seq = args.protein_sequences[i]
            print(f"Protein sequence: {seq}")
            evaluation_proteins[f"custom_{i}"] = seq
    else:
        evaluation_proteins = {
            "short_seen": "MINTQDSSILPLSNCPQLQCCRHIVPGPLWCS*", # Length 32 (in curriculum)
            "medium_unseen": "MKLVRFLMKLSHETVTIELKNGTQVHGTITGVDVSMNTHLKAVKMTLKNREPVQLETLSIRGNNIRYFILPDSLPLDTLLVDVEPKVKSKKREAVAGRGRGRGRGRGRGRGRGRGGPRR*", # Length 120 (unseen)
            "long_seen": "MGASARLLRAVIMGAPGSGKGTVSSRITTHFELKHLSSGDLLRDNMLRGTEIGVLAKAFIDQGKLIPDDVMTRLALHELKNLTQYSWLLDGFPRTLPQAEALDRAYQIDTVINLNVPFEVIKQRLTARWIHPASGRVYNIEFNPPKTVGIDDLTGEPLIQREDDKPETVIKRLKAYEDQTKPVLEYYQKKGVLETFSGTETNKIWPYVYAFLQTKVPQRSQKASVTP*" # Length 228 (in curriculum)
        }

    if args.natural_mRNA_sequences:
        natural_mRNA_sequences = {}
        for i in range(len(args.natural_mRNA_sequences)):
            seq = args.natural_mRNA_sequences[i]
            print(f"Natural mRNA sequence: {seq}")
            natural_mRNA_sequences[f"custom_{i}"] = seq
    else:
        natural_mRNA_sequences = {
        "short_seen": "AUGAUAAACACCCAGGACAGUAGUAUUUUGCCUUUGAGUAACUGUCCCCAGCUCCAGUGCUGCAGGCACAUUGUUCCAGGGCCUCUGUGGUGCUCCUAA",
        "medium_unseen": "AUGAAGCUCGUGAGAUUUUUGAUGAAAUUGAGUCAUGAAACUGUAACCAUUGAAUUGAAGAACGGAACACAGGUCCAUGGAACAAUCACAGGUGUGGAUGUCAGCAUGAAUACACAUCUUAAAGCUGUGAAAAUGACCCUGAAGAACAGAGAACCUGUACAGCUGGAAACGCUGAGUAUUCGAGGAAAUAACAUUCGGUAUUUUAUUCUACCAGACAGUUUACCUCUGGAUACACUACUUGUGGAUGUUGAACCUAAGGUGAAAUCUAAGAAAAGGGAAGCUGUUGCAGGAAGAGGCAGAGGAAGAGGAAGAGGAAGAGGACGUGGCCGUGGCAGAGGAAGAGGGGGUCCUAGGCGAUAA",
        "long_seen": "AUGGGGGCGUCCGCGCGGCUGCUGCGAGCGGUGAUCAUGGGGGCCCCGGGCUCGGGCAAGGGCACCGUGUCGUCGCGCAUCACUACACACUUCGAGCUGAAGCACCUCUCCAGCGGGGACCUGCUCCGGGACAACAUGCUGCGGGGCACAGAAAUUGGCGUGUUAGCCAAGGCUUUCAUUGACCAAGGGAAACUCAUCCCAGAUGAUGUCAUGACUCGGCUGGCCCUUCAUGAGCUGAAAAAUCUCACCCAGUAUAGCUGGCUGUUGGAUGGUUUUCCAAGGACACUUCCACAGGCAGAAGCCCUAGAUAGAGCUUAUCAGAUCGACACAGUGAUUAACCUGAAUGUGCCCUUUGAGGUCAUUAAACAACGCCUUACUGCUCGCUGGAUUCAUCCCGCCAGUGGCCGAGUCUAUAACAUUGAAUUCAACCCUCCCAAAACUGUGGGCAUUGAUGACCUGACUGGGGAGCCUCUCAUUCAGCGUGAGGAUGAUAAACCAGAGACGGUUAUCAAGAGACUAAAGGCUUAUGAAGACCAAACAAAGCCAGUCCUGGAAUAUUACCAGAAAAAAGGGGUGCUGGAAACAUUCUCCGGAACAGAAACCAACAAGAUUUGGCCCUAUGUAUAUGCUUUCCUACAAACUAAAGUUCCACAAAGAAGCCAGAAAGCUUCAGUUACUCCAUGA",
    }

    if args.wandb_project:
        run_name = args.run_name or f"eval_curriculum_{datetime.now().strftime('%Y%m%d_%H%M')}"
        wandb.init(
            project=args.wandb_project,
            config={**vars(args)},
            name=run_name,
            group="Curriculum Evaluation",
            tags=['evaluation', 'curriculum']
        )

    for task_name, protein_seq in evaluation_proteins.items():

        logging.info(f"\n{'='*20} Starting Evaluation for Task: {task_name} {'='*20}")
        logging.info(f"Protein sequence: {protein_seq}")

        eval_weights = torch.tensor([0.3, 0.3, 0.4], device=device)  # weights for the reward function

        task_results = evaluate_model_on_task(gflownet, protein_seq, eval_weights, args, device)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        output_dir = f"EXPERIMENTS/PAPERS/CAGFN/{task_name}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)


        logging.info("Running comprehensive analysis...")
        analysis_results = run_comprehensive_analysis(
            task_results["samples"],
            task_results["gc_list"],
            task_results["mfe_list"],
            task_results["cai_list"],
            natural_mRNA_sequences[task_name],
            output_dir,
            timestamp,
            top_n=args.top_n
        )

        end_time = datetime.now()
        time_taken = end_time - initial_time
        if args.wandb_project:
            logging.info(f"Logging results for task '{task_name}' to WandB...")

            table = wandb.Table(columns=["Sequence", "Reward", "GC", "MFE", "CAI"])
            sorted_samples = sorted(task_results['samples'].items(), key=lambda x: x[1][0], reverse=True)
            for seq, (reward, components) in sorted_samples[:10]:
                table.add_data(seq, reward, components[0], components[1], components[2])

            wandb.log({
                f"{task_name}/protein_length": task_results["protein_length"],
                f"{task_name}/time_taken": time_taken.total_seconds(),
                f"{task_name}/Top_Sequences": table,
                f"{task_name}/pareto_plot": wandb.Image(analysis_results['plot_path']),
                f"{task_name}/reward_mean": analysis_results['quality_metrics']['reward_stats']['mean'],
                f"{task_name}/reward_std": analysis_results['quality_metrics']['reward_stats']['std'],
                f"{task_name}/pareto_efficiency": analysis_results['quality_metrics']['pareto_efficiency'],
                f"{task_name}/diversity_mean_edit_distance": analysis_results['diversity_metrics']['mean_edit_distance'],
                f"{task_name}/uniqueness_ratio": analysis_results['diversity_metrics']['uniqueness_ratio'],
            })

    if args.wandb_project:
        wandb.finish()

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Curriculum Learning GFlowNet model.")

    # --- Essential Arguments ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .pth model file from curriculum training.")
    parser.add_argument('--run_name', type=str, default='', help='Name for this evaluation run in WandB.')
    parser.add_argument('--wandb_project', type=str, default='EVALUATION_Experiments', help='WandB project name.')

    # --- Custom Protein and mRNA Sequences ---
    parser.add_argument('--protein_sequences', nargs='*', help='Custom protein sequences to evaluate (space-separated). If not provided, uses default sequences.')
    parser.add_argument('--natural_mRNA_sequences', nargs='*', help='Custom natural mRNA sequences to evaluate (space-separated). If not provided, uses default sequences.')

    # --- Model Architecture Arguments ---
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=4)
    parser.add_argument('--subTB_lambda', type=float, default=0.9)
    parser.add_argument('--arch', type=str, default='Transformer', help="Model architecture used during training (e.g., MLP, Transformer).")
    parser.add_argument('--tied', action='store_true', help="Whether the policy networks were tied during training.")

    # --- Evaluation Parameters ---
    parser.add_argument('--n_samples', type=int, default=100, help="Number of sequences to generate for evaluation.")
    parser.add_argument('--top_n', type=int, default=50, help="Number of top sequences to use for diversity analysis.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA, run on CPU.")

    args = parser.parse_args()

    main(args)