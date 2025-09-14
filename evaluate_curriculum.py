import sys
import os
import argparse
import logging
from datetime import datetime
import torch
import numpy as np
import wandb
from tqdm import tqdm

# Adjust this path if your project structure is different
sys.path.insert(0, os.path.dirname(__file__))

# Import necessary components from your existing project
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor2 # Using the preprocessor from curriculum trainer
from main_conditional import build_subTB_gflownet, load_config
from reward import compute_simple_reward
from utils import *
from plots import *
from enhanced_comparison import run_comprehensive_analysis
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

    # Reconstruct the model architecture to load the state_dict
    gflownet, _, _ = build_subTB_gflownet(env, preprocessor, args, lamda=args.subTB_lambda)

    # Load the saved state dictionary
    checkpoint = torch.load(model_path, map_location=env.device, weights_only=False)

    # Check for keys in the checkpoint
    if 'model_state_dict' in checkpoint:
        gflownet.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint: # Matches the saving key in your curriculum trainer
        gflownet.load_state_dict(checkpoint['model_state'])
    else:
        raise KeyError("Could not find a valid model state dictionary in the checkpoint file.")

    gflownet.to(env.device)
    gflownet.eval() # Set the model to evaluation mode
    logging.info("Model loaded successfully and set to evaluation mode.")
    return gflownet

def evaluate_model_on_task(
    gflownet: torch.nn.Module,
    protein_seq: str,
    weights: torch.Tensor,
    args,
    config,
    device: torch.device,
) -> dict:
    """
    Evaluates the loaded GFlowNet model on a single protein sequence task.
    This function generates samples, analyzes them, and returns a dictionary of results.
    """
    logging.info(f"Evaluating on protein of length {len(protein_seq)}...")

    # 1. Setup Environment and Sampler for the specific task
    env = CodonDesignEnv(protein_seq=protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor2(len(protein_seq) + 50, embedding_dim=args.embedding_dim, device=device)
    sampler = Sampler(estimator=gflownet.pf)

    # 2. Generate Samples
    conditioning = weights.unsqueeze(0).expand(args.n_samples, *weights.shape)

    with torch.no_grad():
        trajectories = sampler.sample_trajectories(
            env, n=args.n_samples, conditioning=conditioning
        )
        final_states = trajectories.terminating_states.tensor

    # 3. Process and Analyze Samples
    samples = {}
    gc_list, mfe_list, cai_list = [], [], []

    logging.info(f"Processing {len(final_states)} generated sequences...")
    for state in tqdm(final_states, desc="Analyzing sequences"):
        seq_str = "".join([env.idx_to_codon[idx.item()] for idx in state if idx != -1])
        if not seq_str:
            continue

        reward, components = compute_simple_reward(state, env.codon_gc_counts, weights)

        # Store comprehensive data
        samples[seq_str] = (reward, components)
        gc_list.append(components[0])
        mfe_list.append(components[1])
        cai_list.append(components[2])

    logging.info(f"Successfully processed {len(samples)} unique sequences.")

    # Return a dictionary of results for this task
    return {
        "samples": samples,
        "gc_list": gc_list,
        "mfe_list": mfe_list,
        "cai_list": cai_list,
        "protein_length": len(protein_seq),
    }

def main(args, config):
    device = torch.device("cuda") if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")

    # --- Setup: Load a dummy environment to build the model ---
    # The protein sequence doesn't matter here, it's just to initialize the model structure.
    dummy_protein = "A" * 20
    dummy_env = CodonDesignEnv(protein_seq=dummy_protein, device=device)
    dummy_preprocessor = CodonSequencePreprocessor2(250, embedding_dim=args.embedding_dim, device=device)

    # --- Load Model ---
    gflownet = load_curriculum_model(args.model_path, dummy_env, dummy_preprocessor, args)

    # --- Define Evaluation Tasks ---
    # This is where you test your model's capabilities.
    # Include proteins of lengths seen during training AND some it has NOT seen.
    evaluation_proteins = {
        "short_seen": "MINTQDSSILPLSNCPQLQCCRHIVPGPLWCS*", # Length 32 (in curriculum)
        "medium_unseen": "MKLVRFLMKLSHETVTIELKNGTQVHGTITGVDVSMNTHLKAVKMTLKNREPVQLETLSIRGNNIRYFILPDSLPLDTLLVDVEPKVKSKKREAVAGRGRGRGRGRGRGRGRGRGGPRR*", # Length 120 (unseen)
        "long_seen": "MGASARLLRAVIMGAPGSGKGTVSSRITTHFELKHLSSGDLLRDNMLRGTEIGVLAKAFIDQGKLIPDDVMTRLALHELKNLTQYSWLLDGFPRTLPQAEALDRAYQIDTVINLNVPFEVIKQRLTARWIHPASGRVYNIEFNPPKTVGIDDLTGEPLIQREDDKPETVIKRLKAYEDQTKPVLEYYQKKGVLETFSGTETNKIWPYVYAFLQTKVPQRSQKASVTP*" # Length 228 (in curriculum)
    }

    natural_mRNA_sequences = {
        "short_seen": "AUGAUAAACACCCAGGACAGUAGUAUUUUGCCUUUGAGUAACUGUCCCCAGCUCCAGUGCUGCAGGCACAUUGUUCCAGGGCCUCUGUGGUGCUCCUAA",
        "medium_unseen": "AUGAAGCUCGUGAGAUUUUUGAUGAAAUUGAGUCAUGAAACUGUAACCAUUGAAUUGAAGAACGGAACACAGGUCCAUGGAACAAUCACAGGUGUGGAUGUCAGCAUGAAUACACAUCUUAAAGCUGUGAAAAUGACCCUGAAGAACAGAGAACCUGUACAGCUGGAAACGCUGAGUAUUCGAGGAAAUAACAUUCGGUAUUUUAUUCUACCAGACAGUUUACCUCUGGAUACACUACUUGUGGAUGUUGAACCUAAGGUGAAAUCUAAGAAAAGGGAAGCUGUUGCAGGAAGAGGCAGAGGAAGAGGAAGAGGAAGAGGACGUGGCCGUGGCAGAGGAAGAGGGGGUCCUAGGCGAUAA",
        "long_seen": "AUGGGGGCGUCCGCGCGGCUGCUGCGAGCGGUGAUCAUGGGGGCCCCGGGCUCGGGCAAGGGCACCGUGUCGUCGCGCAUCACUACACACUUCGAGCUGAAGCACCUCUCCAGCGGGGACCUGCUCCGGGACAACAUGCUGCGGGGCACAGAAAUUGGCGUGUUAGCCAAGGCUUUCAUUGACCAAGGGAAACUCAUCCCAGAUGAUGUCAUGACUCGGCUGGCCCUUCAUGAGCUGAAAAAUCUCACCCAGUAUAGCUGGCUGUUGGAUGGUUUUCCAAGGACACUUCCACAGGCAGAAGCCCUAGAUAGAGCUUAUCAGAUCGACACAGUGAUUAACCUGAAUGUGCCCUUUGAGGUCAUUAAACAACGCCUUACUGCUCGCUGGAUUCAUCCCGCCAGUGGCCGAGUCUAUAACAUUGAAUUCAACCCUCCCAAAACUGUGGGCAUUGAUGACCUGACUGGGGAGCCUCUCAUUCAGCGUGAGGAUGAUAAACCAGAGACGGUUAUCAAGAGACUAAAGGCUUAUGAAGACCAAACAAAGCCAGUCCUGGAAUAUUACCAGAAAAAAGGGGUGCUGGAAACAUUCUCCGGAACAGAAACCAACAAGAUUUGGCCCUAUGUAUAUGCUUUCCUACAAACUAAAGUUCCACAAAGAAGCCAGAAAGCUUCAGUUACUCCAUGA",
    }

    # Use the specific protein from your baseline for direct comparison
    baseline_protein_seq = config.protein_seq
    evaluation_proteins["baseline_comparison"] = baseline_protein_seq

    # --- W&B Setup ---
    if config.wandb_project:
        run_name = args.run_name or f"eval_curriculum_{datetime.now().strftime('%Y%m%d_%H%M')}"
        wandb.init(
            project=config.wandb_project,
            config={**vars(args), **vars(config)},
            name=run_name,
            group="Curriculum Evaluation",
            tags=['evaluation', 'curriculum']
        )

    # --- Run Evaluation for each task ---
    for task_name, protein_seq in evaluation_proteins.items():

        logging.info(f"\n{'='*20} Starting Evaluation for Task: {task_name} {'='*20}")

        # Define the weights for this evaluation run
        eval_weights = torch.tensor([0.3, 0.3, 0.4], device=device)

        # Get evaluation results
        task_results = evaluate_model_on_task(gflownet, protein_seq, eval_weights, args, config, device)

        # Create output directory for this specific task
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = f"outputs/curriculum_evaluation/{task_name}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Run the same comprehensive analysis as your baseline
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

        # Log results to WandB, namespaced by task
        if config.wandb_project:
            logging.info(f"Logging results for task '{task_name}' to WandB...")

            # Create a summary table for top sequences
            table = wandb.Table(columns=["Sequence", "Reward", "GC", "MFE", "CAI"])
            sorted_samples = sorted(task_results['samples'].items(), key=lambda x: x[1][0], reverse=True)
            for seq, (reward, components) in sorted_samples[:10]:
                table.add_data(seq, reward, components[0], components[1], components[2])

            wandb.log({
                f"{task_name}/protein_length": task_results["protein_length"],
                f"{task_name}/Top_Sequences": table,
                f"{task_name}/pareto_plot": wandb.Image(analysis_results['plot_path']),
                f"{task_name}/reward_mean": analysis_results['quality_metrics']['reward_stats']['mean'],
                f"{task_name}/reward_std": analysis_results['quality_metrics']['reward_stats']['std'],
                f"{task_name}/pareto_efficiency": analysis_results['quality_metrics']['pareto_efficiency'],
                f"{task_name}/diversity_mean_edit_distance": analysis_results['diversity_metrics']['mean_edit_distance'],
                f"{task_name}/uniqueness_ratio": analysis_results['diversity_metrics']['uniqueness_ratio'],
            })

    if config.wandb_project:
        wandb.finish()

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Curriculum Learning GFlowNet model.")

    # --- Essential Arguments ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .pth model file from curriculum training.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the configuration file (for protein sequences, wandb project, etc.).")
    parser.add_argument('--run_name', type=str, default='', help='Name for this evaluation run in WandB.')

    # --- Model Architecture Arguments (MUST MATCH TRAINING) ---
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=4)
    parser.add_argument('--subTB_lambda', type=float, default=0.9)
    parser.add_argument('--arch', type=str, default='Transformer', help="Model architecture used during training (e.g., MLP, Transformer).")
    parser.add_argument('--tied', action='store_true', help="Whether the policy networks were tied during training.")

    # --- Evaluation Parameters ---
    parser.add_argument('--n_samples', type=int, default=200, help="Number of sequences to generate for evaluation.")
    parser.add_argument('--top_n', type=int, default=50, help="Number of top sequences to use for diversity analysis.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA, run on CPU.")

    args = parser.parse_args()
    config = load_config(args.config_path)

    main(args, config)