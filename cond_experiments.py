"""

Reorganized entrypoint for the mRNA GFlowNet experiments.
- Provides a clean CLI with subcommands: `run` and `sweep` (grid or random).
- Keeps model-building helpers (build_tb_gflownet / build_subTB_gflownet) and the training/evaluation wiring.
- `run` executes a single experiment and returns/saves the evaluation score.
- `sweep` runs multiple experiments (grid or random search) and writes CSV with results so you can choose the best parameters.

Usage examples (from shell):
    # Single run, overriding lr
    python mRNA_experiment.py run --config_path config.yaml --lr 0.01 --n_iterations 200

    # Grid sweep (provide param_grid as JSON string)
    python mRNA_experiment.py sweep --param_grid '{"lr": [0.01, 0.001], "subTB_lambda": [0.8, 0.9]}' --n_samples 50

    # Random sweep: sample 10 trials from the distributions you define
    python mRNA_experiment.py sweep --random_search --n_trials 10 --random_space '{"lr": [1e-4, 1e-1], "subTB_lambda": [0.5, 0.99]}' --n_samples 50

The script expects the helper modules you already have in your repo (env, preprocessor, train, evaluate, plots, utils, comparison, gfn.*).
Place this file in the same directory as your existing project and run from there.
"""

import sys
import os
import time
import json
import logging
import argparse
import itertools
import random
import copy
from datetime import datetime
from typing import Dict, Any, Tuple, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

import torch
import numpy as np
import wandb

# Project imports (assumes these modules are available in repo)
from comparison import analyze_sequence_properties
from utils import load_config, tokenize_sequence_to_tensor, analyze_diversity
from plots import (
    plot_training_curves,
    plot_of_weights_over_iterations,
    plot_ternary_plot_of_weights,
    plot_metric_histograms,
    plot_pareto_front,
    plot_cai_vs_mfe,
    plot_gc_vs_mfe,
)
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import train_conditional_gfn
from evaluate import evaluate_conditional

from gfn.gflownet import TBGFlowNet, SubTBGFlowNet
from gfn.estimators import (
    ConditionalDiscretePolicyEstimator,
    ConditionalScalarEstimator,
    ScalarEstimator,
)
from gfn.utils.modules import MLP
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States


# ----------------------------- Model helper wrappers -----------------------------
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
        # Normalize conditioning shapes (support 1D, 2D, or 3D as sampler may expand dims)
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


def build_conditional_pf_pb(env, preprocessor, concat_size=16, tied=False, hidden_dim=256, n_hidden=2) -> Tuple[ConditionalDiscretePolicyEstimator, ConditionalDiscretePolicyEstimator]:
    module_PF = MLP(input_dim=preprocessor.output_dim, output_dim=concat_size, hidden_dim=hidden_dim, n_hidden_layers=n_hidden)
    module_PB = MLP(input_dim=preprocessor.output_dim, output_dim=concat_size, hidden_dim=hidden_dim, n_hidden_layers=n_hidden, trunk=module_PF.trunk if tied else None)

    module_cond = MLP(input_dim=3, output_dim=concat_size, hidden_dim=hidden_dim)

    module_final_PF = MLP(input_dim=concat_size * 2, output_dim=env.n_actions)
    module_final_PB = MLP(input_dim=concat_size * 2, output_dim=env.n_actions - 1, trunk=module_final_PF.trunk if tied else None)

    pf_estimator = ConditionalDiscretePolicyEstimator(module_PF, module_cond, module_final_PF, env.n_actions, preprocessor=preprocessor, is_backward=False)
    pb_estimator = ConditionalDiscretePolicyEstimator(module_PB, module_cond, module_final_PB, env.n_actions, preprocessor=preprocessor, is_backward=True)

    return pf_estimator, pb_estimator


def build_conditional_logF_scalar_estimator(env, preprocessor) -> ConditionalScalarEstimator:
    CONCAT_SIZE = 16
    module_state_logF = MLP(input_dim=preprocessor.output_dim, output_dim=CONCAT_SIZE, hidden_dim=256, n_hidden_layers=1)
    module_conditioning_logF = MLP(input_dim=3, output_dim=CONCAT_SIZE, hidden_dim=256, n_hidden_layers=1)
    module_final_logF = MLP(input_dim=CONCAT_SIZE * 2, output_dim=1, hidden_dim=256, n_hidden_layers=1)

    logF_estimator = ConditionalScalarEstimator(module_state_logF, module_conditioning_logF, module_final_logF, preprocessor=preprocessor)
    return logF_estimator


def build_subTB_gflownet(env, preprocessor, lamda=0.9, tied=False):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env, preprocessor, tied=tied)
    logF_estimator = build_conditional_logF_scalar_estimator(env, preprocessor)
    gflownet = SubTBGFlowNet(logF=logF_estimator, pf=pf_estimator, pb=pb_estimator, lamda=lamda)
    return gflownet


# ----------------------------- Core experiment runner -----------------------------
def run_experiment(args: argparse.Namespace, config: Any, run_name: str = None) -> float:
    """Run a single experiment and return the evaluation average reward.

    This function encapsulates what used to be the __main__ body: building env, models, training,
    sampling and evaluation. It returns Eval_avg_reward so sweep code can compare runs.
    """

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Optional wandb
    if getattr(args, 'wandb_project', None):
        logging.info("Initializing Weights & Biases (wandb)")
        wandb.init(project=args.wandb_project, config=vars(args), name=run_name or args.run_name or None)

    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(env.seq_length, embedding_dim=args.embedding_dim, device=device)

    # Build GFlowNet (SubTB or TB)
    if getattr(args, 'use_tb', False):
        pf_estimator, pb_estimator = build_conditional_pf_pb(env, preprocessor, concat_size=16, tied=getattr(args, 'tied', False), hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        gflownet = build_tb_gflownet(env, pf_estimator, pb_estimator, preprocessor, cond_dim=3)
    else:
        gflownet = build_subTB_gflownet(env, preprocessor, lamda=args.subTB_lambda, tied=getattr(args, 'tied', False))

    sampler = Sampler(estimator=gflownet.pf if hasattr(gflownet, 'pf') else None)
    gflownet = gflownet.to(env.device)

    named_params = dict(gflownet.named_parameters())
    non_logz_params = [v for k, v in named_params.items() if k != 'logZ']
    logz_params = [named_params['logZ']] if 'logZ' in named_params else []

    params = [
        {"params": non_logz_params, "lr": args.lr},
        {"params": logz_params, "lr": args.lr_logz},
    ]
    optimizer = torch.optim.Adam(params)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)

    # Training
    logging.info("Starting training loop...")
    start_time = time.time()

    loss_history, reward_history, reward_components, unique_seqs, sampled_weights = train_conditional_gfn(
        args, env, gflownet, sampler, optimizer, scheduler, device
    )

    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds.")

    # Plotting (safe-guard: only call if functions exist)
    try:
        plot_training_curves(loss_history, reward_components)
        plot_of_weights_over_iterations(sampled_weights)
        plot_ternary_plot_of_weights(sampled_weights)
    except Exception:
        logging.exception("Plotting training curves failed (continuing)")

    # Evaluate (sampling)
    logging.info("Evaluating final model on sampled sequences...")
    start_inference_time = time.time()
    with torch.no_grad():
        samples, gc_list, mfe_list, cai_list = evaluate_conditional(
            env,
            sampler,
            weights=getattr(env, 'weights', [0.3, 0.3, 0.4]),
            n_samples=args.n_samples,
        )
    inference_time = time.time() - start_inference_time
    avg_time_per_seq = inference_time / max(1, args.n_samples)

    eval_mean_gc = float(torch.tensor(gc_list).mean())
    eval_mean_mfe = float(torch.tensor(mfe_list).mean())
    eval_mean_cai = float(torch.tensor(cai_list).mean())

    # Plot evaluation results
    try:
        plot_metric_histograms(gc_list, mfe_list, cai_list, out_path="metric_distributions.png")
        plot_pareto_front(gc_list, mfe_list, cai_list, out_path="pareto_scatter.png")
        plot_cai_vs_mfe(cai_list, mfe_list, out_path="cai_vs_mfe.png")
        plot_gc_vs_mfe(gc_list, mfe_list, out_path="gc_vs_mfe.png")
    except Exception:
        logging.exception("Plotting evaluation figures failed (continuing)")

    # Save generated sequences to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join("outputs_condi")
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f"generated_sequences_{timestamp}.txt")

    sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

    best_gc = max(samples.items(), key=lambda x: x[1][1][0])
    best_mfe = min(samples.items(), key=lambda x: x[1][1][1])
    best_cai = max(samples.items(), key=lambda x: x[1][1][2])

    with open(filename, 'w') as f:
        for i, (seq, reward) in enumerate(sorted_samples):
            f.write(f"Sequence {i+1}: {seq}, Reward: {reward[0]:.4f}, GC: {reward[1][0]:.4f}, MFE: {reward[1][1]:.4f}, CAI: {reward[1][2]:.4f}\n")

    # Small wandb logging (if enabled)
    if getattr(args, 'wandb_project', None):
        try:
            table = wandb.Table(columns=["Index", "Sequence", "Reward", "GC Content", "MFE", "CAI", "Label"])
            for i, (seq, reward) in enumerate(sorted_samples[:5]):
                table.add_data(i+1, seq, reward[0], reward[1][0], reward[1][1], reward[1][2], "Pareto Optimal")
            wandb.log({"Top_Sequences": table})
        except Exception:
            logging.exception("WandB logging failed (continuing)")

    sequences = [seq for seq, _ in sorted_samples[:args.top_n]]
    distances = analyze_diversity(sequences)

    generated_sequences_tensor = [tokenize_sequence_to_tensor(seq) for seq in sequences]
    additional_best_seqs = [best_gc[0], best_mfe[0], best_cai[0]]
    for s in additional_best_seqs:
        generated_sequences_tensor.append(tokenize_sequence_to_tensor(s))

    natural_tensor = tokenize_sequence_to_tensor(config.natural_mRNA_seq)
    sequence_labels = [f"Gen {i+1}" for i in range(min(len(sequences), args.top_n))] + ["Best GC", "Best MFE", "Best CAI"]

    try:
        analyze_sequence_properties(generated_sequences_tensor, natural_tensor, labels=sequence_labels)
    except Exception:
        logging.exception("Sequence property analysis failed (continuing)")

    Eval_avg_reward = sum(w * r for w, r in zip(getattr(env, 'weights', [0.3, -1.0, 0.4]), [eval_mean_gc, -eval_mean_mfe, eval_mean_cai]))

    # Final wandb summary
    if getattr(args, 'wandb_project', None):
        try:
            wandb.summary['final_loss'] = loss_history[-1] if len(loss_history) else None
            wandb.summary['unique_sequences'] = len(unique_seqs)
        except Exception:
            pass
        wandb.finish()

    # Return the main scalar for comparison in sweeps
    return float(Eval_avg_reward)


# ----------------------------- Sweep helpers -----------------------------

def _update_args_from_pair(args: argparse.Namespace, key: str, value: Any) -> argparse.Namespace:
    new_args = copy.deepcopy(args)
    # try to cast to existing type if exists
    if hasattr(args, key):
        old_val = getattr(args, key)
        try:
            new_val = type(old_val)(value)
        except Exception:
            new_val = value
    else:
        new_val = value
    setattr(new_args, key, new_val)
    return new_args


def run_grid_search(base_args: argparse.Namespace, base_config: Any, param_grid: Dict[str, List[Any]], out_csv: str) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    results = []
    logging.info(f"Running grid search: {len(combos)} combinations")

    for idx, combo in enumerate(combos, start=1):
        param_dict = dict(zip(keys, combo))
        run_name = f"grid_{idx}__" + "__".j