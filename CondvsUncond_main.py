import sys
import os
import time
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

import torch
import numpy as np
import wandb

from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import train, train_conditional_gfn
from evaluate import evaluate, evaluate_conditional
from plots import *
from utils import *
from comparison import analyze_sequence_properties

from gfn.gflownet import TBGFlowNet, SubTBGFlowNet
from gfn.estimators import (
    DiscretePolicyEstimator,
    ConditionalDiscretePolicyEstimator,
    ScalarEstimator,
    ConditionalScalarEstimator,
)
from gfn.utils.modules import MLP
from gfn.samplers import Sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def build_unconditional_gflownet(env, preprocessor, args):
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
    module_logF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        trunk=module_PF.trunk if args.tied else None,
    )
    logF_estimator = ScalarEstimator(module=module_logF, preprocessor=preprocessor)
    gflownet = SubTBGFlowNet(
        pf=pf_estimator,
        pb=pb_estimator,
        logF=logF_estimator,
        weighting=args.subTB_weighting,
        lamda=args.subTB_lambda,
    )
    return gflownet

def build_conditional_gflownet(env, preprocessor, args):
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
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        trunk=module_PF.trunk if args.tied else None,
    )
    module_cond = MLP(
        input_dim=3,
        output_dim=CONCAT_SIZE,
        hidden_dim=args.hidden_dim,
    )
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
    module_state_logF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=1,
    )
    module_conditioning_logF = MLP(
        input_dim=3,
        output_dim=CONCAT_SIZE,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=1,
    )
    module_final_logF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=1,
    )
    logF_estimator = ConditionalScalarEstimator(
        module_state_logF,
        module_conditioning_logF,
        module_final_logF,
        preprocessor=preprocessor,
    )
    gflownet = SubTBGFlowNet(
        logF=logF_estimator,
        pf=pf_estimator,
        pb=pb_estimator,
        lamda=args.subTB_lambda,
    )
    return gflownet

def run_experiment(args, config, conditional: bool):
    device = torch.device("cuda") if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(env.seq_length, embedding_dim=args.embedding_dim, device=device)
    if conditional:
        gflownet = build_conditional_gflownet(env, preprocessor, args)
        sampler = Sampler(estimator=gflownet.pf)
        gflownet = gflownet.to(device)
        optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)
        loss_history, reward_history, reward_components, unique_seqs, sampled_weights = train_conditional_gfn(
            args, env, gflownet, sampler, optimizer, scheduler, device
        )
        eval_fn = evaluate_conditional
    else:
        gflownet = build_unconditional_gflownet(env, preprocessor, args)
        sampler = Sampler(estimator=gflownet.pf)
        gflownet = gflownet.to(device)
        optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)
        loss_history, reward_history, reward_components, unique_seqs = train(
            args, env, gflownet, sampler, optimizer, scheduler, device
        )
        eval_fn = evaluate

    # Evaluation
    with torch.no_grad():
        samples, gc_list, mfe_list, cai_list = eval_fn(
            env,
            sampler,
            weights=torch.tensor([0.3, 0.3, 0.4], device=device),
            n_samples=args.n_samples,
        )
    return {
        "loss_history": loss_history,
        "reward_history": reward_history,
        "gc_list": gc_list,
        "mfe_list": mfe_list,
        "cai_list": cai_list,
        "samples": samples,
        "unique_seqs": unique_seqs,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_iterations', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--subTB_lambda', type=float, default=0.8)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument('--tied', action='store_true')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--subTB_weighting', type=str, default="geometric_within")
    parser.add_argument('--wandb_project', type=str, default='mRNA_design')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config_path)

    # Unconditional
    logging.info("Training UNCONDITIONAL GFlowNet...")
    unc_results = run_experiment(args, config, conditional=False)
    # Conditional
    logging.info("Training CONDITIONAL GFlowNet...")
    cond_results = run_experiment(args, config, conditional=True)

    # Plot and compare
    plot_training_curves(unc_results["loss_history"], None, out_path="unc_training_curves.png")
    plot_training_curves(cond_results["loss_history"], None, out_path="cond_training_curves.png")
    plot_metric_histograms(unc_results["gc_list"], unc_results["mfe_list"], unc_results["cai_list"], out_path="unc_metric_distributions.png")
    plot_metric_histograms(cond_results["gc_list"], cond_results["mfe_list"], cond_results["cai_list"], out_path="cond_metric_distributions.png")

if __name__ == "__main__":
    main()