import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

import torch
import torch.nn as nn
from torchgfn.src.gfn.utils.modules import MLP
from torchgfn.src.gfn.modules import DiscretePolicyEstimator
import logging
from typing import Dict, Any, List, Optional
import wandb
import json
from datetime import datetime

from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from train import train
from evaluate import evaluate
from torchgfn.src.gfn.gflownet import TBGFlowNet
from torchgfn.src.gfn.samplers import Sampler


class ResidualMLP(nn.Module):
    """MLP with residual connections"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_hidden_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_hidden_layers):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            self.blocks.append(block)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.trunk = self

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            residual = x
            x = block(x)
            x = torch.relu(x + residual)
        return self.output_layer(x)

class AttentionMLP(nn.Module):
    """MLP with self-attention mechanism"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_hidden_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        layers = []
        for i in range(n_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.trunk = self

    def forward(self, x):
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_att, _ = self.attention(x, x, x)
        x = x + x_att

        if x.size(1) == 1:
            x = x.squeeze(1)

        x = self.mlp(x)
        return self.output_layer(x)

class ConvolutionalMLP(nn.Module):
    """MLP with 1D convolutions for sequence modeling"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_hidden_layers=2,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        layers = []
        for i in range(n_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.trunk = self

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            pass
        else:
            x = x.view(x.size(0), -1).unsqueeze(-1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = self.global_pool(x).squeeze(-1)

        x = self.mlp(x)
        return self.output_layer(x)

class DropoutMLP(nn.Module):
    """Standard MLP with heavy dropout for regularization"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_hidden_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        for i in range(n_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.trunk = self.network[:-1]

    def forward(self, x):
        return self.network(x)

# configurations
ARCHITECTURE_CONFIGS = {
    "standard_mlp": {
        "class": MLP,
        "params": {
            "hidden_dim": 512,
            "n_hidden_layers": 3
        },
        "description": "Standard MLP from torchgfn"
    },
    "deep_mlp": {
        "class": MLP,
        "params": {
            "hidden_dim": 256,
            "n_hidden_layers": 4
        },
        "description": "Deeper standard MLP"
    },
    "wide_mlp": {
        "class": MLP,
        "params": {
            "hidden_dim": 2048,
            "n_hidden_layers": 2
        },
        "description": "Wider standard MLP"
    },
    "residual_mlp": {
        "class": ResidualMLP,
        "params": {
            "hidden_dim": 256,
            "n_hidden_layers": 2,
            "dropout": 0.1
        },
        "description": "MLP with residual connections"
    },
    "deep_residual_mlp": {
        "class": ResidualMLP,
        "params": {
            "hidden_dim": 256,
            "n_hidden_layers": 4,
            "dropout": 0.1
        },
        "description": "Deeper residual MLP"
    },
    "attention_mlp": {
        "class": AttentionMLP,
        "params": {
            "hidden_dim": 256,
            "n_hidden_layers": 2,
            "n_heads": 4,
            "dropout": 0.1
        },
        "description": "MLP with self-attention"
    },
    "conv_mlp": {
        "class": ConvolutionalMLP,
        "params": {
            "hidden_dim": 256,
            "n_hidden_layers": 2,
            "kernel_size": 3,
            "dropout": 0.1
        },
        "description": "Convolutional MLP for sequence modeling"
    },
    "dropout_mlp": {
        "class": DropoutMLP,
        "params": {
            "hidden_dim": 256,
            "n_hidden_layers": 2,
            "dropout": 0.3
        },
        "description": "MLP with heavy dropout regularization"
    },
    "small_mlp": {
        "class": MLP,
        "params": {
            "hidden_dim": 128,
            "n_hidden_layers": 1
        },
        "description": "Small/fast MLP"
    },
    "large_mlp": {
        "class": MLP,
        "params": {
            "hidden_dim": 512,
            "n_hidden_layers": 3
        },
        "description": "Large MLP"
    }
}

def create_architecture(arch_name: str, input_dim: int, output_dim: int, **override_params):

    """Create a model architecture by name with optional parameter overrides"""

    if arch_name not in ARCHITECTURE_CONFIGS:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHITECTURE_CONFIGS.keys())}")

    config = ARCHITECTURE_CONFIGS[arch_name].copy()
    arch_class = config["class"]
    params = config["params"].copy()
    params.update(override_params)

    return arch_class(
        input_dim=input_dim,
        output_dim=output_dim,
        **params
    )

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_architecture_experiment(args, config, arch_name_pf: str, arch_name_pb: Optional[str] = None):
    """Run a single experiment with specified architectures"""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Use same architecture for both PF and PB if not specified
    if arch_name_pb is None:
        arch_name_pb = arch_name_pf

    logging.info(f"Testing architecture: PF={arch_name_pf}, PB={arch_name_pb}")

    # Create environment and preprocessor
    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(
        env.seq_length, embedding_dim=args.embedding_dim, device=device
    )

    # Create architectures
    module_PF = create_architecture(
        arch_name_pf,
        preprocessor.output_dim,
        env.n_actions
    )

    module_PB = create_architecture(
        arch_name_pb,
        preprocessor.output_dim,
        env.n_actions - 1
    )

    if args.tied and hasattr(module_PF, 'trunk') and hasattr(module_PB, 'trunk'):
        logging.warning("Tied parameters requested but may not work with custom architectures")

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )


    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)
    sampler = Sampler(estimator=pf_estimator)
    gflownet = gflownet.to(device)


    pf_params = count_parameters(module_PF)
    pb_params = count_parameters(module_PB)
    total_params = count_parameters(gflownet)

    logging.info(f"PF parameters: {pf_params:,}")
    logging.info(f"PB parameters: {pb_params:,}")
    logging.info(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group(
        {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    loss_history, reward_history, reward_components, unique_seqs = train(
        args, env, gflownet, sampler, optimizer, scheduler, device
    )

    with torch.no_grad():
        samples, gc_list, mfe_list, cai_list = evaluate(
            env,
            sampler,
            weights=torch.tensor([0.3, 0.3, 0.4]),
            n_samples=args.n_samples,
        )

    eval_mean_gc = float(torch.tensor(gc_list).mean())
    eval_mean_mfe = float(torch.tensor(mfe_list).mean())
    eval_mean_cai = float(torch.tensor(cai_list).mean())
    eval_avg_reward = sum(
        w * r for w, r in zip(env.weights, [eval_mean_gc, -eval_mean_mfe, eval_mean_cai])
    )

    results = {
        "architecture_pf": arch_name_pf,
        "architecture_pb": arch_name_pb,
        "pf_description": ARCHITECTURE_CONFIGS[arch_name_pf]["description"],
        "pb_description": ARCHITECTURE_CONFIGS[arch_name_pb]["description"],
        "pf_parameters": pf_params,
        "pb_parameters": pb_params,
        "total_parameters": total_params,
        "final_loss": loss_history[-1],
        "final_reward": reward_history[-1],
        "unique_sequences": len(unique_seqs),
        "eval_mean_gc": eval_mean_gc,
        "eval_mean_mfe": eval_mean_mfe,
        "eval_mean_cai": eval_mean_cai,
        "eval_avg_reward": eval_avg_reward,
        "loss_history": loss_history,
        "reward_history": reward_history
    }

    return results

def run_architecture_sweep(args, config, architectures_to_test: List[str]):
    """Run experiments across multiple architectures"""

    if architectures_to_test == None:
        architectures_to_test = ["standard_mlp", "deep_mlp", "wide_mlp",
                               "residual_mlp", "attention_mlp", "dropout_mlp"]

    all_results = []

    for arch_name in architectures_to_test:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{arch_name}_{timestamp}"

            if config.wandb_project:
                wandb.init(
                    project=config.wandb_project,
                    config={**vars(args), "architecture": arch_name},
                    name=run_name,
                    reinit=True
                )

            # Run experiment
            results = run_architecture_experiment(args, config, arch_name)

            if config.wandb_project:
                wandb.log({
                    "architecture": arch_name,
                    **{k: v for k, v in results.items() if k not in ['loss_history', 'reward_history']}
                })
                wandb.finish()

            all_results.append(results)
            logging.info(f"Completed {arch_name}: reward={results['eval_avg_reward']:.4f}, "
                        f"params={results['total_parameters']:,}")

        except Exception as e:
            logging.error(f"Failed to run {arch_name}: {str(e)}")
            continue

    return all_results

def save_architecture_results(results: List[Dict]):

    """Save architecture comparison results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"architecture_comparison_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved architecture comparison results to {filename}")

    summary_file = filename.replace('.json', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Architecture Comparison Summary\n")
        f.write("=" * 50 + "\n\n")

        sorted_results = sorted(results, key=lambda x: x['eval_avg_reward'], reverse=True)

        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i}. {result['architecture_pf']}\n")
            f.write(f"   Description: {result['pf_description']}\n")
            f.write(f"   Parameters: {result['total_parameters']:,}\n")
            f.write(f"   Final Reward: {result['eval_avg_reward']:.4f}\n")
            f.write(f"   Final Loss: {result['final_loss']:.4f}\n")
            f.write(f"   Unique Sequences: {result['unique_sequences']}\n")
            f.write(f"   GC: {result['eval_mean_gc']:.3f}, "
                   f"MFE: {result['eval_mean_mfe']:.3f}, "
                   f"CAI: {result['eval_mean_cai']:.3f}\n\n")
