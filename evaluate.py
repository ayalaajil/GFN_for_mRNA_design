from utils import compute_reward
import torch
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from torchgfn.src.gfn.gflownet import TBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator
from torchgfn.src.gfn.utils.modules import MLP
from torchgfn.src.gfn.samplers import Sampler
from utils import load_config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_conditional(env, sampler, weights, n_samples=100):

    env.set_weights(weights)
    # build conditioning tensor
    conditioning = (torch.tensor(weights, dtype=torch.get_default_dtype(), device=env.device))
    conditioning = conditioning.unsqueeze(0).expand(n_samples, *conditioning.shape)

    eval_trajectories = sampler.sample_trajectories(env, n=n_samples, conditioning=conditioning)
    final_states = eval_trajectories.terminating_states.tensor
    samples = {}

    gc_list = []
    mfe_list = []
    cai_list = []

    for state in final_states:

        reward, components = compute_reward(state, env.codon_gc_counts, weights)
        seq = "".join([env.idx_to_codon[i.item()] for i in state])
        samples[seq] = [reward, components]

        gc_list.append(components[0])
        mfe_list.append(components[1])
        cai_list.append(components[2])

    return samples, gc_list, mfe_list, cai_list

def evaluate(env, sampler, weights, n_samples=100):

    env.set_weights(weights)

    eval_trajectories = sampler.sample_trajectories(env, n=n_samples)
    final_states = eval_trajectories.terminating_states.tensor
    samples = {}

    gc_list = []
    mfe_list = []
    cai_list = []

    for state in final_states:

        reward, components = compute_reward(state, env.codon_gc_counts, weights)
        seq = "".join([env.idx_to_codon[i.item()] for i in state])
        samples[seq] = [reward, components]

        gc_list.append(components[0])
        mfe_list.append(components[1])
        cai_list.append(components[2])

    return samples, gc_list, mfe_list, cai_list


def is_pareto_efficient_3d(costs):
    """
    Determine Pareto-efficient points for 3 objectives.
    costs: array of shape (N, 3) with objectives:
        - (-CAI)  [we want to maximize CAI]
        -  MFE    [we want to minimize MFE]
        -  GC     [we want to maximize GC]
    Returns: Boolean mask of Pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)

    for i, c in enumerate(costs):

        if is_efficient[i]:

            # Remove dominated points
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] > c, axis=1
            ) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True

    return is_efficient


def sweep_weight_configs(
    env, sampler, configs, n_samples=10, save_path="sweep_results.csv"
):

    all_results = []
    rows = []

    for config_name, weights in configs.items():

        print(f"Evaluating config: {config_name} with weights: {weights}")

        samples, gc_list, mfe_list, cai_list = evaluate(
            env, sampler, weights, n_samples
        )

        result = {
            "name": config_name,
            "weights": weights,
            "samples": samples,
            "metrics": {"GC": gc_list, "MFE": mfe_list, "CAI": cai_list},
        }

        all_results.append(result)

        for i in range(n_samples):

            rows.append(
                {
                    "Config": config_name,
                    "Weight_GC": weights[0],
                    "Weight_MFE": weights[1],
                    "Weight_CAI": weights[2],
                    "Sequence": list(samples.keys())[i],
                    "GC": gc_list[i],
                    "MFE": mfe_list[i],
                    "CAI": cai_list[i],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\nSaved results table to: {save_path}")

    return all_results


def load_trained_model(checkpoint_path: str, config_path: str = "config.yaml"):
    """
    Load trained GFlowNet components from a checkpoint and config file.
    """
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)

    preprocessor = CodonSequencePreprocessor(
        env.seq_length, embedding_dim=config.embedding_dim, device=device
    )

    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=config.hidden_dim,
        n_hidden_layers=config.n_hidden,
    )

    module_PB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=config.hidden_dim,
        n_hidden_layers=config.n_hidden,
        trunk=module_PF.trunk if config.tied else None,
    )

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )

    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gflownet.load_state_dict(checkpoint["model_state"])
    gflownet.logZ = checkpoint.get("logZ", torch.tensor(0.0, device=device))

    gflownet = gflownet.to(device)

    sampler = Sampler(estimator=pf_estimator)

    return env, sampler
