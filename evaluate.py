import torch
import numpy as np
from reward import compute_simple_reward
from utils import *


def evaluate(env, sampler, weights, n_samples=100, conditional=False):
    """Evaluation function that handles both conditional and unconditional evaluation."""

    env.set_weights(weights)

    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32, device=env.device)

    # Sample trajectories
    if conditional:
        conditioning = weights.detach().clone()
        conditioning = conditioning.unsqueeze(0).expand(n_samples, *conditioning.shape).to(env.device)
        eval_trajectories = sampler.sample_trajectories(env, n=n_samples, conditioning=conditioning)

    else:
        eval_trajectories = sampler.sample_trajectories(env, n=n_samples)

    # Process results
    final_states = eval_trajectories.terminating_states.tensor
    samples = {}
    gc_list, mfe_list, cai_list = [], [], []

    for state in final_states:
        reward, components = compute_simple_reward(
            state, env.codon_gc_counts, weights
        )
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

