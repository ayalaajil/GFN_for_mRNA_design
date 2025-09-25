import numpy as np
import torch
from tqdm import tqdm
from reward import compute_simple_reward
import wandb
import time
from plots import *

def train_conditional_gfn(args, env, gflownet, sampler, optimizer, scheduler, device):

    loss_history = []
    reward_history = []
    reward_components_history = []
    sampled_weights = []
    unique_sequences = set()

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):

        iter_start_time = time.time()

        # 1) sample weights
        weights = np.random.dirichlet([1, 1, 1])
        sampled_weights.append(weights.tolist())
        env.set_weights(weights)

        # 2) build conditioning tensor
        weights_tensor = torch.tensor(weights, dtype=torch.get_default_dtype(), device=device)
        conditioning = weights_tensor.unsqueeze(0).expand(args.batch_size, -1)

        # 3) sample trajectories *with conditioning*
        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            conditioning=conditioning,
            save_logprobs=True,
            save_estimator_outputs=True,
            epsilon=args.epsilon,
        )
        optimizer.zero_grad()
        loss = gflownet.loss_from_trajectories(
            env, trajectories, recalculate_all_logprobs=False
        )
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        iter_time = time.time() - iter_start_time

        # Logging rewards
        final_states = trajectories.terminating_states.tensor.to(device)
        rewards, components = [], []

        for state in final_states:

            state = state.to(device)
            r, c = compute_simple_reward(
                state, env.codon_gc_counts, env.weights
            )  # (gc, mfe, cai)
            rewards.append(r)
            components.append(c)

            seq = "".join([env.idx_to_codon[i.item()] for i in state])
            unique_sequences.add(seq)

        avg_reward = sum(rewards) / len(rewards)
        reward_history.append(avg_reward)
        reward_components_history.extend(components)

        components_tensor = torch.tensor(components)
        avg_gc, avg_mfe, avg_cai = components_tensor.mean(dim=0).tolist()

        loss_history.append(loss.item())

        if args.wandb_project:

            wandb.init(project=args.wandb_project, config=args, name=args.run_name)

            wandb.log(
            {
                "iteration": it,
                "loss": loss.item(),
                "avg_reward": avg_reward,
                "avg_gc": avg_gc,
                "avg_mfe": avg_mfe,
                "avg_cai": avg_cai,
                "w_gc": env.weights[0],
                "w_mfe": env.weights[1],
                "w_cai": env.weights[2],
                "iter_time": iter_time,
            }
            )

    sampled_weights = np.array(sampled_weights)
    return loss_history, reward_history, reward_components_history, unique_sequences, sampled_weights

def train(args, env, gflownet, sampler, optimizer, scheduler, device):

    loss_history = []
    reward_history = []
    reward_components_history = []
    unique_sequences = set()

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):

        iter_start_time = time.time()

        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=True,
            epsilon=args.epsilon,
        )

        optimizer.zero_grad()
        loss = gflownet.loss_from_trajectories(
            env, trajectories, recalculate_all_logprobs=False
        )

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        iter_time = time.time() - iter_start_time

        # Logging rewards
        final_states = trajectories.terminating_states.tensor.to(device)
        rewards, components = [], []

        for state in final_states:

            state = state.to(device)
            r, c = compute_simple_reward(state, env.codon_gc_counts, env.weights)  # (gc, mfe, cai)
            rewards.append(r)
            components.append(c)

            seq = "".join([env.idx_to_codon[i.item()] for i in state])
            unique_sequences.add(seq)

        avg_reward = sum(rewards) / len(rewards)
        reward_history.append(avg_reward)
        reward_components_history.extend(components)

        components_tensor = torch.tensor(components)
        avg_gc, avg_mfe, avg_cai = components_tensor.mean(dim=0).tolist()

        loss_history.append(loss.item())

        wandb.log(
            {
                "iteration": it,
                "loss": loss.item(),
                "avg_reward": avg_reward,
                "avg_gc": avg_gc,
                "avg_mfe": avg_mfe,
                "avg_cai": avg_cai,
                "w_gc": env.weights[0],
                "w_mfe": env.weights[1],
                "w_cai": env.weights[2],
                "iter_time": iter_time,
            }
        )
    return loss_history, reward_history, reward_components_history, unique_sequences