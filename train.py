import numpy as np
from tqdm import tqdm
from utils import compute_reward
import wandb
import time

def train(args, config, env, gflownet, sampler, optimizer, scheduler, device):

    loss_history = []
    reward_history = []
    reward_components_history = []
    # sampled_weights = []

    unique_sequences = set()

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):

        iter_start_time = time.time()

        # weights = np.random.dirichlet([1, 1, 1])   # analyze the weights chosen to gain insight into how the model behaves across different reward configurations
        # sampled_weights.append(weights.tolist()
        # env.set_weights(weights)

        trajectories = sampler.sample_trajectories(
            env, args.batch_size, save_logprobs=True, epsilon=args.epsilon
        )

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        iter_time = time.time() - iter_start_time

        # Logging rewards
        final_states = trajectories.terminating_states.tensor.to(device)
        rewards, components = [], []

        for state in final_states:

            state = state.to(device)
            r, c = compute_reward(state, env.codon_gc_counts, env.weights)   #(gc, mfe, cai)
            rewards.append(r)
            components.append(c)

            seq = ''.join([env.idx_to_codon[i.item()] for i in state])
            unique_sequences.add(seq)

        avg_reward = sum(rewards) / len(rewards)
        reward_history.append(avg_reward)
        reward_components_history.extend(components)
        loss_history.append(loss.item())

        wandb.log({
                "iteration": it,
                "loss": loss.item(),
                "avg_reward": avg_reward,
                "w_gc": env.weights[0],
                "w_mfe": env.weights[1],
                "w_cai": env.weights[2]
            })

    # sampled_weights = np.array(sampled_weights)

    return loss_history, reward_history, reward_components_history, unique_sequences
    #return loss_history, reward_history, reward_components_history, unique_sequences, sampled_weights
