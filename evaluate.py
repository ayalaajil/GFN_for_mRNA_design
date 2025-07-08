from utils import compute_reward

import torch
from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from torchgfn.src.gfn.gflownet import TBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator
from torchgfn.src.gfn.utils.modules import MLP, DiscreteUniform, Tabular
from torchgfn.src.gfn.samplers import Sampler
from utils import load_config

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
        seq = ''.join([env.idx_to_codon[i.item()] for i in state])
        samples[seq] = [reward, components]

        gc_list.append(components[0])
        mfe_list.append(components[1])
        cai_list.append(components[2])

    return samples, gc_list, mfe_list, cai_list


def load_trained_model(checkpoint_path: str, config_path: str = "config.yaml"):
    """
    Load trained GFlowNet components from a checkpoint and config file.
    """
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)

    preprocessor = CodonSequencePreprocessor(
        env.seq_length,
        embedding_dim=config.embedding_dim,
        device=device
    )

    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=config.hidden_dim,
        n_hidden_layers=config.n_hidden
    )

    module_PB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=config.hidden_dim,
        n_hidden_layers=config.n_hidden,
        trunk=module_PF.trunk if config.tied else None
    )

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )

    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gflownet.load_state_dict(checkpoint['model_state'])
    gflownet.logZ = checkpoint.get('logZ', torch.tensor(0.0, device=device))

    gflownet = gflownet.to(device)

    sampler = Sampler(estimator=pf_estimator)

    return env, sampler

