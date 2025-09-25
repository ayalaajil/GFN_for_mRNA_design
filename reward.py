from typing import Tuple, Sequence, Dict
from utils import compute_gc_content_vectorized, compute_mfe_energy, compute_cai


def compute_simple_reward(
    state,
    codon_gc_counts,
    weights: Sequence[float],
) -> Tuple[float, Tuple[float, float, float], Dict[str, float]]:

    gc_val, mfe_val, cai_val = compute_reward_components(state, codon_gc_counts)

    # 2. GC Reward
    gc_min = 0.35
    gc_max = 0.65
    gc_val_norm = gc_val/100
    gc_reward = (gc_val_norm - gc_min) / (gc_max - gc_min + 1e-12)

    # 3. MFE Reward
    seq_len = max(1, len(state))
    mfe_reward = - (mfe_val / seq_len)  # kcal/mol per nt

    # 4. CAI Reward
    cai_reward = cai_val  # already in [0,1]

    w = [float(wi) for wi in weights]
    reward = w[0] * gc_reward + w[1] * mfe_reward + w[2] * cai_reward

    return reward, (gc_val, mfe_val, cai_val)


def compute_reward_components(state, codon_gc_counts):
    """Compute raw reward components using existing utility functions."""
    gc_content = compute_gc_content_vectorized(state, codon_gc_counts).item()
    mfe_energy = compute_mfe_energy(state).item()
    cai_score = compute_cai(state).item()
    return gc_content, mfe_energy, cai_score

