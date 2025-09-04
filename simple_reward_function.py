"""
Simplified and Intuitive Reward Function for Conditional GFlowNet Training

This version is designed to be:
1. Easy to understand
2. Efficient to compute
3. Effective for conditioning
4. Well-documented
"""

import torch
import math
import numpy as np
from typing import Tuple, Sequence, Optional, Dict, Any
from utils import compute_gc_content_vectorized, compute_mfe_energy, compute_cai


def compute_simple_reward(
    state,
    codon_gc_counts,
    weights: Sequence[float],
    protein_seq: Optional[str] = None,
    *,
    gc_target: float = 0.50,
    gc_tolerance: float = 0.10,
    mfe_penalty_per_unit: float = 0.01,
    cai_multiplier: float = 1.0,
    protein_length_bonus: float = 0.05,
    reward_scale: float = 5.0,
) -> Tuple[float, Tuple[float, float, float], Dict[str, float]]:
    """
    Reward function that's easy to understand and tune.


    Args:
        state: sequence representation
        codon_gc_counts: GC counts for codons
        weights: [gc_weight, mfe_weight, cai_weight]
        protein_seq: protein sequence for conditioning
        gc_target: target GC content (0.5 = 50%)
        gc_tolerance: acceptable deviation from target
        mfe_penalty_per_unit: penalty per unit of MFE
        cai_multiplier: bonus multiplier for CAI
        protein_length_bonus: bonus per 100 amino acids
        reward_scale: overall scaling factor

    Returns:
        (reward, (gc_val, mfe_val, cai_val))
    """

    gc_val, mfe_val, cai_val = compute_reward_components(state, codon_gc_counts)

    # 2. GC Reward
    gc_deviation = abs(gc_val - gc_target)
    gc_reward = max(0.0, 1.0 - (gc_deviation / gc_tolerance))

    # 3. MFE Reward
    mfe_reward = max(0.0, 1.0 + mfe_val * mfe_penalty_per_unit)

    # 4. CAI Reward
    cai_reward = cai_val * cai_multiplier

    w = [float(wi) for wi in weights]
    base_reward = w[0] * gc_reward + w[1] * mfe_reward + w[2] * cai_reward

    protein_bonus = 1.0
    if protein_seq is not None:
        length_bonus = 1.0 + protein_length_bonus * (len(protein_seq) / 100.0)
        protein_bonus = length_bonus

    reward = base_reward * protein_bonus * reward_scale

    if not math.isfinite(reward):
        reward = 0.0

    return reward, (gc_val, mfe_val, cai_val)


def compute_reward_components(state, codon_gc_counts):
    """Compute raw reward components using existing utility functions."""
    gc_content = compute_gc_content_vectorized(state, codon_gc_counts).item()
    mfe_energy = compute_mfe_energy(state).item()
    cai_score = compute_cai(state).item()
    return gc_content, mfe_energy, cai_score


def compute_adaptive_reward(
    state,
    codon_gc_counts,
    weights: Sequence[float],
    protein_seq: Optional[str] = None,
    *,

    gc_target: float = 0.50,
    gc_tolerance: float = 0.10,
    mfe_penalty_per_unit: float = 0.01,
    cai_multiplier: float = 1.0,
    reward_scale: float = 5.0,

    use_length_adaptation: bool = True,
    use_complexity_adaptation: bool = True,
) -> Tuple[float, Tuple[float, float, float], Dict[str, float]]:
    """
    Adaptive version that adjusts parameters based on protein characteristics.

    This version automatically adjusts:
    - MFE penalty based on protein length
    - GC tolerance based on protein complexity
    - Overall scaling based on protein properties
    """


    gc_val, mfe_val, cai_val = compute_reward_components(state, codon_gc_counts)


    if protein_seq is not None:
        protein_length = len(protein_seq)
        unique_aas = len(set(protein_seq))

        if use_length_adaptation:
            adaptive_mfe_penalty = mfe_penalty_per_unit * (100.0 / protein_length)
        else:
            adaptive_mfe_penalty = mfe_penalty_per_unit

        if use_complexity_adaptation:
            adaptive_gc_tolerance = gc_tolerance * (1.0 + 0.1 * (unique_aas / 20.0))
        else:
            adaptive_gc_tolerance = gc_tolerance

        adaptive_reward_scale = reward_scale * (1.0 + 0.1 * (protein_length / 100.0))

    else:
        adaptive_mfe_penalty = mfe_penalty_per_unit
        adaptive_gc_tolerance = gc_tolerance
        adaptive_reward_scale = reward_scale

    gc_deviation = abs(gc_val - gc_target)
    gc_reward = max(0.0, 1.0 - (gc_deviation / adaptive_gc_tolerance))

    mfe_reward = max(0.0, 1.0 + mfe_val * adaptive_mfe_penalty)

    cai_reward = cai_val * cai_multiplier

    # Weighted combination
    w = [float(wi) for wi in weights]
    base_reward = w[0] * gc_reward + w[1] * mfe_reward + w[2] * cai_reward

    reward = base_reward * adaptive_reward_scale

    if not math.isfinite(reward):
        reward = 0.0


    return reward, (gc_val, mfe_val, cai_val)







# def test_simple_reward():
#     """Test the simple reward function with example data."""

#     print("Testing Simple Reward Function")
#     print("=" * 40)

#     # Mock data
#     state = torch.tensor([1, 2, 3, 4, 5])
#     codon_gc_counts = torch.tensor([0.5, 0.6, 0.4, 0.7, 0.3])
#     weights = [0.3, 0.3, 0.4]
#     protein_seq = "MKLLVL"

#     # Test simple reward
#     reward1, components1 = compute_simple_reward(
#         state, codon_gc_counts, weights, protein_seq
#     )

#     print(f"Simple Reward: {reward1:.4f}")
#     print(f"Components: GC={components1[0]:.3f}, MFE={components1[1]:.3f}, CAI={components1[2]:.3f}")


#     # Test adaptive reward
#     reward2, components2 = compute_adaptive_reward(
#         state, codon_gc_counts, weights, protein_seq
#     )

#     print(f"\nAdaptive Reward: {reward2:.4f}")
#     print(f"Components: GC={components2[0]:.3f}, MFE={components2[1]:.3f}, CAI={components2[2]:.3f}")

#     # Test with different weights
#     print(f"\nTesting different weights:")
#     for w in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]:
#         r ,_,_ = compute_simple_reward(state, codon_gc_counts, w, protein_seq)
#         print(f"Weights {w}: Reward = {r:.4f}")


# if __name__ == "__main__":
#     test_simple_reward()
