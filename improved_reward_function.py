"""
Improved reward function for conditional GFlowNet training.
Addresses the normalization issues that cause similar results between conditional and unconditional models.
"""

import torch
import math
import numpy as np
from typing import Tuple, Sequence, Optional, Dict, Any
from utils import compute_gc_content_vectorized, compute_mfe_energy, compute_cai

def compute_improved_reward(
    state,
    codon_gc_counts,
    weights: Sequence[float],
    protein_seq: Optional[str] = None,
    *,
    # Adaptive normalization parameters
    use_adaptive_norm: bool = True,
    gc_target: float = 0.50,
    gc_width: float = 0.10,
    # MFE bounds - should be estimated from your dataset

    mfe_min: float = -200.0,  # More realistic bounds
    mfe_max: float = -50.0,   # More realistic bounds

    # CAI is already [0,1] so no scaling needed
    cai_min: float = 0.0,
    cai_max: float = 1.0,
    # Reward scaling factors
    reward_scale: float = 10.0,  # Scale final reward to make differences more pronounced
    # Protein-specific parameters
    protein_length_factor: bool = True,
    protein_complexity_factor: bool = True,
) -> Tuple[float, Tuple[float, float, float], Dict[str, float]]:
    """
    Reward function with normalization and protein-specific conditioning.

    Args:
        state: sequence representation
        codon_gc_counts: GC counts for codons
        weights: [gc_weight, mfe_weight, cai_weight]
        protein_seq: protein sequence for conditioning
        use_adaptive_norm: whether to use adaptive normalization
        reward_scale: scaling factor for final reward
        protein_length_factor: whether to include protein length in reward
        protein_complexity_factor: whether to include protein complexity in reward

    Returns:
        (reward, (gc_val, mfe_val, cai_val), debug_info)
    """

    gc_val, mfe_val, cai_val = compute_reward_components(state, codon_gc_counts)

    # 1. GC Reward
    if gc_width == 0:
        gc_reward = 1.0 if math.isclose(gc_val, gc_target) else 0.0
    else:
        gc_reward = math.exp(-0.5 * ((gc_val - gc_target) / (gc_width * 0.5)) ** 2)

    # 2. MFE Reward
    if use_adaptive_norm and protein_seq is not None:
        # Adaptive MFE bounds based on protein length
        protein_length = len(protein_seq)
        adaptive_mfe_min = mfe_min * (protein_length / 100.0)  # Scale with length
        adaptive_mfe_max = mfe_max * (protein_length / 100.0)
    else:
        adaptive_mfe_min, adaptive_mfe_max = mfe_min, mfe_max

    mfe_span = adaptive_mfe_max - adaptive_mfe_min
    if mfe_span == 0:
        mfe_reward = 0.5  # Neutral reward
    else:
        mfe_scaled = (mfe_val - adaptive_mfe_min) / mfe_span
        mfe_scaled = float(np.clip(mfe_scaled, 0.0, 1.0))
        mfe_reward = mfe_scaled ** 2

    # 3. CAI Reward
    cai_span = cai_max - cai_min
    if cai_span == 0:
        cai_reward = 0.0
    else:
        cai_reward = float(np.clip((cai_val - cai_min) / cai_span, 0.0, 1.0))

    # 4. Protein-specific conditioning factors
    protein_factors = {}

    if protein_seq is not None:
        # Length factor
        if protein_length_factor:
            length_factor = 1.0 + 0.1 * (len(protein_seq) / 100.0)  # Up to 10% bonus
            protein_factors['length_factor'] = length_factor
        else:
            protein_factors['length_factor'] = 1.0

        # Complexity factor: proteins with more diverse amino acids get higher rewards
        if protein_complexity_factor:
            unique_aas = len(set(protein_seq))
            complexity_factor = 1.0 + 0.05 * (unique_aas / 20.0)  # Up to 5% bonus
            protein_factors['complexity_factor'] = complexity_factor
        else:
            protein_factors['complexity_factor'] = 1.0
    else:
        protein_factors = {'length_factor': 1.0, 'complexity_factor': 1.0}

    # 5. Weighted combination with protein factors
    if len(weights) != 3:
        raise ValueError("weights must be length-3 for [gc, mfe, cai]")

    w = [float(wi) for wi in weights]
    comp_rewards = [gc_reward, mfe_reward, cai_reward]

    # Base reward
    base_reward = float(w[0] * comp_rewards[0] + w[1] * comp_rewards[1] + w[2] * comp_rewards[2])

    # Apply protein factors
    protein_factor = protein_factors['length_factor'] * protein_factors['complexity_factor']
    reward = base_reward * protein_factor

    # 6. Scale reward to make differences more pronounced
    reward = reward * reward_scale

    # 7. Apply non-linear transformation to amplify differences
    # This makes the reward more sensitive to quality differences
    if reward > 0:
        reward = reward ** 0.8  # Slightly compress high rewards
    else:
        reward = -((-reward) ** 1.2)  # Amplify negative rewards

    # 8. Final clipping and validation
    if not math.isfinite(reward):
        reward = 0.0

    # Debug information
    debug_info = {
        'gc_reward': gc_reward,
        'mfe_reward': mfe_reward,
        'cai_reward': cai_reward,
        'base_reward': base_reward,
        'protein_factors': protein_factors,
        'final_reward': reward,
        'adaptive_mfe_bounds': (adaptive_mfe_min, adaptive_mfe_max)
    }

    return reward, (gc_val, mfe_val, cai_val), debug_info


def compute_reward_components(state, codon_gc_counts):
    """Compute raw reward components - you'll need to implement this based on your existing code."""
    gc_content = compute_gc_content_vectorized(state, codon_gc_counts).item()
    mfe_energy = compute_mfe_energy(state).item()
    cai_score = compute_cai(state).item()
    return gc_content, mfe_energy, cai_score


# Alternative: Less aggressive normalization approach
def compute_conservative_reward(
    state,
    codon_gc_counts,
    weights: Sequence[float],
    protein_seq: Optional[str] = None,
    *,
    # Use raw values with minimal normalization
    gc_target: float = 0.50,
    gc_tolerance: float = 0.05,  # Acceptable deviation from target
    mfe_penalty_factor: float = 0.01,  # Penalty per unit of MFE
    cai_bonus_factor: float = 2.0,  # Bonus multiplier for CAI
) -> Tuple[float, Tuple[float, float, float]]:
    """
    Conservative reward function that uses raw values with minimal normalization.
    This approach preserves more information and makes conditioning more effective.
    """

    gc_val, mfe_val, cai_val = compute_reward_components(state, codon_gc_counts)

    # 1. GC Reward - Linear penalty for deviation from target
    gc_deviation = abs(gc_val - gc_target)
    gc_reward = max(0, 1.0 - (gc_deviation / gc_tolerance))

    # 2. MFE Reward - Linear penalty (more negative MFE = higher penalty)
    mfe_reward = max(0, 1.0 + mfe_val * mfe_penalty_factor)

    # 3. CAI Reward - Linear bonus (higher CAI = higher reward)
    cai_reward = cai_val * cai_bonus_factor

    # 4. Weighted combination
    w = [float(wi) for wi in weights]
    reward = w[0] * gc_reward + w[1] * mfe_reward + w[2] * cai_reward

    # 5. Protein-specific scaling
    if protein_seq is not None:
        # Scale by protein length to make longer proteins more challenging
        length_scale = 1.0 + 0.2 * (len(protein_seq) / 100.0)
        reward *= length_scale

    return reward, (gc_val, mfe_val, cai_val)


# Example usage and testing
def test_reward_functions():
    """Test the improved reward functions."""

    # Mock data
    state = torch.tensor([1, 2, 3, 4, 5])  # Example sequence
    codon_gc_counts = torch.tensor([0.5, 0.6, 0.4, 0.7, 0.3])
    weights = [0.3, 0.3, 0.4]
    protein_seq = "MKLLVL"

    print("Testing improved reward functions...")

    # Test improved reward
    reward1, components1, debug1 = compute_improved_reward(
        state, codon_gc_counts, weights, protein_seq
    )
    print(f"Improved reward: {reward1:.4f}")
    print(f"Components: {components1}")
    print(f"Debug info: {debug1}")

    # Test conservative reward
    reward2, components2 = compute_conservative_reward(
        state, codon_gc_counts, weights, protein_seq
    )
    print(f"Conservative reward: {reward2:.4f}")
    print(f"Components: {components2}")


if __name__ == "__main__":
    test_reward_functions()
