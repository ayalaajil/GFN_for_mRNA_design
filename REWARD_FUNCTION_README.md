# Simple and Intuitive Reward Function for Conditional GFlowNet Training

## Problem with Original Reward Function

The original reward function had several issues that caused conditional and unconditional GFlowNets to produce similar results:

### 1. **Over-Normalization**
- All components (GC, MFE, CAI) were normalized to [0,1] range
- This made the reward insensitive to actual sequence quality differences
- Final reward was always in [0,1] regardless of sequence quality

### 2. **Fixed Normalization Bounds**
- MFE bounds (-500 to 0) were too wide and unrealistic
- Didn't adapt to different protein lengths or types
- One-size-fits-all approach didn't work for diverse proteins

### 3. **Limited Conditioning Effect**
- Since everything was [0,1], weights only affected relative importance
- No protein-specific adaptation
- Conditional GFlowNet couldn't learn protein-specific patterns

## New Simple Reward Function Design

### Core Philosophy
**"Keep it simple, make it work, amplify differences"**

### Key Principles

1. **Simple Linear Transformations**: Easy to understand and tune
2. **Raw Values Where Possible**: Preserve information
3. **Protein-Specific Adjustments**: Adapt to protein characteristics
4. **Clear Scaling**: Make differences visible and meaningful

## Simple Reward Components

### 1. GC Content Reward
```python
# Simple penalty for deviation from target
gc_deviation = abs(gc_val - gc_target)
gc_reward = max(0.0, 1.0 - (gc_deviation / gc_tolerance))
```
- **Target**: 0.5 (50% GC content)
- **Tolerance**: 0.1 (10% acceptable deviation)
- **Range**: [0, 1] with linear penalty
- **Intuition**: Perfect GC content = 1.0, deviation reduces reward linearly

### 2. MFE (Minimum Free Energy) Reward
```python
# Simple linear penalty (more negative MFE = higher penalty)
mfe_reward = max(0.0, 1.0 + mfe_val * mfe_penalty_per_unit)
```
- **Penalty**: 0.01 per unit of MFE
- **Range**: [0, 1] with linear scaling
- **Intuition**: MFE is typically negative, so we want to minimize absolute value

### 3. CAI (Codon Adaptation Index) Reward
```python
# Direct use (CAI is already [0,1])
cai_reward = cai_val * cai_multiplier
```
- **Multiplier**: 1.0 (can be adjusted)
- **Range**: [0, 1] (already well-bounded)
- **Intuition**: Higher CAI = better codon usage = higher reward

### 4. Protein-Specific Bonus
```python
# Simple length bonus
length_bonus = 1.0 + protein_length_bonus * (protein_length / 100.0)
```
- **Bonus**: 0.05 per 100 amino acids
- **Purpose**: Longer proteins are more challenging
- **Intuition**: Reward model for handling complexity

## Final Reward Calculation

```python
# 1. Weighted combination of components
base_reward = w[0] * gc_reward + w[1] * mfe_reward + w[2] * cai_reward

# 2. Apply protein-specific bonus
reward = base_reward * length_bonus * reward_scale

# 3. Ensure finite values
if not math.isfinite(reward):
    reward = 0.0
```

## Adaptive Version (Optional)

For more sophisticated conditioning, use the adaptive version:

```python
# Adaptive parameters based on protein
adaptive_mfe_penalty = mfe_penalty_per_unit * (100.0 / protein_length)
adaptive_gc_tolerance = gc_tolerance * (1.0 + 0.1 * (unique_aas / 20.0))
adaptive_reward_scale = reward_scale * (1.0 + 0.1 * (protein_length / 100.0))
```

- **MFE penalty**: More lenient for longer proteins
- **GC tolerance**: More lenient for complex proteins
- **Reward scale**: Higher scaling for longer proteins

## Why This Works Better

### 1. **Simple and Intuitive**
- Linear transformations are easy to understand
- Clear relationship between sequence quality and reward
- Easy to tune parameters

### 2. **Preserves Information**
- Raw values are used where appropriate
- No over-normalization that destroys information
- Quality differences are amplified, not compressed

### 3. **Protein-Specific Adaptation**
- Length bonus makes longer proteins more challenging
- Adaptive version adjusts parameters based on protein characteristics
- Conditional GFlowNet can learn protein-specific patterns

### 4. **Amplified Differences**
- Reward scale factor (5x) makes differences visible
- Linear penalties create clear gradients
- Protein-specific bonuses create context-dependent rewards

### 5. **Efficient Computation**
- Simple mathematical operations
- No complex non-linear transformations
- Fast to compute during training

## Expected Results

With this improved reward function:

1. **Conditional GFlowNet** will:
   - Adapt to different proteins with varying reward patterns
   - Learn protein-specific optimization strategies
   - Show clear differences from unconditional model

2. **Unconditional GFlowNet** will:
   - Show more consistent but less adaptive behavior
   - Perform well on average but struggle with specific proteins
   - Demonstrate the value of conditioning

3. **Training** will:
   - Converge faster due to clearer reward signals
   - Show more pronounced differences between models
   - Enable better generalization to unseen proteins

## Usage

### Simple Version
```python
# In your environment
def reward(self, states, **kwargs) -> torch.Tensor:
    rewards = []
    for i in range(batch_size):
        r, _, _ = compute_simple_reward(
            states_tensor[i],
            self.codon_gc_counts,
            self.weights,
            protein_seq=self.protein_seq,  # Pass protein sequence
            reward_scale=5.0
        )
        rewards.append(r)
    return torch.tensor(rewards, device=device)
```

### Adaptive Version
```python
# For more sophisticated conditioning
def reward(self, states, **kwargs) -> torch.Tensor:
    rewards = []
    for i in range(batch_size):
        r, _, _ = compute_adaptive_reward(
            states_tensor[i],
            self.codon_gc_counts,
            self.weights,
            protein_seq=self.protein_seq,
            reward_scale=5.0,
            use_length_adaptation=True,
            use_complexity_adaptation=True
        )
        rewards.append(r)
    return torch.tensor(rewards, device=device)
```

## Parameters to Tune

### Simple Version
- **reward_scale**: Overall scaling factor (default: 5.0)
- **gc_tolerance**: Acceptable deviation from GC target (default: 0.1)
- **mfe_penalty_per_unit**: Penalty per unit of MFE (default: 0.01)
- **cai_multiplier**: CAI bonus multiplier (default: 1.0)
- **protein_length_bonus**: Bonus per 100 amino acids (default: 0.05)

### Adaptive Version
- **use_length_adaptation**: Adjust MFE penalty for protein length (default: True)
- **use_complexity_adaptation**: Adjust GC tolerance for protein complexity (default: True)

## Monitoring

Track these metrics during training:
- Reward distribution across different proteins
- Reward variance (should be higher than original)
- Performance differences between conditional/unconditional models
- Generalization to unseen proteins
