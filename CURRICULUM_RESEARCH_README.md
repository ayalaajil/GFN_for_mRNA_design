# Curriculum Learning Configurations for mRNA Design Research

This document describes three different curriculum learning configurations designed to highlight the effectiveness of curriculum learning in mRNA design tasks for your research paper.

## Overview

The three configurations are designed to showcase different aspects of curriculum learning:

1. **Conservative EMA-based Curriculum** - Emphasizes stability and gradual progress
2. **Aggressive Sampling-based Curriculum** - Emphasizes exploration and rapid adaptation
3. **Balanced Proportional Curriculum** - Balances exploration and exploitation for multi-objective optimization

## Configuration Details

### 1. Conservative EMA-based Curriculum (`conservative_ema`)

**Purpose**: Demonstrates the importance of stability in curriculum learning for complex biological optimization tasks.

**Key Characteristics**:
- Uses Online EMA with very slow updates (α=0.05) for stable progress estimation
- Learning Progress (LP) attention computation (simpler than Mastering Rate)
- GreedyProp distribution with 15% uniform exploration
- Lower power settings for less aggressive task selection

**Expected Behavior**:
- Gradual task progression with minimal oscillation
- Lower variance in task selection over time
- Better performance on longer sequences due to stable learning
- May struggle with rapid adaptation to new task difficulties

**Research Value**: Shows how conservative approaches can provide stable learning in noisy reward environments typical of biological optimization.

### 2. Aggressive Sampling-based Curriculum (`aggressive_sampling`)

**Purpose**: Highlights the trade-off between exploration and exploitation in curriculum learning for mRNA design.

**Key Characteristics**:
- Uses Sampling-based progress estimation for rapid adaptation
- Mastering Rate (MR) attention with high power (8) for aggressive task selection
- Boltzmann distribution with high temperature (τ=0.5) for smooth exploration
- High potential emphasis (0.8) focusing on promising tasks

**Expected Behavior**:
- Rapid task switching based on immediate progress signals
- Higher exploration of task space
- Better performance on shorter sequences due to quick adaptation
- May show instability in task selection patterns

**Research Value**: Demonstrates how aggressive exploration can lead to faster adaptation but potentially higher variance.

### 3. Balanced Proportional Curriculum (`balanced_prop`)

**Purpose**: Shows the effectiveness of balanced curriculum learning for complex multi-objective biological optimization problems.

**Key Characteristics**:
- Uses Linear Regression progress estimation (good for noisy rewards)
- Mastering Rate attention with moderate power (4)
- Pure proportional distribution (no additional exploration)
- Balanced parameters for stable yet adaptive learning

**Expected Behavior**:
- Smooth task progression with moderate exploration
- Balanced attention to all objectives (GC, MFE, CAI)
- Consistent performance across different sequence lengths
- Stable convergence with reasonable adaptation speed

**Research Value**: Demonstrates optimal trade-off between stability and adaptation for multi-objective optimization.

## Usage

### Running Individual Experiments

```bash
# Conservative EMA-based curriculum
python curriculum_main_with_configs.py \
    --lpe Online --acp LP --a2d GreedyProp \
    --lpe_alpha 0.05 --a2d_eps 0.15 \
    --run_name conservative_ema_research \
    --wandb_project mRNA_GFN_Curriculum_Research

# Aggressive Sampling-based curriculum
python curriculum_main_with_configs.py \
    --lpe Sampling --acp MR --a2d Boltzmann \
    --lpe_K 10 --a2d_tau 0.5 --acp_MR_power 8 \
    --acp_MR_pot_prop 0.8 --run_name aggressive_sampling_research \
    --wandb_project mRNA_GFN_Curriculum_Research

# Balanced Proportional curriculum
python curriculum_main_with_configs.py \
    --lpe Linreg --acp MR --a2d Prop \
    --lpe_K 25 --acp_MR_power 4 --acp_MR_pot_prop 0.6 \
    --run_name balanced_prop_research \
    --wandb_project mRNA_GFN_Curriculum_Research
```

### Running All Experiments

```bash
# Run all three configurations
python run_curriculum_experiments.py --configs all

# Run specific configurations
python run_curriculum_experiments.py --configs conservative_ema balanced_prop

# Dry run to see configurations without running
python run_curriculum_experiments.py --configs all --dry_run
```

## Research Paper Integration

### Key Metrics to Track

1. **Learning Progress**: How quickly each configuration adapts to new tasks
2. **Task Distribution Stability**: Variance in task selection over time
3. **Multi-objective Performance**: Balance between GC content, MFE, and CAI
4. **Convergence Speed**: Time to reach stable performance
5. **Generalization**: Performance on unseen sequence lengths

### Expected Research Insights

1. **Conservative EMA**: Will show that stability is crucial for complex biological optimization tasks, especially for longer sequences where noise can be high.

2. **Aggressive Sampling**: Will demonstrate that rapid exploration can be beneficial for shorter sequences but may lead to instability in complex multi-objective scenarios.

3. **Balanced Proportional**: Will show that a balanced approach provides the best overall performance across different sequence lengths and objective combinations.

### Paper Sections

These configurations can be used to support several key points in your research paper:

- **Methodology Section**: Explain the different curriculum learning approaches and their theoretical foundations
- **Experimental Design**: Justify the choice of these three configurations as representative of different learning strategies
- **Results Section**: Compare performance across configurations to demonstrate curriculum learning effectiveness
- **Discussion Section**: Analyze the trade-offs between stability and adaptation in biological optimization

## Implementation Notes

### Required Modifications

To use these configurations with your existing code, you need to:

1. **Modify `curriculum_main.py`**: Add command-line arguments for curriculum parameters
2. **Update `MRNDesignCurriculum`**: Make it accept custom configuration dictionaries
3. **Modify `train_with_curriculum`**: Pass curriculum configuration to the curriculum object

### Configuration Files

The `curriculum_configs.py` file contains:
- All three configuration dictionaries
- Comparison tables
- Research insights and hypotheses
- Usage examples

### Experiment Tracking

Use Weights & Biases to track:
- Training curves for each configuration
- Task distribution evolution
- Multi-objective performance metrics
- Curriculum learning effectiveness measures

## Expected Results for Research Paper

Based on the theoretical foundations and parameter settings:

1. **Conservative EMA** should show:
   - Stable but slower convergence
   - Lower variance in task selection
   - Better performance on longer sequences
   - Robustness to reward noise

2. **Aggressive Sampling** should show:
   - Rapid initial adaptation
   - Higher exploration of task space
   - Better performance on shorter sequences
   - Higher variance in performance

3. **Balanced Proportional** should show:
   - Optimal trade-off between stability and adaptation
   - Consistent performance across sequence lengths
   - Good multi-objective balance
   - Stable convergence

These results will provide strong evidence for the effectiveness of curriculum learning in mRNA design and highlight the importance of choosing appropriate curriculum strategies for different optimization scenarios.
