# How to Run Curriculum Learning Experiments

This guide explains how to run the three curriculum learning configurations for your mRNA design research.

## Quick Start

### Option 1: Run All Experiments at Once
```bash
python run_research_experiments.py
```

### Option 2: Run Individual Experiments
```bash
# Conservative EMA-based curriculum
python curriculum_main_complete.py \
    --run_name conservative_ema_research \
    --wandb_project mRNA_GFN_Curriculum_Research \
    --n_iterations 50 \
    --eval_every 5 \
    --train_steps_per_task 100 \
    --lpe Online --acp LP --a2d GreedyProp \
    --lpe_alpha 0.05 --a2d_eps 0.15

# Aggressive Sampling-based curriculum
python curriculum_main_complete.py \
    --run_name aggressive_sampling_research \
    --wandb_project mRNA_GFN_Curriculum_Research \
    --n_iterations 50 \
    --eval_every 5 \
    --train_steps_per_task 100 \
    --lpe Sampling --acp MR --a2d Boltzmann \
    --lpe_K 10 --a2d_tau 0.5 --acp_MR_power 8 \
    --acp_MR_pot_prop 0.8

# Balanced Proportional curriculum
python curriculum_main_complete.py \
    --run_name balanced_prop_research \
    --wandb_project mRNA_GFN_Curriculum_Research \
    --n_iterations 50 \
    --eval_every 5 \
    --train_steps_per_task 100 \
    --lpe Linreg --acp MR --a2d Prop \
    --lpe_K 25 --acp_MR_power 4 --acp_MR_pot_prop 0.6
```

### Option 3: Use the Experiment Runner
```bash
# Run all configurations
python run_curriculum_experiments.py --configs all

# Run specific configurations
python run_curriculum_experiments.py --configs conservative_ema balanced_prop

# Dry run to see what would be executed
python run_curriculum_experiments.py --configs all --dry_run
```

## Configuration Details

### 1. Conservative EMA-based Curriculum
- **LPE**: Online EMA (α=0.05) - very slow, stable updates
- **ACP**: Learning Progress only - simpler attention mechanism
- **A2D**: GreedyProp (ε=0.15) - moderate exploration
- **Expected**: Stable but slower convergence, good for noisy rewards

### 2. Aggressive Sampling-based Curriculum
- **LPE**: Sampling-based (K=10) - rapid adaptation
- **ACP**: Mastering Rate (power=8) - aggressive task selection
- **A2D**: Boltzmann (τ=0.5) - high exploration
- **Expected**: Rapid adaptation but higher variance

### 3. Balanced Proportional Curriculum
- **LPE**: Linear Regression (K=25) - balanced, good for noisy rewards
- **ACP**: Mastering Rate (power=4) - moderate task selection
- **A2D**: Proportional (ε=0.0) - no additional exploration
- **Expected**: Optimal trade-off for multi-objective optimization

## Parameters Explained

### Learning Progress Estimator (LPE)
- `--lpe`: Type of progress estimator
  - `Online`: Simple online learning progress
  - `Sampling`: Sampling-based progress estimation
  - `Linreg`: Linear regression progress estimation
- `--lpe_alpha`: EMA alpha for Online/Naive/Window (0.05 = very slow)
- `--lpe_K`: Window size for progress estimation

### Attention Computer (ACP)
- `--acp`: Type of attention computation
  - `LP`: Learning Progress only
  - `MR`: Mastering Rate (more sophisticated)
- `--acp_MR_power`: Power for MR computation (higher = more aggressive)
- `--acp_MR_pot_prop`: Potential proportion (0.8 = focus on promising tasks)
- `--acp_MR_att_pred`: Predecessor attention (stability)
- `--acp_MR_att_succ`: Successor attention (forward push)

### Attention-to-Distribution (A2D)
- `--a2d`: Distribution mapping type
  - `Prop`: Pure proportional
  - `GreedyProp`: Proportional with exploration
  - `Boltzmann`: Boltzmann distribution
- `--a2d_eps`: Epsilon for greedy converters (exploration level)
- `--a2d_tau`: Temperature for Boltzmann (higher = more exploration)

## Monitoring Results

### Weights & Biases
All experiments are logged to Weights & Biases with the project name `mRNA_GFN_Curriculum_Research`. You can monitor:

- Training curves and loss evolution
- Task distribution over time
- Multi-objective performance (GC, MFE, CAI)
- Curriculum learning effectiveness metrics

### Local Outputs
Results are saved in:
- `curriculum_model/` - Trained models
- Console output - Real-time progress and statistics

## Expected Results for Research Paper

### Conservative EMA
- **Learning curves**: Gradual, stable convergence
- **Task distribution**: Low variance, smooth progression
- **Performance**: Better on longer sequences, robust to noise

### Aggressive Sampling
- **Learning curves**: Rapid initial adaptation, higher variance
- **Task distribution**: High exploration, frequent task switching
- **Performance**: Better on shorter sequences, faster adaptation

### Balanced Proportional
- **Learning curves**: Smooth convergence with moderate adaptation
- **Task distribution**: Balanced exploration-exploitation
- **Performance**: Consistent across sequence lengths, good multi-objective balance

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `--batch_size` or `--train_steps_per_task`
2. **Dataset not found**: Ensure protein dataset files are in the current directory
3. **WandB errors**: Check your WandB login and project permissions

### Performance Tips
1. **Faster experiments**: Reduce `--n_iterations` and `--train_steps_per_task`
2. **More thorough experiments**: Increase `--n_iterations` and `--eval_every`
3. **Debugging**: Use `--verbose` flag for detailed output

## Research Paper Integration

These experiments will provide data for:

1. **Methodology Section**: Explain curriculum learning approaches
2. **Results Section**: Compare performance across configurations
3. **Discussion Section**: Analyze trade-offs between stability and adaptation
4. **Conclusion**: Demonstrate curriculum learning effectiveness in mRNA design

The three configurations are designed to showcase different aspects of curriculum learning effectiveness, providing strong evidence for your research paper's claims.
