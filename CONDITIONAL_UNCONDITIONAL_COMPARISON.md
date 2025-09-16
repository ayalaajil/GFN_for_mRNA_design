# Conditional vs Unconditional GFlowNet Comparison

This document describes the comprehensive experiment designed to compare conditional and unconditional GFlowNet models for mRNA design using identical architecture and hyperparameters.

## Overview

The experiment ensures a fair comparison between conditional and unconditional models by using:
- **Identical architecture**: Transformer-based GFlowNet with same dimensions
- **Same hyperparameters**: Learning rates, batch size, training iterations, etc.
- **Same evaluation sequences**: Using the exact protein sequences from curriculum learning evaluation
- **Comprehensive metrics**: GC content, MFE, CAI, diversity, and Pareto efficiency

## Files

### Main Experiment Script
- `run_conditional_unconditional_comparison.py`: Core experiment implementation
- `run_cond_uncond_experiment.py`: Simple runner script with predefined configurations

### Key Features

1. **Identical Architecture**: Both models use the same Transformer architecture with:
   - Embedding dimension: 32
   - Hidden dimension: 256
   - Number of hidden layers: 4
   - SubTB lambda: 0.9

2. **Same Training Parameters**:
   - Learning rate: 0.005
   - LogZ learning rate: 0.1
   - Batch size: 16
   - Epsilon: 0.25
   - Gradient clipping: 1.0
   - LR scheduler patience: 10

3. **Evaluation Sequences**:
   - **Short seen** (32 aa): `MINTQDSSILPLSNCPQLQCCRHIVPGPLWCS*`
   - **Medium unseen** (120 aa): `MKLVRFLMKLSHETVTIELKNGTQVHGTITGVDVSMNTHLKAVKMTLKNREPVQLETLSIRGNNIRYFILPDSLPLDTLLVDVEPKVKSKKREAVAGRGRGRGRGRGRGRGRGRGGPRR*`
   - **Long seen** (228 aa): `MGASARLLRAVIMGAPGSGKGTVSSRITTHFELKHLSSGDLLRDNMLRGTEIGVLAKAFIDQGKLIPDDVMTRLALHELKNLTQYSWLLDGFPRTLPQAEALDRAYQIDTVINLNVPFEVIKQRLTARWIHPASGRVYNIEFNPPKTVGIDDLTGEPLIQREDDKPETVIKRLKAYEDQTKPVLEYYQKKGVLETFSGTETNKIWPYVYAFLQTKVPQRSQKASVTP*`

## Usage

### Quick Start

```bash
# Run with interactive configuration selection
python run_cond_uncond_experiment.py

# Run directly with specific parameters
python run_conditional_unconditional_comparison.py \
    --n_iterations 200 \
    --eval_every 20 \
    --n_samples 100 \
    --wandb_project "My_Comparison_Experiment"
```

### Configuration Options

The runner script provides three predefined configurations:

1. **Quick Test** (50 iterations, 50 samples): For debugging and quick validation
2. **Standard** (200 iterations, 100 samples): Balanced performance vs time
3. **Comprehensive** (300 iterations, 200 samples): Full evaluation with maximum samples

### Command Line Arguments

```bash
# Model Architecture
--arch Transformer                    # Model architecture
--embedding_dim 32                   # Embedding dimension
--hidden_dim 256                     # Hidden dimension
--n_hidden 4                         # Number of hidden layers
--subTB_lambda 0.9                   # SubTB lambda parameter
--tied                              # Use tied parameters

# Training Parameters
--lr 0.005                          # Learning rate
--lr_logz 0.1                       # LogZ learning rate
--n_iterations 300                  # Training iterations
--eval_every 10                     # Evaluation frequency
--batch_size 16                     # Batch size
--epsilon 0.25                      # Exploration epsilon
--clip_grad_norm 1.0                # Gradient clipping
--lr_patience 10                    # LR scheduler patience

# Evaluation
--n_samples 200                     # Evaluation samples
--top_n 50                          # Top N for analysis

# System
--no_cuda                           # Disable CUDA
--seed 42                           # Random seed
--config_path config.yaml           # Config file
--wandb_project PROJECT_NAME        # WandB project
--run_name RUN_NAME                 # Run name
```

## Output

The experiment generates comprehensive outputs in the `outputs/conditional_unconditional_comparison_TIMESTAMP/` directory:

### Files Generated

1. **`comparison_report.txt`**: Human-readable summary of results
2. **`detailed_results.json`**: Complete results in JSON format
3. **`conditional_TASK_NAME/`**: Conditional model results for each task
4. **`unconditional_TASK_NAME/`**: Unconditional model results for each task

### Metrics Evaluated

For each model and task combination:

1. **Training Metrics**:
   - Loss history
   - Reward history
   - Final training reward

2. **Generation Quality**:
   - GC content (mean, std)
   - Minimum Free Energy (mean, std)
   - Codon Adaptation Index (mean, std)
   - Unique sequences generated

3. **Comprehensive Analysis**:
   - Pareto efficiency
   - Diversity metrics
   - Sequence uniqueness
   - Visualizations (Pareto plots)

### WandB Integration

If enabled, the experiment logs to WandB with:
- Training curves for both models
- Comparison tables
- Pareto plots
- Detailed metrics for each task
- Grouped runs for easy comparison

## Key Differences Between Models

### Conditional Model
- **Training**: Uses randomly sampled weight configurations during training
- **Evaluation**: Uses fixed weights [0.3, 0.3, 0.4] for evaluation
- **Advantage**: Learns to adapt to different objective preferences
- **Use case**: When you need flexibility in objective weighting

### Unconditional Model
- **Training**: Uses fixed weight configuration [0.3, 0.3, 0.4] throughout
- **Evaluation**: Uses same fixed weights [0.3, 0.3, 0.4]
- **Advantage**: Focused learning on single objective configuration
- **Use case**: When you have a specific, fixed objective preference

## Expected Results

The experiment will help answer:

1. **Performance**: Which model achieves higher rewards on each task?
2. **Consistency**: Which model is more stable across different protein lengths?
3. **Generalization**: How do models perform on unseen vs seen protein lengths?
4. **Efficiency**: Which model converges faster and more reliably?

## Comparison with Curriculum Learning

This experiment uses the same:
- **Architecture**: Identical to curriculum learning models
- **Hyperparameters**: Same learning rates, dimensions, etc.
- **Evaluation sequences**: Exact same protein sequences
- **Metrics**: Same comprehensive evaluation framework

This ensures fair comparison between:
- Curriculum Learning (progressive training)
- Conditional Training (fixed architecture, random weights)
- Unconditional Training (fixed architecture, fixed weights)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` or `--n_samples`
2. **Slow training**: Reduce `--n_iterations` or use `--eval_every` less frequently
3. **WandB errors**: Check API key and project permissions

### Performance Tips

1. **For quick testing**: Use the "quick_test" configuration
2. **For full evaluation**: Use the "comprehensive" configuration
3. **For debugging**: Disable WandB and use smaller sample sizes

## Example Output

```
CONDITIONAL vs UNCONDITIONAL MODEL COMPARISON REPORT
============================================================

Experiment Date: 2025-01-15 14:30:00
Architecture: Transformer
Embedding Dim: 32
Hidden Dim: 256
Hidden Layers: 4
Learning Rate: 0.005
LogZ Learning Rate: 0.1
SubTB Lambda: 0.9
Training Iterations: 300
Batch Size: 16
Evaluation Samples: 200

TASK: SHORT_SEEN
----------------------------------------
Protein Length: 32

CONDITIONAL MODEL:
  Final Reward: 0.8234
  Final Loss: 0.1234
  Unique Sequences: 187
  GC Content - Mean: 0.4567, Std: 0.1234
  MFE - Mean: -12.3456, Std: 2.1234
  CAI - Mean: 0.7890, Std: 0.0567

UNCONDITIONAL MODEL:
  Final Reward: 0.7890
  Final Loss: 0.1456
  Unique Sequences: 156
  GC Content - Mean: 0.4234, Std: 0.0987
  MFE - Mean: -11.2345, Std: 1.8765
  CAI - Mean: 0.7654, Std: 0.0432

COMPARISON:
  Reward Difference (Cond - Uncond): 0.0344
  Conditional Better: Yes
  Improvement: 0.0344 (4.36%)
```

This experiment provides a comprehensive, fair comparison between conditional and unconditional approaches for mRNA design using GFlowNets.
