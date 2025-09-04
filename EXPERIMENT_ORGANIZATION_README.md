# Experiment Organization Guide

This guide explains the new organized directory structure for running conditional vs unconditional GFlowNet experiments on different protein sizes.

## Directory Structure

All experiment outputs are now organized in the following structure:

```
outputs/
├── conditional/
│   ├── small/
│   │   ├── conditional_small_20241201_143022/
│   │   │   ├── experiment_summary.txt
│   │   │   ├── trained_gflownet_conditional_small_20241201_143022.pth
│   │   │   ├── generated_sequences_20241201_143022.txt
│   │   │   ├── metric_distributions_20241201_143022.png
│   │   │   ├── pareto_scatter_20241201_143022.png
│   │   │   ├── cai_vs_mfe_20241201_143022.png
│   │   │   └── gc_vs_mfe_20241201_143022.png
│   │   └── conditional_small_20241201_150145/
│   ├── medium/
│   │   └── conditional_medium_20241201_143022/
│   └── large/
│       └── conditional_large_20241201_143022/
└── unconditional/
    ├── small/
    │   └── unconditional_small_20241201_143022/
    ├── medium/
    │   └── unconditional_medium_20241201_143022/
    └── large/
        └── unconditional_large_20241201_143022/
```

## Running Experiments

### Method 1: Using the Organized Runner Script

The easiest way to run multiple experiments:

```bash
# Run all experiments (conditional and unconditional on all protein sizes)
python run_experiments_organized.py

# Run only conditional experiments on all protein sizes
python run_experiments_organized.py --experiment_type conditional --protein_size all

# Run only small protein experiments (both conditional and unconditional)
python run_experiments_organized.py --experiment_type both --protein_size small

# Run with additional arguments
python run_experiments_organized.py --additional_args --n_iterations 500 --batch_size 16
```

### Method 2: Manual Execution

Run individual experiments manually:

```bash
# Conditional experiment on small protein
python main.py --config_path config_small.yaml --conditional --run_name "my_conditional_small_run"

# Unconditional experiment on medium protein
python main.py --config_path config_medium.yaml --run_name "my_unconditional_medium_run"

# Conditional experiment on large protein
python main.py --config_path config_large.yaml --conditional --run_name "my_conditional_large_run"
```

## Configuration Files

We provide separate config files for different protein sizes:

- `config_small.yaml` - Small protein (~75 amino acids)
- `config_medium.yaml` - Medium protein (~141 amino acids)
- `config_large.yaml` - Large protein (~547 amino acids)

Each config file contains:
- Protein sequence and natural mRNA sequence
- WandB project name
- Protein type classification
- Model architecture settings

## Output Files

Each experiment creates a timestamped directory containing:

### Core Files
- `experiment_summary.txt` - Complete experiment details and parameters
- `trained_gflownet_*.pth` - Trained model weights
- `generated_sequences_*.txt` - Generated mRNA sequences with metrics

### Visualizations
- `metric_distributions_*.png` - Histograms of GC, MFE, and CAI distributions
- `pareto_scatter_*.png` - Pareto front visualization
- `cai_vs_mfe_*.png` - CAI vs MFE scatter plot
- `gc_vs_mfe_*.png` - GC content vs MFE scatter plot
- `enhanced_diversity_analysis_*.png` - Comprehensive diversity analysis plots

### Enhanced Analysis Files
- `comprehensive_comparison_*.txt` - Detailed comparison table with natural sequence
- `metrics_summary_*.txt` - Comprehensive metrics summary with diversity and quality metrics

## Experiment Summary File

Each experiment directory contains an `experiment_summary.txt` file with:

```
Experiment Summary
==================

Timestamp: 2024-12-01_14:30:22
Experiment Type: conditional
Protein Size: small
Run Name: conditional_small_20241201_143022
Architecture: MLP
Protein Sequence Length: 75
Training Iterations: 300
Batch Size: 8
Learning Rate: 0.005
Hidden Dimension: 128
Number of Hidden Layers: 2
SubTB Lambda: 0.8
Epsilon: 0.25
WandB Project: GFN_Organized_Experiments
Device: cuda

Protein Sequence: MDSEVQRDGRILDLIDDAWREDKLPYEDVAIPLNELPEPEQDNGGTTESVKEQEMKWTDLALQYLHENVPPIGN*
Natural mRNA Sequence: AUGGACAGUGAGGUUCAGAGAGAUGGAAGGAUCUUGGAUUUGAUUGAUGAUGCUUGGCGAGAAGACAAGCUGCCUUAUGAGGAUGUCGCAAUACCACUGAAUGAGCUUCCUGAACCUGAACAAGACAAUGGUGGCACCACAGAAUCUGUCAAAGAACAAGAAAUGAAGUGGACAGACUUAGCCUUACAGUACCUCCAUGAGAAUGUUCCCCCCAUUGGAAACUGA
```

## Enhanced Analysis Features

### Comprehensive Comparison Table
- **Direct comparison** with natural mRNA sequence
- **Levenshtein distance** and **identity percentage** for each generated sequence
- **Length differences** from natural sequence
- **Best-by-objective sequences** (highest GC, lowest MFE, highest CAI)

### Diversity Metrics
- **Mean edit distance** between generated sequences
- **Uniqueness ratio** (unique sequences / total sequences)
- **Sequence length distribution** analysis
- **GC content distribution** analysis
- **MFE energy distribution** analysis

### Quality Metrics
- **Pareto efficiency** (fraction of sequences on Pareto front)
- **Reward statistics** (mean, std deviation)
- **Objective statistics** (GC, MFE, CAI distributions)
- **Comprehensive metrics summary** with detailed statistics

### Enhanced Visualizations
- **4-panel diversity analysis** plot showing:
  - Edit distance distribution
  - Sequence length distribution
  - GC content distribution
  - MFE energy distribution

## Benefits of This Organization

1. **Easy Comparison**: Results from conditional vs unconditional experiments are clearly separated
2. **Protein Size Organization**: Results are grouped by protein complexity
3. **Timestamped Runs**: Each experiment has a unique timestamp for easy identification
4. **Complete Documentation**: Each run includes a summary file with all parameters
5. **Enhanced Analysis**: Comprehensive diversity and quality metrics for thorough evaluation
6. **Scalable**: Easy to add new protein sizes or experiment types
7. **WandB Integration**: All plots and metrics are logged to WandB with organized project names

## Tips for Analysis

1. **Compare Results**: Look in `outputs/conditional/small/` vs `outputs/unconditional/small/` for the same protein size
2. **Check Experiment Summary**: Always read `experiment_summary.txt` to understand the exact parameters used
3. **Use WandB**: All experiments are logged to WandB with organized project names for easy comparison
4. **Batch Analysis**: Use the organized structure to write scripts that analyze results across multiple experiments

## Example Analysis Script

```python
import os
import glob

# Find all conditional small protein experiments
conditional_small_dirs = glob.glob("outputs/conditional/small/*/")

# Find all unconditional small protein experiments
unconditional_small_dirs = glob.glob("outputs/unconditional/small/*/")

# Compare results
for cond_dir in conditional_small_dirs:
    # Load and analyze conditional results
    pass

for unc_dir in unconditional_small_dirs:
    # Load and analyze unconditional results
    pass
```
