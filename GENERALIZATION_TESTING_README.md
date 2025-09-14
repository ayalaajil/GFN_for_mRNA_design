# Generalization Testing for mRNA Design Models

This document describes the comprehensive generalization testing framework added to evaluate how well your trained mRNA design models perform across different extreme weight configurations.

## Overview

The generalization testing framework allows you to:

1. **Test model robustness** across extreme weight configurations
2. **Compare different models** on their ability to generalize
3. **Identify model weaknesses** in specific weight regimes
4. **Generate comprehensive reports** with visualizations and metrics

## Quick Start

### 1. During Training (Integrated)

Run training with generalization tests enabled:

```bash
# Unconditional model
python main.py --run_generalization_tests --generalization_n_samples 50 --run_name my_test

# Conditional model
python main_conditional.py --run_generalization_tests --generalization_n_samples 50 --run_name my_test
```

### 2. On Pre-trained Models (Standalone)

Test an already trained model:

```bash
python run_generalization_tests.py --model_path path/to/trained_model.pth --n_samples 50
```

## Weight Configurations Tested

The framework automatically tests 15 different extreme weight configurations:

### Single Objective Extremes
- `GC_only`: [1.0, 0.0, 0.0] - Only optimize GC content
- `MFE_only`: [0.0, 1.0, 0.0] - Only optimize MFE
- `CAI_only`: [0.0, 0.0, 1.0] - Only optimize CAI

### Two Objective Extremes
- `GC_MFE`: [0.5, 0.5, 0.0] - Balance GC and MFE
- `GC_CAI`: [0.5, 0.0, 0.5] - Balance GC and CAI
- `MFE_CAI`: [0.0, 0.5, 0.5] - Balance MFE and CAI

### Balanced Configurations
- `Balanced`: [0.33, 0.33, 0.34] - Equal weights
- `Slightly_GC`: [0.5, 0.25, 0.25] - Slight GC preference
- `Slightly_MFE`: [0.25, 0.5, 0.25] - Slight MFE preference
- `Slightly_CAI`: [0.25, 0.25, 0.5] - Slight CAI preference

### Extreme Unbalanced Configurations
- `Very_GC`: [0.8, 0.1, 0.1] - Strong GC preference
- `Very_MFE`: [0.1, 0.8, 0.1] - Strong MFE preference
- `Very_CAI`: [0.1, 0.1, 0.8] - Strong CAI preference

### Edge Cases
- `Almost_GC`: [0.9, 0.05, 0.05] - Nearly pure GC optimization
- `Almost_MFE`: [0.05, 0.9, 0.05] - Nearly pure MFE optimization
- `Almost_CAI`: [0.05, 0.05, 0.9] - Nearly pure CAI optimization

## Metrics Computed

For each weight configuration, the framework computes:

### Basic Statistics
- Number of samples generated
- Number of unique sequences
- Uniqueness ratio

### Metric Performance
- Mean, std, min, max, median for GC, MFE, CAI
- Reward statistics (mean, std, min, max)

### Quality Metrics
- Pareto efficiency (fraction of Pareto-optimal solutions)
- Number of Pareto-optimal solutions
- Sequence diversity (edit distance)

## Output Files

### 1. Summary CSV
`generalization_results_YYYY-MM-DD_HH-MM.csv` - Contains all statistics for all configurations

### 2. Individual Configuration Files
`{config_name}_sequences_YYYY-MM-DD_HH-MM.txt` - Detailed sequence results per configuration

### 3. Visualizations
- `generalization_analysis_YYYY-MM-DD_HH-MM.png` - Comprehensive analysis plots
- `{config_name}_plots_YYYY-MM-DD_HH-MM.png` - Individual configuration plots

### 4. Text Report
`generalization_report_YYYY-MM-DD_HH-MM.txt` - Human-readable summary report

## Command Line Options

### Main Scripts (main.py, main_conditional.py)
```bash
--run_generalization_tests          # Enable generalization testing
--generalization_n_samples N        # Number of samples per configuration (default: 50)
```

### Standalone Testing (run_generalization_tests.py)
```bash
--model_path PATH                   # Path to trained model (required)
--config_path PATH                  # Path to config file (default: config.yaml)
--conditional                       # Whether model is conditional
--n_samples N                       # Samples per configuration (default: 50)
--output_dir DIR                    # Output directory (default: generalization_results)
--custom_weights PATH               # Path to custom weight configurations JSON
--arch ARCH                         # Model architecture (MLP, MLP_EHH, Transformer)
--hidden_dim N                      # Hidden dimension
--n_hidden N                        # Number of hidden layers
--tied                              # Whether parameters are tied
--subTB_lambda FLOAT                # SubTB lambda parameter
--subTB_weighting STR               # SubTB weighting scheme
--seed N                            # Random seed (default: 42)
--device DEVICE                     # Device (auto, cpu, cuda)
```

## Custom Weight Configurations

You can define custom weight configurations by creating a JSON file:

```json
{
    "Custom_GC_Heavy": [0.8, 0.1, 0.1],
    "Custom_MFE_Heavy": [0.1, 0.8, 0.1],
    "Custom_CAI_Heavy": [0.1, 0.1, 0.8],
    "Custom_Balanced": [0.4, 0.3, 0.3]
}
```

Then use it with:
```bash
python run_generalization_tests.py --model_path model.pth --custom_weights custom_configs.json
```

## Programmatic Usage

### Basic Usage
```python
from generalization_tests import GeneralizationTester, run_generalization_tests

# Quick testing
tester = run_generalization_tests(env, sampler, device, "unconditional", n_samples=50)

# Advanced usage
tester = GeneralizationTester(env, sampler, device, "conditional")
results = tester.test_generalization(n_samples=100, output_dir="my_results")
report_path = tester.generate_report("my_results")
```

### Custom Weight Configurations
```python
custom_configs = {
    "My_Config_1": [0.7, 0.2, 0.1],
    "My_Config_2": [0.2, 0.7, 0.1],
    "My_Config_3": [0.1, 0.2, 0.7]
}

tester = GeneralizationTester(env, sampler, device, "unconditional")
results = tester.test_generalization(
    weight_configs=custom_configs,
    n_samples=100,
    output_dir="custom_results"
)
```

## Interpreting Results

### Good Generalization
- High Pareto efficiency across all configurations
- Consistent sequence diversity
- Balanced performance across all metrics

### Poor Generalization
- Low Pareto efficiency on extreme configurations
- High variance in diversity across configurations
- Poor performance when weights don't match training

### Model Comparison
- Compare average Pareto efficiency across models
- Look at performance on extreme configurations
- Check consistency of diversity metrics

## Examples

See `example_generalization_tests.py` for comprehensive examples of:
- Running tests during training
- Testing pre-trained models
- Using custom configurations
- Comparative analysis between models

## Integration with WandB

When enabled, generalization metrics are automatically logged to WandB:
- `generalization_avg_pareto_efficiency`
- `generalization_std_pareto_efficiency`
- `generalization_avg_diversity`
- `generalization_std_diversity`
- `generalization_n_configs`

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure model architecture parameters match the saved model
2. **Memory issues**: Reduce `n_samples` for large models
3. **CUDA errors**: Use `--device cpu` for CPU-only testing

### Performance Tips

1. Use smaller `n_samples` for quick testing
2. Test on a subset of configurations first
3. Use CPU for very large models if memory is limited

## Future Enhancements

Potential improvements to the framework:
- Support for more complex weight configurations
- Integration with hyperparameter optimization
- Automated model comparison reports
- Support for multi-objective optimization metrics

