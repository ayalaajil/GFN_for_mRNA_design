# Generalization Analysis: Conditional vs Unconditional GFlowNet Training

This guide explains how to compare conditional and unconditional GFlowNet training for mRNA design and verify whether conditional training helps with generalization.

## Overview

The key question is: **Does conditional training (where the model learns to adapt to different weight configurations) generalize better than unconditional training (where the model is trained on a fixed weight configuration)?**

## What is Generalization in This Context?

Generalization refers to how well a model performs on **unseen weight configurations** that were not used during training. A model with good generalization should:

1. **Perform well on new weight combinations** (e.g., if trained on [0.3, 0.3, 0.4], perform well on [0.7, 0.1, 0.2])
2. **Maintain consistent performance** across different weight configurations
3. **Handle extreme cases** (e.g., weights like [0.9, 0.05, 0.05] or [0.0, 0.5, 0.5])

## How to Verify Generalization

### Method 1: Quick Comparison (Recommended for initial testing)

Run the quick comparison script to get a fast assessment:

```bash
python quick_comparison.py
```

This script will:
- Train both conditional and unconditional models on 3 weight configurations
- Test them on 4 unseen weight configurations
- Provide a simple comparison of generalization performance
- Generate a plot showing the results

### Method 2: Comprehensive Analysis (Recommended for thorough evaluation)

For a detailed analysis, use the full comparison script:

```bash
python compare_conditional_vs_unconditional.py \
    --n_iterations 200 \
    --n_samples 100 \
    --hidden_dim 256 \
    --n_hidden 2 \
    --save_dir generalization_analysis \
    --wandb_project "mRNA_generalization_study"
```

This comprehensive analysis includes:

#### Training Phase
- **Unconditional Model**: Trained on 4 weight configurations, but doesn't see the weights during training
- **Conditional Model**: Trained on the same 4 weight configurations, with weights provided as conditioning

#### Evaluation Phase
- **Standard Test**: 8 unseen weight configurations
- **Extreme Test**: 6 extreme weight configurations (e.g., [0.9, 0.05, 0.05])
- **Multiple Metrics**: Reward, Pareto efficiency, sequence diversity, consistency

#### Analysis Metrics

1. **Average Test Reward**: How well the model performs on unseen weight configurations
2. **Pareto Efficiency**: Percentage of generated sequences that are Pareto optimal
3. **Sequence Diversity**: Average edit distance between generated sequences
4. **Consistency**: Standard deviation of rewards across different weight configurations (lower is better)
5. **Extreme Case Performance**: How well the model handles very extreme weight configurations

## Understanding the Results

### Key Indicators of Better Generalization

1. **Higher Average Test Reward**: The conditional model should achieve higher rewards on unseen weight configurations
2. **Lower Standard Deviation**: More consistent performance across different weight configurations
3. **Better Extreme Case Performance**: Superior performance on extreme weight configurations
4. **Higher Pareto Efficiency**: More of the generated sequences are Pareto optimal
5. **Better Sequence Diversity**: More diverse sequence generation while maintaining quality

### Statistical Significance

The analysis includes statistical tests (t-tests) to determine if the differences are statistically significant (p < 0.05).

## Example Results Interpretation

```
=== RESULTS COMPARISON ===
Test Performance (Generalization):
  Unconditional - Mean Reward: 0.2345 ± 0.1234
  Conditional   - Mean Reward: 0.3456 ± 0.0987
  Improvement: +47.3%

  → Conditional training shows better generalization!
```

This would indicate that conditional training helps with generalization.

## Files Generated

### Quick Comparison
- `quick_comparison_results.png`: Simple comparison plot

### Comprehensive Analysis
- `generalization_analysis/raw_results_YYYYMMDD_HHMMSS.json`: Raw experimental data
- `generalization_analysis/analysis_YYYYMMDD_HHMMSS.json`: Statistical analysis results
- `generalization_analysis/summary_report_YYYYMMDD_HHMMSS.txt`: Human-readable summary
- `generalization_analysis/training_comparison.png`: Training curves comparison
- `generalization_analysis/weight_configuration_performance.png`: Performance across weight configurations
- `generalization_analysis/pareto_efficiency_comparison.png`: Pareto efficiency comparison
- `generalization_analysis/diversity_comparison.png`: Sequence diversity comparison

## Weight Configurations Used

### Training Weights (seen during training)
- `[0.3, 0.3, 0.4]`: Balanced configuration
- `[0.5, 0.3, 0.2]`: GC-focused
- `[0.2, 0.5, 0.3]`: MFE-focused
- `[0.2, 0.3, 0.5]`: CAI-focused

### Test Weights (unseen during training)
- `[0.4, 0.4, 0.2]`: GC+MFE focused
- `[0.1, 0.4, 0.5]`: MFE+CAI focused
- `[0.6, 0.2, 0.2]`: High GC
- `[0.1, 0.7, 0.2]`: High MFE
- `[0.1, 0.2, 0.7]`: High CAI
- `[0.33, 0.33, 0.34]`: Nearly equal
- `[0.7, 0.1, 0.2]`: Very high GC
- `[0.1, 0.1, 0.8]`: Very high CAI

### Extreme Test Weights (very challenging)
- `[0.9, 0.05, 0.05]`: Extreme GC
- `[0.05, 0.9, 0.05]`: Extreme MFE
- `[0.05, 0.05, 0.9]`: Extreme CAI
- `[0.0, 0.5, 0.5]`: Zero GC
- `[0.5, 0.0, 0.5]`: Zero MFE
- `[0.5, 0.5, 0.0]`: Zero CAI

## Expected Outcomes

### If Conditional Training Helps Generalization:
- Conditional model should perform better on test weights
- Lower standard deviation in test performance
- Better performance on extreme weight configurations
- Higher Pareto efficiency across different weight configurations

### If Conditional Training Doesn't Help:
- Similar performance between conditional and unconditional models
- No significant statistical difference
- Similar performance on extreme cases

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all required modules are available
2. **Memory Issues**: Reduce `n_samples` or `hidden_dim` for lower memory usage
3. **Training Time**: Use `quick_comparison.py` for faster results
4. **CUDA Issues**: The scripts automatically fall back to CPU if CUDA is not available

### Performance Tips

1. **Start with Quick Comparison**: Use `quick_comparison.py` to get initial results
2. **Use GPU**: Ensure CUDA is available for faster training
3. **Adjust Parameters**: Modify `n_iterations` and `n_samples` based on your needs
4. **Monitor Resources**: Use `nvidia-smi` to monitor GPU memory usage

## Next Steps

After running the comparison:

1. **Analyze the Results**: Check the summary report and visualizations
2. **Interpret Statistical Significance**: Look at p-values in the analysis
3. **Consider Practical Implications**: Even if statistically significant, is the improvement practically meaningful?
4. **Iterate**: Try different training configurations or weight sets
5. **Document**: Record your findings for future reference

## Example Workflow

```bash
# 1. Quick test
python quick_comparison.py

# 2. If results look promising, run comprehensive analysis
python compare_conditional_vs_unconditional.py \
    --n_iterations 300 \
    --n_samples 100 \
    --wandb_project "mRNA_generalization"

# 3. Analyze results
cat generalization_analysis/summary_report_*.txt

# 4. View visualizations
ls generalization_analysis/*.png
```

This systematic approach will help you determine whether conditional training provides meaningful improvements in generalization for your mRNA design task.
