# Short Sequence Specialist Training System

This system trains a Conditional GFlowNet to become a specialist on short protein sequences (30-40 amino acids) and evaluates its performance on unseen proteins.

## Overview

The system consists of three main components:

1. **Training Script** (`train_short_specialist.py`): Trains a Conditional GFN on a collection of short protein sequences
2. **Evaluation Script** (`evaluate_short_specialist.py`): Tests the trained specialist on unseen proteins
3. **Runner Script** (`run_short_specialist.py`): Easy-to-use interface for both training and evaluation

## Key Features

- **Specialized Training**: Trains on 50 different protein sequences in the 30-40 AA range
- **Model Persistence**: Saves trained models for later evaluation
- **Comprehensive Evaluation**: Tests on different types of unseen sequences
- **Multiple Weight Combinations**: Evaluates performance under different reward weightings (GC, MFE, CAI)
- **Detailed Logging**: Comprehensive logging and result tracking
- **Weights & Biases Integration**: Optional experiment tracking

## Quick Start

### Basic Usage

```bash
# Train a specialist on 50 short sequences
python run_short_specialist.py --mode train --num_sequences 50

# Train and then evaluate
python run_short_specialist.py --mode both --num_sequences 50 --num_test_sequences 20

# Only evaluate (uses latest trained model)
python run_short_specialist.py --mode eval --test_type unseen_short --num_test_sequences 20
```

### Advanced Usage

```bash
# Train with custom parameters
python run_short_specialist.py --mode train \
    --num_sequences 30 \
    --n_iterations 500 \
    --batch_size 32 \
    --lr 0.01 \
    --hidden_dim 512

# Evaluate on medium-length sequences
python run_short_specialist.py --mode eval \
    --test_type medium \
    --num_test_sequences 15 \
    --num_samples 200
```

## Detailed Usage

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_sequences` | 50 | Number of short sequences to train on |
| `--min_length` | 30 | Minimum protein sequence length |
| `--max_length` | 40 | Maximum protein sequence length |
| `--n_iterations` | 200 | Training iterations per sequence |
| `--batch_size` | 16 | Batch size |
| `--lr` | 0.005 | Learning rate |
| `--lr_logz` | 0.1 | LogZ learning rate |
| `--embedding_dim` | 32 | Embedding dimension |
| `--hidden_dim` | 256 | Hidden dimension |
| `--n_hidden_layers` | 4 | Number of hidden layers |

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--test_type` | "unseen_short" | Type of test sequences |
| `--num_test_sequences` | 20 | Number of test sequences |
| `--num_samples` | 100 | Samples per weight combination |
| `--model_path` | None | Path to model (auto-detects latest) |

### Test Types

- **`unseen_short`**: Short sequences (30-40 AA) not seen during training
- **`medium`**: Medium-length sequences (50-100 AA)
- **`long`**: Long sequences (>100 AA)
- **`mixed`**: Mixed length sequences

## File Structure

```
outputs/
├── short_specialist/
│   └── Short_Specialist_YYYY-MM-DD_HH-MM/
│       ├── training_results.json
│       ├── training_metadata.json
│       ├── training_summary.txt
│       ├── trained_short_specialist_model.pth
│       └── training.log
└── short_specialist_evaluation/
    └── Evaluation_YYYY-MM-DD_HH-MM/
        ├── evaluation_results.json
        ├── evaluation_summary.txt
        └── evaluation.log
```

## Output Files

### Training Outputs

- **`training_results.json`**: Complete training results with loss/reward histories
- **`training_metadata.json`**: Training configuration and sequence information
- **`training_summary.txt`**: Human-readable training summary
- **`trained_short_specialist_model.pth`**: Saved model for evaluation
- **`training.log`**: Detailed training logs

### Evaluation Outputs

- **`evaluation_results.json`**: Complete evaluation results
- **`evaluation_summary.txt`**: Human-readable evaluation summary
- **`evaluation.log`**: Detailed evaluation logs

## Model Architecture

The Conditional GFlowNet uses:

- **Conditional Policy Estimators**: Forward and backward policies conditioned on reward weights
- **Conditional LogZ Estimator**: Partition function estimator
- **SubTB Loss**: Subtrajectory balance loss with λ=0.9
- **MLP_ENN Architecture**: Multi-layer perceptron with enhanced neural networks

## Training Process

1. **Data Loading**: Loads 50 short protein sequences (30-40 AA) from existing datasets
2. **Sequential Training**: Trains on each sequence individually with different reward weightings
3. **Weight Sampling**: Uses Dirichlet distribution to sample GC/MFE/CAI weights
4. **Model Saving**: Saves model configuration and metadata for evaluation

## Evaluation Process

1. **Model Loading**: Loads trained model configuration
2. **Test Sequence Selection**: Selects unseen sequences based on test type
3. **Multi-Weight Evaluation**: Tests with different weight combinations:
   - GC-focused: [1.0, 0.0, 0.0]
   - MFE-focused: [0.0, 1.0, 0.0]
   - CAI-focused: [0.0, 0.0, 1.0]
   - Balanced: [0.33, 0.33, 0.34]
4. **Sequence Generation**: Generates mRNA sequences for each weight combination
5. **Analysis**: Computes rewards, diversity, and other metrics

## Metrics

### Training Metrics
- **Loss**: SubTB loss per iteration
- **Reward**: Average reward per iteration
- **Components**: GC content, MFE, CAI scores
- **Diversity**: Number of unique sequences generated

### Evaluation Metrics
- **Average Reward**: Mean reward across all weight combinations
- **Sequence Diversity**: Ratio of unique to total sequences
- **Component Scores**: Average GC, MFE, CAI scores
- **Generalization**: Performance on unseen sequence lengths

## Example Results

### Training Summary
```
Short Sequence Specialist Training Summary
==================================================

Training Date: 2025-01-16 14:30:00
Number of sequences trained on: 50
Sequence length range: 30-40 AA
Average sequence length: 35.2 AA
Total training time: 1250.5 seconds
Average time per sequence: 25.0 seconds
Total unique sequences generated: 2,450

Final Loss Statistics:
  Average: 0.0234
  Min: 0.0156
  Max: 0.0345
  Std: 0.0045

Final Reward Statistics:
  Average: 0.8765
  Min: 0.7234
  Max: 0.9456
  Std: 0.0456
```

### Evaluation Summary
```
Short Sequence Specialist Evaluation Summary
==================================================

Evaluation Date: 2025-01-16 15:45:00
Test Type: unseen_short
Number of test sequences: 20
Samples per weight combination: 100
Total evaluation time: 450.2 seconds

Overall Performance Statistics:
  Average Reward: 0.8234 ± 0.0567
  Reward Range: [0.7123, 0.9234]
  Average Diversity: 0.8567 ± 0.0234
  Average Unique Sequences per Test: 85.7
  Total Unique Sequences Generated: 1,714
```

## Troubleshooting

### Common Issues

1. **No training data found**: Ensure the training dataset CSV files exist
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Model not found**: Check that training completed successfully
4. **Import errors**: Ensure all dependencies are installed

### Performance Tips

1. **GPU Usage**: Use CUDA for faster training and evaluation
2. **Batch Size**: Increase batch size if you have sufficient memory
3. **Learning Rate**: Adjust learning rate based on convergence
4. **Sequence Count**: More training sequences generally improve performance

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- tqdm
- Weights & Biases (optional)
- Custom modules: `torchgfn`, `env`, `preprocessor`, `reward`, etc.

## Citation

If you use this system in your research, please cite the relevant papers and acknowledge this implementation.

## License

This code is provided for research purposes. Please ensure compliance with any applicable licenses for the underlying GFlowNet implementation.
