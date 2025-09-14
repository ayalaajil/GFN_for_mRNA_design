# Methodology

## Environment Design

### Codon Design Environment
We developed a specialized environment (`CodonDesignEnv`) for mRNA sequence design that operates on codon-level granularity. The environment is built as a discrete environment where:

- **State Space**: Each state represents a partially constructed mRNA sequence as a tensor of codon indices, with `-1` indicating unassigned positions
- **Action Space**: The action space consists of all 64 possible codons plus an exit action (total of 65 actions)
- **Sequence Length**: The environment is initialized with a target protein sequence, and the mRNA sequence length is determined by the number of amino acids in the protein

### Dynamic Masking Strategy
A critical component of our methodology is the implementation of dynamic masking during sequence generation:

- **Synonymous Codon Masking**: At each step `t < seq_length`, only codons that encode the amino acid at position `t` of the target protein sequence are allowed. This ensures biological validity by constraining the action space to synonymous codons only
- **Termination Masking**: When the sequence reaches the target length (`t == seq_length`), only the exit action is permitted, forcing the agent to terminate sequence generation
- **Backward Masking**: For backward sampling, only the last added codon can be removed, maintaining the sequential construction property

The masking is implemented in the `update_masks()` method, which dynamically updates forward and backward action masks based on the current state of sequence construction.

## Sampling Policy and Training

### GFlowNet Architecture
We employed a SubTB (Sub-Trajectory Balance) GFlowNet with three key components:

1. **Forward Policy (PF)**: A neural network that estimates the probability of selecting each codon given the current partial sequence
2. **Backward Policy (PB)**: A neural network that estimates the probability of removing the last codon from a sequence
3. **Log State Flow (LogF)**: A scalar estimator that approximates the logarithm of the state flow function

### Network Architectures
We experimented with multiple neural network architectures:

- **MLP**: Standard multi-layer perceptron with configurable hidden dimensions and layers
- **MLP_ENN**: Enhanced MLP with additional architectural improvements
- **Transformer**: Attention-based architecture for capturing long-range dependencies in codon sequences

### Training Process
The training procedure follows the SubTB algorithm:

1. **Trajectory Sampling**: At each iteration, we sample `batch_size` trajectories using the current forward policy with epsilon-greedy exploration
2. **Loss Computation**: The SubTB loss is computed using the sampled trajectories, incorporating both forward and backward policy estimates
3. **Gradient Updates**: Parameters are updated using Adam optimizer with separate learning rates for logZ and other parameters
4. **Learning Rate Scheduling**: We employ ReduceLROnPlateau scheduler to adaptively reduce learning rates based on loss convergence

### Conditional vs Unconditional Training
Our framework supports both training paradigms:

- **Unconditional Training**: The model learns to generate sequences without specific objective preferences, using fixed weight configurations
- **Conditional Training**: The model is trained with randomly sampled weight configurations, enabling it to generate sequences optimized for different objective preferences

## Evaluation Framework

### Multi-Objective Reward Function
We designed a comprehensive reward function that balances three key biological objectives:

1. **GC Content (GC)**: Normalized reward based on optimal GC content range (35-65%)
   ```
   gc_reward = (gc_content/100 - 0.35) / (0.65 - 0.35)
   ```

2. **Minimum Free Energy (MFE)**: Reward based on RNA secondary structure stability
   ```
   mfe_reward = -mfe_energy / sequence_length
   ```

3. **Codon Adaptation Index (CAI)**: Reward based on codon usage efficiency
   ```
   cai_reward = cai_score  # already normalized to [0,1]
   ```

The final reward is a weighted combination:
```
reward = w_gc * gc_reward + w_mfe * mfe_reward + w_cai * cai_reward
```

### Evaluation Metrics
Our evaluation framework includes:

- **Pareto Efficiency**: Identification of sequences that are Pareto-optimal across all three objectives
- **Diversity Analysis**: Measurement of sequence diversity using edit distance distributions
- **Quality Metrics**: Statistical analysis of reward components and their distributions
- **Inference Performance**: Timing analysis for sequence generation efficiency

### Biological Validation
We incorporate several biological validation steps:

- **MFE Calculation**: Using the Zucker algorithm for RNA secondary structure prediction
- **CAI Computation**: Calculating codon adaptation index based on reference codon usage tables
- **GC Content Analysis**: Vectorized computation of GC content using precomputed codon GC counts

## Generalization Testing

### Weight Configuration Sweep
To assess model generalization, we designed a comprehensive testing framework that evaluates performance across diverse objective preferences:

1. **Single Objective Extremes**: Testing with weights [1,0,0], [0,1,0], [0,0,1] for GC, MFE, and CAI respectively
2. **Two Objective Combinations**: Balanced combinations like [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]
3. **Balanced Configurations**: Various balanced weight distributions
4. **Extreme Unbalanced**: Highly skewed weight distributions to test robustness

### Generalization Metrics
For each weight configuration, we measure:

- **Statistical Performance**: Mean and standard deviation of each objective metric
- **Sample Diversity**: Number of unique sequences generated
- **Reward Distribution**: Analysis of reward value distributions
- **Cross-Configuration Consistency**: Performance stability across different weight settings

### Generalization Analysis
The generalization testing framework:

1. **Generates sequences** using the trained model for each weight configuration
2. **Computes comprehensive statistics** for all biological metrics
3. **Creates detailed reports** with performance summaries and visualizations
4. **Saves results** in structured formats (CSV, summary files) for further analysis

## Implementation Details

### Computational Environment
- **Hardware**: CUDA-enabled GPU for efficient neural network training and inference
- **Software**: PyTorch-based implementation with custom GFlowNet components
- **Reproducibility**: Fixed random seeds and deterministic operations for consistent results

### Hyperparameter Configuration
Key hyperparameters include:
- Learning rates: 0.005 for main parameters, 0.1 for logZ
- Batch size: 16 sequences per training iteration
- Network architecture: 128 hidden dimensions, 3 hidden layers
- SubTB parameters: λ=0.9, geometric weighting scheme
- Training iterations: 300 epochs with early stopping based on loss convergence

### Experimental Tracking
We employ Weights & Biases (WandB) for comprehensive experiment tracking, logging:
- Training curves and loss evolution
- Reward component distributions
- Generated sequence statistics
- Generalization test results
- Model performance metrics

This methodology provides a robust framework for mRNA sequence design that balances biological validity, computational efficiency, and multi-objective optimization capabilities.
