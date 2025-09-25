## Curriculum-Augmented GFlowNets for mRNA Design
This repository implements Curriculum-Augmented GFlowNets (CAGFN) for multi-objective mRNA design, enabling the generation of diverse codon sequences optimized for biologically relevant properties (GC content, minimum free energy, Codon Adaptation Index).

We provide three training modes:

- Unconditional GFlowNet
- Conditional GFlowNet
- Curriculum-augmented Conditional GFlowNet (CAGFN)

### 1) Environment setup

### Step 1 : Create main environment
We provide a YAML file for dependencies:
```bash
conda env create -f mRNA_env.yml
conda activate mRNA_env
```

### Step 2 : Install TorchGFN
We use torchgfn library, please refer to the torchgn github for installation details. We cloned the repository in our codebase :

```
git clone https://github.com/GFNOrg/torchgfn.git
cd torchgfn
pip install -e ".[all]"
```
### Step 3 : Install Codon Adaptation Index (CAI) library
We use CodonAdaptationIndex library to compute CAI. Please refer to the CAI repository for installation details :
```pip install git+https://github.com/Benjamin-Lee/CodonAdaptationIndex.git```

### Step 4 : Clone the automatic-curriculum
Follow the instructions of this repository : ```https://github.com/lcswillems/automatic-curriculum.git```

### 2) Repository structure (key files)
- `main.py`: Unconditional training entry-point
- `main_conditional.py`: Conditional training entry-point
- `main_curriculum.py`: Curriculum learning driver (builds on conditional)
- `config.yaml`, `config_small.yaml`, `config_medium.yaml`: Example experiment configs
- `env.py`, `preprocessor.py`, `reward.py`, `utils.py`: Core components
- `evaluate.py`, `evaluate_curriculum.py`, `plots.py`, `comparison_utils.py`: Evaluation and plotting utilities

### 3) Configuration
Edit a YAML config to set the target protein and defaults. Example `config.yaml` contains:
- `protein_seq`: amino acid sequence (use `*` to include stop)
- `natural_mRNA_seq`: reference sequence for comparison
- `wandb_project`, `run_name`, `type`, `arch`, `device`

You can also pass `--config_path` to each script to point to another config (e.g., `config_small.yaml`).

### 4) Unconditional GFlowNet

Run training and evaluation:

For example :
```bash
python main.py \
  --config_path config.yaml \
  --n_iterations 200 \
  --batch_size 16 \
  --lr 0.005 \
  --hidden_dim 256 \
  --n_hidden 4 \
  --n_samples 100 \
  --top_n 50 \
  --subTB_lambda 0.9 \
  --epsilon 0.25 \
  --run_name RUN_UNCONDITIONAL_GFN
```

Outputs are saved under `outputs/unconditional/<type>/<run_name>_<timestamp>/`, including:
- Trained weights `trained_gflownet_*.pth`
- Generated sequences and plots (Pareto, histograms, etc.)
- `experiment_summary.txt`

Useful flags (subset):
- `--wandb_project <name>`: enable logging to Weights & Biases
- `--tied`: share PF trunk with PB (when using MLP)

### 5) Conditional GFlowNet

Runs a conditional GFlowNet with weights over objectives (GC, MFE, CAI). Example:
```bash
python main_conditional.py \
  --config_path config_small.yaml \
  --n_iterations 200 \
  --batch_size 64 \
  --lr 0.005 \
  --hidden_dim 256 \
  --n_hidden 4 \
  --n_samples 100 \
  --top_n 50 \
  --subTB_lambda 0.9 \
  --epsilon 0.25 \
  --run_name RUN_CONDITIONAL_GFN \
  --arch Transformer
```

Outputs are saved under `outputs/conditional/<type>/<run_name>_<timestamp>/` with the same artifacts as unconditional.

### 6) Curriculum-augmented Conditional GFlowNet

Trains across a sequence of protein-length tasks using an automatic curriculum.

Default task ranges are defined via `--curriculum_tasks` (as `[min,max]` strings):
```bash
python main_curriculum.py \
  --config_path config_small.yaml \
  --curriculum_tasks "[25,40]" "[45,60]" "[65,80]" "[85,120]" "[125,180]" \
  --n_iterations 10 \
  --train_steps_per_task 200 \
  --eval_every 5 \
  --batch_size 64 \
  --lr 0.005 \
  --hidden_dim 256 \
  --n_hidden 4 \
  --subTB_lambda 0.9 \
  --epsilon 0.25 \
  --run_name RUN_CURRICULUM_GFN \
  --arch Transformer
```

Artifacts:
- Final model under `curriculum_model/<run_name>_<timestamp>/final_curriculum_gflownet.pth`
- Console logs include task distribution and performance summaries
- Optional Weights & Biases logging via `--wandb_project`

Notes:
- Curriculum builds on the conditional setup; it samples tasks and conditions the policy accordingly.
- You can supply single integers (e.g., `30`) or ranges (`"[25,40]"`).

To run the evaluation experiments, use ```run_evaluation.py```

### 7) Reproducibility and devices
- Set seeds with `--seed` (default 42). Scripts call `set_seed` internally.
- Device is auto-detected (CUDA if available). Override in config if needed.


### 8) Minimal quickstart
```bash
# Setup
conda env create -f mRNA_env.yml && conda activate mRNA_env

# Unconditional
python main.py --config_path config.yaml --run_name quick_uncond

# Conditional
python main_conditional.py --config_path config.yaml --run_name quick_cond

# Curriculum
python main_curriculum.py --config_path config.yaml --run_name quick_curri
```


