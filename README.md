
# nanoGPT: Scaling Studies Fork

This repository is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) designed specifically for **Scaling Studies** and **Hyperparameter Optimization (HPO)**. It extends the original framework with modern architectural features, advanced parametrization support, and an automated experiment runner to facilitate research into scaling laws.

![scaling](https://github.com/karpathy/nanoGPT/assets/49915909/c2759972-2d88-4ca4-86a0-2f102558f62f)
*(Example scaling curve visualization)*

## Key Features

### 🏗️ Modern Architecture (LLaMA-Style)
This fork adds support for components found in modern LLMs like LLaMA:
- **RMSNorm**: Replaces LayerNorm for stability (`--use_rmsnorm=True`).
- **Rotary Embeddings (RoPE)**: Replaces absolute positional embeddings (`--use_rope=True`).
- **SwiGLU**: Replaces GELU MLP, with standard $8/3d$ scaling (`--use_swiglu=True`).

### ⚡ Optimization
- **Muon Optimizer**: Integration of the Muon optimizer for 2D parameters, combined with AdamW for 1D parameters (embeddings/norms).
- **Scaling Metrics**:
  - Precise FLOPs estimation (MFU) adapted for new architectures.
  - "Tokens Seen" logging for consistent tracking across batch size changes.
  - Detailed parameter count breakdowns and gradient/activation statistics logging.

### 🎛️ Advanced Parametrization
Supports flexible parametrization schemes to stabilize training at scale:
- **Standard Parametrization (SP)**: Default PyTorch initialization.
- **Maximal Update Parametrization (MuP)**: Scaling rules for infinite-width transfer.
- **CompleteP**: Deep transformer parametrization.
- *Configurable via `parametrization.py`.*

### 🧪 Experiment Runner
A built-in framework for automation:
- **`experiments.py`**: Programatically define grid searches (e.g., batch size vs. learning rate).
- **`scaling_studies.ipynb`**: Jupyter notebook to drive experiments and visualize scaling laws (Loss vs. Compute, Depth scaling, etc.).
- **`metrics.json`**: Automatically saved at the end of every run for easy analysis.

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib pandas seaborn pyyaml
```

## Usage

### 1. Standard Training
You can run training just like standard nanoGPT, but with new flags for the modern features:

```bash
python train.py config/train_shakespeare_char.py \
    --dataset=shakespeare_char \
    --use_rope=True \
    --use_swiglu=True \
    --use_rmsnorm=True \
    --parametrization.mode=SP \
    --batch_size=32 \
    --compile=True
```

### 2. Running Scaling Experiments
To run a sweep or scaling study, use the `ExperimentRunner` in a script or notebook.

**Example: `scaling_studies.ipynb`**
Open the notebook to run pre-configured studies:
- **Batch Size Scaling**: Observer throughput and loss convergence.
- **Depth Scaling**: Train models with varying `n_layer`.
- **HPO**: Grid search Learning Rate vs Weight Decay.

**Programmatic Usage:**
```python
from experiments import ExperimentRunner

runner = ExperimentRunner(base_config_path='config/train_shakespeare_char.py', output_root='experiments_out')

# Run a grid search
runner.run_grid(
    grid_name='depth_study',
    base_params={'dataset': 'shakespeare_char', 'max_iters': 1000},
    grid_params={'n_layer': [2, 4, 8]}
)
```

## Directory Structure
- `train.py`: Main training script (refactored to use `ExperimentConfig`).
- `model.py`: GPT model definition with LLaMA features.
- `config.py`: Configuration dataclasses (replacing global configs).
- `parametrization.py`: Initialization and LR scaling logic.
- `experiments.py`: Experiment automation engine.
- `muon.py`: Muon optimizer implementation.
