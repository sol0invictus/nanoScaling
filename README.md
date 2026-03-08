# nanoScaling

A fork of [nanoGPT](https://github.com/karpathy/nanoGPT) built for **scaling studies and optimizer research**. The primary focus is empirical investigation of training dynamics — comparing **Muon** and **Scion** optimizers against AdamW — through spectral geometry analysis and activation statistics.

---

## Features

### Modern Architecture
Fully configurable decoder-only Transformer with LLaMA-style components:

| Flag | Default | Description |
|---|---|---|
| `use_rmsnorm` | `false` | RMSNorm instead of LayerNorm |
| `use_rope` | `false` | Rotary positional embeddings (RoPE) |
| `use_swiglu` | `false` | SwiGLU FFN with 8/3·d hidden dim |
| `use_hybrid` | `false` | Interleave GatedDeltaNet linear-attention layers |
| `norm_position` | `'pre'` | `'pre'`, `'post'`, or `'none'` (norm-free) |
| `norm_affine` | `true` | Learnable γ/β on norm layers |

### Optimizers
- **AdamW** — standard baseline
- **Muon** — Newton-Schulz orthogonalization on 2D weight gradients; AdamW on 1D params (embeddings, norms). Controlled by `muon_lr`, `muon_momentum`, `muon_ns_steps`.
- **Scion** — spectral normalization variants. Controlled by `--scion_norm` (`Spectral`, `ColNorm`, `RowNorm`, etc.).

### Data Pipeline
- **Parquet-native**: OpenWebText, FineWeb-Edu, and all validation sets are stored as raw-text parquet shards, tokenized on-the-fly during training. No pre-tokenized `.bin` files needed.
- **BOS-aligned best-fit packing**: 100% token utilization, no padding.
- **Auto-detection**: `create_dataloader()` routes to parquet or legacy `.bin` automatically based on what's present in the dataset directory.
- **Val split convention**: last shard in the dataset folder = val, all others = train.

### DDP-Safe Validation
All ranks participate in `estimate_loss()` using a weighted `all_reduce(SUM) / effective_world` that remains unbiased even when `world_size` is not divisible by the number of val row groups. No deadlocks even with tiny val shards.

### Spectral Metrics
`utils/spectral_metrics.py` logs weight geometry, gradient statistics, and activation stats to TensorBoard and CSV at every `log_interval`. Full SVD spectra saved as JSON at each checkpoint.

### Multi-Dataset Validation
- `val_splits` — evaluate on named splits within the training dataset
- `val_datasets` — evaluate on separate external dataset folders (parquet or bin)

### Parametrization
`utils/parametrization.py` applies initialization scaling and per-parameter LR multipliers. Modes: `SP` (Standard), `MuP` (Maximal Update), `CompleteP`.

---

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib pandas seaborn pyyaml pyarrow requests

# Optional — Mixture of Experts (configs/moe.yaml):
pip install megablocks einops

# Optional — Hybrid GatedDeltaNet (configs/hybrid_gated_delta_net.yaml):
pip install flash-linear-attention
```

---

## Quick Start

```bash
# Single GPU, AdamW
python train.py --batch_size=32 --compile=False --optimizer=adamw

# Muon optimizer
python train.py --optimizer=muon --learning_rate=0.02

# Scion optimizer
python train.py --optimizer=scion --learning_rate=0.001 --scion_norm=Spectral

# Full config file
python train.py configs/train_full.yaml

# Multi-GPU DDP (8 GPUs)
torchrun --standalone --nproc_per_node=8 train.py configs/train_full.yaml
```

---

## Data Preparation

All datasets output `shard_?????.parquet` files. The **last shard is always the val split**; all others are train. No tokenization step required — the dataloader tokenizes on-the-fly using GPT-2 BPE.

```bash
# OpenWebText (~8M docs → ~320 train shards + 1 val shard)
python data/openwebtext/prepare.py                        # full dataset
python data/openwebtext/prepare.py --max_shards 10        # subset for experiments
python data/openwebtext/prepare.py --num_workers 16       # faster parallel writes

# FineWeb-Edu — Option A: download pre-built shards (fast if bandwidth allows)
python data/fineweb_edu/download.py -n 10                 # 10 shards (~1 GB)
python data/fineweb_edu/download.py -n 1823 -w 8          # full 180 GB

# FineWeb-Edu — Option B: stream and repackage from HuggingFace
python data/fineweb_edu/prepare.py --max_shards 10
python data/fineweb_edu/prepare.py                        # full dataset (hours)

# Validation datasets (for val_datasets: in config)
python data/wikitext103/prepare.py     # WikiText-103 val
python data/pile_val/prepare.py        # The Pile val (10k docs)
python data/code_val/prepare.py        # Python code (codeparrot, 5k files)
python data/math_val/prepare.py        # GSM8K math problems (~1.3k)

# Shakespeare (smoke tests)
python data/shakespeare/prepare_parquet.py
```

Then set `dataset: openwebtext` (or `fineweb_edu`) in your config YAML.

### Val Split Details

After `data/openwebtext/prepare.py`:

```
data/openwebtext/
  shard_00000.parquet   ← train
  shard_00001.parquet   ← train
  ...
  shard_00319.parquet   ← train
  shard_00320.parquet   ← val  (last shard, ~4007 docs)
```

No additional config is needed — `create_dataloader('val', ...)` automatically picks up the last shard. The `val_datasets` config field is for *additional* held-out sets (code, math, wikitext, etc.).

---

## Configuration

Configs are YAML files; CLI flags override any field (`--key=value`). Nested fields use dot notation (`--parametrization.mode=MuP`).

```yaml
# configs/train_full.yaml (excerpt)
dataset: "data/openwebtext"
val_datasets:
  - /home/nanoScaling/val/code_val
  - /home/nanoScaling/val/math_val
  - /home/nanoScaling/val/wikitext_val

optimizer: "muon"
learning_rate: 3e-3
n_layer: 12
n_head: 12
n_embd: 768
use_rmsnorm: true
use_rope: true
use_swiglu: true
```

Key config fields:

| Field | Description |
|---|---|
| `optimizer` | `adamw`, `muon`, `scion` |
| `muon_lr` / `muon_momentum` / `muon_ns_steps` | Muon-specific hyperparameters |
| `scion_norm` | Scion normalization variant |
| `norm_position` | `pre`, `post`, `none` |
| `norm_affine` | Learnable norm scale/bias |
| `data_format` | `auto`, `bin`, `parquet` |
| `val_splits` | Split names within training dataset to eval |
| `val_datasets` | Paths to external eval dataset folders |
| `checkpoint_interval` | Save every N steps (0 = tied to eval_interval) |
| `tensorboard_log` | Enable TensorBoard logging |
| `use_hybrid` | Enable GatedDeltaNet hybrid model |
| `parametrization.mode` | `SP`, `MuP`, `CompleteP` |

---

## Research Experiments

### RQ1: Spectral Geometry (Muon vs AdamW)

```bash
python experiments/rq1_spectral_geometry/train.py \
    experiments/rq1_spectral_geometry/configs/125m_muon.yaml

# Run all 5 conditions (baselines + ns_steps/momentum ablations)
bash experiments/rq1_spectral_geometry/run_rq1.sh

# Quick smoke test
bash experiments/rq1_spectral_geometry/run_rq1.sh --quick
```

Logs weight geometry (`stable_rank`, top-k singular values), gradient norms, cosine similarity, and activation stats to TensorBoard + CSV at every `log_interval`.

### RQ2: Normalization Ablations (can Muon train without LayerNorm?)

```bash
python experiments/rq2_normalization/train.py \
    experiments/rq2_normalization/configs/125m_muon_no_norm.yaml

# Run all 12 conditions (6 norm variants × 2 optimizers)
bash experiments/rq2_normalization/run_rq2.sh
```

Adds `StabilityMonitor` (grad norm spikes, NaN detection, loss monotonicity) and writes `is_stable` + `stability_log` to `metrics.json`.

### Programmatic Grid Search

```python
from experiments import ExperimentRunner

runner = ExperimentRunner(
    base_config_path='configs/train_full.yaml',
    output_root='experiments_out'
)
runner.run_grid(
    grid_name='depth_study',
    base_params={'dataset': 'shakespeare_char', 'max_iters': 1000},
    grid_params={'n_layer': [2, 4, 8]}
)
```

---

## TensorBoard Logging

```bash
tensorboard --logdir out/runs
```

Tag hierarchy:

| Prefix | Content |
|---|---|
| `train/` | Loss, LR, MFU, tokens seen |
| `val/` | Val loss per split and dataset |
| `weight/` | Stable rank, top-k singular values |
| `grad/` | Gradient norms, cosine similarity |
| `activations/` | Per-block mean/std |
| `sharpness/` | Loss sharpness (disabled by default) |
| `stability/` | RQ2 stability monitor |
| `checkpoint/weight/` | Full SVD spectra at checkpoints |

CSV files are also written to `{out_dir}/logs/` and are safe to read mid-run with `pd.read_csv(...)`.

---

## Directory Structure

```
train.py                        # Main pretraining loop (DDP, compile, Muon/Scion/AdamW)
train_sft.py                    # Supervised fine-tuning
models/
  gpt.py                        # Decoder-only Transformer (RMSNorm, RoPE, SwiGLU, MoE, Hybrid)
  gated_delta_net.py            # GatedDeltaNet linear-attention layer
optimizers/
  muon.py                       # Muon optimizer
  scion.py                      # Scion optimizer
utils/
  config.py                     # ExperimentConfig dataclass
  data.py                       # create_dataloader() factory, bin-path get_batch()
  dataloader.py                 # Parquet on-the-fly dataloader with best-fit packing
  spectral_metrics.py           # SpectralLogger, StabilityMonitor
  parametrization.py            # SP / MuP / CompleteP
data/
  openwebtext/prepare.py        # → parquet shards (parallel, multiprocess)
  fineweb_edu/prepare.py        # → parquet shards (streaming)
  fineweb_edu/download.py       # Download pre-built shards
  wikitext103/prepare.py        # Val: WikiText-103
  pile_val/prepare.py           # Val: The Pile
  code_val/prepare.py           # Val: Python code (codeparrot)
  math_val/prepare.py           # Val: GSM8K math
  shakespeare/prepare_parquet.py # Smoke test dataset
configs/
  train_full.yaml               # Full training config (Muon, openwebtext, val datasets)
  moe.yaml                      # Mixture of Experts
  hybrid_gated_delta_net.yaml   # Hybrid linear+causal attention
experiments/
  rq1_spectral_geometry/        # Muon vs AdamW weight geometry study
  rq2_normalization/            # Normalization ablations
  structured_weight_decay/      # Structured weight decay experiment
helper_notebooks/
  scaling_studies.ipynb         # Drive experiments, plot scaling curves
  scaling_laws.ipynb            # Scaling law fitting
  transformer_sizing.ipynb      # Model size / compute calculator
```

---

## Other Scripts

```bash
python sample.py        # Generate text from a checkpoint
python bench.py         # Benchmark throughput
python eval_sft.py      # Evaluate SFT models
python debug_yaml.py    # Debug YAML config loading
```
