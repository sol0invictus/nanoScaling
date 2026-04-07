# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nanoScaling** is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) designed for scaling studies and hyperparameter optimization of LLMs. The primary research focus is empirical investigation of training dynamics—particularly comparing the **Muon** and **Scion** optimizers against AdamW—through spectral geometry analysis and activation statistics.

See [research_plan/muon_research_proposal.md](research_plan/muon_research_proposal.md) for the full research agenda, experimental hypotheses (RQ1, RQ2, RQ4), compute budget, and 4-week execution plan.

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib pandas seaborn pyyaml pyarrow requests

# MoE training (configs/moe.yaml) — no extra dependencies needed (pure PyTorch)

# Optional — required for Hybrid Gated-Delta-Net (configs/hybrid_gated_delta_net.yaml):
pip install flash-linear-attention
```

## Common Commands

### Training

```bash
# Basic training (single GPU)
python train.py --batch_size=32 --compile=False --optimizer=adamw

# With a YAML config file (uses muon optimizer, shakespeare training data, val/ eval datasets)
python train.py configs/train_full.yaml

# With Muon optimizer
python train.py --optimizer=muon --learning_rate=0.02

# With Scion optimizer
python train.py --optimizer=scion --learning_rate=0.001 --scion_norm=Spectral

# Multi-GPU DDP
torchrun --standalone --nproc_per_node=4 train.py configs/train_full.yaml

# SFT data preparation (downloads from HuggingFace, packs into .bin blocks)
python data/prepare_sft.py --datasets alpaca                  # ~52k examples, fast
python data/prepare_sft.py --datasets alpaca ultrachat        # recommended default
python data/prepare_sft.py --datasets alpaca --max_samples 5000  # quick smoke test

# Supervised fine-tuning (from scratch or from a pretrained checkpoint)
python train_sft.py configs/sft.yaml
python train_sft.py configs/sft.yaml --init_from=out/ckpt.pt  # fine-tune from PT ckpt

# Multi-GPU SFT
torchrun --standalone --nproc_per_node=4 train_sft.py configs/sft.yaml

# Mixture of Experts training (pure PyTorch, no extra deps)
python train.py configs/moe.yaml

# Hybrid Gated-Delta-Net + Attention (requires: pip install flash-linear-attention)
python train.py configs/hybrid_gated_delta_net.yaml
```

### Research Experiments (RQ1 / RQ2)

```bash
# RQ1: Spectral geometry — Muon vs. AdamW weight matrix structure
python experiments/rq1_spectral_geometry/train.py \
    experiments/rq1_spectral_geometry/configs/125m_muon.yaml

# Run all RQ1 conditions (5 configs: baselines + ns_steps/momentum ablations)
bash experiments/rq1_spectral_geometry/run_rq1.sh

# RQ2: Normalization ablations — can Muon train without LayerNorm?
python experiments/rq2_normalization/train.py \
    experiments/rq2_normalization/configs/125m_muon_no_norm.yaml

# Run all RQ2 conditions (12 configs: 6 norm variants × 2 optimizers)
bash experiments/rq2_normalization/run_rq2.sh

# Quick smoke test (500 iters, shakespeare_char)
bash experiments/rq1_spectral_geometry/run_rq1.sh --quick
bash experiments/rq2_normalization/run_rq2.sh --quick

# Structured weight decay experiment
python experiments/structured_weight_decay/train.py \
    experiments/structured_weight_decay/configs/...yaml
```

### Data Preparation

```bash
# OpenWebText (~8M docs) — produces parquet shards (raw text, tokenized on-the-fly)
python data/openwebtext/prepare.py                          # full dataset (~320 train shards + 1 val)
python data/openwebtext/prepare.py --max_shards 10          # quick experiment subset
python data/openwebtext/prepare.py --num_workers 16         # more parallel shard writers

# FineWeb-Edu — two options:

# Option A: Download pre-built parquet shards from karpathy/fineweb-edu-100b-shuffle (fast)
python data/fineweb_edu/download.py -n 10           # 10 shards (~1 GB, good for experiments)
python data/fineweb_edu/download.py -n 1823 -w 8    # all 1823 shards (~180 GB)

# Option B: Stream from HuggingFace and repackage as parquet (no bulk download needed)
python data/fineweb_edu/prepare.py --max_shards 10  # 10 shards (~250K docs each)
python data/fineweb_edu/prepare.py                  # full dataset (hours)

# Then set dataset: fineweb_edu in your config yaml.
# The dataloader auto-detects parquet shards and tokenizes on-the-fly (no .bin needed).

# WikiText-103 validation set (parquet, for eval_only runs)
python data/wikitext103/prepare.py

# The Pile validation set (parquet, first 10k docs from monology/pile-uncopyrighted)
python data/pile_val/prepare.py
python data/pile_val/prepare.py --per-domain  # also write per-component subdirs

# Code validation set (parquet, first 5k Python files from codeparrot-clean-valid)
python data/code_val/prepare.py

# Math validation set (parquet, ~1.3k GSM8K problems from openai/gsm8k)
python data/math_val/prepare.py

# Shakespeare as parquet (for smoke tests with the parquet dataloader)
python data/shakespeare/prepare_parquet.py

# Legacy: produce extra .bin val sets for openwebtext (wikitext, lambada, wikitext_test)
python tools/produce_val_sets.py
```

### Evaluation

```bash
# Step 1: Pre-download all eval data (run once before first eval)
python data/prepare_eval.py               # all tasks + BPB corpora
python data/prepare_eval.py --skip_bpb    # HF tasks only (faster)
python data/prepare_eval.py --tasks mmlu arc hellaswag gsm8k  # specific tasks

# Step 2: Run benchmarks against a checkpoint
python eval_sft.py out-sft/ckpt_sft.pt

# Select specific tasks
python eval_sft.py out-sft/ckpt_sft.pt --tasks mmlu arc_easy arc_challenge hellaswag gsm8k bpb

# Quick run: 200 samples per task, 50 BPB batches
python eval_sft.py out-sft/ckpt_sft.pt --tasks all --max_samples 200 --max_batches 50

# Save results to JSON
python eval_sft.py out-sft/ckpt_sft.pt --out results.json
```

### Other Scripts

```bash
python sample.py         # Generate text from a trained checkpoint
python bench.py          # Benchmark throughput
python experiments.py    # Run grid searches / scaling studies
python debug_yaml.py     # Debug YAML config loading
```

### Experiment Runner (programmatic)

```python
from experiments import ExperimentRunner

runner = ExperimentRunner(base_config_path='configs/train_full.yaml', output_root='experiments_out')
runner.run_grid(
    grid_name='depth_study',
    base_params={'dataset': 'shakespeare_char', 'max_iters': 1000},
    grid_params={'n_layer': [2, 4, 8]}
)
```

### Tests

```bash
# SFT correctness suite (6 tests: tokenisation, packing, mask shift, attn mask, loss, round-trip)
python tests/test_sft.py
```

**Gitignored data and artifacts**: `.bin`, `.parquet`, `.pt`, `.pkl`, `*.pyc`, `out/`, `experiments_out/`, `wandb/`, `val/`, `research_plan/` are all excluded from version control. Run the relevant `data/*/prepare.py` scripts to regenerate data locally.

**`val/` directory**: `configs/train_full.yaml` points `val_datasets` at `/home/nanoScaling/val/{code_val,math_val,openweb_val,wikitext_val}` using absolute paths. This directory holds pre-built parquet evaluation shards and is gitignored. Populate it by running the relevant `data/*/prepare.py` scripts and copying/symlinking output into `val/`.

## Architecture Overview

### Configuration System (`utils/config.py`)

Central `ExperimentConfig` dataclass holds all hyperparameters. Configs are loaded from YAML files and overridable via CLI flags (`--key=value`). Nested `ParametrizationConfig` controls initialization and LR scaling.

Key fields added for research experiments:
- `norm_position` – `'pre'` (default Pre-LN), `'post'` (Post-LN), `'none'` (no normalization)
- `norm_affine` – `True` (learnable γ/β), `False` (fixed-scale norm, no affine params)
- `norm_free_scaled_init` – tighter residual init (`std=0.02/√n_layer`) for norm-free training
- `muon_lr`, `muon_momentum`, `muon_ns_steps` – Muon hyperparameters (were previously hardcoded)
- `data_format` – `'auto'` (detect), `'bin'` (force memmap), `'parquet'` (force on-the-fly)
- `dataloader_buffer_size` – number of docs buffered for best-fit packing (default 1000)
- `tokenizer_batch_size` – reserved for future threaded tokenization (default 128)
- `val_splits` – list of split names evaluated each `eval_interval` within the training dataset (default `['val']`)
- `val_datasets` – list of external dataset folder paths for additional held-out evaluation (e.g. `['data/wikitext103', 'data/pile_val']`); both parquet and bin formats are auto-detected per folder
- `checkpoint_interval` – save checkpoint every N steps independent of eval (0 = tie to eval_interval)
- `tensorboard_log` / `tensorboard_run_name` – TensorBoard logging toggle and run label
- `multiple_of` – round SwiGLU hidden dim up to nearest multiple (default 256)
- `use_hybrid`, `delta_net_every`, `delta_net_chunk_size` – Hybrid GatedDeltaNet model fields (see Model section)
- `moe_hidden_dim` – MoE expert hidden dim (0 = default 4×n_embd)
- `moe_block_size` – Triton tile size for block-sparse MoE kernels (default 128)
- `load_balance_loss_weight` / `router_z_loss_weight` – MoE auxiliary loss coefficients (default 0.01 / 0.001)
- `use_wandb` / `wandb_project` / `wandb_run_name` – WandB logging (opt-in, used by `train_sft.py`)

### Model (`models/gpt.py`)

Decoder-only Transformer with toggleable modern components:
- `use_rmsnorm` – RMSNorm instead of LayerNorm
- `use_rope` – Rotary positional embeddings (RoPE) instead of absolute PE
- `use_swiglu` – SwiGLU FFN (8/3·d scaling) instead of GELU MLP; hidden dim rounded to `multiple_of`
- `use_moe` – Switch Transformer-style Mixture of Experts with auxiliary load-balancing and router-z losses; pure-PyTorch implementation in `models/moe.py` (no megablocks required)
- `use_hybrid` – Interleave `GatedDeltaNetLayer` (linear attention, O(T) memory) with standard causal attention every `delta_net_every` layers. Requires `pip install flash-linear-attention`. See `models/gated_delta_net.py` and `configs/hybrid_gated_delta_net.yaml`.

Normalization variants (controlled by `norm_position` and `norm_affine`):
- `build_norm(config, for_final_ln=False)` — factory function that returns the right module
- `FixedRMSNorm` / `FixedLayerNorm` — norm layers without learnable affine parameters
- Post-LN applies norm *after* the residual addition inside each `Block`; the final `ln_f` becomes `nn.Identity()` in post/none modes

### Optimizers (`optimizers/`)

- **`muon.py`** – Muon: Newton-Schulz orthogonalization applied to 2D weight gradients; AdamW is used for 1D parameters (embeddings, norms). `ns_steps` controls orthogonalization quality.
- **`scion.py`** – Scion: Spectral normalization with column/row-wise and bias-RMS variants. Controlled by `--scion_norm`.

The combined Muon+AdamW optimizer returns a `CombinedOptimizer` whose `param_groups` list interleaves groups from both optimizers. The experiment training scripts apply LR decay as a **multiplier** on each group's initial LR (so Muon's `0.02` and AdamW's `3e-4` decay in proportion, not to a common value).

### Spectral Metrics (`utils/spectral_metrics.py`)

Measurement infrastructure for RQ1/RQ2/RQ4. Metrics are logged to **TensorBoard** and **CSV files** (`{out_dir}/logs/`).

All metrics fire together every time `log_step()` is called — there are no separate per-metric frequencies. The training scripts call `log_step()` inside the `log_interval` block so everything is tied to a single cadence. The only separate gate is `sharpness_freq` (disabled by default, `0`) because sharpness requires N full forward passes.

| Metric group | TB prefix | CSV file | Function |
|---|---|---|---|
| Weight geometry | `weight/` | `logs/weight.csv` | `stable_rank`, `topk_svd_metrics` |
| Gradients | `grad/` | `logs/grad.csv` | `grad_norms`, `cosine_sim` |
| Activations | `activations/` | `logs/activations.csv` | `activation_stats` |
| Loss/LR/MFU | `train/` | `logs/training.csv` | train loop |
| Val loss | `val/` | `logs/training.csv` | `estimate_loss` |
| Sharpness | `sharpness/` | `logs/sharpness.csv` | `measure_sharpness` |
| Stability (RQ2) | `stability/` | — | `StabilityMonitor` |
| Full SVD | `checkpoint/weight/` | `svd_step_NNNNNNN.json` | `log_checkpoint` |

`SpectralLogger` constructor takes: `model`, `tb_writer`, `out_dir`, `csv_dir`, `sharpness_freq=0`, `topk=32`, `stability_monitor`. The removed per-metric freq params (`spectral_freq`, `grad_freq`, etc.) no longer exist.

`StabilityMonitor` tracks the RQ2 stability criterion (grad norm spikes, NaN detection, loss monotonicity).

**DDP data loading**:
- *Bin path*: all ranks `mmap` the same `.bin` file, draw random positions using per-rank RNG seeds. Diversity is probabilistic.
- *Parquet path*: rank `r` reads row groups at indices `r, r+W, r+2W, ...`. When `world_size > total_row_groups` (common for small val shards), `effective_world = min(world_size, total_rg)` is used and ranks wrap via `ddp_rank % effective_world` to avoid deadlock. `create_parquet_dataloader()` returns a `_LoaderWrapper` that exposes `loader.effective_world` so `estimate_loss` can apply correct per-rank weights in `all_reduce(SUM)` even when `world_size % total_rg != 0`.
- *DDP eval*: all ranks call `estimate_loss()` together. Per-loader weighted `all_reduce(SUM) / effective_world` gives an unbiased mean regardless of how row groups divide across ranks. Only `master_process` logs/checkpoints.

### Training Scripts

- **`train.py`** – Main pretraining loop. Supports DDP, mixed precision, `torch.compile`, gradient accumulation, cosine LR schedule with warmup. Emits `metrics.json` at the end of each run.
- **`train_sft.py`** – SFT loop. Packed conversations with block-diagonal causal attention (doc_ids), instruction masking (loss only on assistant tokens), block-aligned batch sampling, AdamW/Muon/Scion optimizers, TensorBoard + optional WandB, decoupled checkpointing. See SFT section below.
- **`experiments/rq1_spectral_geometry/train.py`** – Adds `SpectralLogger` + CSV logging to the standard loop; correct per-optimizer LR decay for CombinedOptimizer. Spectral/grad/activation metrics logged at `log_interval` cadence alongside train loss.
- **`experiments/rq2_normalization/train.py`** – Same as RQ1, plus `StabilityMonitor`, NaN early-stopping, and stability verdict in `metrics.json`.

### Data Loading (`utils/data.py`, `utils/dataloader.py`)

Two data paths, selected automatically via `data_format` config:

**Parquet path** (`utils/dataloader.py`) — activated when `shard_?????.parquet` files are present in the dataset directory or `data_format='parquet'`:
- `create_parquet_dataloader()` — infinite generator yielding `(x, y)` LongTensor`[B, T]`
- BOS-aligned best-fit packing: finds the largest buffered doc that fits each row slot; crops shortest doc when nothing fits. Gives 100% token utilization, no padding, ~35% of tokens discarded by cropping.
- GPT-2 EOT token (50256) prepended to each document as BOS
- DDP: rank `r` reads row groups at indices `r, r+W, r+2W, ...` from shared parquet files (no pre-splitting)
- Val split convention: last shard = val, all others = train

**Bin path** (`utils/data.py`) — used for Shakespeare and any dataset with pre-tokenized `.bin` files:
- `get_batch()` — memory-mapped uint16 arrays, random position sampling per batch
- DDP: all ranks read from the same `.bin` file with per-rank RNG seeds

**Factory**: `create_dataloader(split, config, ...)` returns an infinite generator for either path. All three training scripts call `next(loader)` through the same `get_batch(split)` local function — the call sites are unchanged.

**Evaluation datasets** (parquet, used via `val_datasets` config field):
- `data/wikitext103/prepare.py` — WikiText-103 raw val set; produces `shard_00000.parquet` (train) + `shard_00001.parquet` (val)
- `data/pile_val/prepare.py` — First 10k docs from `monology/pile-uncopyrighted` train split (original `EleutherAI/pile` is broken — the-eye.eu is offline); `--per-domain` writes per-component subdirs under `data/pile_val/domains/`
- `data/code_val/prepare.py` — First 5k Python files from `codeparrot/codeparrot-clean-valid`; standard code perplexity benchmark
- `data/math_val/prepare.py` — `openai/gsm8k` test split (~1.3k grade-school math problems + chain-of-thought answers); widely cited math benchmark
- `data/shakespeare/prepare_parquet.py` — converts Shakespeare `input.txt` to parquet shards for smoke-testing the parquet dataloader

**OpenWebText data utilities**:
- `data/openwebtext/prepare.py` — downloads `Skylion007/openwebtext` from HuggingFace, splits train/val (0.05%, seed 2357), and writes raw text as parquet shards. Parallel shard writing via `--num_workers` (default: `os.cpu_count()`). Last shard = val. Supports `--max_shards`, `--docs_per_shard`, `--output_dir`.

**FineWeb-Edu data utilities**:
- `data/fineweb_edu/download.py` — downloads pre-built parquet shards from `karpathy/fineweb-edu-100b-shuffle` (1823 shards, ~180 GB total)
- `data/fineweb_edu/prepare.py` — streams `HuggingFaceFW/fineweb-edu` and saves raw text to parquet shards (no tokenization); output format: `shard_00000.parquet`, ..., zstd-compressed, 1024 docs/row-group

### SFT Pipeline (`data/prepare_sft.py`, `train_sft.py`)

**Data preparation** (`data/prepare_sft.py`) — consolidated script that downloads, formats, packs, and writes all SFT data:

Supported datasets (all public, no auth required):

| Name | Flag | Source | Size |
|---|---|---|---|
| Alpaca | `alpaca` | `tatsu-lab/alpaca` | 52k single-turn |
| UltraChat | `ultrachat` | `HuggingFaceH4/ultrachat_200k` | 207k multi-turn |
| OASST1 | `oasst1` | `OpenAssistant/oasst1` | multilingual, EN-filtered |
| OpenHermes | `openhermes` | `teknium/OpenHermes-2.5` | 1M mixed-source |

Chat format (GPT-2 tokenizer, no special tokens added):
```
<|endoftext|>User: {turn1_user}\nAssistant: {turn1_response}\nUser: {turn2_user}\nAssistant: {turn2_response}<|endoftext|>
```
Loss mask = 1 for assistant content tokens and the final EOT; 0 for everything else (BOS, user turns, "Assistant: " prefix, padding).

Packing: conversations are shuffled and packed sequentially into blocks of `block_size`. Each block begins with a BOS (EOT_ID=50256) token, so block-aligned sampling in `get_batch` always starts at a conversation boundary. `doc_ids` increment per conversation within a block for block-diagonal attention.

Output files (default: `data/sft/`):
- `{split}.bin` — uint16 tokens, flattened (N_blocks × block_size)
- `{split}_mask.bin` — uint8 loss mask, same shape
- `{split}_doc_ids.bin` — uint32 conversation ids, same shape

**Training loop** (`train_sft.py`) key design points:

- **Block-aligned sampling**: `get_batch` samples at `block_idx * block_size` offsets, so every sequence starts with BOS and never mid-conversation.
- **Mask shift**: The loss mask is applied as `y[t] = -1 when mask[t+1] == 0`. This is the correct shift — `y[t] = x[t+1]` is the target, and `mask[t+1]` says whether token `x[t+1]` should be predicted. Applying `mask[t]` instead would be off by one (loses the first assistant token, trains on wrong tokens).
- **Block-diagonal attention**: `make_attn_mask(doc_ids)` builds a `(B, 1, T, T)` bool mask — tokens attend only within their own conversation, causally. Passed as `attn_mask` to `model.forward()`.
- **Optimizer**: dispatched through `model.configure_optimizers()` — same as `train.py`. Supports AdamW, Muon (2D weights), and Scion. `CombinedOptimizer` param groups decay proportionally via per-group initial LR multipliers.
- **`raw_model`**: assigned after both `torch.compile` and DDP are applied (`raw_model = model.module` for DDP, else the compiled/plain model). Avoids the crash from accessing `.module` on `OptimizedModule` before DDP wrapping.
- **WandB**: opt-in via `use_wandb: True` in config (fields: `wandb_project`, `wandb_run_name`).

Config: `configs/sft.yaml`. Set `init_from` to a checkpoint path for fine-tuning, or `"scratch"` for testing.

**Test suite** (`tests/test_sft.py`): 6 tests covering tokenisation alignment, packing correctness, mask shift verification, block-diagonal attention structure, forward-pass loss isolation, and full binary round-trip. Run with `python tests/test_sft.py`.

### Eval Suite (`evals/`, `eval_sft.py`)

Mirrors the nanochat ChatCORE benchmark. All tasks download from HuggingFace automatically on first run.

| Task | File | Method | Metric | Random baseline |
|---|---|---|---|---|
| MMLU | `evals/tasks/mmlu.py` | logit at answer pos (single fwd pass/Q) | accuracy | 0.25 |
| ARC-Easy | `evals/tasks/arc.py` | logit at answer pos | accuracy | 0.25 |
| ARC-Challenge | `evals/tasks/arc.py` | logit at answer pos | accuracy | 0.25 |
| HellaSwag | `evals/tasks/hellaswag.py` | per-continuation CE loss (4 fwd passes/Q) | accuracy | 0.25 |
| GSM8K | `evals/tasks/gsm8k.py` | greedy generation + regex extraction | exact-match | 0.00 |
| BPB | `evals/tasks/bpb.py` | per-token CE / bytes | bits/byte | n/a |

**Scoring utils** (`evals/scoring.py`):
- `logit_mc(model, prompt_ids, letters, device)` — single forward pass, compares log-probs of answer-letter tokens (` A`=317, ` B`=347, ` C`=327, ` D`=360 in GPT-2). Used for MMLU and ARC.
- `completion_loss_mc(model, context_ids, continuations, device, block_size)` — one forward pass per multi-token continuation, picks min mean CE loss. Used for HellaSwag.
- `generate_greedy(model, prompt_ids, max_new_tokens, device)` — calls `model.generate(top_k=1)` with KV cache, strips EOS. Used for GSM8K.
- `centered_accuracy(raw, baseline)` — ChatCORE normalisation: `(raw − baseline) / (1 − baseline)`.

**ChatCORE** = mean centered accuracy over {MMLU, ARC-Easy, ARC-Challenge, HellaSwag, GSM8K}. Printed at the end of every `eval_sft.py` run.

### Parametrization (`utils/parametrization.py`)

Applies initialization scaling and per-parameter LR multipliers based on `ParametrizationConfig.mode` (`SP`, `MuP`, `CompleteP`). Enables HP transfer across model widths.

### Experiment Automation (`experiments.py`)

`ExperimentRunner` launches training subprocesses, collects `metrics.json` outputs, and supports grid searches. Helper notebooks live in `helper_notebooks/`:
- `scaling_studies.ipynb` — drives experiments and plots scaling curves
- `scaling_laws.ipynb` — scaling law fitting and analysis
- `transformer_sizing.ipynb` — model size / compute budget calculator

## Key Design Decisions

- **Config precedence**: YAML file values are base; CLI flags (`--key=value`) override them. Nested keys use dot notation (`--parametrization.mode=MuP`).
- **Metrics output**: Every run writes `metrics.json` to `out_dir`. RQ2 runs additionally write `is_stable` and a `stability_log`. The experiment scripts also write per-step CSV files to `{out_dir}/logs/` for offline analysis.
- **Logging cadence**: All spectral/grad/activation metrics in the experiment scripts fire at the same `log_interval` — no separate per-metric frequencies. Call `spectral_logger.log_step()` once per `log_interval` inside the training loop. Sharpness (`sharpness_freq`, default 0/disabled) is the only exception due to its cost (N forward passes).
- **TensorBoard tag hierarchy**: `train/`, `val/`, `weight/` (weight geometry), `grad/`, `activations/`, `sharpness/`, `stability/`, `checkpoint/weight/`. The old `spectral/` prefix no longer exists.
- **CSV raw data**: Experiment scripts write `{out_dir}/logs/training.csv`, `weight.csv`, `grad.csv`, `activations.csv`, `sharpness.csv`. Line-buffered, append-mode — safe to read mid-run with `pd.read_csv(...)`.
- **Optimizer routing**: `train.py` dispatches via `--optimizer` flag (`adamw`, `muon`, `scion`). Muon separates 2D vs. 1D parameters internally; `muon_lr/momentum/ns_steps` are now config-driven.
- **LR scheduling with CombinedOptimizer**: The experiment scripts scale each optimizer's LR by a cosine decay *multiplier* applied to its individual initial LR, so Muon and AdamW decay proportionally rather than converging to the same absolute value. TB tags: `train/lr/muon` and `train/lr/adamw`.
- **SVD checkpoints**: Full singular value spectra are written as `svd_step_NNNNNNN.json` in `out_dir` at each major checkpoint for offline analysis.
- **FineWeb-Edu dataset**: Two preparation paths: `download.py` fetches pre-built parquet shards from `karpathy/fineweb-edu-100b-shuffle`; `prepare.py` streams `HuggingFaceFW/fineweb-edu` and saves raw text as parquet. Both produce `shard_?????.parquet` files consumed by `utils/dataloader.py` with on-the-fly GPT-2 tokenization. The old pre-tokenized `.bin` workflow is removed for FineWeb-Edu.
- **Data format auto-detection**: `create_dataloader()` in `utils/data.py` checks for `shard_?????.parquet` files and routes to the parquet path automatically. OpenWebText now uses parquet (run `data/openwebtext/prepare.py`). Shakespeare still uses the `.bin` memmap path. Force a specific path with `data_format: 'bin'` or `data_format: 'parquet'` in config.
- **MoE auxiliary loss**: Two coefficients: `load_balance_loss_weight` (default 0.01) and `router_z_loss_weight` (default 0.001); both are added directly to the main cross-entropy loss. `models/moe.py` is a pure-PyTorch implementation using token permutation dispatch (no megablocks / stk required); `einops` dependency is also removed.
- **Multi-dataset validation**: `val_splits` lists split names within the training dataset; `val_datasets` lists paths to separate dataset folders (parquet or bin). Both are evaluated every `eval_interval` and logged to TensorBoard under `val/<split>` tags.
- **Hybrid model**: `use_hybrid=True` replaces every `delta_net_every`-th layer (starting at 0) with `GatedDeltaNetLayer` from `models/gated_delta_net.py`. Requires `flash-linear-attention`. Config: `configs/hybrid_gated_delta_net.yaml`.
- **SFT loss masking**: The mask stored in `{split}_mask.bin` marks tokens that *are* assistant tokens (mask=1). In `get_batch`, this is applied with a +1 shift: `y[t]` (which equals `x[t+1]`) is kept when `mask[t+1]=1`. Using `mask[t]` directly would be off by one — it would skip the first assistant token and include one spurious non-assistant target.
- **SFT block-diagonal attention**: `doc_ids` are assigned per conversation during packing and stored in `{split}_doc_ids.bin`. At training time `make_attn_mask(doc_ids)` produces a `(B,1,T,T)` bool mask preventing cross-conversation attention within a packed sequence. This is passed as `attn_mask` to `model.forward()` and used by `scaled_dot_product_attention`.
- **SFT `raw_model` assignment**: must be done after both `torch.compile` and DDP wrapping. `torch.compile` returns `OptimizedModule` which has no `.module` attribute, so accessing `model.module` before DDP is applied crashes. Correct order: compile → DDP → `raw_model = model.module`.
