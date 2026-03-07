"""
RQ1: Spectral Geometry of Weight Matrices During Training.

Measures how Muon vs. AdamW shapes the singular value structure of weight
matrices throughout training. Instruments the standard training loop with
SpectralLogger for comprehensive per-step and checkpoint-level logging of:

  - Stable rank (every spectral_freq steps)
  - Gradient Frobenius + spectral norms (every grad_freq steps)
  - Gradient direction cosine similarity between consecutive steps
  - Nuclear/Frobenius ratio + spectral entropy via top-k SVD (every topk_svd_freq)
  - Activation statistics: mean, var, kurtosis, dead fraction, isotropy
  - Sharpness via random perturbation (every sharpness_freq steps)
  - Full SVD dumps at each checkpoint (JSON)

Usage:
    python experiments/rq1_spectral_geometry/train.py \
        experiments/rq1_spectral_geometry/configs/125m_muon.yaml

    # Override specific params:
    python experiments/rq1_spectral_geometry/train.py \
        experiments/rq1_spectral_geometry/configs/125m_muon.yaml \
        --muon_ns_steps=3

LR scheduling for the CombinedOptimizer (Muon + AdamW) is handled correctly:
each optimizer's LR is scaled by the same cosine decay *multiplier* applied
to its own initial LR, so Muon's 0.02 and AdamW's 3e-4 decay in sync.
"""

import os
import sys
import time
import math
import pickle
import json
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Resolve project root from this file's location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from models import GPTConfig, GPT
from utils.config import ExperimentConfig
from utils.parametrization import apply_parametrization
from utils.data import get_batch as get_batch_fn, create_dataloader
from utils.spectral_metrics import SpectralLogger, StabilityMonitor

# ---------------------------------------------------------------------------
# Experiment-specific hyperparameters (overridable via YAML or CLI)
# ---------------------------------------------------------------------------

spectral_freq = 100       # stable rank logging interval
grad_freq = 50            # gradient norm + direction cosine interval
topk_svd_freq = 500       # nuclear/frob + spectral entropy interval
activation_freq = 100     # activation stats interval
sharpness_freq = 5000     # sharpness measurement interval (0 = disabled)
svd_checkpoint_freq = 2000  # full SVD dump interval (aligns with eval_interval)
topk = 32                 # k for top-k SVD entropy

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

config = ExperimentConfig()

for arg in sys.argv[1:]:
    if '=' not in arg:
        if arg.endswith('.yaml') or arg.endswith('.yml'):
            try:
                config = ExperimentConfig.from_yaml(arg)
                print(f"Loaded config from {arg}")
            except Exception as e:
                print(f"Error loading config {arg}: {e}")
    else:
        assert arg.startswith('--'), f"Unknown argument format: {arg}"
        key, val = arg[2:].split('=', 1)

        # Handle experiment-level overrides first
        for exp_var in ['spectral_freq', 'grad_freq', 'topk_svd_freq',
                        'activation_freq', 'sharpness_freq', 'svd_checkpoint_freq', 'topk']:
            if key == exp_var:
                globals()[exp_var] = int(val)
                print(f"Overriding experiment param: {key} = {val}")
                break
        else:
            # Standard config override
            obj = config
            while '.' in key:
                sub, key = key.split('.', 1)
                obj = getattr(obj, sub)
            target_type = type(getattr(obj, key))
            if target_type == bool:
                v = val.lower() == 'true'
            elif target_type == int:
                v = int(val)
            elif target_type == float:
                v = float(val)
            else:
                v = val
            setattr(obj, key, v)
            print(f"Overriding config: {key} = {v}")

# ---------------------------------------------------------------------------
# System setup (DDP, device, seed)
# ---------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=getattr(config, 'backend', 'nccl'))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert config.gradient_accumulation_steps % ddp_world_size == 0
    config.gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = config.device

tokens_per_iter = (config.gradient_accumulation_steps * ddp_world_size
                   * config.batch_size * config.block_size)
if master_process:
    print(f"tokens per iteration: {tokens_per_iter:,}")
    os.makedirs(config.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
           'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data_dir = os.path.join(project_root, 'data', config.dataset)

import warnings as _warnings

_train_loader = create_dataloader('train', config, device_type, device, data_dir,
                                  ddp_rank=ddp_rank if ddp else 0,
                                  ddp_world_size=ddp_world_size)
_val_loaders = {}
for _split in sorted(set(config.val_splits)):
    try:
        _val_loaders[_split] = create_dataloader(_split, config, device_type, device,
                                                  data_dir,
                                                  ddp_rank=ddp_rank if ddp else 0,
                                                  ddp_world_size=ddp_world_size)
    except (ValueError, FileNotFoundError) as _e:
        _warnings.warn(f"Skipping data split '{_split}': {_e}")

_extra_val_loaders = {}  # key: basename of folder path
for _ds_path in config.val_datasets:
    _ds_abs = _ds_path if os.path.isabs(_ds_path) else os.path.join(project_root, _ds_path)
    _ds_key = os.path.basename(_ds_path.rstrip('/\\'))
    try:
        _extra_val_loaders[_ds_key] = create_dataloader(
            'val', config, device_type, device, _ds_abs,
            ddp_rank=ddp_rank if ddp else 0,
            ddp_world_size=ddp_world_size,
        )
        print(f"Loaded extra val dataset: {_ds_key} ({_ds_abs})")
    except (ValueError, FileNotFoundError) as _e:
        _warnings.warn(f"Skipping extra val dataset '{_ds_path}': {_e}")

def get_batch(split):
    if split == 'train':
        return next(_train_loader)
    loader = _val_loaders.get(split)
    if loader is not None:
        return next(loader)
    return get_batch_fn(split, config, device_type, device, data_dir)

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"vocab_size = {meta_vocab_size}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9

config.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
model = GPT(config)
apply_parametrization(model, config)

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

scaler = torch.amp.GradScaler(device='cuda', enabled=(config.dtype == 'float16'))
optimizer = model.configure_optimizers(
    config.weight_decay, config.learning_rate,
    (config.beta1, config.beta2), device_type
)

# Store initial LR for each param group (needed for proper decay of Muon+AdamW)
initial_lrs = [pg['lr'] for pg in optimizer.param_groups]

if config.compile:
    print("compiling the model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# ---------------------------------------------------------------------------
# Spectral Logger (only on master process, only raw model)
# ---------------------------------------------------------------------------

tb_writer = None
spectral_logger = None
if master_process and config.tensorboard_log:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(
        log_dir=os.path.join(config.out_dir, 'runs', config.tensorboard_run_name))
    spectral_logger = SpectralLogger(
        model=raw_model,
        tb_writer=tb_writer,
        out_dir=config.out_dir,
        sharpness_freq=sharpness_freq,
        topk=topk,
    )

# ---------------------------------------------------------------------------
# LR scheduling helpers
# ---------------------------------------------------------------------------

def get_lr_multiplier(it: int) -> float:
    """Returns cosine decay multiplier in [min_lr/lr, 1.0]."""
    min_ratio = config.min_lr / config.learning_rate
    if it < config.warmup_iters:
        return (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return min_ratio
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_ratio + coeff * (1.0 - min_ratio)


def apply_lr(it: int):
    """Apply cosine decay to all param groups using their individual initial LRs."""
    if not config.decay_lr:
        return
    mult = get_lr_multiplier(it)
    for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
        pg['lr'] = init_lr * mult


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in sorted(set(['train'] + config.val_splits)):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            try:
                X, Y = get_batch(split)
            except FileNotFoundError:
                continue
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        key = split[:-4] if split.endswith('.bin') else split
        out[key] = losses.mean()
    for ds_name, loader in _extra_val_loaders.items():
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = next(loader)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[ds_name] = losses.mean()
    model.train()
    return out

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
tokens_seen = iter_num * tokens_per_iter

# Keep a fixed sharpness batch so measurements are comparable across steps
sharpness_X, sharpness_Y = get_batch('val') if 'val' in config.val_splits else (X, Y)

while True:
    apply_lr(iter_num)

    # Evaluation + checkpointing
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        loss_msg = f"step {iter_num}:"
        for k, v in losses.items():
            loss_msg += f" {k}_loss={v:.4f}"
        print(loss_msg)

        if tb_writer:
            for k, v in losses.items():
                tb_writer.add_scalar(f'val/loss_{k}', v, iter_num)

        current_val_loss = losses.get('val', next(iter(losses.values())))

        if current_val_loss < best_val_loss or config.always_save_checkpoint:
            best_val_loss = current_val_loss
            if iter_num > 0:
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config.to_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(ckpt, os.path.join(config.out_dir, 'ckpt.pt'))
                print(f"Saved checkpoint at step {iter_num}")

        # Full SVD checkpoint dump
        if spectral_logger and iter_num % svd_checkpoint_freq == 0 and iter_num > 0:
            spectral_logger.log_checkpoint(iter_num)

    if iter_num == 0 and config.eval_only:
        break

    # Forward + backward
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # Gradient clipping
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    elif master_process and spectral_logger:
        # Need to unscale before reading grads for logging
        scaler.unscale_(optimizer)

    # --- Spectral + dynamics logging (grads available here) ---
    if master_process and spectral_logger:
        spectral_logger.log_step(
            step=iter_num,
            X=sharpness_X,
            Y=sharpness_Y,
            ctx=ctx,
        )

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing + basic metrics
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_seen += tokens_per_iter

    if iter_num % config.log_interval == 0 and master_process:
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(
                config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
              f"mfu {running_mfu*100:.2f}%, tokens {tokens_seen:,}")

        if tb_writer:
            tb_writer.add_scalar('train/loss', lossf, iter_num)
            tb_writer.add_scalar('train/lr_adamw',
                                 optimizer.param_groups[-1]['lr'], iter_num)
            tb_writer.add_scalar('train/lr_special',
                                 optimizer.param_groups[0]['lr'], iter_num)
            tb_writer.add_scalar('train/mfu', running_mfu * 100, iter_num)
            tb_writer.add_scalar('train/tokens_seen', tokens_seen, iter_num)
            tb_writer.flush()

    iter_num += 1
    local_iter_num += 1

    if iter_num > config.max_iters:
        break

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

if spectral_logger:
    spectral_logger.remove_hooks()
    # Final SVD dump
    spectral_logger.log_checkpoint(iter_num)

if ddp:
    destroy_process_group()

if master_process:
    def to_serializable(val):
        return val.item() if hasattr(val, 'item') else val

    final_losses = estimate_loss() if not config.eval_only else {}
    metrics = {
        'best_val_loss': to_serializable(best_val_loss),
        'iter_num': iter_num,
        'tokens_seen': tokens_seen,
        'total_params': total_params,
        'config': config.to_dict(),
        'val_loss': to_serializable(final_losses.get('val', float('nan'))),
        'train_loss': to_serializable(final_losses.get('train', float('nan'))),
    }
    metrics_path = os.path.join(config.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
