"""
This training script creates and trains a GPT model for character-level language modeling.
It supports distributed training (DDP), mixed precision (bfloat16/float16), 
and compiled models (torch.compile).

It is modular and uses components from:
- models/: GPT model definition
- optimizers/: Muon and Scion optimizers
- utils/: Configuration, data loading, and parametrization

Usage:

1. Single GPU training:
   $ python train.py --batch_size=32 --compile=False --optimizer=adamw

2. Training with Muon optimizer:
   $ python train.py --optimizer=muon --learning_rate=0.02

3. Training with Scion optimizer:
   $ python train.py --optimizer=scion --learning_rate=0.001 --scion_norm=Spectral

4. Distributed Data Parallel (DDP) usage:
   $ torchrun --standalone --nproc_per_node=4 train.py

Configuration:
Configuration is handled by `utils.config.ExperimentConfig`. 
Defaults can be overridden via command line arguments (e.g., --key=value) 
or by passing a YAML config file (e.g., python train.py config/params.yaml).
"""

import os
import time
import math
import pickle
import json
import csv
import sys
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import GPTConfig, GPT
from utils.config import ExperimentConfig
from utils.parametrization import apply_parametrization
from utils.data import get_batch as get_batch_fn, create_dataloader

# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------

config = ExperimentConfig()

# Quick and dirty argument parsing to override config
# Supports --key=value or config file paths
for arg in sys.argv[1:]:
    if '=' not in arg:
        try:
            # Assume config file (YAML)
            if arg.endswith('.yaml') or arg.endswith('.yml'):
                config = ExperimentConfig.from_yaml(arg)
                print(f"Loaded config from {arg}")
            else:
                pass
        except Exception as e:
            print(f"Error loading config {arg}: {e}")
    else:
        # CLI override: --key=value
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        # Traverse nested config (e.g. parametrization.mode)
        obj = config
        while '.' in key:
            sub, key = key.split('.', 1)
            obj = getattr(obj, sub)
        
        # Infer type from default value in config
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
        print(f"Overriding: {key} = {v}")

# -----------------------------------------------------------------------------
# System Setup (DDP, Device, Seed)
# -----------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=getattr(config, 'backend', 'nccl'))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # adjust gradient accumulation steps
    assert config.gradient_accumulation_steps % ddp_world_size == 0
    config.gradient_accumulation_steps //= ddp_world_size
else:
    # vanilla, non-DDP run
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = config.device

tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(config.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu'

# Precision setup
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

data_dir = os.path.join('data', config.dataset)

# Build infinite data generators (parquet on-the-fly or bin memmap, auto-detected)
_train_loader = create_dataloader('train', config, device_type, device, data_dir,
                                  ddp_rank=ddp_rank if ddp else 0,
                                  ddp_world_size=ddp_world_size)
_val_loaders = {}
import warnings as _warnings
for _split in sorted(set(['train'] + config.val_splits)):
    try:
        _val_loaders[_split] = create_dataloader(_split, config, device_type, device,
                                                  data_dir,
                                                  ddp_rank=ddp_rank if ddp else 0,
                                                  ddp_world_size=ddp_world_size)
    except (ValueError, FileNotFoundError) as _e:
        _warnings.warn(f"Skipping data split '{_split}': {_e}")

def get_batch(split):
    if split == 'train':
        return next(_train_loader)
    loader = _val_loaders.get(split)
    if loader is not None:
        return next(loader)
    # bin-mode fallback for custom split names not in loaders dict
    return get_batch_fn(split, config, device_type, device, data_dir)

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9

if config.init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    config.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    model = GPT(config)
    
    # Apply advanced parametrization (MuP, CompleteP, etc.)
    apply_parametrization(model, config)
    
elif config.init_from == 'resume':
    print(f"Resuming training from {config.out_dir}")
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Force architecture params from checkpoint
    checkpoint_config = checkpoint.get('config', {})
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'use_rmsnorm', 'use_rope', 'use_swiglu']:
        if k in checkpoint_config and hasattr(config, k):
            setattr(config, k, checkpoint_config[k])
            
    model = GPT(config)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary (remove '_orig_mod.')
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
elif config.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
    override_args = dict(dropout=config.dropout)
    model = GPT.from_pretrained(config.init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        setattr(config, k, getattr(model.config, k))

# model surgery: crop down the block size if desired
if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    config.block_size = config.block_size

model.to(device)

# Printable Parameter Count
print("Model Parameter Breakdown:")
total_params = 0
for name, p in model.named_parameters():
    print(f"{name}: {p.shape} ({p.numel()} params)")
    total_params += p.numel()
print(f"Total parameters: {total_params}")

# -----------------------------------------------------------------------------
# Optimizer & Compilation
# -----------------------------------------------------------------------------

# initialize a GradScaler
scaler = torch.amp.GradScaler(device='cuda', enabled=(config.dtype == 'float16'))

# Optimizer selection and configuration is handled inside the model's method
# to allow for fine-grained control (e.g. different optimizers for different layers)
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
_initial_lrs = [pg['lr'] for pg in optimizer.param_groups]  # for proportional LR decay

if config.init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if config.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model  # unwrap DDP container if needed

# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    """ Helps estimate an arbitrarily accurate loss over either split using many batches """
    out = {}
    model.eval()
    
    # Identify splits to evaluate
    # Use config.val_splits if provided, otherwise default to ['train', 'val'] (though config default is ['val'])
    # We always evaluate on 'train' as well.
    splits = ['train'] + config.val_splits
    # Remove duplicates
    splits = sorted(list(set(splits)))
    
    for split in splits:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            try:
                X, Y = get_batch(split)
            except FileNotFoundError:
                # e.g. if 'val' split doesn't strictly exist as 'val.bin' or custom logic fails
                continue
                
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        # Clean up key name for log
        key = split
        if split.endswith('.bin'):
            key = split[:-4] # remove .bin
        out[key] = losses.mean()
        
    model.train()
    return out

def get_lr(it):
    """ Learning rate decay scheduler (cosine with warmup) """
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def _get_lr_multiplier(it):
    """Cosine decay multiplier in [min_lr/learning_rate, 1.0]. Used for proportional per-group LR decay."""
    min_ratio = config.min_lr / config.learning_rate
    if it < config.warmup_iters:
        return (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return min_ratio
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    return min_ratio + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (1.0 - min_ratio)

# Tensorboard Logging
if config.tensorboard_log and master_process:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'runs', config.tensorboard_run_name))

# Activation Logging Hooks
activation_stats = {}
if config.tensorboard_log and master_process:
    def get_activation_hook(name):
        def hook(model, input, output):
            if name not in activation_stats:
                activation_stats[name] = []
            if isinstance(output, torch.Tensor):
                out = output.detach()
                mean = out.mean().item()
                std = out.std().item()
                activation_stats[name].append({'mean': mean, 'std': std})
        return hook

    target_model = model.module if ddp else model
    for i, block in enumerate(target_model.transformer.h):
         block.register_forward_hook(get_activation_hook(f'block_{i}'))

# Research Logging Initialization (RQ1 / RQ2)
_spectral_logger = None
_stability_monitor = None
_csv_writer = None
_csv_file = None
_sharpness_X = None
_sharpness_Y = None
_is_stable = True
_stability_log = []

if (config.research.enable_spectral_logging or config.research.enable_stability_monitor) and master_process:
    from utils.spectral_metrics import SpectralLogger, StabilityMonitor

    _logs_dir = os.path.join(config.out_dir, 'logs')
    os.makedirs(_logs_dir, exist_ok=True)

    _csv_fields = ['step', 'train_loss', 'val_loss', 'lr_muon', 'lr_adamw', 'mfu', 'tokens_seen']
    if config.research.enable_stability_monitor:
        _csv_fields.append('is_stable')
    _csv_path = os.path.join(_logs_dir, 'training.csv')
    _csv_file = open(_csv_path, 'a', newline='', buffering=1)
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields, extrasaction='ignore')
    if _csv_file.tell() == 0:
        _csv_writer.writeheader()

    if config.research.enable_stability_monitor:
        _stability_monitor = StabilityMonitor()

    if config.research.enable_spectral_logging:
        _spectral_logger = SpectralLogger(
            model=raw_model,
            tb_writer=tb_writer if config.tensorboard_log else None,
            out_dir=config.out_dir,
            csv_dir=_logs_dir,
            sharpness_freq=config.research.sharpness_freq,
            topk=config.research.spectral_topk,
            stability_monitor=_stability_monitor,
        )

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
tokens_seen = iter_num * tokens_per_iter

# Fixed sharpness reference batch (fetched once for consistent sharpness measurements)
if _spectral_logger is not None and config.research.sharpness_freq > 0:
    _sharpness_X, _sharpness_Y = get_batch('train')

while True:

    # 1. Update Learning Rate
    if config.research.proportional_lr_decay and config.decay_lr:
        # Per-optimizer proportional decay (Muon+AdamW: each group decays from its own init LR)
        _mult = _get_lr_multiplier(iter_num)
        for _pg, _init_lr in zip(optimizer.param_groups, _initial_lrs):
            _pg['lr'] = _init_lr * _mult
        lr = optimizer.param_groups[0]['lr']  # for logging fallback
    else:
        lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 2. Evaluation and Checkpointing
    # 2. Evaluation and Checkpointing
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        
        loss_msg = f"step {iter_num}:"
        for k, v in losses.items():
            loss_msg += f" {k} loss {v:.4f},"
        print(loss_msg)
        
        # Write to TensorBoard
        if config.tensorboard_log:
             for k, v in losses.items():
                 tb_writer.add_scalar(f"val/loss_{k}", v, iter_num)

        # Track best 'val' loss (assumes 'val' exists, or use first available)
        current_val_loss = losses.get('val', list(losses.values())[0] if losses else 0.0)
        
        if current_val_loss < best_val_loss or config.always_save_checkpoint:
            best_val_loss = current_val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config.to_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))

        # SVD checkpoint dump (RQ1)
        if (_spectral_logger is not None and config.research.svd_checkpoint_freq > 0
                and iter_num > 0 and iter_num % config.research.svd_checkpoint_freq == 0):
            _spectral_logger.log_checkpoint(iter_num)

        # Stability check (RQ2)
        if _stability_monitor is not None and _spectral_logger is not None:
            _stability_result = _spectral_logger.check_stability(
                iter_num, current_val_loss.item(),
                tb_writer if config.tensorboard_log else None)
            _is_stable = _stability_result['is_stable']
            _stability_log.append({'step': iter_num, **_stability_result})

        # Write val row to training CSV
        if _csv_writer is not None:
            _csv_row = {'step': iter_num, 'val_loss': current_val_loss.item()}
            if config.research.enable_stability_monitor:
                _csv_row['is_stable'] = int(_is_stable)
            _csv_writer.writerow(_csv_row)

        # RQ2: early stop on divergence
        if config.research.enable_stability_monitor and not math.isfinite(current_val_loss.item()):
            print(f"DIVERGED at step {iter_num}. Stopping.")
            _is_stable = False
            break

    if iter_num == 0 and config.eval_only:
        break

    # 3. Forward + Backward Pass (with Gradient Accumulation)
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps # scale loss
        
        # Async prefetch next batch
        X, Y = get_batch('train')
        
        # Backward pass
        scaler.scale(loss).backward()
    
    # 4. Gradient Clipping + Spectral Logging (before optimizer step, grads available here)
    _do_log = iter_num % config.log_interval == 0 and master_process
    _needs_unscale = config.grad_clip != 0.0 or (_do_log and _spectral_logger is not None)
    if _needs_unscale:
        scaler.unscale_(optimizer)
    if config.grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # Spectral logging (RQ1: weight geometry, grad norms, activations)
    if _do_log and _spectral_logger is not None:
        _spectral_logger.log_step(step=iter_num, X=_sharpness_X, Y=_sharpness_Y, ctx=ctx)

    # RQ2: NaN/Inf train loss early stop
    if config.research.enable_stability_monitor:
        _lossf_check = loss.item() * config.gradient_accumulation_steps
        if not math.isfinite(_lossf_check):
            print(f"NaN/Inf train loss at step {iter_num}. Stopping early.")
            _is_stable = False
            break

    # 5. Logging Gradients (optional)
    grad_norms = {}
    if _do_log and config.tensorboard_log:
        for name, p in raw_model.named_parameters():
             if p.grad is not None:
                grad_norms[name] = p.grad.norm().item()

    # 6. Optimizer Step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True) # Flush gradients

    # 7. Timing and Metrics
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_seen += tokens_per_iter
    
    if _do_log:
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tokens {tokens_seen:,}")

        if config.tensorboard_log:
            tb_writer.add_scalar("iter", iter_num, iter_num)
            tb_writer.add_scalar("train/loss", lossf, iter_num)
            # Per-optimizer LR logging for Muon+AdamW combined optimizer
            if config.research.proportional_lr_decay and len(optimizer.param_groups) > 1:
                tb_writer.add_scalar('train/lr/muon',  optimizer.param_groups[0]['lr'], iter_num)
                tb_writer.add_scalar('train/lr/adamw', optimizer.param_groups[-1]['lr'], iter_num)
            else:
                tb_writer.add_scalar("lr", lr, iter_num)
            tb_writer.add_scalar("mfu", running_mfu*100, iter_num)
            tb_writer.add_scalar("tokens_seen", tokens_seen, iter_num)

            for name, val in grad_norms.items():
                tb_writer.add_scalar(f"grad_norm/{name}", val, iter_num)

            # Log weights occasionally
            if iter_num % 100 == 0:
                for name, p in raw_model.named_parameters():
                    tb_writer.add_scalar(f"weight_norm/{name}", p.norm().item(), iter_num)

            for name, stats in activation_stats.items():
                if stats:
                    means = [s['mean'] for s in stats]
                    stds = [s['std'] for s in stats]
                    tb_writer.add_scalar(f"act_mean/{name}", sum(means)/len(means), iter_num)
                    tb_writer.add_scalar(f"act_std/{name}", sum(stds)/len(stds), iter_num)
            activation_stats = {}
            tb_writer.flush()

        # Write train step row to CSV
        if _csv_writer is not None:
            _csv_row = {'step': iter_num, 'train_loss': lossf,
                        'mfu': running_mfu * 100, 'tokens_seen': tokens_seen}
            if config.research.proportional_lr_decay and len(optimizer.param_groups) > 1:
                _csv_row['lr_muon']  = optimizer.param_groups[0]['lr']
                _csv_row['lr_adamw'] = optimizer.param_groups[-1]['lr']
            else:
                _csv_row['lr_adamw'] = lr
            _csv_writer.writerow(_csv_row)

    iter_num += 1
    local_iter_num += 1

    # 8. Termination
    if iter_num > config.max_iters:
        break

# -----------------------------------------------------------------------------
# Clean up
# -----------------------------------------------------------------------------

# Research logging cleanup
if _spectral_logger is not None:
    _spectral_logger.remove_hooks()
    _spectral_logger.log_checkpoint(iter_num)  # Final SVD dump
    _spectral_logger.close()
if _csv_file is not None:
    _csv_file.close()

if ddp:
    destroy_process_group()

if master_process:
    # Save final metrics
    def to_serializable(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    metrics = {
        'best_val_loss': to_serializable(best_val_loss),
        'iter_num': iter_num,
        'tokens_seen': tokens_seen,
        'total_params': total_params,
        'config': config.to_dict(),
        'val_loss': to_serializable(losses['val']) if 'losses' in locals() else None,
        'train_loss': to_serializable(losses['train']) if 'losses' in locals() else None,
    }
    if config.research.enable_stability_monitor:
        metrics['is_stable'] = _is_stable
        metrics['stability_log'] = _stability_log[-20:]
        print(f"Final stability: {'STABLE' if _is_stable else 'UNSTABLE'}")
    metrics_path = os.path.join(config.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
