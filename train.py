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
import sys
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import GPTConfig, GPT
from utils.config import ExperimentConfig
from utils.parametrization import apply_parametrization
from utils.data import get_batch as get_batch_fn

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
def get_batch(split):
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

# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    """ Helps estimate an arbitrarily accurate loss over either split using many batches """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
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

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
tokens_seen = iter_num * tokens_per_iter

while True:

    # 1. Update Learning Rate
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2. Evaluation and Checkpointing
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or config.always_save_checkpoint:
            best_val_loss = losses['val']
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
    
    # 4. Gradient Clipping
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    # 5. Logging Gradients (optional)
    grad_norms = {}
    if iter_num % config.log_interval == 0 and master_process and config.tensorboard_log:
        if config.grad_clip == 0.0:
            scaler.unscale_(optimizer)
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
    
    if iter_num % config.log_interval == 0 and master_process:
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tokens {tokens_seen:,}")
        
        if config.tensorboard_log:
            tb_writer.add_scalar("iter", iter_num, iter_num)
            tb_writer.add_scalar("train/loss", lossf, iter_num)
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

    iter_num += 1
    local_iter_num += 1

    # 8. Termination
    if iter_num > config.max_iters:
        break

# -----------------------------------------------------------------------------
# Clean up
# -----------------------------------------------------------------------------

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
    metrics_path = os.path.join(config.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
