"""
Experiment: Structured Weight Decay
This script trains a GPT model using StructuredAdamW, which applies group-lasso style
regularization (row-wise or column-wise) to weight matrices.
"""

import os
import time
import math
import pickle
import json
import sys
from contextlib import nullcontext

# Add project root to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
# Add experiment dir to sys.path to allow direct import of optimizer avoiding namespace conflict with experiments.py
sys.path.append(current_dir)

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import GPTConfig, GPT
from utils.config import ExperimentConfig
from utils.parametrization import apply_parametrization
from utils.data import get_batch as get_batch_fn

# Import the custom optimizer
# Now we can import directly as we added current_dir to sys.path
from optimizer import StructuredAdamW

# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------

import yaml

config = ExperimentConfig()

# Add experiment-specific configs
group_mode = 'row'
epsilon_decay = 1e-5

# Quick and dirty argument parsing to override config
for arg in sys.argv[1:]:
    if '=' not in arg:
        try:
            if arg.endswith('.yaml') or arg.endswith('.yml'):
                # Load yaml manually to extract experiment-specific keys
                with open(arg, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Extract and remove custom keys
                if 'group_mode' in data:
                    group_mode = data.pop('group_mode')
                    print(f"Loaded group_mode from config: {group_mode}")
                if 'epsilon_decay' in data:
                    epsilon_decay = data.pop('epsilon_decay')
                    print(f"Loaded epsilon_decay from config: {epsilon_decay}")
                
                # Re-construct config using cleaned data
                # We can't use from_yaml anymore because we modified the dict.
                # We reuse the logic from ExperimentConfig.from_yaml roughly, 
                # or just instantiate. ExperimentConfig(**data) works if data matches fields.
                
                # Handle nested parametrization same as from_yaml
                if 'parametrization' in data:
                    from utils.config import ParametrizationConfig
                    data['parametrization'] = ParametrizationConfig(**data['parametrization'])
                
                config = ExperimentConfig(**data)
                print(f"Loaded config from {arg}")

        except Exception as e:
            print(f"Error loading config {arg}: {e}")
    else:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        
        # Handle experiment-specific args
        if key == 'group_mode':
            group_mode = val
            print(f"Overriding: {key} = {val}")
            continue
        if key == 'epsilon_decay':
            epsilon_decay = float(val)
            print(f"Overriding: {key} = {val}")
            continue

        # Traverse nested config
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
        print(f"Overriding: {key} = {v}")

# -----------------------------------------------------------------------------
# System Setup
# -----------------------------------------------------------------------------

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

tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(config.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in config.device else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

data_dir = os.path.join(project_root, 'data', config.dataset) # Adjusted path
def get_batch(split):
    return get_batch_fn(split, config, device_type, device, data_dir)

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
    config.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    model = GPT(config)
    apply_parametrization(model, config)
    
elif config.init_from == 'resume':
    print(f"Resuming training from {config.out_dir}")
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_config = checkpoint.get('config', {})
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'use_rmsnorm', 'use_rope', 'use_swiglu']:
        if k in checkpoint_config and hasattr(config, k):
            setattr(config, k, checkpoint_config[k])
    model = GPT(config)
    state_dict = checkpoint['model']
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

if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    config.block_size = config.block_size

model.to(device)

print("Model Parameter Breakdown:")
total_params = 0
for name, p in model.named_parameters():
    print(f"{name}: {p.shape} ({p.numel()} params)")
    total_params += p.numel()
print(f"Total parameters: {total_params}")

# -----------------------------------------------------------------------------
# Optimizer (Custom for Experiment)
# -----------------------------------------------------------------------------

scaler = torch.amp.GradScaler('cuda', enabled=(config.dtype == 'float16'))

# Manually configure optimizer with StructuredAdamW
param_dict = {pn: p for pn, p in model.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

print(f"Using StructuredAdamW(AdamW) with group_mode='{group_mode}' and weight_decay={config.weight_decay}")

# We pass 'structured_weight_decay' explicitly in the group to match our new implementation's expectation
optim_groups = [
    {'params': param_dict.values(), 'structured_weight_decay': config.weight_decay, 
     'group_mode': group_mode, 'epsilon_decay': epsilon_decay}
]

# Note: The optimizer init will see 'weight_decay' in defaults as 0.0 (inside class init), 
# but param groups override defaults. We must ensure the group has 'weight_decay': 0.0 
# OR relies on the class to handle it.
# Our new class init sets global defaults weight_decay=0. 
# BUT if we pass a dict in optim_groups without 'weight_decay', it inherits global default (0.0). Correct.
# We pass 'structured_weight_decay' which the step() method looks for. Correct.

optimizer = StructuredAdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))


if config.init_from == 'resume' and 'optimizer' in checkpoint:
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except Exception as e:
        print(f"WARNING: Could not load optimizer state (expected if switching optimizers): {e}")

checkpoint = None 

if config.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    splits = ['train'] + config.val_splits
    splits = sorted(list(set(splits)))
    for split in splits:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
             try:
                 X, Y = get_batch(split)
             except FileNotFoundError:
                 continue
             with ctx:
                 logits, loss = model(X, Y)
             losses[k] = loss.item()
        key = split
        if split.endswith('.bin'):
            key = split[:-4] 
        out[key] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

if config.tensorboard_log and master_process:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'runs', config.tensorboard_run_name))

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

X, Y = get_batch('train') 
t0 = time.time()
local_iter_num = 0 
raw_model = model.module if ddp else model 
running_mfu = -1.0
tokens_seen = iter_num * tokens_per_iter

while True:
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        loss_msg = f"step {iter_num}:"
        for k, v in losses.items():
            loss_msg += f" {k} loss {v:.4f},"
        print(loss_msg)
        
        if config.tensorboard_log:
             for k, v in losses.items():
                 tb_writer.add_scalar(f"val/loss_{k}", v, iter_num)

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
    
    if iter_num == 0 and config.eval_only:
        break

    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps
        
        X, Y = get_batch('train')
        
        scaler.scale(loss).backward()
    
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_seen += tokens_per_iter
    
    if iter_num % config.log_interval == 0 and master_process:
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5: 
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tokens {tokens_seen:,}")
        
        if config.tensorboard_log:
            tb_writer.add_scalar("iter", iter_num, iter_num)
            tb_writer.add_scalar("train/loss", lossf, iter_num)
            tb_writer.add_scalar("lr", lr, iter_num)
            tb_writer.add_scalar("mfu", running_mfu*100, iter_num)
            tb_writer.add_scalar("tokens_seen", tokens_seen, iter_num)
            tb_writer.flush()

    iter_num += 1
    local_iter_num += 1

    if iter_num > config.max_iters:
        break

if ddp:
    destroy_process_group()

if master_process:
    def to_serializable(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    metrics = {
        'best_val_loss': to_serializable(best_val_loss),
        'iter_num': iter_num,
        'tokens_seen': tokens_seen,
        'config': config.to_dict(),
        'val_loss': to_serializable(losses['val']) if 'losses' in locals() else None,
        'train_loss': to_serializable(losses['train']) if 'losses' in locals() else None,
    }
    metrics_path = os.path.join(config.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
