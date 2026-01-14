"""
SFT Training Script (train_sft.py)
Supports:
- Packed sequences
- Masked Attention (Block-Diagonal)
- Instruction masking (loss masking)
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

# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------

config = ExperimentConfig()

# Quick argument parsing
for arg in sys.argv[1:]:
    if '=' not in arg:
        try:
            if arg.endswith('.yaml') or arg.endswith('.yml'):
                config = ExperimentConfig.from_yaml(arg)
                print(f"Loaded config from {arg}")
        except Exception as e:
            print(f"Error loading config {arg}: {e}")
    else:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
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
    print(f"tokens per iteration: {tokens_per_iter:,}")
    os.makedirs(config.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in config.device else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data Loading (SFT Specific)
# -----------------------------------------------------------------------------

data_dir = os.path.join('data', config.dataset)

def get_batch(split):
    # SFT Data Loader
    # Loads: train.bin (tokens), train_mask.bin (loss mask), train_doc_ids.bin (doc ids)
    # Assumes packed data in binary files
    
    # Recreate memmap each time to avoid leaks
    if split == 'train':
        bin_path = os.path.join(data_dir, 'train.bin')
        mask_path = os.path.join(data_dir, 'train_mask.bin')
        doc_path = os.path.join(data_dir, 'train_doc_ids.bin')
    else:
        bin_path = os.path.join(data_dir, 'val.bin')
        mask_path = os.path.join(data_dir, 'val_mask.bin')
        doc_path = os.path.join(data_dir, 'val_doc_ids.bin')
        
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    
    # Optimization: Only load mask/doc if needed?
    # For training we need all.
    mask_data = np.memmap(mask_path, dtype=np.uint8, mode='r')
    doc_data = np.memmap(doc_path, dtype=np.uint32, mode='r') # Assumed uint32 per prepare_sft.py
    
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    
    x_stack = []
    y_stack = []
    doc_stack = []
    
    for i in ix:
        # Inputs
        x_chunk = torch.from_numpy((data[i:i+config.block_size]).astype(np.int64))
        # Targets (shifted)
        y_chunk = torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64))
        
        # Loss Mask (aligned with targets)
        # mask data corresponds to tokens.
        # usually mask[i] tells if token[i] is a target to be learned?
        # In prepare_sft: mask aligns with tokens.
        # if token[i] is user input, we don't want to predict it.
        # Prediction happens at position i-1 to predict i.
        # So if token[i] is masked (0), then the target y at position i-1 should be -1.
        # y_chunk is x[i+1]. So we check mask[i+1].
        m_chunk = torch.from_numpy((mask_data[i+1:i+1+config.block_size]).astype(np.int64))
        
        # Apply mask to targets
        y_chunk[m_chunk == 0] = -1
        
        # Doc IDs
        # We need doc ids for the input sequence x to build attention mask
        d_chunk = torch.from_numpy((doc_data[i:i+config.block_size]).astype(np.int64))
        
        x_stack.append(x_chunk)
        y_stack.append(y_chunk)
        doc_stack.append(d_chunk)
        
    x = torch.stack(x_stack)
    y = torch.stack(y_stack)
    doc_ids = torch.stack(doc_stack)
    
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        doc_ids = doc_ids.pin_memory().to(device, non_blocking=True)
    else:
        x, y, doc_ids = x.to(device), y.to(device), doc_ids.to(device)
        
    return x, y, doc_ids

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9

print(f"Initializing model for SFT from {config.init_from}")
# Assuming loading from a pretrained checkpoint or similar
if config.init_from == 'scratch':
    # Default vocab size for GPT-2
    if not hasattr(config, 'vocab_size'):
        config.vocab_size = 50304
    model = GPT(config)
    apply_parametrization(model, config)
elif config.init_from == 'resume':
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    config_dict = checkpoint['config']
    # Override config from checkpoint? usually yes
    # But for SFT we might change things (e.g. dropout).
    # Ideally reuse model config.
    # For now assuming compatible.
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
     model = GPT.from_pretrained(config.init_from, dict(dropout=config.dropout))
     # Update config
     for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size']:
        setattr(config, k, getattr(model.config, k))

if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    
model.to(device)

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------

scaler = torch.amp.GradScaler(device='cuda', enabled=(config.dtype == 'float16'))
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)

if config.compile:
    print("compiling model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_attn_mask(doc_ids):
    # doc_ids: (B, T)
    # Returns: (B, 1, T, T) bool tensor
    # True where attention is allowed, False where blocked.
    # Block diagonal: allowed if doc_ids[i] == doc_ids[j] AND i >= j (causal)
    
    B, T = doc_ids.size()
    # (B, T, 1) == (B, 1, T) -> (B, T, T)
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(1)
    
    # Causal mask: i >= j
    # This is standardly handled by passing is_causal=True, but here we are combining.
    # Construct explicit causal mask
    causal = torch.tril(torch.ones(T, T, device=doc_ids.device, dtype=torch.bool))
    
    # Combine
    mask = same_doc & causal
    
    # Add head dimension: (B, 1, T, T)
    return mask.unsqueeze(1)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    splits = ['train', 'val']
    for split in splits:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            try:
                X, Y, DocIds = get_batch(split)
            except Exception:
                continue
            
            mask = make_attn_mask(DocIds)
            
            with ctx:
                logits, loss = model(X, Y, attn_mask=mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

X, Y, DocIds = get_batch('train')
t0 = time.time()

while True:
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss or config.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config.to_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt_sft.pt'))

    if iter_num > config.max_iters:
        break
        
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
            
        attn_mask = make_attn_mask(DocIds)
        
        with ctx:
            logits, loss = model(X, Y, attn_mask=attn_mask)
            loss = loss / config.gradient_accumulation_steps
            
        X, Y, DocIds = get_batch('train')
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
    
    if iter_num % config.log_interval == 0 and master_process:
        lossf = loss.item() * config.gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        
    iter_num += 1

if ddp:
    destroy_process_group()
