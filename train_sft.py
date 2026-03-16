"""
SFT Training Script (train_sft.py)
Supports:
- Packed sequences with block-diagonal masked attention
- Instruction masking (loss masking on prompt tokens)
- DDP multi-GPU training
- Mixed precision (bfloat16/float16)
- torch.compile
- Cosine LR schedule with warmup
- TensorBoard logging
- Decoupled checkpointing with keep_last_n_checkpoints
- eval_only mode
- MFU estimation
- metrics.json output

Usage:
1. Single GPU:
   $ python train_sft.py configs/sft.yaml

2. Multi-GPU DDP:
   $ torchrun --standalone --nproc_per_node=4 train_sft.py configs/sft.yaml
"""

import os
import time
import gc
import math
import pickle
import json
import sys
import shutil
import glob
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
# System Setup (DDP, Device, Seed)
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
torch.backends.cudnn.benchmark = True
device_type = 'cuda' if 'cuda' in device else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data Loading (SFT — packed .bin files with mask and doc_ids)
# -----------------------------------------------------------------------------

data_dir = os.path.join('data', config.dataset)

def get_batch(split):
    """Load a batch from packed SFT binary files.

    Files expected in data_dir:
        {split}.bin          — uint16 token ids
        {split}_mask.bin     — uint8  loss mask (1=predict, 0=ignore prompt)
        {split}_doc_ids.bin  — uint32 document ids (for block-diagonal attention)
    """
    bin_path  = os.path.join(data_dir, f'{split}.bin')
    mask_path = os.path.join(data_dir, f'{split}_mask.bin')
    doc_path  = os.path.join(data_dir, f'{split}_doc_ids.bin')

    data      = np.memmap(bin_path,  dtype=np.uint16, mode='r')
    mask_data = np.memmap(mask_path, dtype=np.uint8,  mode='r')
    doc_data  = np.memmap(doc_path,  dtype=np.uint32, mode='r')

    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    x_stack, y_stack, doc_stack = [], [], []
    for i in ix:
        x_chunk = torch.from_numpy(data[i     : i +     config.block_size].astype(np.int64))
        y_chunk = torch.from_numpy(data[i + 1 : i + 1 + config.block_size].astype(np.int64))
        # mask[i]==0 means token i is a prompt token — suppress its prediction target
        m_chunk = torch.from_numpy(mask_data[i + 1 : i + 1 + config.block_size].astype(np.int64))
        y_chunk[m_chunk == 0] = -1
        d_chunk = torch.from_numpy(doc_data[i : i + config.block_size].astype(np.int64))
        x_stack.append(x_chunk)
        y_stack.append(y_chunk)
        doc_stack.append(d_chunk)

    x       = torch.stack(x_stack)
    y       = torch.stack(y_stack)
    doc_ids = torch.stack(doc_stack)

    if device_type == 'cuda':
        x       = x.pin_memory().to(device, non_blocking=True)
        y       = y.pin_memory().to(device, non_blocking=True)
        doc_ids = doc_ids.pin_memory().to(device, non_blocking=True)
    else:
        x, y, doc_ids = x.to(device), y.to(device), doc_ids.to(device)

    return x, y, doc_ids

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9
checkpoint = None

print(f"Initializing model for SFT from: {config.init_from}")

if config.init_from == 'scratch':
    if not hasattr(config, 'vocab_size') or config.vocab_size is None:
        config.vocab_size = 50304
    model = GPT(config)
    apply_parametrization(model, config)

elif config.init_from == 'resume':
    ckpt_path = os.path.join(config.out_dir, 'ckpt_sft.pt')
    print(f"Resuming SFT from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_config = checkpoint.get('config', {})
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'use_rmsnorm', 'use_rope', 'use_swiglu']:
        if k in checkpoint_config and hasattr(config, k):
            setattr(config, k, checkpoint_config[k])
    model = GPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif os.path.isfile(config.init_from):
    # Load a pretrained PT checkpoint for SFT fine-tuning
    print(f"Loading PT checkpoint from {config.init_from}")
    checkpoint = torch.load(config.init_from, map_location=device)
    checkpoint_config = checkpoint.get('config', {})
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'use_rmsnorm', 'use_rope', 'use_swiglu']:
        if k in checkpoint_config and hasattr(config, k):
            setattr(config, k, checkpoint_config[k])
    model = GPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', 1e9)

elif config.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
    model = GPT.from_pretrained(config.init_from, dict(dropout=config.dropout))
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        setattr(config, k, getattr(model.config, k))

if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)

model.to(device)

print("Model Parameter Breakdown:")
total_params = 0
for name, p in model.named_parameters():
    print(f"  {name}: {p.shape} ({p.numel():,} params)")
    total_params += p.numel()
print(f"Total parameters: {total_params:,}")

# -----------------------------------------------------------------------------
# Optimizer & Compilation
# -----------------------------------------------------------------------------

scaler = torch.amp.GradScaler(device='cuda', enabled=(config.dtype == 'float16'))
optimizer = model.configure_optimizers(
    config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type
)

if checkpoint is not None and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free memory

if config.compile:
    print("Compiling model... (takes ~1 minute)")
    unoptimized_model = model
    model = torch.compile(model)

# Define raw_model before DDP wrapping so we can access weights directly
raw_model = model.module if ddp else model

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------

def make_attn_mask(doc_ids):
    """Build block-diagonal causal attention mask from doc ids.

    Returns (B, 1, T, T) bool tensor: True where attention is allowed.
    Tokens attend only within their own document, causally.
    """
    B, T = doc_ids.size()
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(1)  # (B, T, T)
    causal   = torch.tril(torch.ones(T, T, device=doc_ids.device, dtype=torch.bool))
    return (same_doc & causal).unsqueeze(1)                    # (B, 1, T, T)

@torch.no_grad()
def estimate_loss():
    """Evaluate loss on all configured splits. All DDP ranks participate."""
    out = {}
    model.eval()

    for split in sorted(set(['train', 'val'] + config.val_splits)):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            try:
                X, Y, DocIds = get_batch(split)
            except Exception:
                continue
            mask = make_attn_mask(DocIds)
            with ctx:
                _, loss, _ = model(X, Y, attn_mask=mask)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        if ddp:
            t = torch.tensor(mean_loss, device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
            mean_loss = t.item()
        out[split] = mean_loss

    if device_type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()

    model.train()
    return out

def get_lr(it):
    """Cosine LR schedule with linear warmup."""
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# TensorBoard
if config.tensorboard_log and master_process:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(
        log_dir=os.path.join(config.out_dir, 'runs', config.tensorboard_run_name)
    )

# Activation logging hooks
activation_stats = {}
if config.tensorboard_log and master_process:
    def _make_hook(name):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                o = out.detach()
                activation_stats.setdefault(name, []).append(
                    {'mean': o.mean().item(), 'std': o.std().item()}
                )
        return hook
    for i, block in enumerate(raw_model.transformer.h):
        block.register_forward_hook(_make_hook(f'block_{i}'))

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

X, Y, DocIds = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
tokens_seen = iter_num * tokens_per_iter
_metrics_interval = config.metrics_log_interval if config.metrics_log_interval > 0 else config.log_interval

while True:

    # 1. LR update
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2. Evaluation — all ranks participate so allreduce in estimate_loss works
    if iter_num % config.eval_interval == 0:
        losses = estimate_loss()
        if master_process:
            loss_msg = f"step {iter_num}:"
            for k, v in losses.items():
                loss_msg += f" {k} loss {v:.4f},"
            print(loss_msg)

            current_val_loss = losses.get('val', list(losses.values())[0])
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss

            if config.tensorboard_log:
                for k, v in losses.items():
                    tb_writer.add_scalar(f"val/loss_{k}", v, iter_num)

    # 3. Checkpointing (decoupled from eval)
    _ckpt_interval = config.checkpoint_interval if config.checkpoint_interval > 0 else config.eval_interval
    if iter_num % _ckpt_interval == 0 and master_process and iter_num > 0:
        ckpt_data = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config.to_dict(),
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
        }
        numbered_ckpt = os.path.join(config.out_dir, f'ckpt_sft_{iter_num:07d}.pt')
        latest_ckpt   = os.path.join(config.out_dir, 'ckpt_sft.pt')
        print(f"saving checkpoint to {numbered_ckpt}")
        torch.save(ckpt_data, numbered_ckpt)
        shutil.copy2(numbered_ckpt, latest_ckpt)
        if config.keep_last_n_checkpoints > 0:
            all_ckpts = sorted(glob.glob(os.path.join(config.out_dir, 'ckpt_sft_???????.pt')))
            for old in all_ckpts[:-config.keep_last_n_checkpoints]:
                os.remove(old)
                print(f"removed old checkpoint {os.path.basename(old)}")

    if iter_num == 0 and config.eval_only:
        break

    # 4. Forward + Backward (with gradient accumulation)
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        attn_mask = make_attn_mask(DocIds)
        with ctx:
            _, loss, _ = model(X, Y, attn_mask=attn_mask)
            loss = loss / config.gradient_accumulation_steps
        X, Y, DocIds = get_batch('train')
        scaler.scale(loss).backward()

    # 5. Gradient clipping
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # 6. Grad norm logging (before optimizer step so gradients are still live)
    grad_norms = {}
    if iter_num % _metrics_interval == 0 and master_process and config.tensorboard_log:
        if config.grad_clip == 0.0:
            scaler.unscale_(optimizer)
        for name, p in raw_model.named_parameters():
            if p.grad is not None:
                grad_norms[name] = p.grad.norm().item()

    # 7. Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 8. Timing and metrics
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_seen += tokens_per_iter

    if iter_num % config.log_interval == 0 and master_process:
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tokens {tokens_seen:,}")

    if iter_num % _metrics_interval == 0 and master_process and config.tensorboard_log:
        lossf = loss.item() * config.gradient_accumulation_steps
        tb_writer.add_scalar("train/loss", lossf, iter_num)
        tb_writer.add_scalar("lr", lr, iter_num)
        tb_writer.add_scalar("mfu", running_mfu * 100, iter_num)
        tb_writer.add_scalar("tokens_seen", tokens_seen, iter_num)

        for name, val in grad_norms.items():
            tb_writer.add_scalar(f"grad_norm/{name}", val, iter_num)

        if iter_num % 100 == 0:
            for name, p in raw_model.named_parameters():
                tb_writer.add_scalar(f"weight_norm/{name}", p.norm().item(), iter_num)

        for name, stats in activation_stats.items():
            if stats:
                means = [s['mean'] for s in stats]
                stds  = [s['std']  for s in stats]
                tb_writer.add_scalar(f"act_mean/{name}", sum(means) / len(means), iter_num)
                tb_writer.add_scalar(f"act_std/{name}",  sum(stds)  / len(stds),  iter_num)
        activation_stats.clear()
        tb_writer.flush()

    iter_num += 1
    local_iter_num += 1

    # 9. Termination
    if iter_num > config.max_iters:
        break

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

if ddp:
    destroy_process_group()

if master_process:
    def _to_serializable(v):
        return v.item() if hasattr(v, 'item') else v

    metrics = {
        'best_val_loss': _to_serializable(best_val_loss),
        'iter_num': iter_num,
        'tokens_seen': tokens_seen,
        'total_params': total_params,
        'config': config.to_dict(),
        'val_loss':   _to_serializable(losses.get('val'))   if 'losses' in locals() else None,
        'train_loss': _to_serializable(losses.get('train')) if 'losses' in locals() else None,
    }
    metrics_path = os.path.join(config.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
