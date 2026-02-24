"""
RQ2: Does Muon's spectral control reduce or eliminate the need for normalization?

Full factorial experiment: 6 normalization conditions × 2 optimizers (Muon, AdamW).
Focuses on stability monitoring in addition to spectral diagnostics.

Stability criterion (per proposal):
  1. Val loss decreases monotonically over any 2000-step window (5% tolerance)
  2. Per-layer grad norms bounded: no spike > 100× rolling median (500-step window)
  3. No NaN/Inf in weights or activations

All 6 normalization conditions are supported via config:
  - Pre-RMSNorm    (norm_position=pre, use_rmsnorm=true, norm_affine=true)
  - Pre-LayerNorm  (norm_position=pre, use_rmsnorm=false, norm_affine=true)
  - Post-LayerNorm (norm_position=post, use_rmsnorm=false, norm_affine=true)
  - No Norm        (norm_position=none)
  - No Norm + scaled init (norm_position=none, norm_free_scaled_init=true)
  - Fixed-scale Norm (norm_position=pre, norm_affine=false)

Usage:
    python experiments/rq2_normalization/train.py \
        experiments/rq2_normalization/configs/125m_muon_no_norm.yaml

    bash experiments/rq2_normalization/run_rq2.sh --quick
"""

import os
import sys
import time
import math
import pickle
import json
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from models import GPTConfig, GPT
from utils.config import ExperimentConfig
from utils.parametrization import apply_parametrization
from utils.data import get_batch as get_batch_fn
from utils.spectral_metrics import SpectralLogger, StabilityMonitor

# ---------------------------------------------------------------------------
# Experiment-level logging frequencies
# ---------------------------------------------------------------------------

spectral_freq = 100
grad_freq = 50
topk_svd_freq = 500
activation_freq = 100
sharpness_freq = 5000
svd_checkpoint_freq = 2000

# ---------------------------------------------------------------------------
# Configuration
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
        assert arg.startswith('--'), f"Unknown argument: {arg}"
        key, val = arg[2:].split('=', 1)
        for exp_var in ['spectral_freq', 'grad_freq', 'topk_svd_freq',
                        'activation_freq', 'sharpness_freq', 'svd_checkpoint_freq']:
            if key == exp_var:
                globals()[exp_var] = int(val)
                print(f"Experiment param: {key} = {val}")
                break
        else:
            obj = config
            while '.' in key:
                sub, key = key.split('.', 1)
                obj = getattr(obj, sub)
            t = type(getattr(obj, key))
            v = (val.lower() == 'true') if t == bool else (int(val) if t == int else
                 (float(val) if t == float else val))
            setattr(obj, key, v)
            print(f"Config override: {key} = {v}")

# ---------------------------------------------------------------------------
# System setup
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
    print(f"norm_position={config.norm_position}, norm_affine={config.norm_affine}, "
          f"norm_free_scaled_init={config.norm_free_scaled_init}")
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

def get_batch(split):
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
# Optimizer + LR
# ---------------------------------------------------------------------------

scaler = torch.amp.GradScaler(device='cuda', enabled=(config.dtype == 'float16'))
optimizer = model.configure_optimizers(
    config.weight_decay, config.learning_rate,
    (config.beta1, config.beta2), device_type
)
initial_lrs = [pg['lr'] for pg in optimizer.param_groups]

if config.compile:
    print("compiling the model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model


def get_lr_multiplier(it: int) -> float:
    min_ratio = config.min_lr / config.learning_rate
    if it < config.warmup_iters:
        return (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return min_ratio
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_ratio + coeff * (1.0 - min_ratio)


def apply_lr(it: int):
    if not config.decay_lr:
        return
    mult = get_lr_multiplier(it)
    for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
        pg['lr'] = init_lr * mult

# ---------------------------------------------------------------------------
# Spectral Logger + Stability Monitor
# ---------------------------------------------------------------------------

tb_writer = None
spectral_logger = None
stability_monitor = StabilityMonitor()

if master_process and config.tensorboard_log:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(
        log_dir=os.path.join(config.out_dir, 'runs', config.tensorboard_run_name))
    spectral_logger = SpectralLogger(
        model=raw_model,
        tb_writer=tb_writer,
        out_dir=config.out_dir,
        spectral_freq=spectral_freq,
        grad_freq=grad_freq,
        topk_svd_freq=topk_svd_freq,
        activation_freq=activation_freq,
        sharpness_freq=sharpness_freq,
        stability_monitor=stability_monitor,
    )

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    splits = sorted(set(['train'] + config.val_splits))
    for split in splits:
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
    model.train()
    return out

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
tokens_seen = 0

sharpness_X, sharpness_Y = get_batch('val') if 'val' in config.val_splits else (X, Y)

# Stability tracking
is_stable = True
stability_log = []

while True:
    apply_lr(iter_num)

    # Evaluation + checkpointing
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        loss_msg = f"step {iter_num}:"
        for k, v in losses.items():
            loss_msg += f" {k}_loss={v:.4f}"

        current_val_loss = losses.get('val', next(iter(losses.values())))

        # --- RQ2 Stability check ---
        if spectral_logger:
            stability_result = spectral_logger.check_stability(
                iter_num, current_val_loss.item(), tb_writer)
            is_stable = stability_result['is_stable']
            stability_log.append({'step': iter_num,
                                   'val_loss': current_val_loss.item(),
                                   **stability_result})
            if not is_stable:
                loss_msg += f" [UNSTABLE: {stability_result['issues'][:2]}]"

        print(loss_msg)

        if tb_writer:
            for k, v in losses.items():
                tb_writer.add_scalar(f'val/loss_{k}', v, iter_num)

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

        if spectral_logger and iter_num % svd_checkpoint_freq == 0 and iter_num > 0:
            spectral_logger.log_checkpoint(iter_num)

        # Early stop on catastrophic instability (NaN loss or weight explosion)
        if not math.isfinite(current_val_loss.item()):
            print(f"DIVERGED at step {iter_num}. Stopping.")
            is_stable = False
            break

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

    # Check for NaN in loss before clipping
    lossf = loss.item() * config.gradient_accumulation_steps
    if not math.isfinite(lossf):
        print(f"NaN/Inf loss at step {iter_num}. Stopping early.")
        is_stable = False
        break

    # Gradient clipping (or unscale for logging)
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    elif master_process and spectral_logger:
        scaler.unscale_(optimizer)

    # Spectral + stability logging
    if master_process and spectral_logger:
        spectral_logger.log_step(
            step=iter_num,
            X=sharpness_X,
            Y=sharpness_Y,
            ctx=ctx,
        )

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_seen += tokens_per_iter

    if iter_num % config.log_interval == 0 and master_process:
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
            tb_writer.add_scalar('train/tokens_seen', tokens_seen, iter_num)
            tb_writer.flush()

    iter_num += 1
    local_iter_num += 1

    if iter_num > config.max_iters:
        break

# ---------------------------------------------------------------------------
# Cleanup + final metrics
# ---------------------------------------------------------------------------

if spectral_logger:
    spectral_logger.remove_hooks()
    spectral_logger.log_checkpoint(iter_num)

if ddp:
    destroy_process_group()

if master_process:
    def to_serializable(val):
        return val.item() if hasattr(val, 'item') else val

    final_losses = estimate_loss()
    metrics = {
        'best_val_loss': to_serializable(best_val_loss),
        'iter_num': iter_num,
        'tokens_seen': tokens_seen,
        'total_params': total_params,
        'config': config.to_dict(),
        'val_loss': to_serializable(final_losses.get('val', float('nan'))),
        'train_loss': to_serializable(final_losses.get('train', float('nan'))),
        # RQ2 stability verdict
        'is_stable': is_stable,
        'stability_log': stability_log[-20:],  # last 20 eval checkpoints
    }
    metrics_path = os.path.join(config.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
    print(f"Final stability: {'STABLE' if is_stable else 'UNSTABLE'}")
