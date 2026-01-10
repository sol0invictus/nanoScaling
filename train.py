"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import json
from contextlib import nullcontext
import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from config import ExperimentConfig
from parametrization import apply_parametrization

# -----------------------------------------------------------------------------
# Configuration setup
config = ExperimentConfig()

# Quick and dirty argument parsing to override config
# Supports --key=value or config file paths
for arg in sys.argv[1:]:
    if '=' not in arg:
        try:
            # Assume config file (Python or YAML)
            # If standard key=val failed, try loading as yaml if extension matches
            if arg.endswith('.yaml') or arg.endswith('.yml'):
                config = ExperimentConfig.from_yaml(arg)
                print(f"Loaded config from {arg}")
            else:
                # Support legacy python config files by executing them in a dummy dict context
                # and updating config? Or just warn.
                # The user might stick to CLI args mostly.
                # Let's support simple naive yaml loading for now as strictly requested.
                pass
        except Exception as e:
            print(f"Error loading config {arg}: {e}")
    else:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        # Traverse nested config (e.g. parametrization.mode)
        obj = config
        while '.' in key:
            sub, key = key.split('.', 1)
            obj = getattr(obj, sub)
        
        # Infer type
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

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=config.backend) # config needs backend field? It wasn't in ExperimentConfig def. I need to add it or default it.
    # Ah, I missed 'backend' in ExperimentConfig. I'll add it or just hardcode 'nccl' here as it was.
    # It was: backend = 'nccl'
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert config.gradient_accumulation_steps % ddp_world_size == 0
    config.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = config.device
tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', config.dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
if config.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    config.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    model = GPT(config)
    
    # Apply advanced parametrization (MuP, CompleteP, etc.)
    apply_parametrization(model, config)
    
elif config.init_from == 'resume':
    print(f"Resuming training from {config.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # We must strict load config because architecture might have changed?
    # Actually, usually we want to respect the checkpoint's config for architecture info.
    # But for now let's assume valid resume.
    # We can update config from checkpoint if we stored it?
    # checkpoint['config'] is dict. ExperimentConfig.from_dict would be nice.
    # For now, let's just warn or assume config matches.
    
    # Force architecture params from checkpoint
    checkpoint_config = checkpoint['config'] # this was a dict
    # We should update our config object with these values if possible
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'use_rmsnorm', 'use_rope', 'use_swiglu']:
        if k in checkpoint_config and hasattr(config, k):
            setattr(config, k, checkpoint_config[k])
            
    model = GPT(config)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
elif config.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=config.dropout)
    model = GPT.from_pretrained(config.init_from, override_args)
    # read off the created config params
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        setattr(config, k, getattr(model.config, k))
        
# crop down the model block size if desired, using model surgery
if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    config.block_size = config.block_size # update config to match

model.to(device)

# Detailed parameter count logging
print("Model Parameter Breakdown:")
total_params = 0
for name, p in model.named_parameters():
    print(f"{name}: {p.shape} ({p.numel()} params)")
    total_params += p.numel()
print(f"Total parameters: {total_params}")


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device='cuda', enabled=(config.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
if config.init_from == 'resume':
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


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
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

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
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

# logging
if config.wandb_log and master_process:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config.to_dict())

if config.tensorboard_log and master_process:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'runs', config.tensorboard_run_name))


# training loop
# Activation logging hooks
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

    # Register hooks
    target_model = model.module if ddp else model
    # Apply to blocks (transformer.h)
    for i, block in enumerate(target_model.transformer.h):
         block.register_forward_hook(get_activation_hook(f'block_{i}'))

X, Y = get_batch('train') # fetch the very first batch

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# QoL: total tokens seen
tokens_seen = iter_num * tokens_per_iter

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        # Support per-parameter scaling if set by parametrization
        # Logic: param_group['lr'] = lr * scaling_factor
        # But we don't have per-parameter groups usually unless we created them.
        # CombinedOptimizer might handle multiple groups.
        # If we have Muon + AdamW, we have 2 groups (or more).
        # We need to preserve the relative scaling if we set it in `configure_optimizers`.
        # Standard: param_group['lr'] = lr. 
        # Advanced: param_group['lr'] = lr * group.get('lr_scale', 1.0)
        # We need to make sure `configure_optimizers` set `lr_scale` or `initial_lr`.
        # For now, simplistic update:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if config.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "tokens_seen": tokens_seen,
            })
        if losses['val'] < best_val_loss or config.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': None, # We rely on config now? Or save generic args dict?
                    # Ideally save config object.
                    'config': config.to_dict(), # Save the full experiment config
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
    if iter_num == 0 and config.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    
    # clip the gradient
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    # log gradient norms (must be done before zero_grad)
    grad_norms = {}
    if iter_num % config.log_interval == 0 and master_process and config.tensorboard_log:
        # ensure unscaled if it wasn't clipped (which unscales)
        if config.grad_clip == 0.0:
            scaler.unscale_(optimizer)
        for name, p in raw_model.named_parameters():
            if p.grad is not None:
                grad_norms[name] = p.grad.norm().item()

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    tokens_seen += tokens_per_iter
    
    if iter_num % config.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tokens {tokens_seen:,}")
        
        if config.tensorboard_log:
            tb_writer.add_scalar("iter", iter_num, iter_num)
            tb_writer.add_scalar("train/loss", lossf, iter_num)
            tb_writer.add_scalar("lr", lr, iter_num)
            tb_writer.add_scalar("mfu", running_mfu*100, iter_num)
            tb_writer.add_scalar("tokens_seen", tokens_seen, iter_num)
            
            # log gradients
            for name, val in grad_norms.items():
                tb_writer.add_scalar(f"grad_norm/{name}", val, iter_num)
            
            # log weights
            for name, p in raw_model.named_parameters():
                tb_writer.add_scalar(f"weight_norm/{name}", p.norm().item(), iter_num)
            
            # log activations
            for name, stats in activation_stats.items():
                if stats:
                    means = [s['mean'] for s in stats]
                    stds = [s['std'] for s in stats]
                    tb_writer.add_scalar(f"act_mean/{name}", sum(means)/len(means), iter_num)
                    tb_writer.add_scalar(f"act_std/{name}", sum(stds)/len(stds), iter_num)
            activation_stats = {} # clear statistics after logging
            tb_writer.flush()

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > config.max_iters:
        break

if ddp:
    destroy_process_group()

if master_process:
    # ensure metrics are serializable
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
