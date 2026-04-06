"""
SFT Training Script (train_sft.py)

Supervised fine-tuning loop over packed conversation data produced by
data/prepare_sft.py.  Key features:

  - Block-aligned batch sampling: every sampled sequence starts at a
    conversation boundary (BOS token), avoiding partial-conversation waste.
  - Block-diagonal causal attention via doc_ids: tokens attend only within
    their own conversation, preventing cross-conversation information leakage.
  - Loss masking: only assistant-response tokens contribute to the loss
    (user prompts and padding are masked with ignore_index=-1).
  - Optimizer: AdamW (default), Muon, or Scion — dispatched through the
    model's configure_optimizers() method, identical to train.py.
  - DDP multi-GPU training, mixed precision (bfloat16/float16), torch.compile.
  - Cosine LR schedule with linear warmup.
  - TensorBoard + optional WandB logging.
  - Decoupled checkpointing with keep_last_n_checkpoints.
  - metrics.json output at the end of the run.

Data format (produced by data/prepare_sft.py):
  {split}.bin          uint16  flattened (N_blocks × block_size) token ids
  {split}_mask.bin     uint8   loss mask (1=predict, 0=ignore)
  {split}_doc_ids.bin  uint32  conversation id per token (for block-diag attn)

Usage:
  # Single GPU
  python train_sft.py configs/sft.yaml

  # Multi-GPU DDP
  torchrun --standalone --nproc_per_node=4 train_sft.py configs/sft.yaml

  # Override any config key
  python train_sft.py configs/sft.yaml --learning_rate=3e-4 --batch_size=8
"""

import gc
import glob
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from models import GPT, GPTConfig
from utils.config import ExperimentConfig
from utils.parametrization import apply_parametrization

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = ExperimentConfig()

for arg in sys.argv[1:]:
    if "=" not in arg:
        if arg.endswith(".yaml") or arg.endswith(".yml"):
            try:
                config = ExperimentConfig.from_yaml(arg)
                print(f"Loaded config from {arg}")
            except Exception as e:
                print(f"Error loading config {arg}: {e}")
    else:
        assert arg.startswith("--")
        key, val = arg.split("=", 1)
        key = key[2:]
        obj = config
        while "." in key:
            sub, key = key.split(".", 1)
            obj = getattr(obj, sub)
        target_type = type(getattr(obj, key))
        if target_type == bool:
            v = val.lower() == "true"
        elif target_type == int:
            v = int(val)
        elif target_type == float:
            v = float(val)
        else:
            v = val
        setattr(obj, key, v)
        print(f"Override: {key} = {v}")

# -----------------------------------------------------------------------------
# System setup (DDP, device, seed)
# -----------------------------------------------------------------------------

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=getattr(config, "backend", "nccl"))
    ddp_rank       = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device         = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset    = ddp_rank
    assert config.gradient_accumulation_steps % ddp_world_size == 0
    config.gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset    = 0
    ddp_world_size = 1
    device         = config.device

tokens_per_iter = (
    config.gradient_accumulation_steps * ddp_world_size
    * config.batch_size * config.block_size
)
if master_process:
    print(f"tokens per iteration: {tokens_per_iter:,}")
    os.makedirs(config.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device_type = "cuda" if "cuda" in device else "cpu"

ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

data_dir = os.path.join("data", config.dataset)

# Memory-map all splits once at startup; re-open lazily if needed.
_mmap_cache: dict = {}

def _open_split(split: str):
    """Return (tokens, mask, doc_ids) memory-mapped arrays for a split."""
    if split in _mmap_cache:
        return _mmap_cache[split]
    bin_path  = os.path.join(data_dir, f"{split}.bin")
    mask_path = os.path.join(data_dir, f"{split}_mask.bin")
    doc_path  = os.path.join(data_dir, f"{split}_doc_ids.bin")
    data     = np.memmap(bin_path,  dtype=np.uint16, mode="r")
    mask_arr = np.memmap(mask_path, dtype=np.uint8,  mode="r")
    doc_arr  = np.memmap(doc_path,  dtype=np.uint32, mode="r")
    _mmap_cache[split] = (data, mask_arr, doc_arr)
    return _mmap_cache[split]


def get_batch(split: str):
    """
    Sample a batch from the packed SFT data.

    Sampling is block-aligned: each sampled sequence starts at the beginning
    of a pre-packed block (which begins with a BOS/EOT token).  This avoids
    starting mid-conversation and ensures the doc_id / attention mask is
    coherent from the very first token.

    The target for the last token in each sequence is set to -1 (ignored)
    since we do not have the first token of the next block's continuation.
    """
    data, mask_arr, doc_arr = _open_split(split)
    T = config.block_size
    n_blocks = len(data) // T  # number of complete pre-packed blocks

    # Sample block indices (uniform over complete blocks)
    block_idx = torch.randint(n_blocks, (config.batch_size,))

    x_list, y_list, d_list = [], [], []
    for bi in block_idx:
        start = int(bi) * T
        # Input: tokens [start, start+T)
        x_chunk = torch.from_numpy(data[start : start + T].astype(np.int64))
        # Target: tokens [start+1, start+T+1) — but the +T token is the BOS of
        # the next block (or out-of-bounds), so we grab the interior shift and
        # mask the last position.
        y_chunk = x_chunk.clone()
        y_chunk[:-1] = x_chunk[1:]          # y[t] = x[t+1] for t < T-1
        y_chunk[-1]  = -1                   # ignore last position

        # Loss mask: mask[t]=1 means token t is an assistant token that should
        # be predicted.  The model predicts token t at output position t-1,
        # so y[t-1] should be kept when mask[t]=1 — i.e. use mask shifted +1.
        m_chunk = torch.from_numpy(mask_arr[start : start + T].astype(np.int64))
        # y[i] = x[i+1]; keep y[i] only when mask[i+1] = 1
        # m_chunk[1:] gives mask for positions 1..T-1 (the targets at y[0..T-2])
        y_chunk[:-1][m_chunk[1:] == 0] = -1
        # y[-1] is already -1 (set above)

        d_chunk = torch.from_numpy(doc_arr[start : start + T].astype(np.int64))

        x_list.append(x_chunk)
        y_list.append(y_chunk)
        d_list.append(d_chunk)

    x       = torch.stack(x_list)
    y       = torch.stack(y_list)
    doc_ids = torch.stack(d_list)

    if device_type == "cuda":
        x       = x.pin_memory().to(device, non_blocking=True)
        y       = y.pin_memory().to(device, non_blocking=True)
        doc_ids = doc_ids.pin_memory().to(device, non_blocking=True)
    else:
        x, y, doc_ids = x.to(device), y.to(device), doc_ids.to(device)

    return x, y, doc_ids


# -----------------------------------------------------------------------------
# Model initialisation
# -----------------------------------------------------------------------------

iter_num      = 0
best_val_loss = 1e9
checkpoint    = None

print(f"Initialising model for SFT — init_from: {config.init_from}")

if config.init_from == "scratch":
    if not hasattr(config, "vocab_size") or config.vocab_size is None:
        config.vocab_size = 50304
    model = GPT(config)
    apply_parametrization(model, config)

elif config.init_from == "resume":
    ckpt_path = os.path.join(config.out_dir, "ckpt_sft.pt")
    print(f"Resuming SFT from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    _ckpt_cfg = checkpoint.get("config", {})
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size",
              "use_rmsnorm", "use_rope", "use_swiglu"]:
        if k in _ckpt_cfg and hasattr(config, k):
            setattr(config, k, _ckpt_cfg[k])
    model = GPT(config)
    _sd = checkpoint["model"]
    for k in list(_sd.keys()):
        if k.startswith("_orig_mod."):
            _sd[k[len("_orig_mod."):]] = _sd.pop(k)
    model.load_state_dict(_sd)
    iter_num      = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

elif os.path.isfile(config.init_from):
    # Fine-tune from a pretrained checkpoint
    print(f"Loading pretrained checkpoint: {config.init_from}")
    checkpoint = torch.load(config.init_from, map_location=device)
    _ckpt_cfg = checkpoint.get("config", {})
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size",
              "use_rmsnorm", "use_rope", "use_swiglu"]:
        if k in _ckpt_cfg and hasattr(config, k):
            setattr(config, k, _ckpt_cfg[k])
    model = GPT(config)
    _sd = checkpoint["model"]
    for k in list(_sd.keys()):
        if k.startswith("_orig_mod."):
            _sd[k[len("_orig_mod."):]] = _sd.pop(k)
    model.load_state_dict(_sd)
    iter_num      = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", 1e9)

elif config.init_from.startswith("gpt2"):
    print(f"Initialising from OpenAI GPT-2 weights: {config.init_from}")
    model = GPT.from_pretrained(config.init_from, dict(dropout=config.dropout))
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        setattr(config, k, getattr(model.config, k))

else:
    raise ValueError(f"Unknown init_from: {config.init_from!r}")

if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
if master_process:
    print(f"Model parameters: {total_params / 1e6:.2f}M")

# -----------------------------------------------------------------------------
# Optimizer, scaler, compile, DDP
# -----------------------------------------------------------------------------

scaler = torch.amp.GradScaler(device="cuda", enabled=(config.dtype == "float16"))

optimizer = model.configure_optimizers(
    config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type
)
# Store per-group initial LRs so CombinedOptimizer (Muon+AdamW) decays proportionally
initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

if checkpoint is not None and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free memory

raw_model = model  # unwrapped GPT — updated below after wrapping

if config.compile:
    print("Compiling model… (~1 min)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    # DDP.module is compile(GPT) or GPT depending on whether compile is on.
    # Either way, .state_dict() / .named_parameters() work correctly.
    raw_model = model.module
elif config.compile:
    raw_model = model  # OptimizedModule — state_dict() proxies to underlying GPT

# -----------------------------------------------------------------------------
# WandB (optional)
# -----------------------------------------------------------------------------

use_wandb = getattr(config, "use_wandb", False)
wandb_run  = None
if use_wandb and master_process:
    try:
        import wandb
        wandb_project = getattr(config, "wandb_project", "nanoscaling-sft")
        wandb_run_name = getattr(config, "wandb_run_name", config.out_dir)
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name,
                               config=config.to_dict())
        print(f"WandB run: {wandb_run.url}")
    except ImportError:
        print("WandB not installed; skipping.")

# -----------------------------------------------------------------------------
# TensorBoard (optional)
# -----------------------------------------------------------------------------

tb_writer = None
if getattr(config, "tensorboard_log", False) and master_process:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(
        log_dir=os.path.join(config.out_dir, "runs",
                             getattr(config, "tensorboard_run_name", "sft"))
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_attn_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """
    Build a block-diagonal causal attention mask from doc_ids.

    Tokens attend only to earlier tokens in the same conversation.
    Returns a (B, 1, T, T) bool tensor (True = attend).
    """
    B, T = doc_ids.size()
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(1)  # (B, T, T)
    causal   = torch.tril(torch.ones(T, T, device=doc_ids.device, dtype=torch.bool))
    return (same_doc & causal).unsqueeze(1)                   # (B, 1, T, T)


@torch.no_grad()
def estimate_loss() -> dict:
    """Evaluate mean loss on train and val splits.  All DDP ranks participate."""
    out = {}
    model.eval()
    for split in sorted({"train", "val"} | set(config.val_splits)):
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
        mean_loss = losses[losses != 0].mean().item() if (losses != 0).any() else float("nan")
        if ddp:
            t = torch.tensor(mean_loss, device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
            mean_loss = t.item()
        out[split] = mean_loss
    if device_type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    model.train()
    return out


def get_lr(it: int) -> float:
    """Cosine LR schedule with linear warmup."""
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def apply_lr(it: int):
    """
    Apply cosine decay to each param group using its own initial LR.
    Ensures CombinedOptimizer (Muon + AdamW) groups decay proportionally.
    """
    if not config.decay_lr:
        return
    decay = get_lr(it)
    base  = config.learning_rate if config.learning_rate > 0 else 1.0
    ratio = decay / base
    for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
        pg["lr"] = init_lr * ratio


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

X, Y, DocIds = get_batch("train")
t0 = time.time()
local_iter_num = 0
running_mfu    = -1.0
tokens_seen    = iter_num * tokens_per_iter
_metrics_interval = (
    config.metrics_log_interval if config.metrics_log_interval > 0
    else config.log_interval
)

while True:
    # 1. LR update
    apply_lr(iter_num)

    # 2. Evaluation (all ranks participate so all_reduce in estimate_loss works)
    if iter_num % config.eval_interval == 0:
        losses = estimate_loss()
        if master_process:
            parts = [f"{k} loss {v:.4f}" for k, v in losses.items()]
            print(f"step {iter_num}: {', '.join(parts)}")

            current_val = losses.get("val", next(iter(losses.values())))
            if current_val < best_val_loss:
                best_val_loss = current_val

            if tb_writer:
                for k, v in losses.items():
                    tb_writer.add_scalar(f"val/{k}", v, iter_num)

            if wandb_run:
                wandb_run.log({"val/" + k: v for k, v in losses.items()}, step=iter_num)

    # 3. Checkpointing (decoupled from eval)
    _ckpt_interval = config.checkpoint_interval if config.checkpoint_interval > 0 else config.eval_interval
    if iter_num % _ckpt_interval == 0 and master_process and iter_num > 0:
        ckpt_data = {
            "model":          raw_model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "config":         config.to_dict(),
            "iter_num":       iter_num,
            "best_val_loss":  best_val_loss,
        }
        numbered = os.path.join(config.out_dir, f"ckpt_sft_{iter_num:07d}.pt")
        latest   = os.path.join(config.out_dir, "ckpt_sft.pt")
        print(f"saving checkpoint to {numbered}")
        torch.save(ckpt_data, numbered)
        shutil.copy2(numbered, latest)
        if config.keep_last_n_checkpoints > 0:
            all_ckpts = sorted(glob.glob(os.path.join(config.out_dir, "ckpt_sft_???????.pt")))
            for old in all_ckpts[: -config.keep_last_n_checkpoints]:
                os.remove(old)
                print(f"  removed {os.path.basename(old)}")

    if iter_num == 0 and config.eval_only:
        break

    # 4. Forward + backward with gradient accumulation
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == config.gradient_accumulation_steps - 1
            )
        attn_mask = make_attn_mask(DocIds)
        with ctx:
            _, loss, _ = model(X, Y, attn_mask=attn_mask)
            loss = loss / config.gradient_accumulation_steps
        X, Y, DocIds = get_batch("train")
        scaler.scale(loss).backward()

    # 5. Gradient clipping
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # 6. Grad norm logging (before step so gradients are still live)
    grad_norms: dict = {}
    if iter_num % _metrics_interval == 0 and master_process and tb_writer:
        if config.grad_clip == 0.0:
            scaler.unscale_(optimizer)
        for name, p in raw_model.named_parameters():
            if p.grad is not None:
                grad_norms[name] = p.grad.norm().item()

    # 7. Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 8. Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_seen += tokens_per_iter

    lossf = loss.item() * config.gradient_accumulation_steps

    if iter_num % config.log_interval == 0 and master_process:
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(
                config.batch_size * config.gradient_accumulation_steps, dt
            )
            running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
        lr_display = optimizer.param_groups[0]["lr"]
        print(
            f"iter {iter_num}: loss {lossf:.4f}, lr {lr_display:.2e}, "
            f"time {dt*1000:.1f}ms, mfu {running_mfu*100:.2f}%, "
            f"tokens {tokens_seen:,}"
        )

    if iter_num % _metrics_interval == 0 and master_process:
        if tb_writer:
            tb_writer.add_scalar("train/loss", lossf, iter_num)
            tb_writer.add_scalar("train/lr",   optimizer.param_groups[0]["lr"], iter_num)
            tb_writer.add_scalar("train/mfu",  running_mfu * 100, iter_num)
            tb_writer.add_scalar("train/tokens_seen", tokens_seen, iter_num)
            for name, val in grad_norms.items():
                tb_writer.add_scalar(f"grad/{name}", val, iter_num)
            if iter_num % 100 == 0:
                for name, p in raw_model.named_parameters():
                    tb_writer.add_scalar(f"weight/{name}", p.norm().item(), iter_num)
            tb_writer.flush()

        if wandb_run:
            wandb_run.log({
                "train/loss":         lossf,
                "train/lr":           optimizer.param_groups[0]["lr"],
                "train/mfu":          running_mfu * 100,
                "train/tokens_seen":  tokens_seen,
            }, step=iter_num)

    iter_num       += 1
    local_iter_num += 1

    if iter_num > config.max_iters:
        break

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

if ddp:
    destroy_process_group()

if master_process:
    metrics = {
        "best_val_loss": float(best_val_loss),
        "iter_num":      iter_num,
        "tokens_seen":   tokens_seen,
        "total_params":  total_params,
        "config":        config.to_dict(),
    }
    try:
        metrics["val_loss"]   = float(losses.get("val",   float("nan")))
        metrics["train_loss"] = float(losses.get("train", float("nan")))
    except NameError:
        pass  # never evaluated (e.g. max_iters=0)

    metrics_path = os.path.join(config.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

    if wandb_run:
        wandb_run.log(metrics)
        wandb_run.finish()
