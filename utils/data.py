import glob
import os
import warnings

import torch
import numpy as np

def get_config_val(config, key, default=None):
    return getattr(config, key, default)


def _should_use_parquet(config, data_dir: str) -> bool:
    """Return True if the parquet dataloader should be used for this dataset/dir.

    Dispatch rules (in order):
      data_format='bin'     → always False (force bin path)
      data_format='parquet' → always True  (force parquet path)
      data_format='auto'    → True if shard_?????.parquet files exist in data_dir
    """
    fmt = getattr(config, 'data_format', 'auto')
    if fmt == 'bin':
        return False
    if fmt == 'parquet':
        return True
    # 'auto': detect by presence of named shard files
    pattern = os.path.join(data_dir, 'shard_?????.parquet')
    return len(glob.glob(pattern)) > 0


def create_dataloader(split, config, device_type, device, data_dir=None,
                      ddp_rank=0, ddp_world_size=1):
    """Return an infinite generator yielding (x, y) LongTensor[B, T] pairs.

    Dispatches to the parquet on-the-fly tokenizing dataloader when parquet
    shards are detected (or data_format='parquet'), otherwise wraps the
    existing get_batch() in an infinite generator for backward compatibility.

    Args:
        split:          'train', 'val', or a custom split name (bin mode only)
        config:         ExperimentConfig
        device_type:    'cuda' or 'cpu'
        device:         torch device
        data_dir:       path to dataset directory (default: data/<config.dataset>)
        ddp_rank:       this rank's DDP index (0 for single-GPU)
        ddp_world_size: total DDP ranks (1 for single-GPU)
    """
    if data_dir is None:
        data_dir = os.path.join('data', config.dataset)

    if _should_use_parquet(config, data_dir):
        if split not in ('train', 'val'):
            warnings.warn(
                f"Parquet dataloader does not support split '{split}'. "
                "Only 'train' and 'val' are available. Falling back to 'val'.",
                UserWarning,
            )
            split = 'val'
        from utils.dataloader import create_parquet_dataloader
        return create_parquet_dataloader(
            data_dir=data_dir,
            split=split,
            B=config.batch_size,
            T=config.block_size,
            device=device,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            buffer_size=getattr(config, 'dataloader_buffer_size', 1000),
            tokenizer_batch_size=getattr(config, 'tokenizer_batch_size', 128),
        )
    else:
        # Wrap existing get_batch in an infinite generator (no functional change)
        def _bin_gen():
            while True:
                yield get_batch(split, config, device_type, device, data_dir)
        return _bin_gen()


def get_batch(split, config, device_type, device, data_dir=None):
    # poor man's data loader
    if data_dir is None:
        data_dir = os.path.join('data', config.dataset)
        
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        # Fallback for custom files (e.g. val_wikitext.bin)
        # If split is a filename (ends with .bin), use it directly
        path = os.path.join(data_dir, split)
        if not os.path.exists(path) and not split.endswith('.bin'):
             path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(path):
            data = np.memmap(path, dtype=np.uint16, mode='r')
        else:
             # Default fallback if not found? Or raise error?
             # Let's try to look for valid bin file
             raise FileNotFoundError(f"Could not find data file for split '{split}' in {data_dir}")
    
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
