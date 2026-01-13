import os
import torch
import numpy as np

def get_config_val(config, key, default=None):
    return getattr(config, key, default)

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
