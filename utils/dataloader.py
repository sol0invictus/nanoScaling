"""
Parquet-based on-the-fly tokenizing dataloader with BOS-aligned best-fit packing.

Ported from https://github.com/karpathy/nanochat/blob/master/nanochat/dataloader.py
and adapted for nanoScaling's architecture (tiktoken GPT-2, no nanochat deps).

Convention:
  - Parquet shards are named shard_00000.parquet, shard_00001.parquet, ...
  - Last shard = val split; all others = train split
  - Each shard has a single 'text' column of raw document strings
  - GPT-2 EOT token (50256) is prepended to each document as BOS

BOS-aligned best-fit packing algorithm:
  For each row of T+1 tokens:
    1. Find the LARGEST buffered document that fits entirely in the remaining space
    2. Repeat until no document fits entirely
    3. Crop the shortest buffered document to fill the remaining space exactly
  Result: every row starts with BOS, 100% utilization (no padding), ~35% tokens discarded by cropping
"""

import glob
import os

import numpy as np
import tiktoken
import torch

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError(
        "pyarrow is required for the parquet dataloader. "
        "Install it with: pip install pyarrow"
    )

EOT_TOKEN = 50256         # GPT-2 <|endoftext|>, used as both BOS and EOS
VOCAB_SIZE = 50304        # GPT-2 vocab padded to a nice multiple


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _list_parquet_files(data_dir: str) -> list:
    """Return sorted list of shard_?????.parquet paths in data_dir."""
    pattern = os.path.join(data_dir, "shard_?????.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No shard_?????.parquet files found in {data_dir}.\n"
            "Run data/fineweb_edu/prepare.py or data/fineweb_edu/download.py first."
        )
    return files


def _get_split_files(data_dir: str, split: str) -> list:
    """Return parquet paths for the given split.
    Convention: last shard = val, all others = train.
    """
    all_files = _list_parquet_files(data_dir)
    if split == 'train':
        if len(all_files) < 2:
            raise ValueError(
                f"Only {len(all_files)} shard(s) found in {data_dir}. "
                "Need at least 2 shards (1 for val, rest for train). "
                "Add more shards or use split='val'."
            )
        return all_files[:-1]
    elif split == 'val':
        return all_files[-1:]
    else:
        raise ValueError(f"Parquet dataloader only supports 'train' or 'val', got '{split}'")


# ---------------------------------------------------------------------------
# DDP-aware row-group iterator
# ---------------------------------------------------------------------------

def _iter_row_groups(files: list, ddp_rank: int = 0, ddp_world_size: int = 1):
    """Infinite generator over parquet row groups, DDP-sharded.

    Rank r takes row groups at global indices r, r+W, r+2W, ... cycling
    through epochs indefinitely.  Yields the 'text' column as a list of strings.
    """
    while True:  # infinite epochs
        rg_global_idx = 0
        for filepath in files:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                if rg_global_idx % ddp_world_size == ddp_rank:
                    batch = pf.read_row_group(rg_idx, columns=['text'])
                    yield batch.column('text').to_pylist()
                rg_global_idx += 1


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize_texts(texts: list, enc) -> list:
    """Tokenize a list of document strings.  Prepends EOT as BOS.

    Returns a list of token-id lists.
    """
    bos = enc.eot_token  # 50256
    return [[bos] + enc.encode_ordinary(t) for t in texts]


# ---------------------------------------------------------------------------
# Best-fit packing
# ---------------------------------------------------------------------------

def _pack_tokens(doc_buffer: list, B: int, T: int) -> list:
    """Pack B rows of exactly T+1 tokens from doc_buffer using best-fit algorithm.

    Mutates doc_buffer in place (consumes docs; re-inserts remainders).
    Returns a list of B token-id lists, each of length T+1.
    The caller slices [:T] for inputs and [1:T+1] for targets.
    """
    row_capacity = T + 1
    rows = []

    for _ in range(B):
        row = []
        pos = 0

        while pos < row_capacity:
            remaining = row_capacity - pos

            # Find the largest document that fits entirely in remaining space
            best_idx = -1
            best_len = -1
            for i, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = i
                    best_len = doc_len

            if best_idx >= 0:
                # Document fits — consume it whole
                doc = doc_buffer.pop(best_idx)
                row.extend(doc)
                pos += len(doc)
            else:
                # No document fits — crop the shortest one to fill remaining space
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer[shortest_idx]
                row.extend(doc[:remaining])
                remainder = doc[remaining:]
                if remainder:
                    doc_buffer[shortest_idx] = remainder
                else:
                    doc_buffer.pop(shortest_idx)
                pos = row_capacity  # done with this row

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_parquet_dataloader(
    data_dir: str,
    split: str,
    B: int,
    T: int,
    device,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    buffer_size: int = 1000,
    tokenizer_batch_size: int = 128,
):
    """Infinite generator yielding (inputs, targets) for training.

    inputs:  LongTensor [B, T]  — token ids
    targets: LongTensor [B, T]  — inputs shifted by 1

    Uses BOS-aligned best-fit packing: for each row slot in the batch,
    the largest available tokenized document that fits is placed first.
    When no document fits, the shortest document is cropped to fill the space.
    This gives 100% token utilization with no padding, discarding ~35% of tokens.

    Args:
        data_dir:            Directory containing shard_?????.parquet files
        split:               'train' or 'val'
        B:                   Batch size
        T:                   Sequence length (block_size)
        device:              Torch device string or device object
        ddp_rank:            This rank's index (0 for single-GPU)
        ddp_world_size:      Total number of DDP ranks (1 for single-GPU)
        buffer_size:         Number of documents to keep in the token buffer
        tokenizer_batch_size: Reserved (not used; tiktoken is fast enough inline)
    """
    enc = tiktoken.get_encoding("gpt2")
    files = _get_split_files(data_dir, split)
    rg_iter = _iter_row_groups(files, ddp_rank, ddp_world_size)

    doc_buffer = []  # list of token-id lists

    device_type = 'cuda' if ('cuda' in str(device)) else 'cpu'
    use_cuda = (device_type == 'cuda')

    def _fill_buffer():
        """Pull the next row group and append its tokenized docs to doc_buffer."""
        texts = next(rg_iter)
        tokenized = _tokenize_texts(texts, enc)
        doc_buffer.extend(tokenized)

    # Initial fill
    while len(doc_buffer) < buffer_size:
        _fill_buffer()

    while True:
        # Refill when buffer drops below half capacity
        while len(doc_buffer) < buffer_size // 2:
            _fill_buffer()

        rows = _pack_tokens(doc_buffer, B, T)

        # rows: list of B lists, each of length T+1
        arr = np.array(rows, dtype=np.int64)  # [B, T+1]
        x_np = arr[:, :T]
        y_np = arr[:, 1:T + 1]

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        if use_cuda:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

        yield x, y
