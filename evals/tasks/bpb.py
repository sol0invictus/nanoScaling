"""
BPB — Bits Per Byte
(Vocabulary-size-independent language-model quality metric)

Formula:
    BPB = sum_t( CE_loss(t) ) / ( ln(2) · sum_t( byte_len(t) ) )

where the sum is over all non-special, non-padding tokens in the corpus.

Special tokens (id >= 50257 in GPT-2) have byte_len=0 and are skipped.
BOS/EOS (50256) is also excluded from the byte count but contributes to
context (it's included in the input).

Sources evaluated:
  - data/wikitext103/   (if prepared)
  - data/pile_val/      (if prepared)
  - Any user-specified parquet or .bin folder

Falls back gracefully if a corpus is unavailable.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# Pre-compute byte lengths for GPT-2 vocab (vocab_size = 50257)
# We decode each token id and measure the UTF-8 byte length.
# Tokens that cannot be decoded cleanly get byte_len=0 (excluded).
def _build_byte_lengths(enc, vocab_size: int = 50257) -> torch.Tensor:
    byte_lens = torch.zeros(vocab_size, dtype=torch.float32)
    for tok_id in range(vocab_size):
        try:
            text = enc.decode([tok_id])
            byte_lens[tok_id] = len(text.encode("utf-8"))
        except Exception:
            byte_lens[tok_id] = 0
    # EOT / BOS (50256) – don't count its bytes since it's a delimiter
    if vocab_size > 50256:
        byte_lens[50256] = 0
    return byte_lens


def _eval_bpb_bin(model, bin_path: str, enc, device: str, block_size: int,
                  byte_lens: torch.Tensor, max_batches: int) -> Optional[Dict]:
    """Evaluate BPB on a .bin file (uint16 mmap)."""
    if not os.path.isfile(bin_path):
        return None
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_blocks = len(data) // block_size
    if n_blocks == 0:
        return None

    indices = list(range(n_blocks))
    if max_batches and len(indices) > max_batches:
        import random
        rng = random.Random(42)
        indices = rng.sample(indices, max_batches)

    total_nats  = 0.0
    total_bytes = 0.0

    for b in tqdm(indices, desc=f"BPB {os.path.basename(bin_path)}", leave=False):
        start  = b * block_size
        tokens = torch.from_numpy(data[start : start + block_size].astype(np.int64)).to(device)
        x = tokens[:-1].unsqueeze(0)
        y = tokens[1:]

        with torch.no_grad():
            logits, _, _ = model(x)

        losses = F.cross_entropy(logits[0], y, reduction="none")   # (T-1,)
        blens  = byte_lens[y.cpu()].to(device)                     # (T-1,)

        valid        = blens > 0
        total_nats  += losses[valid].sum().item()
        total_bytes += blens[valid].sum().item()

    if total_bytes == 0:
        return None
    bpb = total_nats / (math.log(2) * total_bytes)
    return {"bpb": bpb, "total_bytes": total_bytes}


def _eval_bpb_parquet(model, data_dir: str, enc, device: str, block_size: int,
                      byte_lens: torch.Tensor, max_batches: int) -> Optional[Dict]:
    """Evaluate BPB on parquet shards (last shard = val by convention)."""
    import glob as gl
    shards = sorted(gl.glob(os.path.join(data_dir, "shard_?????.parquet")))
    if not shards:
        return None

    # Use the last shard (val by convention)
    shard = shards[-1]
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(shard)
        texts = table.column("text").to_pylist()
    except Exception as e:
        print(f"  [BPB] Could not read {shard}: {e}")
        return None

    if max_batches:
        texts = texts[:max_batches]

    total_nats  = 0.0
    total_bytes = 0.0
    EOT_ID = 50256

    for text in tqdm(texts, desc=f"BPB {os.path.basename(shard)}", leave=False):
        if not text or not isinstance(text, str):
            continue
        token_ids = enc.encode(text)
        if len(token_ids) < 2:
            continue

        # Process in block_size chunks
        for start in range(0, len(token_ids) - 1, block_size - 1):
            chunk = token_ids[start : start + block_size]
            if len(chunk) < 2:
                continue
            tokens = torch.tensor(chunk, dtype=torch.long, device=device)
            x = tokens[:-1].unsqueeze(0)
            y = tokens[1:]

            with torch.no_grad():
                logits, _, _ = model(x)

            losses = F.cross_entropy(logits[0], y, reduction="none")
            blens  = byte_lens[y.cpu()].to(device)
            valid  = blens > 0

            total_nats  += losses[valid].sum().item()
            total_bytes += blens[valid].sum().item()

    if total_bytes == 0:
        return None
    bpb = total_nats / (math.log(2) * total_bytes)
    return {"bpb": bpb, "total_bytes": total_bytes}


class BPBTask:
    """
    Bits-per-byte evaluation across all available held-out corpora.

    Corpora checked (in order):
      data/wikitext103/       — WikiText-103 val shard
      data/pile_val/          — The Pile val shard
      data/shakespeare/       — Shakespeare val.bin (binary)
      Extra paths from extra_data_dirs

    Set max_batches to limit evaluation time.
    """

    name = "bpb"
    random_baseline = 0.0  # not applicable; lower BPB is better

    DEFAULT_DIRS = [
        "data/wikitext103",
        "data/pile_val",
    ]
    DEFAULT_BIN_FILES = [
        ("data/shakespeare", "val.bin"),
    ]

    def __init__(
        self,
        max_batches: int = 100,
        extra_data_dirs: Optional[List[str]] = None,
    ):
        self.max_batches    = max_batches
        self.extra_data_dirs = extra_data_dirs or []

    def run(self, model, enc, device: str, block_size: int) -> Dict[str, Any]:
        model.eval()
        byte_lens = _build_byte_lengths(enc).to(device)

        results: Dict[str, Any] = {"task": self.name, "corpora": {}}

        # Parquet dirs
        for data_dir in self.DEFAULT_DIRS + self.extra_data_dirs:
            if not os.path.isdir(data_dir):
                continue
            name = os.path.basename(data_dir)
            r = _eval_bpb_parquet(
                model, data_dir, enc, device, block_size, byte_lens, self.max_batches
            )
            if r:
                results["corpora"][name] = r
                print(f"  {name}: BPB = {r['bpb']:.4f}")

        # Binary files
        for data_dir, fname in self.DEFAULT_BIN_FILES:
            bin_path = os.path.join(data_dir, fname)
            name = f"{os.path.basename(data_dir)}_{fname}"
            r = _eval_bpb_bin(
                model, bin_path, enc, device, block_size, byte_lens, self.max_batches
            )
            if r:
                results["corpora"][name] = r
                print(f"  {name}: BPB = {r['bpb']:.4f}")

        if results["corpora"]:
            # Mean BPB across all available corpora
            bpb_vals = [v["bpb"] for v in results["corpora"].values()]
            results["mean_bpb"] = sum(bpb_vals) / len(bpb_vals)
        else:
            results["mean_bpb"] = None
            print("  [BPB] No corpora available. Run data/*/prepare.py scripts first.")

        return results
