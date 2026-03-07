"""
Download OpenWebText from HuggingFace and save raw text as parquet shards.

Each shard is a parquet file with a single 'text' column, written in row groups
of --rows_per_rg documents (default 1024), compressed with zstd level 3.
Shards are named shard_00000.parquet, shard_00001.parquet, ...

By convention the LAST shard is reserved for validation; all others are train.
The val split is drawn from the dataset using the same seed as the original
prepare.py (test_size=0.0005, seed=2357, ~4007 docs).

Shard writing is parallelised across --num_workers processes so that disk I/O
and zstd compression are fully pipelined with dataset iteration.

This script replaces the old tokenization-based prepare.py. No tiktoken, no .bin
files. The resulting parquet shards are ready to use with utils/dataloader.py
which tokenizes on-the-fly during training.

Usage:
    # Full dataset (~8M docs, ~320 train shards + 1 val shard)
    python data/openwebtext/prepare.py

    # Smaller run for quick experiments
    python data/openwebtext/prepare.py --max_shards 10

    # Custom parallelism
    python data/openwebtext/prepare.py --num_workers 16

Output:
    <output_dir>/shard_00000.parquet  ← train
    ...
    <output_dir>/shard_NNNNN.parquet  ← val (last shard)
"""

import os
import sys

def _early_get(flag: str, default: str) -> str:
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default

_temp_dir = _early_get("--temp_dir", "/tmp/openwebtext_prep")
os.makedirs(_temp_dir, exist_ok=True)
os.environ.setdefault("HF_HOME",           _temp_dir)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_temp_dir, "datasets"))
os.environ.setdefault("TMPDIR",            _temp_dir)

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pyarrow is required. Install with: pip install pyarrow")

from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME       = "Skylion007/openwebtext"
VAL_FRACTION       = 0.0005
VAL_SEED           = 2357
SCHEMA             = pa.schema([pa.field("text", pa.string())])
COMPRESSION        = "zstd"
COMPRESSION_LEVEL  = 3
ROWS_PER_RG_DEFAULT    = 1024
DOCS_PER_SHARD_DEFAULT = 25_000


# Top-level so it's picklable by multiprocessing
def _write_shard_worker(texts: list, output_path: str, rows_per_rg: int):
    """Write one shard in a worker process. Returns (output_path, n_docs, size_mb)."""
    tmp_path = output_path + ".tmp"
    writer = pq.ParquetWriter(
        tmp_path,
        SCHEMA,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
    )
    for i in range(0, len(texts), rows_per_rg):
        chunk = texts[i : i + rows_per_rg]
        table = pa.table({"text": chunk}, schema=SCHEMA)
        writer.write_table(table)
    writer.close()
    os.rename(tmp_path, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    return output_path, len(texts), size_mb


def _drain_completed(pending: set, min_drain: int = 1) -> set:
    """Wait for at least min_drain futures to finish, print results, return remaining."""
    done, pending = wait(pending, return_when=FIRST_COMPLETED)
    for fut in done:
        out_path, n_docs, size_mb = fut.result()
        shard_name = os.path.basename(out_path)
        tqdm.write(f"  Wrote {shard_name}  ({n_docs:,} docs, {size_mb:.1f} MB)")
    return pending


def main():
    parser = argparse.ArgumentParser(
        description="Save OpenWebText raw text to parquet shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max_shards", type=int, default=None,
        help="Maximum number of train shards to produce (default: unlimited). "
             "The val shard is always written as the final shard.",
    )
    parser.add_argument(
        "--docs_per_shard", type=int, default=DOCS_PER_SHARD_DEFAULT,
        help="Documents per train shard",
    )
    parser.add_argument(
        "--rows_per_rg", type=int, default=ROWS_PER_RG_DEFAULT,
        help="Documents per parquet row group",
    )
    parser.add_argument(
        "--num_workers", type=int, default=os.cpu_count(),
        help="Worker processes for parallel shard writing",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write .parquet files",
    )
    parser.add_argument(
        "--temp_dir", type=str, default=_temp_dir,
        help="HuggingFace cache / tmp directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.temp_dir != _temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)
        os.environ["HF_HOME"]           = args.temp_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(args.temp_dir, "datasets")
        os.environ["TMPDIR"]            = args.temp_dir

    # Max in-flight futures: keep at most 2× workers worth of shard buffers in memory
    max_pending = max(args.num_workers * 2, 4)

    print(f"Dataset    : {DATASET_NAME}")
    print(f"Output dir : {args.output_dir}")
    print(f"Docs/shard : {args.docs_per_shard:,}")
    print(f"Rows/RG    : {args.rows_per_rg:,}")
    print(f"Workers    : {args.num_workers}")
    if args.max_shards:
        print(f"Max shards : {args.max_shards} train shards")
    print()

    # ------------------------------------------------------------------
    # Load and split dataset
    # ------------------------------------------------------------------
    print("Loading dataset (this may take a while on first run)...")
    dataset = load_dataset(DATASET_NAME, num_proc=args.num_workers)

    split_dataset = dataset["train"].train_test_split(
        test_size=VAL_FRACTION, seed=VAL_SEED, shuffle=True
    )
    train_split = split_dataset["train"]
    val_split   = split_dataset["test"]

    print(f"Train docs : {len(train_split):,}")
    print(f"Val docs   : {len(val_split):,}")
    print()

    # ------------------------------------------------------------------
    # Write train shards (parallel)
    # ------------------------------------------------------------------
    shard_idx  = 0
    total_docs = 0
    shard_buf  = []
    pending    = set()
    stop       = False

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:

        for example in tqdm(train_split, desc="Iterating train", unit="doc"):
            text = example.get("text", "")
            if not text:
                continue

            shard_buf.append(text)
            total_docs += 1

            if len(shard_buf) >= args.docs_per_shard:
                out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")
                pending.add(executor.submit(_write_shard_worker, shard_buf, out_path, args.rows_per_rg))
                shard_buf  = []
                shard_idx += 1

                # Backpressure: drain one completed future if queue is full
                if len(pending) >= max_pending:
                    pending = _drain_completed(pending)

                if args.max_shards and shard_idx >= args.max_shards:
                    tqdm.write(f"Reached max_shards={args.max_shards}, stopping train shards.")
                    stop = True
                    break

        # Flush remaining partial train shard
        if shard_buf:
            out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")
            pending.add(executor.submit(_write_shard_worker, shard_buf, out_path, args.rows_per_rg))
            shard_idx += 1

        # Write val shard (still inside executor so it runs in parallel with last train shards)
        print(f"\nQueuing val shard ({len(val_split):,} docs)...")
        val_texts = [ex["text"] for ex in tqdm(val_split, desc="Val documents", unit="doc") if ex.get("text")]
        val_path  = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")
        pending.add(executor.submit(_write_shard_worker, val_texts, val_path, args.rows_per_rg))
        shard_idx += 1

        # Drain all remaining futures
        print("Waiting for workers to finish...")
        while pending:
            pending = _drain_completed(pending)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_train_shards = shard_idx - 1
    print()
    print("=" * 60)
    print(f"  Total shards : {shard_idx}")
    print(f"  Total docs   : {total_docs + len(val_texts):,}")
    print(f"  Train split  : {n_train_shards} shard(s)  "
          f"(shard_00000 … shard_{n_train_shards-1:05d})")
    print(f"  Val split    : 1 shard  (shard_{shard_idx-1:05d})")
    print("=" * 60)
    print()
    print("To start training, set dataset=openwebtext in your config YAML.")
    print("The dataloader will auto-detect the parquet shards (data_format=auto).")


if __name__ == "__main__":
    main()
