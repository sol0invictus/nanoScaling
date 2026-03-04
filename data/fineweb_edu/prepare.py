"""
Stream HuggingFaceFW/fineweb-edu sample-100BT and save raw text as parquet shards.

Each shard is a parquet file with a single 'text' column, written in row groups
of --rows_per_rg documents (default 1024), compressed with zstd level 3.
Shards are named shard_00000.parquet, shard_00001.parquet, ...

By convention the LAST shard is reserved for validation; all others are train.
Make sure to produce at least 2 shards so there is one for each split.

This script replaces the old tokenization-based prepare.py. No tiktoken, no .bin
files. The resulting parquet shards are ready to use with utils/dataloader.py
which tokenizes on-the-fly during training.

Usage:
    # Stream and package ~250K docs into 10 shards (quick experiment)
    python data/fineweb_edu/prepare.py --max_shards 10

    # Stream the full 100BT sample (will produce ~1800+ shards, may take hours)
    python data/fineweb_edu/prepare.py

    # Limit to a specific number of total documents
    python data/fineweb_edu/prepare.py --max_docs 500000 --output_dir /data/fw

    # Quick test: 3 tiny shards
    python data/fineweb_edu/prepare.py --max_shards 3 --docs_per_shard 500

Output:
    <output_dir>/shard_00000.parquet
    <output_dir>/shard_00001.parquet
    ...
    Last shard = val, rest = train (convention used by utils/dataloader.py)

Alternatively, use data/fineweb_edu/download.py to directly download pre-built
parquet shards from karpathy/fineweb-edu-100b-shuffle (faster if bandwidth allows).
"""

# ---------------------------------------------------------------------------
# Set HF cache before importing datasets
# ---------------------------------------------------------------------------
import os
import sys

def _early_get(flag: str, default: str) -> str:
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default

_temp_dir = _early_get("--temp_dir", "/tmp/fineweb_edu_prep")
os.makedirs(_temp_dir, exist_ok=True)
os.environ.setdefault("HF_HOME",           _temp_dir)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_temp_dir, "datasets"))
os.environ.setdefault("TMPDIR",            _temp_dir)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pyarrow is required. Install with: pip install pyarrow")

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME   = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-100BT"
SCHEMA         = pa.schema([pa.field("text", pa.string())])
COMPRESSION    = "zstd"
COMPRESSION_LEVEL = 3
ROWS_PER_RG_DEFAULT    = 1024
DOCS_PER_SHARD_DEFAULT = 25_000  # ~25K docs ≈ 1–2 GB per shard at avg 250 chars/doc


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def write_shard(texts: list, output_path: str, rows_per_rg: int) -> None:
    """Write a list of document strings to a parquet file.

    Uses an atomic write (write to .tmp, then rename) to avoid partial files.
    Documents are split into row groups of `rows_per_rg` each.
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stream FineWeb-Edu and save raw text to parquet shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max_shards", type=int, default=None,
        help="Maximum number of shards to produce (default: unlimited)",
    )
    parser.add_argument(
        "--max_docs", type=int, default=None,
        help="Stop after this many total documents (default: unlimited)",
    )
    parser.add_argument(
        "--docs_per_shard", type=int, default=DOCS_PER_SHARD_DEFAULT,
        help="Documents per shard",
    )
    parser.add_argument(
        "--rows_per_rg", type=int, default=ROWS_PER_RG_DEFAULT,
        help="Documents per parquet row group",
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

    print(f"Dataset    : {DATASET_NAME}  config={DATASET_CONFIG}")
    print(f"Output dir : {args.output_dir}")
    print(f"Docs/shard : {args.docs_per_shard:,}")
    print(f"Rows/RG    : {args.rows_per_rg:,}")
    if args.max_shards:
        print(f"Max shards : {args.max_shards}")
    if args.max_docs:
        print(f"Max docs   : {args.max_docs:,}")
    print()

    # ------------------------------------------------------------------
    # Stream dataset
    # ------------------------------------------------------------------
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    shard_idx   = 0
    doc_count   = 0
    total_docs  = 0
    shard_buf   = []
    stop        = False

    pbar = tqdm(desc="Documents", unit="doc")

    for example in dataset:
        text = example.get("text", "")
        if not text:
            continue

        shard_buf.append(text)
        doc_count  += 1
        total_docs += 1
        pbar.update(1)

        # Flush shard when it's full
        if len(shard_buf) >= args.docs_per_shard:
            out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")
            write_shard(shard_buf, out_path, args.rows_per_rg)
            shard_size_mb = os.path.getsize(out_path) / 1e6
            tqdm.write(
                f"  Wrote shard_{shard_idx:05d}.parquet  "
                f"({doc_count:,} docs, {shard_size_mb:.1f} MB)"
            )
            shard_buf  = []
            doc_count  = 0
            shard_idx += 1

            # Check limits
            if args.max_shards and shard_idx >= args.max_shards:
                tqdm.write(f"Reached max_shards={args.max_shards}, stopping.")
                stop = True
                break

        if args.max_docs and total_docs >= args.max_docs:
            tqdm.write(f"Reached max_docs={args.max_docs:,}, stopping.")
            stop = True
            break

    pbar.close()

    # Flush any remaining documents as a final (partial) shard
    if shard_buf:
        out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")
        write_shard(shard_buf, out_path, args.rows_per_rg)
        shard_size_mb = os.path.getsize(out_path) / 1e6
        label = "partial, " if stop else ""
        print(f"  Wrote shard_{shard_idx:05d}.parquet  "
              f"({label}{len(shard_buf):,} docs, {shard_size_mb:.1f} MB)")
        shard_idx += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  Total shards : {shard_idx}")
    print(f"  Total docs   : {total_docs:,}")
    if shard_idx >= 2:
        print(f"  Train split  : {shard_idx - 1} shard(s)  "
              f"(shard_00000 … shard_{shard_idx-2:05d})")
        print(f"  Val split    : 1 shard  (shard_{shard_idx-1:05d})")
    elif shard_idx == 1:
        print("  WARNING: Only 1 shard produced — need >= 2 for train+val splits.")
        print("  Re-run with larger --max_shards, or reduce --docs_per_shard.")
    else:
        print("  WARNING: No shards produced. Check dataset access.")
    print("=" * 60)
    print()
    if shard_idx >= 2:
        print("To start training, set dataset=fineweb_edu in your config YAML.")
        print("The dataloader will auto-detect the parquet shards (data_format=auto).")


if __name__ == "__main__":
    main()
