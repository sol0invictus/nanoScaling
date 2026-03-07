"""
Prepare a code validation set for perplexity evaluation as a parquet shard.

Uses `codeparrot/codeparrot-clean-valid`, a curated Python code dataset with
an explicit `validation` split. It is the standard code perplexity benchmark
used in the CodeParrot and similar pretraining papers.

The `content` field (raw source file text) is used as the document text.
Tokenization happens on-the-fly in the dataloader.

Output:
  shard_00000.parquet  — first VAL_DOCS Python files (used as val split)

Usage:
    python data/code_val/prepare.py

Then reference in a config:
    val_datasets:
      - data/code_val

Citation:
    Tunstall et al., "The CodeParrot Dataset", 2022.
    https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid
"""

import argparse
import os

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pyarrow is required. Install with: pip install pyarrow")

from datasets import load_dataset

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA = pa.schema([pa.field("text", pa.string())])
ROWS_PER_RG = 1024
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3
VAL_DOCS = 5_000  # number of files to keep


def write_shard(texts: list, output_path: str) -> None:
    tmp_path = output_path + ".tmp"
    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression=COMPRESSION,
                              compression_level=COMPRESSION_LEVEL)
    for i in range(0, len(texts), ROWS_PER_RG):
        chunk = texts[i : i + ROWS_PER_RG]
        writer.write_table(pa.table({"text": chunk}, schema=SCHEMA))
    writer.close()
    os.rename(tmp_path, output_path)


def prepare():
    # Only a train split exists; take the first VAL_DOCS files as a fixed val set.
    print(f"Streaming first {VAL_DOCS:,} files from codeparrot/codeparrot-clean-valid (train)...")
    dataset = load_dataset(
        "codeparrot/codeparrot-clean-valid",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )

    texts = []
    print("Collecting documents (streaming)...")
    for doc in dataset:
        if len(texts) >= VAL_DOCS:
            break
        content = doc.get("content", "")
        if not content.strip():
            continue
        texts.append(content)

    out_path = os.path.join(OUT_DIR, "shard_00000.parquet")
    print(f"Writing {len(texts):,} files -> shard_00000.parquet")
    write_shard(texts, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  shard_00000.parquet: {size_mb:.1f} MB")

    print("\nDone. Example eval config:")
    print("  val_datasets:")
    print("    - data/code_val")


if __name__ == "__main__":
    prepare()
