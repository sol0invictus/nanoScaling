"""
Prepare WikiText-103 for perplexity evaluation as parquet shards.

WikiText-103 is the standard benchmark for language model perplexity comparison.
Uses the raw (untokenized) version — tokenization happens on-the-fly in the dataloader,
matching the FineWeb-Edu parquet pipeline exactly.

Output (shard_?????.parquet convention — last shard = val):
  shard_00000.parquet  — train articles (~28K docs)
  shard_00001.parquet  — val articles  (~218 docs)  ← val split

Usage:
    python data/wikitext103/prepare.py

Then set in your eval config:
    dataset: wikitext103
    eval_only: true
    init_from: <path/to/ckpt.pt>

Citation:
    Merity et al., "Pointer Sentinel Mixture Models", ICLR 2017.
    https://arxiv.org/abs/1609.07843
"""

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


def write_shard(texts: list, output_path: str) -> None:
    """Write a list of raw document strings to a parquet file (atomic write)."""
    tmp_path = output_path + ".tmp"
    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression=COMPRESSION,
                              compression_level=COMPRESSION_LEVEL)
    for i in range(0, len(texts), ROWS_PER_RG):
        chunk = texts[i : i + ROWS_PER_RG]
        writer.write_table(pa.table({"text": chunk}, schema=SCHEMA))
    writer.close()
    os.rename(tmp_path, output_path)


def prepare():
    print("Downloading WikiText-103-raw-v1 from HuggingFace...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", trust_remote_code=True)

    # shard_00000 = train, shard_00001 = val (last shard = val convention)
    splits = [
        ("train",      "train",      "shard_00000.parquet"),
        ("val",        "validation", "shard_00001.parquet"),
    ]

    for split_name, hf_split, filename in splits:
        texts = dataset[hf_split]["text"]
        articles = [t for t in texts if t.strip()]
        out_path = os.path.join(OUT_DIR, filename)
        print(f"Writing {split_name}: {len(articles):,} articles -> {filename}")
        write_shard(articles, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {filename}: {size_mb:.1f} MB")

    print("\nDone. Example eval config:")
    print("  dataset: wikitext103")
    print("  data_format: parquet")
    print("  eval_only: true")
    print("  init_from: resume  # or path to ckpt.pt")


if __name__ == "__main__":
    prepare()
