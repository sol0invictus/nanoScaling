"""
Prepare a math validation set for perplexity evaluation as a parquet shard.

Uses `openai/gsm8k` (Grade School Math), a widely cited math reasoning benchmark
with chain-of-thought solutions. The test split has ~1.3k problems; each document
is the question concatenated with its step-by-step answer.

Output:
  shard_00000.parquet  — all test problems (used as val split)

Usage:
    python data/math_val/prepare.py

Then reference in a config:
    val_datasets:
      - data/math_val

Citation:
    Cobbe et al., "Training Verifiers to Solve Math Word Problems"
    https://arxiv.org/abs/2110.14168
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
    tmp_path = output_path + ".tmp"
    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression=COMPRESSION,
                              compression_level=COMPRESSION_LEVEL)
    for i in range(0, len(texts), ROWS_PER_RG):
        chunk = texts[i : i + ROWS_PER_RG]
        writer.write_table(pa.table({"text": chunk}, schema=SCHEMA))
    writer.close()
    os.rename(tmp_path, output_path)


def prepare():
    print("Downloading openai/gsm8k test split...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    texts = []
    for doc in dataset:
        question = doc.get("question", "").strip()
        answer   = doc.get("answer", "").strip()
        if not question:
            continue
        # Concatenate as a natural Q&A block
        texts.append(f"Question: {question}\n\nAnswer: {answer}")

    out_path = os.path.join(OUT_DIR, "shard_00000.parquet")
    print(f"Writing {len(texts):,} problems -> shard_00000.parquet")
    write_shard(texts, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  shard_00000.parquet: {size_mb:.1f} MB")

    print("\nDone. Example eval config:")
    print("  val_datasets:")
    print("    - data/math_val")


if __name__ == "__main__":
    prepare()
