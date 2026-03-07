"""
Convert Shakespeare input.txt to parquet shards for the on-the-fly tokenizing dataloader.

Output: shard_00000.parquet (train, ~90%), shard_00001.parquet (val, ~10%)
Each shard has a single 'text' column of raw document strings.
Documents are natural speech/scene blocks split on double newlines.

Convention matches utils/dataloader.py:
  - Files named shard_?????.parquet in the dataset directory
  - Last shard = val split, all others = train split
  - Raw text only; tokenization is done on-the-fly by the dataloader
"""

import os
import re

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_FILE = os.path.join(os.path.dirname(__file__), 'input.txt')
OUT_DIR = os.path.dirname(__file__)
ROW_GROUP_SIZE = 128  # docs per row group


def load_and_split(path: str) -> list:
    """Split Shakespeare text into documents at double-newline boundaries.
    Filters out empty/whitespace-only chunks."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    docs = re.split(r'\n{2,}', text)
    return [d.strip() for d in docs if d.strip()]


def write_shard(docs: list, path: str):
    table = pa.table({'text': docs})
    pq.write_to_dataset(
        table,
        root_path=path,
        row_group_size=ROW_GROUP_SIZE,
        existing_data_behavior='overwrite_or_ignore',
    )


def write_shard_file(docs: list, path: str):
    """Write a single parquet file with row_group_size rows per group."""
    writer = None
    schema = pa.schema([('text', pa.string())])
    try:
        with pq.ParquetWriter(path, schema, compression='snappy') as writer:
            for i in range(0, len(docs), ROW_GROUP_SIZE):
                batch = pa.record_batch(
                    [pa.array(docs[i:i + ROW_GROUP_SIZE], type=pa.string())],
                    schema=schema,
                )
                writer.write_batch(batch)
    except Exception as e:
        raise RuntimeError(f"Failed to write {path}: {e}")


if __name__ == '__main__':
    docs = load_and_split(INPUT_FILE)
    n = len(docs)
    split_idx = int(n * 0.9)
    train_docs = docs[:split_idx]
    val_docs = docs[split_idx:]

    train_path = os.path.join(OUT_DIR, 'shard_00000.parquet')
    val_path = os.path.join(OUT_DIR, 'shard_00001.parquet')

    write_shard_file(train_docs, train_path)
    write_shard_file(val_docs, val_path)

    print(f"Total documents: {n}")
    print(f"Train docs: {len(train_docs)} -> {train_path}")
    print(f"Val docs:   {len(val_docs)} -> {val_path}")
    print("Done. Set dataset: shakespeare in your config to use parquet auto-detection.")
