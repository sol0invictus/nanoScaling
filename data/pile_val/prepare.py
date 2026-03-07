"""
Prepare The Pile validation set for perplexity evaluation as parquet shards.

The Pile (EleutherAI, 2021) is the most comprehensive open LM benchmark.
Reporting perplexity on The Pile val set is increasingly expected for
publishable pretraining results.

Uses `monology/pile-uncopyrighted` (HuggingFace-hosted, train split only).
Takes the first 10,000 documents as a fixed held-out val set. The original
`EleutherAI/pile` is unavailable (the-eye.eu is offline) and
`monology/pile-uncopyrighted` has no validation split.

Raw text is stored as parquet — tokenization happens on-the-fly in the
dataloader, matching the FineWeb-Edu parquet pipeline exactly.

Output:
  shard_00000.parquet  — all val documents (used as val split)

  Per-domain output (--per-domain):
  domains/<domain_name>/shard_00000.parquet  — one dir per Pile component

Usage:
    python data/pile_val/prepare.py
    python data/pile_val/prepare.py --per-domain

Then set in your eval config:
    dataset: pile_val
    data_format: parquet
    eval_only: true
    init_from: <path/to/ckpt.pt>

Citation:
    Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling"
    https://arxiv.org/abs/2101.00027
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


VAL_DOCS = 10_000  # number of train docs to use as a fixed val set


def prepare(per_domain: bool = False):
    # monology/pile-uncopyrighted only has a train split; take the first VAL_DOCS
    # documents as a held-out val set. Streaming avoids downloading the full corpus.
    print(f"Streaming first {VAL_DOCS:,} docs from monology/pile-uncopyrighted (train)...")
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )

    all_texts = []
    domain_texts: dict = {}

    print("Collecting documents (streaming)...")
    for doc in dataset:
        if len(all_texts) >= VAL_DOCS:
            break
        text = doc["text"]
        if not text.strip():
            continue
        all_texts.append(text)

        if per_domain:
            meta = doc.get("meta", {})
            pile_set = meta.get("pile_set_name", "unknown") if isinstance(meta, dict) else "unknown"
            domain_texts.setdefault(pile_set, []).append(text)

    # Write combined val shard
    out_path = os.path.join(OUT_DIR, "shard_00000.parquet")
    print(f"Writing {len(all_texts):,} documents -> shard_00000.parquet")
    write_shard(all_texts, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  shard_00000.parquet: {size_mb:.1f} MB")

    # Write per-domain shards into separate subdirectories
    if per_domain:
        domains_dir = os.path.join(OUT_DIR, "domains")
        os.makedirs(domains_dir, exist_ok=True)
        for domain, texts in sorted(domain_texts.items()):
            safe_name = domain.lower().replace(" ", "_").replace("/", "_")
            domain_dir = os.path.join(domains_dir, safe_name)
            os.makedirs(domain_dir, exist_ok=True)
            domain_path = os.path.join(domain_dir, "shard_00000.parquet")
            write_shard(texts, domain_path)
            print(f"  domains/{safe_name}/shard_00000.parquet: {len(texts):,} docs")

    print("\nDone. Example eval config:")
    print("  dataset: pile_val")
    print("  data_format: parquet")
    print("  eval_only: true")
    print("  init_from: resume  # or path to ckpt.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", action="store_true",
                        help="Also write per-Pile-component shards to domains/<name>/")
    args = parser.parse_args()
    prepare(per_domain=args.per_domain)
