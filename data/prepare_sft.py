"""
Consolidated SFT Data Preparation Script

Downloads and processes multiple public SFT datasets, applies a unified chat
template, packs conversations into fixed-length blocks (with block-diagonal
doc IDs for masked attention), and writes binary files ready for train_sft.py.

Supported datasets:
  alpaca      — tatsu-lab/alpaca (52k single-turn instruction-following)
  ultrachat   — HuggingFaceH4/ultrachat_200k (207k filtered multi-turn chat)
  oasst1      — OpenAssistant/oasst1 (multi-turn, multilingual; filtered to EN)
  openhermes  — teknium/OpenHermes-2.5 (1M mixed-source, high quality)

Chat format (plain text, GPT-2 tokenizer compatible):
  <|endoftext|>User: {turn1_user}
  Assistant: {turn1_assistant}
  User: {turn2_user}
  Assistant: {turn2_assistant}<|endoftext|>

  Loss mask = 1 for assistant content tokens (and the final EOT).
  Loss mask = 0 for BOS, "User: …\\n", "Assistant: " headers, and padding.

Packing:
  Conversations are shuffled and packed sequentially into blocks of
  `block_size` tokens.  Each block begins with a BOS (EOT) token so that
  cross-block sampling (ix = randint(n_blocks) * block_size) is safe.
  doc_ids track conversation boundaries within each block for block-diagonal
  attention masking at training time.

Output files (written to --output_dir, default data/sft):
  train.bin          uint16  (N_train_blocks × block_size, flattened)
  train_mask.bin     uint8   same shape
  train_doc_ids.bin  uint32  same shape
  val.bin / val_mask.bin / val_doc_ids.bin  — validation split

Usage:
  # Alpaca only (fast, ~52k examples, good for testing)
  python data/prepare_sft.py --datasets alpaca

  # Alpaca + UltraChat (recommended default)
  python data/prepare_sft.py --datasets alpaca ultrachat

  # All datasets
  python data/prepare_sft.py --datasets alpaca ultrachat oasst1

  # Limit samples per dataset (useful for quick experiments)
  python data/prepare_sft.py --datasets alpaca --max_samples 5000

  # Custom block size and output directory
  python data/prepare_sft.py --datasets alpaca --block_size 2048 --output_dir data/sft_2k
"""

import argparse
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import tiktoken
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------

ENC = tiktoken.get_encoding("gpt2")
EOT_ID = ENC.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]  # 50256

# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def _encode(text: str) -> List[int]:
    return ENC.encode(text)

def _encode_user_prefix() -> List[int]:
    return _encode("User: ")

def _encode_assistant_prefix() -> List[int]:
    return _encode("Assistant: ")

def _encode_newline() -> List[int]:
    return _encode("\n")

def conversation_to_tokens(
    messages: List[dict],
) -> Tuple[List[int], List[int]]:
    """
    Convert a list of {"role": ..., "content": ...} messages to token ids
    and a corresponding loss mask (1 = compute loss, 0 = ignore).

    Format:
      BOS User: {msg}\\nAssistant: {msg}[\\nUser: {msg}\\nAssistant: {msg}...]EOT

    Only assistant content tokens (and the final EOT) get mask=1.
    """
    tokens: List[int] = [EOT_ID]  # BOS
    mask:   List[int] = [0]

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "").lower()

        if role == "system":
            # Prepend system content to the first user turn if present
            i += 1
            continue  # skip system messages (fold into first user turn later)

        if role == "user":
            user_toks = _encode_user_prefix() + _encode(msg["content"]) + _encode_newline()
            tokens.extend(user_toks)
            mask.extend([0] * len(user_toks))

            # Expect the next message to be assistant
            if i + 1 < len(messages) and messages[i + 1].get("role", "").lower() == "assistant":
                asst_msg = messages[i + 1]
                prefix_toks = _encode_assistant_prefix()
                content_toks = _encode(asst_msg["content"])
                # Prefix header: no loss
                tokens.extend(prefix_toks)
                mask.extend([0] * len(prefix_toks))
                # Assistant content: compute loss
                tokens.extend(content_toks)
                mask.extend([1] * len(content_toks))
                i += 2
            else:
                # Malformed — user with no assistant response; skip
                i += 1
                continue

            # Add separator between turns (but not after last turn)
            if i < len(messages):
                tokens.extend(_encode_newline())
                mask.extend([0])
        else:
            i += 1

    # Final EOT — train the model to predict end-of-sequence
    tokens.append(EOT_ID)
    mask.append(1)

    return tokens, mask


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_alpaca(max_samples: int) -> List[List[dict]]:
    """tatsu-lab/alpaca — 52k single-turn instruction-following."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    conversations = []
    for ex in tqdm(ds, desc="alpaca"):
        instruction = ex["instruction"].strip()
        inp = ex.get("input", "").strip()
        output = ex["output"].strip()
        if not instruction or not output:
            continue
        if inp:
            user_content = f"{instruction}\n\n{inp}"
        else:
            user_content = instruction
        conversations.append([
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ])
    return conversations


def load_ultrachat(max_samples: int) -> List[List[dict]]:
    """HuggingFaceH4/ultrachat_200k — 207k filtered multi-turn chat."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    conversations = []
    for ex in tqdm(ds, desc="ultrachat"):
        msgs = ex.get("messages", [])
        if len(msgs) < 2:
            continue
        conversations.append(msgs)
    return conversations


def load_oasst1(max_samples: int) -> List[List[dict]]:
    """OpenAssistant/oasst1 — multi-turn, filtered to English."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    # Build message trees to extract conversation threads
    by_id = {ex["message_id"]: ex for ex in ds}
    roots = [ex for ex in ds if ex["parent_id"] is None and ex["lang"] == "en"]

    def build_thread(node) -> List[List[dict]]:
        """Recursively collect all root→leaf paths."""
        children = [
            ex for ex in ds
            if ex.get("parent_id") == node["message_id"] and ex["lang"] == "en"
        ]
        if not children:
            return [[{"role": node["role"], "content": node["text"]}]]
        threads = []
        for child in children:
            for sub in build_thread(child):
                threads.append([{"role": node["role"], "content": node["text"]}] + sub)
        return threads

    conversations = []
    for root in tqdm(roots, desc="oasst1"):
        for thread in build_thread(root):
            # Only keep threads that start with prompter→assistant
            if (len(thread) >= 2
                    and thread[0]["role"] == "prompter"
                    and thread[1]["role"] == "assistant"):
                # Rename roles to match our format
                cleaned = [
                    {"role": "user" if m["role"] == "prompter" else "assistant",
                     "content": m["content"]}
                    for m in thread
                ]
                conversations.append(cleaned)
                if max_samples and len(conversations) >= max_samples:
                    return conversations
    return conversations


def load_openhermes(max_samples: int) -> List[List[dict]]:
    """teknium/OpenHermes-2.5 — 1M mixed-source, high quality."""
    from datasets import load_dataset
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    conversations = []
    for ex in tqdm(ds, desc="openhermes"):
        convs = ex.get("conversations", [])
        if len(convs) < 2:
            continue
        msgs = []
        for turn in convs:
            role = turn.get("from", "").lower()
            value = turn.get("value", "").strip()
            if not value:
                continue
            if role in ("human", "user"):
                msgs.append({"role": "user", "content": value})
            elif role in ("gpt", "assistant", "chatgpt"):
                msgs.append({"role": "assistant", "content": value})
        if len(msgs) >= 2:
            conversations.append(msgs)
    return conversations


DATASET_LOADERS = {
    "alpaca":      load_alpaca,
    "ultrachat":   load_ultrachat,
    "oasst1":      load_oasst1,
    "openhermes":  load_openhermes,
}


# ---------------------------------------------------------------------------
# Block packing
# ---------------------------------------------------------------------------

def pack_into_blocks(
    conversations: List[Tuple[List[int], List[int]]],
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack (tokens, mask) pairs into fixed-length blocks using greedy
    sequential packing.

    Each block begins with a BOS (EOT_ID) token.  Conversations are appended
    until the block is full; when a conversation doesn't fit the current block,
    it begins the next one.  Conversations longer than block_size are cropped.

    doc_ids are assigned per conversation within a block to enable
    block-diagonal causal attention masking at training time.

    Returns:
        tokens   (N_blocks, block_size) uint16
        masks    (N_blocks, block_size) uint8
        doc_ids  (N_blocks, block_size) uint32
    """
    all_block_tokens:   List[List[int]] = []
    all_block_masks:    List[List[int]] = []
    all_block_doc_ids:  List[List[int]] = []

    cur_t: List[int] = [EOT_ID]
    cur_m: List[int] = [0]
    cur_d: List[int] = [0]   # doc_id 0 = BOS / padding
    doc_counter: int = 0

    def flush():
        """Pad current block to block_size and append to output lists."""
        pad = block_size - len(cur_t)
        all_block_tokens.append(cur_t + [EOT_ID] * pad)
        all_block_masks.append(cur_m + [0] * pad)
        all_block_doc_ids.append(cur_d + [0] * pad)

    for conv_tokens, conv_mask in conversations:
        # Crop conversations that are longer than a full block (keep first block_size-1 tokens)
        if len(conv_tokens) >= block_size:
            conv_tokens = conv_tokens[: block_size - 1]
            conv_mask   = conv_mask[: block_size - 1]

        if len(cur_t) + len(conv_tokens) <= block_size:
            # Fits in current block
            doc_counter += 1
            cur_t.extend(conv_tokens)
            cur_m.extend(conv_mask)
            cur_d.extend([doc_counter] * len(conv_tokens))
        else:
            # Flush current block and start a new one
            flush()
            doc_counter += 1
            cur_t = [EOT_ID] + list(conv_tokens)
            cur_m = [0]      + list(conv_mask)
            cur_d = [0]      + [doc_counter] * len(conv_tokens)

    # Flush final block if it has any content beyond just BOS
    if len(cur_t) > 1:
        flush()

    tokens  = np.array(all_block_tokens,  dtype=np.uint16).reshape(-1)
    masks   = np.array(all_block_masks,   dtype=np.uint8).reshape(-1)
    doc_ids = np.array(all_block_doc_ids, dtype=np.uint32).reshape(-1)
    return tokens, masks, doc_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT binary datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_LOADERS.keys()),
        default=["alpaca"],
        help="Which datasets to include (default: alpaca)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Max samples per dataset (0 = all)",
    )
    parser.add_argument(
        "--block_size", type=int, default=1024,
        help="Sequence length for packed blocks (default: 1024)",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.05,
        help="Fraction of data to use for validation (default: 0.05)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/sft",
        help="Output directory for .bin files (default: data/sft)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load and tokenize conversations
    # ------------------------------------------------------------------
    all_conversations: List[Tuple[List[int], List[int]]] = []

    for name in args.datasets:
        loader = DATASET_LOADERS[name]
        raw_convs = loader(args.max_samples)
        print(f"  {name}: {len(raw_convs):,} conversations loaded")

        for msgs in tqdm(raw_convs, desc=f"tokenizing {name}"):
            toks, mask = conversation_to_tokens(msgs)
            # Skip conversations with zero assistant tokens (no learning signal)
            if sum(mask) == 0:
                continue
            all_conversations.append((toks, mask))

    print(f"\nTotal conversations after filtering: {len(all_conversations):,}")

    # ------------------------------------------------------------------
    # 2. Shuffle and split
    # ------------------------------------------------------------------
    random.shuffle(all_conversations)
    n_val = max(1, int(len(all_conversations) * args.val_fraction))
    val_convs   = all_conversations[:n_val]
    train_convs = all_conversations[n_val:]
    print(f"Train: {len(train_convs):,}  Val: {len(val_convs):,}")

    # ------------------------------------------------------------------
    # 3. Pack into blocks
    # ------------------------------------------------------------------
    print("\nPacking training conversations into blocks…")
    train_tokens, train_masks, train_doc_ids = pack_into_blocks(train_convs, args.block_size)
    n_train_blocks = len(train_tokens) // args.block_size
    print(f"  {n_train_blocks:,} blocks × {args.block_size} tokens = {len(train_tokens):,} tokens")

    print("Packing validation conversations into blocks…")
    val_tokens, val_masks, val_doc_ids = pack_into_blocks(val_convs, args.block_size)
    n_val_blocks = len(val_tokens) // args.block_size
    print(f"  {n_val_blocks:,} blocks × {args.block_size} tokens = {len(val_tokens):,} tokens")

    # Compute mask density (fraction of tokens with loss)
    train_density = train_masks.mean() * 100
    val_density   = val_masks.mean()   * 100
    print(f"\nLoss-mask density — train: {train_density:.1f}%  val: {val_density:.1f}%")

    # ------------------------------------------------------------------
    # 4. Write binary files
    # ------------------------------------------------------------------
    def write_split(prefix: str, tokens, masks, doc_ids):
        tokens.tofile(os.path.join(args.output_dir, f"{prefix}.bin"))
        masks.tofile(os.path.join(args.output_dir,  f"{prefix}_mask.bin"))
        doc_ids.tofile(os.path.join(args.output_dir, f"{prefix}_doc_ids.bin"))
        print(f"  wrote {prefix}.bin  ({len(tokens):,} tokens)")

    write_split("train", train_tokens, train_masks, train_doc_ids)
    write_split("val",   val_tokens,   val_masks,   val_doc_ids)

    print(f"\nDone. Files written to: {args.output_dir}")
    print("Next step:")
    print(f"  python train_sft.py configs/sft.yaml --dataset=sft")


if __name__ == "__main__":
    main()
