# Parquet Dataloader (`utils/dataloader.py`)

On-the-fly tokenizing dataloader for parquet-sharded datasets. Designed as a drop-in replacement for the pre-tokenized `.bin` memmap path, trading storage efficiency for flexibility — raw text is stored once and tokenized at training time.

---

## Overview

The dataloader reads raw text documents from parquet shards, tokenizes them with the GPT-2 BPE tokenizer (tiktoken), and packs them into fixed-length training batches with zero padding and 100% token utilization. It is an infinite generator — it never raises `StopIteration`, cycling through the dataset indefinitely.

---

## Data Format

Shards must follow a strict naming convention:

```
data/<dataset>/
    shard_00000.parquet   ← train
    shard_00001.parquet   ← train
    ...
    shard_NNNNN.parquet   ← val  (always the last shard)
```

Each parquet file must have a single `text` column containing raw document strings. The last shard is always the validation split; all others are training. A minimum of 2 shards is required.

Row groups within each shard are the unit of I/O — the DDP sharding operates at the row-group level (see below).

---

## Pipeline

Each `next()` call on the generator executes this pipeline:

```
Parquet shards
    │
    ▼ _iter_row_groups()          read one row group at a time (lazy)
Row group (list of raw text strings)
    │
    ▼ _tokenize_texts()           tiktoken GPT-2 BPE, prepend EOT as BOS
List of token-id lists (doc_buffer)
    │
    ▼ _pack_tokens()              best-fit bin packing
B rows of T+1 tokens each
    │
    ▼ numpy → torch, pin_memory
(x, y): LongTensor [B, T]
```

### 1. File Discovery

`_list_parquet_files(data_dir)` globs for `shard_?????.parquet` and returns them sorted. `_get_split_files(data_dir, split)` slices this list: `train` gets `[:-1]`, `val` gets `[-1:]`.

### 2. Row-Group Iterator

`_iter_row_groups(files, ddp_rank, ddp_world_size)` is an infinite generator. It assigns row groups to DDP ranks by global index:

```
rank r reads row groups at global indices:  r,  r+W,  r+2W, ...
```

where `W` is the world size. This strided assignment means all ranks see different data without pre-splitting the files. At the end of the dataset, the iterator wraps back to the start (infinite epochs).

### 3. Tokenization

`_tokenize_texts(texts, enc)` tokenizes each document with `tiktoken.encode_ordinary` (no special tokens) and prepends the GPT-2 EOT token (50256) as a BOS marker:

```python
[EOT] + enc.encode_ordinary(text)
```

This ensures every document boundary in a packed row is marked by an EOT token, which the model can learn to associate with context resets.

### 4. Document Buffer

A list of tokenized documents (`doc_buffer`) is maintained in memory. It is pre-filled to `buffer_size` documents at startup, then refilled whenever it drops below `buffer_size // 2`. The default `buffer_size=1000` keeps ~1000 documents available for the packing algorithm to choose from.

### 5. BOS-Aligned Best-Fit Packing

This is the core of the dataloader. `_pack_tokens(doc_buffer, B, T)` fills `B` rows of exactly `T+1` tokens each using a greedy best-fit algorithm:

**For each position in a row:**
1. Find the **largest document in the buffer that fits entirely** in the remaining space.
2. Place it, advance the position, repeat.
3. When no document fits entirely, take the **shortest document** in the buffer and **crop it** to fill the remaining space exactly. The unread tail is put back into the buffer.

**Properties of this scheme:**
- Every row starts with a BOS (EOT) token — document boundaries are always aligned to the start of a row.
- **100% token utilization** — no padding tokens ever appear.
- **~35% of tokens are discarded** by cropping. This is the cost of the zero-padding guarantee.
- The `doc_buffer` is mutated in place: consumed documents are popped, cropped remainders are re-inserted.

**Output shape:** `_pack_tokens` returns a list of `B` lists, each of length `T+1`. The caller slices `[:T]` for inputs and `[1:T+1]` for targets (next-token prediction).

### 6. Tensor Construction

The `B × (T+1)` list is converted to a `numpy` int64 array, then split into `x` (inputs) and `y` (targets). For CUDA devices, tensors are pinned and transferred with `non_blocking=True` to overlap data transfer with compute.

---

## Public API

```python
from utils.dataloader import create_parquet_dataloader

loader = create_parquet_dataloader(
    data_dir   = "data/fineweb_edu",
    split      = "train",          # 'train' or 'val'
    B          = 12,               # batch size
    T          = 1024,             # sequence length (block_size)
    device     = "cuda:0",
    ddp_rank   = 0,                # 0 for single-GPU
    ddp_world_size = 1,            # 1 for single-GPU
    buffer_size = 1000,            # docs to keep in the packing buffer
)

x, y = next(loader)  # LongTensor [B, T] each
```

The loader is normally not called directly. `utils/data.py::create_dataloader()` auto-detects parquet shards and routes to this loader transparently.

---

## DDP Behaviour

With `ddp_world_size=W` ranks:
- All ranks open the same parquet files.
- Rank `r` reads row groups at global indices `r, r+W, r+2W, ...`
- No file pre-splitting or coordination between ranks is needed.
- Data diversity between ranks is guaranteed at the row-group level. Within a row group, all documents go to a single rank's buffer.

---

## Constants

| Name | Value | Description |
|---|---|---|
| `EOT_TOKEN` | 50256 | GPT-2 `<\|endoftext\|>`, used as BOS and EOS |
| `VOCAB_SIZE` | 50304 | GPT-2 vocab padded to a multiple of 64 |

---

## Trade-offs vs. Pre-tokenized `.bin`

| | Parquet (this loader) | `.bin` memmap |
|---|---|---|
| Storage | Raw text, compact | Pre-tokenized, ~2× larger |
| Startup | Tokenizes on-the-fly | Instant (memmap) |
| Flexibility | Any tokenizer, any split logic | Fixed tokenizer at prep time |
| Token utilization | 100% (no padding) | Random-position sampling |
| Document boundaries | BOS-aligned, preserved | Ignored (random crops) |
| DDP | Strided row-group assignment | Per-rank RNG seeds |

---

## Known Limitations

- Only `'train'` and `'val'` splits are supported. Custom split names fall back to `'val'` with a warning (parquet split convention is positional, not named).
- `tokenizer_batch_size` is accepted as a parameter but not used — tiktoken is fast enough to tokenize inline. It is reserved for a future threaded tokenization path.
- The `~35% token discard rate` from cropping is an empirical estimate that depends on document length distribution. Shorter average documents → more cropping.
