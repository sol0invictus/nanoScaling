# FineWeb-Edu 100BT

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is a high-quality educational web corpus curated by HuggingFace.

The `sample-100BT` configuration contains ~100 billion GPT-2 BPE tokens.

## Prepare

```bash
# Full 100B token run (~200 GB, takes many hours)
python data/fineweb_edu/prepare.py

# Smaller run for experiments (e.g. 10B tokens, ~20 GB)
python data/fineweb_edu/prepare.py --max_train_tokens 10_000_000_000

# Custom val split size
python data/fineweb_edu/prepare.py --num_val_docs 10000
```

## Output

| File        | Size (full) | Tokens        |
|-------------|-------------|---------------|
| `train.bin` | ~200 GB     | ~100B         |
| `val.bin`   | ~30 MB      | ~10-15M       |

Files are flat `uint16` arrays of GPT-2 BPE token IDs, identical in format to `data/openwebtext/`.

## Training

```yaml
# configs/train_full.yaml
dataset: fineweb_edu
```

or

```bash
python train.py --dataset=fineweb_edu
```
