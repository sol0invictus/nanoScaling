"""
nanoScaling evaluation suite.

Implements the same benchmarks used in nanochat:
  - MMLU         (57-subject multiple choice, logit-based)
  - ARC-Easy     (science MC, logit-based)
  - ARC-Challenge (science MC, logit-based)
  - HellaSwag    (commonsense completion, per-continuation loss)
  - GSM8K        (grade-school math, greedy generation + regex)
  - BPB          (bits-per-byte on held-out text)

Usage:
  python eval_sft.py out-sft/ckpt_sft.pt
  python eval_sft.py out-sft/ckpt_sft.pt --tasks mmlu arc gsm8k
  python eval_sft.py out-sft/ckpt_sft.pt --tasks all --max_samples 200
"""
