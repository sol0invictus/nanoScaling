"""
HellaSwag — commonsense NLI completion
(Zellers et al., 2019 — https://arxiv.org/abs/1905.07830)

Given an activity + partial sentence, choose which of 4 endings is most
likely.  Evaluated with per-continuation mean CE loss (lower = more likely).
This is the standard approach used in lm-evaluation-harness and nanochat's
CORE eval.

Metric: raw accuracy + centered accuracy (random baseline = 0.25).
Dataset: Rowan/hellaswag  (HuggingFace)
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from evals.scoring import completion_loss_mc, centered_accuracy

RANDOM_BASELINE = 0.25


def _preprocess(text: str) -> str:
    """Remove HellaSwag-specific HTML/wiki markup."""
    text = text.strip()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"</s>", "", text)
    return text.strip()


def _build_context_and_continuations(
    enc,
    activity_label: str,
    ctx_a: str,
    ctx_b: str,
    endings: List[str],
) -> tuple:
    """
    Prompt format:
        Activity: {label}
        {ctx_a} {ctx_b}

    Each continuation is one of the four endings (full sentence).
    Returns:
        context_ids    LongTensor (T_ctx,)
        continuations  list of 4 int-lists
    """
    ctx_text  = f"Activity: {_preprocess(activity_label)}\n"
    ctx_text += _preprocess(ctx_a)
    if ctx_b:
        ctx_text += " " + _preprocess(ctx_b)

    context_ids   = torch.tensor(enc.encode(ctx_text), dtype=torch.long)
    continuations = [enc.encode(" " + _preprocess(e)) for e in endings]
    return context_ids, continuations


class HellaSwagTask:
    name = "hellaswag"
    random_baseline = RANDOM_BASELINE

    def __init__(self, max_samples: Optional[int] = None):
        self.max_samples = max_samples

    def run(self, model, enc, device: str, block_size: int) -> Dict[str, Any]:
        from datasets import load_dataset

        model.eval()
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
        examples = list(ds)
        if self.max_samples:
            rng = random.Random(42)
            examples = rng.sample(examples, min(self.max_samples, len(examples)))

        correct = 0
        for ex in tqdm(examples, desc="HellaSwag"):
            gold = int(ex["label"])

            context_ids, continuations = _build_context_and_continuations(
                enc,
                ex["activity_label"],
                ex["ctx_a"],
                ex["ctx_b"],
                ex["endings"],
            )

            pred, scores = completion_loss_mc(
                model, context_ids, continuations, device, block_size
            )
            if pred == gold:
                correct += 1

        n       = len(examples)
        raw_acc = correct / n if n > 0 else 0.0
        return {
            "task":              self.name,
            "raw_accuracy":      raw_acc,
            "centered_accuracy": centered_accuracy(raw_acc, self.random_baseline),
            "random_baseline":   self.random_baseline,
            "n":                 n,
        }
