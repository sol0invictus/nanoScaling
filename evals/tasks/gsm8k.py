"""
GSM8K — Grade School Math
(Cobbe et al., 2021 — https://arxiv.org/abs/2110.14168)

~1.3k grade-school word problems.  Evaluated by greedy generation followed
by regex extraction of the final numeric answer.

Answer extraction:
  1. Look for '#### {number}' (GSM8K gold format).
  2. Fall back to the last integer / decimal in the generated response.

Metric: exact-match accuracy on the numeric answer.
        Centered accuracy uses random baseline = 0.0 (open-ended).
Dataset: openai/gsm8k  (HuggingFace)
"""

from __future__ import annotations

import re
import random
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from evals.scoring import generate_greedy, centered_accuracy

RANDOM_BASELINE  = 0.0    # open-ended, no random chance
EOT_ID           = 50256  # GPT-2 <|endoftext|>
MAX_NEW_TOKENS   = 300


def _extract_answer(text: str) -> Optional[str]:
    """
    Extract the numeric answer from a model response.

    Priority:
      1. '#### {number}' (GSM8K chain-of-thought format)
      2. Last standalone number in the text
    """
    # 1. Gold-format marker
    m = re.search(r"####\s*([-]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    # 2. Last number in the text
    nums = re.findall(r"[-]?\d[\d,]*\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "").strip()

    return None


def _normalise_number(s: str) -> Optional[str]:
    """Normalise numeric string: strip trailing zeros, unify decimals."""
    try:
        f = float(s.replace(",", ""))
        # Return integer string if whole number, else up to 4 decimal places
        if f == int(f):
            return str(int(f))
        return f"{f:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return None


def _build_prompt(enc, question: str) -> torch.Tensor:
    """
    Chat-format prompt. We use a 'step by step' instruction to encourage
    chain-of-thought reasoning, matching common GSM8K evaluation practice.
    """
    text = (
        f"User: {question}\n"
        f"Assistant: Let me solve this step by step.\n"
    )
    return torch.tensor(enc.encode(text), dtype=torch.long)


class GSM8KTask:
    name = "gsm8k"
    random_baseline = RANDOM_BASELINE

    def __init__(
        self,
        max_samples: Optional[int] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        self.max_samples    = max_samples
        self.max_new_tokens = max_new_tokens

    def run(self, model, enc, device: str, block_size: int) -> Dict[str, Any]:
        from datasets import load_dataset

        model.eval()
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        examples = list(ds)
        if self.max_samples:
            rng = random.Random(42)
            examples = rng.sample(examples, min(self.max_samples, len(examples)))

        correct = 0
        failures: List[Dict] = []   # store first few wrong for diagnostics

        for ex in tqdm(examples, desc="GSM8K"):
            gold_raw = ex["answer"]
            gold_ans = _extract_answer(gold_raw)
            gold_norm = _normalise_number(gold_ans) if gold_ans else None

            prompt_ids = _build_prompt(enc, ex["question"])

            # Truncate prompt if needed (leave room for generation)
            max_prompt = block_size - self.max_new_tokens
            if prompt_ids.shape[0] > max_prompt:
                prompt_ids = prompt_ids[-max_prompt:]

            new_tokens  = generate_greedy(model, prompt_ids, self.max_new_tokens, device, EOT_ID)
            response    = enc.decode(new_tokens)

            pred_ans  = _extract_answer(response)
            pred_norm = _normalise_number(pred_ans) if pred_ans else None

            is_correct = (pred_norm is not None and gold_norm is not None
                          and pred_norm == gold_norm)
            if is_correct:
                correct += 1
            elif len(failures) < 3:
                failures.append({
                    "question": ex["question"][:80],
                    "gold":     gold_ans,
                    "pred":     pred_ans,
                    "response": response[:200],
                })

        n       = len(examples)
        raw_acc = correct / n if n > 0 else 0.0
        return {
            "task":              self.name,
            "raw_accuracy":      raw_acc,
            "centered_accuracy": centered_accuracy(raw_acc, self.random_baseline),
            "random_baseline":   self.random_baseline,
            "n":                 n,
            "sample_failures":   failures,
        }
