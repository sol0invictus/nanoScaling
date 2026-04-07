"""
ARC — AI2 Reasoning Challenge
(Clark et al., 2018 — https://arxiv.org/abs/1803.05457)

Two difficulties: ARC-Easy and ARC-Challenge.
Questions have 3 or 4 answer choices labelled A/B/C/D (or 1/2/3/4).
We normalise all labels to A/B/C/D.

Metric: raw accuracy + centered accuracy (random baseline = 0.25).
Dataset: allenai/ai2_arc  (HuggingFace)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from evals.scoring import logit_mc, centered_accuracy

RANDOM_BASELINE = 0.25
NUM_FEWSHOT = 0    # ARC is typically evaluated 0-shot
CHOICES = ["A", "B", "C", "D"]


def _normalise_label(label: str) -> Optional[str]:
    """Map numeric labels (1,2,3,4) to letters (A,B,C,D)."""
    mapping = {"1": "A", "2": "B", "3": "C", "4": "D",
               "A": "A", "B": "B", "C": "C", "D": "D"}
    return mapping.get(str(label).strip().upper())


def _build_prompt(
    enc,
    question: str,
    choices: List[str],   # text of each choice (len 3 or 4)
    labels: List[str],    # normalised letter labels per choice
    fewshot_examples: List[Dict],
) -> Tuple[torch.Tensor, List[str]]:
    """
    Build the tokenised prompt and return the ordered answer letters.
    Returns:
        prompt_ids  — LongTensor
        letters     — answer letters in the same order as choices
    """
    shots = ""
    for ex in fewshot_examples:
        shots += f"Question: {ex['question']}\n"
        for lbl, txt in zip(ex["labels"], ex["choices"]):
            shots += f"{lbl}) {txt}\n"
        shots += f"Answer: {ex['answer_letter']}\n\n"

    prompt = shots
    prompt += f"Question: {question}\n"
    for lbl, txt in zip(labels, choices):
        prompt += f"{lbl}) {txt}\n"
    prompt += "Answer:"

    return torch.tensor(enc.encode(prompt), dtype=torch.long), labels


def _load_arc(config_name: str, max_samples: Optional[int]) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config_name, split="test", trust_remote_code=True)
    examples = []
    for ex in ds:
        labels  = [_normalise_label(l) for l in ex["choices"]["label"]]
        texts   = ex["choices"]["text"]
        gold    = _normalise_label(ex["answerKey"])
        if None in labels or gold is None:
            continue
        examples.append({
            "question":      ex["question"],
            "choices":       texts,
            "labels":        labels,
            "answer_letter": gold,
        })
    if max_samples:
        rng = random.Random(42)
        examples = rng.sample(examples, min(max_samples, len(examples)))
    return examples


class _ARCTask:
    random_baseline = RANDOM_BASELINE

    def __init__(self, config_name: str, max_samples: Optional[int] = None):
        self._config_name = config_name
        self.max_samples  = max_samples

    def run(self, model, enc, device: str, block_size: int) -> Dict[str, Any]:
        model.eval()
        examples = _load_arc(self._config_name, self.max_samples)
        correct  = 0

        for idx, ex in enumerate(tqdm(examples, desc=self.name)):
            prompt_ids, letters = _build_prompt(
                enc,
                ex["question"],
                ex["choices"],
                ex["labels"],
                fewshot_examples=[],   # 0-shot
            )

            if prompt_ids.shape[0] >= block_size:
                prompt_ids = prompt_ids[-(block_size - 1):]

            pred, scores = logit_mc(model, prompt_ids, letters, device)
            if letters[pred] == ex["answer_letter"]:
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


class ARCEasyTask(_ARCTask):
    name = "arc_easy"
    def __init__(self, max_samples: Optional[int] = None):
        super().__init__("ARC-Easy", max_samples)


class ARCChallengeTask(_ARCTask):
    name = "arc_challenge"
    def __init__(self, max_samples: Optional[int] = None):
        super().__init__("ARC-Challenge", max_samples)
