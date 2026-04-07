"""
MMLU — Massive Multitask Language Understanding
(Hendrycks et al., 2021 — https://arxiv.org/abs/2009.03300)

57 academic subjects, 4-way multiple choice.
Evaluated with 5-shot in-context examples from the dev split.

Metric: raw accuracy + centered accuracy (random baseline = 0.25).
Dataset: cais/mmlu  (HuggingFace)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from evals.scoring import logit_mc, centered_accuracy

# All 57 MMLU subjects
SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

CHOICES = ["A", "B", "C", "D"]
RANDOM_BASELINE = 0.25   # 1/4 chance for 4-way MC
NUM_FEWSHOT = 5


def _format_example(question: str, choices: List[str], answer: Optional[str] = None) -> str:
    """Format a single MMLU example as plain text."""
    text = f"Question: {question}\n"
    for letter, choice in zip(CHOICES, choices):
        text += f"{letter}) {choice}\n"
    if answer is not None:
        text += f"Answer: {answer}\n"
    return text


def _build_prompt(
    enc,
    subject: str,
    question: str,
    choices: List[str],
    fewshot_examples: List[Dict],
) -> torch.Tensor:
    """
    Build a tokenized prompt with num_fewshot demonstrations followed by
    the query question.  Ends with 'Answer:' so the model predicts the letter.
    """
    header = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
    shots  = ""
    for ex in fewshot_examples:
        shots += _format_example(ex["question"], ex["choices"], ex["answer_letter"]) + "\n"
    query = _format_example(question, choices)
    query += "Answer:"   # model should predict the next token = " A"/" B" etc.

    full_text = header + shots + query
    return torch.tensor(enc.encode(full_text), dtype=torch.long)


class MMLUTask:
    """
    MMLU evaluation task.

    Args:
        max_samples: cap on test examples per subject (None = all)
        subjects:    list of subjects to evaluate (None = all 57)
        num_fewshot: number of few-shot dev examples (default 5)
    """

    name = "mmlu"
    random_baseline = RANDOM_BASELINE

    def __init__(
        self,
        max_samples: Optional[int] = None,
        subjects: Optional[List[str]] = None,
        num_fewshot: int = NUM_FEWSHOT,
    ):
        self.max_samples = max_samples
        self.subjects    = subjects or SUBJECTS
        self.num_fewshot = num_fewshot

    def run(self, model, enc, device: str, block_size: int) -> Dict[str, Any]:
        from datasets import load_dataset

        model.eval()
        total_correct = 0
        total_count   = 0
        per_subject: Dict[str, Dict] = {}

        for subject in tqdm(self.subjects, desc="MMLU subjects"):
            # Load dev (few-shot pool) and test splits
            try:
                dev_ds  = load_dataset("cais/mmlu", subject, split="dev",        trust_remote_code=True)
                test_ds = load_dataset("cais/mmlu", subject, split="test",       trust_remote_code=True)
            except Exception as e:
                print(f"  [MMLU] Could not load {subject}: {e}")
                continue

            # Build few-shot pool
            letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            fewshot_pool = [
                {
                    "question":      ex["question"],
                    "choices":       ex["choices"],
                    "answer_letter": letter_map[ex["answer"]],
                }
                for ex in dev_ds
            ]

            examples = list(test_ds)
            if self.max_samples:
                rng = random.Random(42)
                examples = rng.sample(examples, min(self.max_samples, len(examples)))

            correct = 0
            for idx, ex in enumerate(examples):
                # Sample num_fewshot examples from dev (seeded per question)
                rng = random.Random(1234 + idx)
                shots = rng.sample(fewshot_pool, min(self.num_fewshot, len(fewshot_pool)))

                prompt_ids = _build_prompt(
                    enc, subject, ex["question"], ex["choices"], shots
                )

                # Truncate prompt from the left if too long (keep fewshot tail)
                if prompt_ids.shape[0] >= block_size:
                    prompt_ids = prompt_ids[-(block_size - 1):]

                pred, scores = logit_mc(model, prompt_ids, CHOICES, device)
                gold = ex["answer"]   # int 0-3
                if pred == gold:
                    correct += 1

            n = len(examples)
            acc = correct / n if n > 0 else 0.0
            per_subject[subject] = {"accuracy": acc, "n": n, "correct": correct}
            total_correct += correct
            total_count   += n

        raw_acc = total_correct / total_count if total_count > 0 else 0.0
        return {
            "task":             self.name,
            "raw_accuracy":     raw_acc,
            "centered_accuracy": centered_accuracy(raw_acc, self.random_baseline),
            "random_baseline":  self.random_baseline,
            "n":                total_count,
            "per_subject":      per_subject,
        }
