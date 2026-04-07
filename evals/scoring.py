"""
Low-level scoring utilities shared across eval tasks.

Two evaluation strategies:

  1. logit_mc   — single forward pass; compare log-probs of answer-letter tokens
                  at the final prompt position. Used for MMLU and ARC where
                  every choice maps to a single token (A / B / C / D).

  2. completion_loss_mc — one forward pass per choice; score each multi-token
                  continuation by its mean per-token CE loss, pick the minimum.
                  Used for HellaSwag where choices are full sentences.

Both return the predicted 0-based choice index and the score vector.
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Tuple


# ─── answer letter tokens in GPT-2 tokenizer ─────────────────────────────────
# enc.encode(" A") = [317], enc.encode(" B") = [347], etc.
# We use the space-prefixed versions since the prompt ends with a word and the
# next natural token has a leading space in BPE.
LETTER_TOKEN_IDS = {
    "A": 317,   # " A"
    "B": 347,   # " B"
    "C": 327,   # " C"
    "D": 360,   # " D"
    "E": 412,   # " E"
}


@torch.no_grad()
def logit_mc(
    model,
    prompt_ids: torch.Tensor,          # (T,) or (1, T)
    answer_letters: List[str],         # e.g. ["A","B","C","D"]
    device: str,
) -> Tuple[int, List[float]]:
    """
    Single forward pass.  Compare log-probs of answer-letter tokens at the
    last prompt position.  Picks the letter with highest log-prob.

    Returns:
        pred  — 0-based predicted choice index
        scores — log-prob of each answer letter
    """
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)            # (1, T)
    prompt_ids = prompt_ids.to(device)

    logits, _, _ = model(prompt_ids)                   # (1, T, V)
    last_logits = logits[0, -1, :]                     # (V,)
    log_probs   = F.log_softmax(last_logits, dim=-1)

    scores = []
    for letter in answer_letters:
        tok_id = LETTER_TOKEN_IDS.get(letter.upper())
        if tok_id is None:
            raise ValueError(f"No single-token id for answer letter {letter!r}. "
                             f"Supported: {list(LETTER_TOKEN_IDS)}")
        scores.append(log_probs[tok_id].item())

    pred = int(torch.tensor(scores).argmax())
    return pred, scores


@torch.no_grad()
def completion_loss_mc(
    model,
    context_ids: torch.Tensor,           # (T_ctx,) — shared context (no choices)
    continuations: List[List[int]],       # one token list per choice
    device: str,
    block_size: int,
) -> Tuple[int, List[float]]:
    """
    Per-continuation scoring.  For each choice, concatenate context + continuation
    tokens, run a forward pass, and compute the mean CE loss over the continuation
    tokens only.  Picks the choice with the lowest mean loss (highest likelihood).

    Returns:
        pred   — 0-based predicted choice index
        scores — mean CE loss per choice (lower = more likely)
    """
    context_ids = context_ids.to(device)
    ctx_len     = context_ids.shape[-1]
    scores      = []

    for cont in continuations:
        cont_t = torch.tensor(cont, dtype=torch.long, device=device)
        cont_len = cont_t.shape[0]

        # Build full sequence; truncate from the left if over block_size
        full = torch.cat([context_ids, cont_t], dim=-1)
        if full.shape[-1] > block_size:
            full = full[-block_size:]
            # Recompute where the continuation starts after truncation
            eff_ctx = block_size - cont_len
        else:
            eff_ctx = ctx_len

        x = full[:-1].unsqueeze(0)              # (1, L-1)
        y = full[1:].unsqueeze(0)               # (1, L-1)

        logits, _, _ = model(x)                 # (1, L-1, V)

        # Loss over continuation positions only
        cont_logits  = logits[0, eff_ctx - 1 :, :]       # (cont_len, V)
        cont_targets = y[0, eff_ctx - 1 :]               # (cont_len,)

        loss = F.cross_entropy(cont_logits, cont_targets, reduction="mean")
        scores.append(loss.item())

    pred = int(torch.tensor(scores).argmin())    # lowest loss = best
    return pred, scores


@torch.no_grad()
def generate_greedy(
    model,
    prompt_ids: torch.Tensor,    # (T,) or (1, T)
    max_new_tokens: int,
    device: str,
    eos_token_id: int = 50256,
) -> List[int]:
    """
    Greedy (temperature=0) autoregressive generation with KV cache.
    Stops at eos_token_id or max_new_tokens.
    Returns only the generated token ids (not the prompt).
    """
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids.to(device)

    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=1.0,      # temperature=1.0 with top_k=1 → greedy
        top_k=1,
        use_kv_cache=True,
    )
    # model.generate returns the full sequence (prompt + new tokens)
    new_tokens = generated[0, prompt_ids.shape[1]:].tolist()
    # Trim at EOS
    if eos_token_id in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(eos_token_id)]
    return new_tokens


def centered_accuracy(raw_acc: float, random_baseline: float) -> float:
    """
    ChatCORE-style centered accuracy.
    Normalises raw accuracy against random-chance baseline so that
    random performance maps to 0 and perfect performance maps to 1.

    centered = (raw - baseline) / (1 - baseline)
    """
    if random_baseline >= 1.0:
        return 0.0
    return (raw_acc - random_baseline) / (1.0 - random_baseline)
