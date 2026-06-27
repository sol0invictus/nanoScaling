"""
Numerical diff of our compression against the DeepSeek-V4 reference implementation
(`huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference/model.py`).

Why only the compressor?
    The full reference forward cannot be RUN in most environments: it requires
    FP8/FP4 tensor types, custom Triton kernels (`sparse_attn`, `fp8_gemm`,
    `act_quant`) and `fast_hadamard_transform`, i.e. Hopper/Blackwell GPUs. So a
    literal "run both, compare outputs" is not portable. Instead we transcribe the
    *math* of the most intricate, bug-prone piece — the `Compressor` — into plain
    fp64 (no quant / RoPE / Hadamard / KV-cache) directly from the reference source
    and assert our `overlapped_compress` / `blockwise_compress` produce identical
    results. The Hadamard rotation in the reference is orthonormal and applied to
    both indexer q and k, so it preserves their dot product and is a no-op in full
    precision; the rest of the attention (indexer ReLU scoring, shared-KV MQA,
    attention sink, grouped output, RoPE) is covered by test_correctness.py and the
    structural audit in REFERENCE_AUDIT.md.

Run from the repo root:
    python -m experiments.csa_hca.test_reference_diff
    (or: python experiments/csa_hca/test_reference_diff.py)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import overlapped_compress, blockwise_compress


def _check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok: {name}")


# ── faithful fp64 transcription of reference Compressor.forward (prefill) ──────
#
# Reference (model.py, Compressor.forward, start_pos==0, seqlen % ratio == 0):
#   kv    = wkv(x)                       # (B, T, coff*d); coff = 2 when overlap
#   score = wgate(x)
#   kv    = kv.unflatten(1, (-1, ratio))            # (B, nb, ratio, coff*d)
#   score = score.unflatten(1, (-1, ratio)) + ape   # ape: (ratio, coff*d)
#   if overlap:  kv, score = overlap_transform(...)  # -> (B, nb, 2*ratio, d)
#   kv = (kv * score.softmax(dim=2)).sum(dim=2)      # softmax over the (2*)ratio axis
# overlap_transform(t, value):
#   new[:, :, ratio:]  = t[:, :,   :, d:]     # current block, second-half dims
#   new[:, 1:, :ratio] = t[:, :-1, :, :d]     # previous block, first-half dims (shifted)
#
# Mapping to our split tensors: reference first-half dims (:d) == our "overlap/
# previous" series (Cb / Zb / Bb); reference second-half dims (d:) == our "current"
# series (Ca / Za / Ba).

def ref_overlap_compress(Ca, Cb, Za, Zb, Ba, Bb, m):
    B, Tt, d = Ca.shape
    nb = Tt // m
    cur_kv, cur_sc = Ca.view(B, nb, m, d), Za.view(B, nb, m, d) + Ba
    prev_kv, prev_sc = Cb.view(B, nb, m, d), Zb.view(B, nb, m, d) + Bb
    kv2 = Ca.new_zeros(B, nb, 2 * m, d)
    sc2 = Za.new_full((B, nb, 2 * m, d), float("-inf"))
    kv2[:, :, m:] = cur_kv                 # current block
    sc2[:, :, m:] = cur_sc
    kv2[:, 1:, :m] = prev_kv[:, :-1]       # previous block (shifted by one)
    sc2[:, 1:, :m] = prev_sc[:, :-1]
    return (kv2 * sc2.softmax(dim=2)).sum(dim=2)


def ref_block_compress(C, Z, B, m):
    # Reference non-overlap path: kv = (kv.unflatten(1,(-1,ratio)) * (score+ape).softmax(2)).sum(2)
    Bsz, Tt, d = C.shape
    nb = Tt // m
    sc = Z.view(Bsz, nb, m, d) + B
    return (C.view(Bsz, nb, m, d) * sc.softmax(dim=2)).sum(dim=2)


def test_overlap_compress_matches_reference():
    torch.manual_seed(0)
    B, T, d, m = 3, 24, 6, 4                       # T % m == 0
    Ca = torch.randn(B, T, d, dtype=torch.float64)
    Cb = torch.randn(B, T, d, dtype=torch.float64)
    Za = torch.randn(B, T, d, dtype=torch.float64)
    Zb = torch.randn(B, T, d, dtype=torch.float64)
    Ba = torch.randn(m, d, dtype=torch.float64)
    Bb = torch.randn(m, d, dtype=torch.float64)

    ours = overlapped_compress(Ca, Cb, Za, Zb, Ba, Bb, m)
    ref = ref_overlap_compress(Ca, Cb, Za, Zb, Ba, Bb, m)
    err = (ours - ref).abs().max().item()
    print(f"  [overlap] max abs err vs reference transcription = {err:.3e}")
    _check("overlapped_compress == reference Compressor (overlap)", err < 1e-12)


def test_block_compress_matches_reference():
    torch.manual_seed(1)
    B, T, d, m = 3, 32, 5, 8
    C = torch.randn(B, T, d, dtype=torch.float64)
    Z = torch.randn(B, T, d, dtype=torch.float64)
    Bias = torch.randn(m, d, dtype=torch.float64)

    ours = blockwise_compress(C, Z, Bias, m)
    ref = ref_block_compress(C, Z, Bias, m)
    err = (ours - ref).abs().max().item()
    print(f"  [block]   max abs err vs reference transcription = {err:.3e}")
    _check("blockwise_compress == reference Compressor (non-overlap)", err < 1e-12)


def test_compression_reduces_sequence():
    # Reference: "compresses the sequence length to 1/m times".
    B, T, d = 2, 32, 4
    C = torch.randn(B, T, d, dtype=torch.float64)
    Z = torch.randn(B, T, d, dtype=torch.float64)
    _check("HCA m=128 -> ceil(T/128) entries",
           blockwise_compress(C, Z, torch.zeros(8, d, dtype=torch.float64), 8).shape[1] == T // 8)


if __name__ == "__main__":
    print("[reference compression diff]")
    test_overlap_compress_matches_reference()
    test_block_compress_matches_reference()
    test_compression_reduces_sequence()
    print("\nCompression matches the DeepSeek-V4 reference transcription.")
