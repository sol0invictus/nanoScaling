# Faithfulness audit vs. the DeepSeek-V4 reference

Reference: `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`
(`model.py`, `kernel.py`, `config.json`). This documents how our modules line up
with the official code, what was verified numerically, and what is deliberately
left out.

## Can we run the reference end-to-end here? No.

The reference is an **inference** implementation built around:
- FP8 (`float8_e4m3fn`) weights / activations and FP4 (`float4_e2m1fn_x2`) experts,
- custom Triton kernels: `sparse_attn`, `fp8_gemm`, `fp4_gemm`, `act_quant`,
- `fast_hadamard_transform`,
- a KV-cache / incremental-decode state machine.

These require Hopper/Blackwell-class GPUs (FP8/FP4) and extra packages. On this
box (RTX 3090 Ti, Ampere) it cannot run, so a literal "run both, diff the tensors"
is not possible. Instead we did a **structural audit** plus a **numerical diff of
the compressor** (the most intricate, bug-prone part) against a faithful fp64
transcription — see `test_reference_diff.py` (overlap err 2e-16, non-overlap err 0).

## What matches (verified)

| Component | Reference (`model.py`) | Ours | Evidence |
|---|---|---|---|
| Overlapped compression (CSA) | `Compressor` overlap=True: single `wkv: dim→2·d` split into prev/current halves, `ape` bias, softmax over 2m | `overlapped_compress` with two `w_kv_a/b`, `bias_a/b` (algebraically identical: stacking two `dim→d` == one `dim→2d`) | `test_reference_diff.py` → **2.2e-16** |
| Non-overlap compression (HCA) | `Compressor` overlap=False: softmax over m | `blockwise_compress` | `test_reference_diff.py` → **0.0** |
| Low-rank query | `wq_a → RMSNorm(q_lora_rank) → wq_b → per-head rms-normalize` | `w_dq → q_latent_norm → w_uq → rms_normalize` | matched after this audit |
| Shared latent → indexer | `qr` (normalised latent) feeds the indexer | `_query_latent` returns the normalised latent, passed to the indexer | matched |
| Lightning indexer | `wq_b`→heads, RoPE, `relu(q·k)`, per-head `weights_proj`, sum, top-k | `_indexer_scores`: ReLU, per-head `w_w`, top-k | structural + `test_correctness` selection test |
| Shared-KV MQA | one KV head as both key & value; window + compressed concatenated | `_core_attention` single softmax over `[compressed; window]` | brute-force oracle (1e-17) |
| Attention sink | learnable per-head logit in the denominator | `sink_logit`, extra softmax column | `test_correctness` sink test |
| Partial RoPE | last `rope_head_dim=64` dims on q / KV entries; **inverse** RoPE on output at query pos; compressed entry pos = block start | `apply_rope` + `_inverse_rope_output`; compressed pos = `i*m` | `test_correctness` RoPE tests |
| Grouped output proj | `wo_a` per-group → `o_lora_rank`, `wo_b` → dim | `out_group` + `out_final` | brute-force oracle |
| Causality | query sees compressed blocks `s < floor(t/m)` + window | `preceding_block_mask` + `causal_window_mask` | `test_correctness` causality (0 drift) |

## Differences (deliberate, documented)

1. **No low-precision / kernels.** We run bf16/fp32/fp64 with plain PyTorch — no
   FP8/FP4, no `sparse_attn`/`fp8_gemm`, no `act_quant`. The top-k is realised by
   masking (correct) rather than a gather kernel, so we don't get the FLOP/memory
   savings, only the same numerical result.
2. **No Hadamard rotation.** The reference applies `rotate_activation` (Hadamard)
   to indexer q and k before FP4 quant. It is orthonormal and applied to *both*
   operands, so it preserves `q·k` and is a no-op in full precision — safely omitted.
3. **YaRN off by default.** `precompute_freqs_cis` ports the reference's YaRN
   ramp, but defaults to plain RoPE (`rope_original_seq_len=0`). The reference uses
   `compress_rope_theta=160000` for compressed layers and base `rope_theta=10000`
   for the pure-window layer; expose via `rope_theta`.
4. **Indexer compression weight-sharing.** The report says the indexer reuses "the
   same compression operation"; the reference gives the indexer its **own**
   `Compressor`. We follow the reference (separate indexer projections + `idx_norm`).
5. **Per-head query norm has no learnable gain** in the reference
   (`q *= rsqrt(mean(q²)+eps)`); we match that with `rms_normalize` (the learnable
   `RMSNorm` is on the latent, `q_latent_norm`).
6. **Compressor remainder handling.** In prefill with `T % m == 0` (our test
   regime) results are identical. The reference additionally stashes a partial
   trailing block in `kv_state` for incremental decode; we pad-and-mask the
   trailing block (which is never visible to any query anyway).
7. **Not modelled** (out of scope — not part of CSA/HCA): MLA-style KV cache,
   Manifold-Constrained Hyper-Connections (mHC) residuals, MoE/MTP, the last
   layer's `compress_ratio=0` pure-window mode.

## Bottom line

The **compression is numerically identical** to the reference, and every other
CSA/HCA component matches it structurally and is covered by the correctness suite.
What we cannot claim is end-to-end bit-equivalence — that needs the FP8/FP4 +
Triton reference on Hopper/Blackwell hardware, which isn't available here.
