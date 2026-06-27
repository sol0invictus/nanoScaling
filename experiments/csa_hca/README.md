# DeepSeek-V4 Hybrid Attention — CSA & HCA (simple PyTorch)

Readable, research-oriented reimplementations of the two attention variants from
the **DeepSeek-V4 technical report** (Sec. 2.3, *Hybrid Attention with CSA and
HCA*):

- **CSA** — Compressed Sparse Attention (`csa.py`)
- **HCA** — Heavily Compressed Attention (`hca.py`)
- Shared machinery (`common.py`): low-rank queries, sliding window, attention
  sink, grouped output projection, the compression kernels, and the base class.

These are written for *understanding and experimentation*, not for production —
they use dense masking instead of gather/scatter kernels and fp32/fp64 math
instead of FP8/FP4. Source: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro`.

## The idea in one paragraph

As context length grows, attention dominates compute. DeepSeek-V4 attacks this by
**compressing the KV cache along the sequence**: every `m` tokens are merged into
one KV entry with *learned, data-dependent* softmax weights. **CSA** uses light
compression (`m=4`) and then keeps things cheap by having each query attend to
only the **top-k** compressed entries chosen by a lightweight **Lightning
Indexer** (DeepSeek Sparse Attention). **HCA** uses *heavy* compression
(`m'=128`) so few entries remain, and just attends to **all** of them densely.
Both add a small **sliding window** of recent uncompressed tokens for local
detail. Interleaving CSA and HCA layers makes 1M-token context tractable
(~2% of a BF16 GQA8 KV cache at 1M).

## CSA — Compressed Sparse Attention

1. **Overlapped compression** (report Eq. 11–12). Two value series `Cᵃ, Cᵇ` and
   two weight series `Zᵃ, Zᵇ` are projected from the hidden states. Compressed
   entry `i` pools the **current** block `i` of `Cᵃ` with the **previous** block
   `i-1` of `Cᵇ` under a single softmax over the `2m` elements (per channel),
   plus learnable positional biases `Bᵃ, Bᵇ`. The overlap smooths block
   boundaries; net compression is still `1/m`. → `overlapped_compress` in
   `common.py`.
2. **Lightning Indexer** (Eq. 13–17). Low-rank indexer queries `qᴵ` (from the
   shared query latent `cQ`), per-head weights `wᴵ`, and compressed indexer keys
   `Kᴵ` (same compression op, own low-dim projections). Score:
   `I[t,s] = Σ_h wᴵ[t,h] · ReLU(qᴵ[t,h] · Kᴵ[s])`. A **top-k** selector keeps the
   k highest-scoring *preceding* blocks (`s < ⌊t/m⌋`).
3. **Shared-KV MQA core attention** (Eq. 18–19). One shared KV head, many query
   heads; each selected compressed entry is both key and value.
4. **Sliding window + attention sink** (Sec. 2.3.3), fused into the same softmax.

## HCA — Heavily Compressed Attention

Same compression *procedure* but **non-overlapped** and aggressive (`m' ≫ m`,
report Eq. 22–23), then **dense** MQA over all preceding compressed entries — no
indexer, no top-k. Same sliding-window + sink + grouped output projection as CSA.

## Shared "other details" (Sec. 2.3.3)

- **Sliding-window branch**: `n_win` recent uncompressed KV entries, concatenated
  with the compressed entries into **one** softmax (not a gated sum).
- **Attention sink**: a learnable per-head logit added to the softmax
  denominator, so a head's total attention can be `< 1`.
- **Q / KV-entry RMSNorm** before core attention (latent `RMSNorm` + unweighted
  per-head normalize on queries, matching the reference).
- **Partial RoPE** (`rope_head_dim`, last 64 dims in V4) on queries, window-KV and
  compressed-KV entries, with an **inverse RoPE on the output** at the query
  position — the reference's relative-position trick (a compressed entry is both
  key and value, so its value contribution must be de-rotated). Compressed-entry
  position = block start (`i·m`). YaRN scaling is supported (`rope_original_seq_len`,
  `rope_factor`) but off by default. Set `rope_head_dim=0` to disable RoPE.
- **Grouped output projection**: split `n_h` heads into `g` groups, project each
  to `d_g`, concat, then to `d`.

## V4-Pro reference hyperparameters

| | symbol | value |
|---|---|---|
| CSA compression rate | `m` | 4 |
| CSA attention top-k | `k` | 1024 |
| indexer query heads / dim | `n^I_h`, `c_I` | 64, 128 |
| HCA compression rate | `m'` | 128 |
| query heads / head dim | `n_h`, `c` | 128, 512 |
| query compression dim | `d_c` | 1536 |
| output groups / inter dim | `g`, `d_g` | 16, 1024 |
| sliding window | `n_win` | 128 |
| layers | | 61 (first 2 = HCA, rest interleave CSA/HCA) |

The classes default to small values so the tests exercise every path; pass the
table values for a faithful-scale module.

## Faithfulness vs. the reference

This implementation was audited against the official inference code
(`huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`) — see
[REFERENCE_AUDIT.md](REFERENCE_AUDIT.md). The **compression is numerically
identical** to the reference `Compressor` (`test_reference_diff.py`: overlap err
2e-16, non-overlap err 0); every other component matches structurally.

Deliberate non-faithfulness (does not change the algorithm's result):

- **Dense masking, not gather.** Sparse selection masks non-selected blocks to
  `-inf` — correct, but without the FLOP/memory savings of a real gather kernel.
- **fp32/fp64 only** — no FP8 KV / FP4 indexer/expert quantization, no Triton
  kernels (`sparse_attn`, `fp8_gemm`), no Hadamard rotation (orthonormal and
  applied to both q & k, so it preserves `q·k` and is a no-op in full precision).
- **YaRN off by default** (supported via `rope_*` args); the reference uses
  `compress_rope_theta=160000` for compressed layers.
- A literal end-to-end numerical diff against the reference needs FP8/FP4 + Triton
  on Hopper/Blackwell hardware, which is unavailable here.

## Training the Lightning Indexer (auxiliary distillation loss)

Top-k selection is **non-differentiable**: the gather indices are detached, so no
gradient reaches the indexer parameters (`w_ik_*`, `w_iz_*`, `ibias_*`, `w_iuq`,
`w_w`) from the main attention output. In DeepSeek-V4 (inheriting DSA from V3.2,
and per the report's "warm up the lightning indexer" stage, Sec. 4.2.2) the
indexer is trained by a **separate auxiliary objective** that distils its scores
toward the real attention distribution.

`CompressedSparseAttention.indexer_distillation_loss(H)` implements it:

- **Target** — the *dense* main-attention distribution over all preceding
  compressed blocks, averaged over heads and detached (computed as if attention
  were dense, matching DeepSeek's dense-warmup recipe).
- **Student** — softmax over the indexer scores for the same preceding blocks.
- **Loss** — `KL(target ‖ student)`, averaged over query tokens with ≥1 preceding
  block.
- **Gradient isolation** — the indexer's inputs (hidden states + shared query
  latent) are detached, so the loss trains *only* the indexer's own parameters and
  never perturbs the main model.

Use it during training:

```python
out, aux = csa(H, return_aux=True)          # aux == indexer_distillation_loss(H)
loss = main_task_loss(out) + lam * aux      # lam ~ small, e.g. 1e-3 .. 1e-2
```

`forward(..., return_aux=True)` returns `(output, aux)`; HCA returns `aux = 0`
(it has no indexer). The correctness suite verifies the KL is finite and ≥ 0,
that backprop reaches all 8 indexer params and **no** core params, and that a few
Adam steps on the indexer reduce the KL.

## Tests

```bash
python experiments/csa_hca/test_correctness.py     # 38 checks
python experiments/csa_hca/test_reference_diff.py   # vs reference
```

`test_correctness.py` covers: output shapes/finiteness; **brute-force equivalence**
(a loop oracle that reuses the module's submodules — incl. RoPE — but recomputes
core attention explicitly, matches to ~1e-17); **strict causality** (perturbing
token `p` leaves all outputs `< p` bit-identical); indexer **top-k selection**
equals an independent top-k; `top_k ≥ n_blocks` ⇒ dense over preceding; compression
weights; the attention **sink**; **gradient flow** (main loss reaches all core
params but no indexer params); the **indexer distillation loss**; **RoPE**
(inverse round-trip, norm preservation, relative-position property, and that it
changes the output / is optional); and single-head MQA structure.

`test_reference_diff.py` transcribes the reference `Compressor` math (fp64, no
quant/kernels) and asserts our compression is identical — see
[REFERENCE_AUDIT.md](REFERENCE_AUDIT.md).

## Usage

```python
from experiments.csa_hca.csa import CompressedSparseAttention
from experiments.csa_hca.hca import HeavilyCompressedAttention

csa = CompressedSparseAttention(dim=256, n_head=8, head_dim=64, q_compress_dim=128,
                                window=128, compress_ratio=4, top_k=16)
hca = HeavilyCompressedAttention(dim=256, n_head=8, head_dim=64, q_compress_dim=128,
                                 window=128, compress_ratio=128)
y = csa(torch.randn(2, 512, 256))   # (2, 512, 256)
```
