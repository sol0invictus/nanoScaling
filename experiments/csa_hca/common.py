"""
Shared building blocks for the DeepSeek-V4 hybrid attention modules
(Compressed Sparse Attention + Heavily Compressed Attention).

Readable PyTorch reimplementation for research/investigation, aligned with the
DeepSeek-V4 technical report (Sec. 2.3) AND the official inference reference at
`huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference` (`model.py`).

Faithful to the reference algorithm (in fp32/bf16):
  - Low-rank query: wq_a -> RMSNorm(q_lora_rank) -> wq_b -> per-head normalize.
    The normalised latent is shared with the CSA indexer.
  - Data-dependent learned softmax compression with learnable positional bias
    (`ape`). CSA uses overlapped compression (entry i pools the previous block's
    "overlap" features with the current block's features over 2m elements); HCA
    uses non-overlapped compression over m' elements. Verified numerically equal
    to the reference `Compressor` in test_reference_diff.py.
  - Shared-KV Multi-Query Attention (one KV head, many query heads); the window
    and compressed entries are concatenated into ONE softmax with a learnable
    per-head attention sink.
  - Partial RoPE (last `rope_head_dim` dims) on queries / window-KV / compressed
    KV entries, with an INVERSE RoPE on the output at the query position — the
    reference's relative-position trick (a compressed entry is both key and
    value, so its value contribution must be de-rotated). Optional YaRN scaling.
  - Grouped output projection; strict causality (a query sees only compressed
    blocks s < floor(t/m), plus its local window).

Deliberate non-faithfulness (documented in README.md): no FP8/FP4 quantization,
no Triton kernels, no Hadamard rotation (an orthogonal, score-preserving quant
trick), and a dense-mask implementation of the top-k instead of a real gather.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── normalisation helpers ─────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """RMSNorm over the last dim, with a learnable per-channel gain."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalisation WITHOUT a learnable gain (matches the reference's
    per-head query normalisation `q *= rsqrt(q.square().mean(-1) + eps)`)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# ── rotary positional embedding (partial, complex, YaRN-capable) ──────────────

def precompute_freqs_cis(rope_dim, max_pos, theta=10000.0, original_seq_len=0,
                         factor=1.0, beta_fast=32, beta_slow=1) -> torch.Tensor:
    """
    (max_pos, rope_dim/2) complex rotation table. Ported from the reference
    `precompute_freqs_cis`; YaRN frequency interpolation is applied only when
    original_seq_len > 0 (otherwise this is plain RoPE).
    """
    def correction_dim(num_rotations):
        return rope_dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(theta))

    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    if original_seq_len > 0:
        low = math.floor(correction_dim(beta_fast))
        high = math.ceil(correction_dim(beta_slow))
        low, high = max(low, 0), min(high, rope_dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(rope_dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)        # (max_pos, rope_dim/2)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, rope_dim: int,
               inverse: bool = False) -> torch.Tensor:
    """
    Rotate the last `rope_dim` dims of `x` by `freqs_cis`. The position axis is
    dim -2 of `x`; `freqs_cis` has shape (L, rope_dim/2) complex with L == x.shape[-2].
    Adjacent dims are paired into complex numbers (the reference's convention).
    `inverse=True` conjugates the rotation (de-rotation of the output).
    """
    if rope_dim == 0:
        return x
    nope, rot = x[..., :-rope_dim], x[..., -rope_dim:]
    lead = rot.shape[:-2]
    L, rd = rot.shape[-2], rot.shape[-1]
    xc = torch.view_as_complex(rot.float().reshape(*lead, L, rd // 2, 2))
    f = freqs_cis.conj() if inverse else freqs_cis
    f = f.reshape(*([1] * len(lead)), L, rd // 2)
    rotated = torch.view_as_real(xc * f).reshape(*lead, L, rd).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


# ── masks ──────────────────────────────────────────────────────────────────--

def preceding_block_mask(T: int, m: int, n_blocks: int, device) -> torch.Tensor:
    """
    (T, n_blocks) bool. allow[t, s] == True iff compressed block s is strictly
    preceding query token t, i.e. s < floor(t / m). This guarantees causality
    (the whole of block s lies before t) and excludes the query's own block
    (which is instead covered by the sliding window).
    """
    block_idx = torch.arange(n_blocks, device=device)          # (n_blocks,)
    t_block = torch.arange(T, device=device) // m              # (T,)
    return block_idx[None, :] < t_block[:, None]               # (T, n_blocks)


def causal_window_mask(T: int, window: int, device) -> torch.Tensor:
    """
    (T, T) bool. allow[t, i] == True iff key token i is in the sliding window
    of query t: i in (t - window, t], i.e. the most recent `window` tokens
    including t itself.
    """
    t = torch.arange(T, device=device)
    return (t[None, :] <= t[:, None]) & (t[None, :] > t[:, None] - window)


# ── compression kernels ───────────────────────────────────────────────────────

def overlapped_compress(Ca, Cb, Za, Zb, Ba, Bb, m):
    """
    CSA overlapped compression (report Eq. 11-12; reference `Compressor` with
    overlap=True). Compressed entry i pools the *current* block i of C^a with the
    *previous* block i-1 of C^b under one softmax over the 2m elements (per
    channel). Entry 0 has no previous block (padded so it contributes 0).

        Ca, Cb : (B, T, c)  value features (a = current, b = overlap/previous)
        Za, Zb : (B, T, c)  compression-weight logits
        Ba, Bb : (m, c)     learnable intra-block positional biases
    Returns comp : (B, ceil(T/m), c).
    """
    B, T, c = Ca.shape
    n_blocks = math.ceil(T / m)
    pad = n_blocks * m - T

    def to_blocks(x, pad_value=0.0):
        if pad:
            x = F.pad(x, (0, 0, 0, pad), value=pad_value)
        return x.view(B, n_blocks, m, c)

    Ca_b = to_blocks(Ca)
    Za_b = to_blocks(Za, pad_value=float("-inf"))
    Cb_b = to_blocks(Cb)
    Zb_b = to_blocks(Zb, pad_value=float("-inf"))

    Cb_prev = torch.zeros_like(Cb_b)
    Zb_prev = torch.full_like(Zb_b, float("-inf"))
    Cb_prev[:, 1:] = Cb_b[:, :-1]
    Zb_prev[:, 1:] = Zb_b[:, :-1]

    Z_cat = torch.cat([Za_b + Ba, Zb_prev + Bb], dim=2)        # (B, nb, 2m, c)
    C_cat = torch.cat([Ca_b, Cb_prev], dim=2)
    S = torch.softmax(Z_cat, dim=2)
    return (S * C_cat).sum(dim=2)                              # (B, nb, c)


def blockwise_compress(C, Z, B, m):
    """
    HCA non-overlapped compression (report Eq. 22-23; reference `Compressor` with
    overlap=False). Each entry pools its own block of m tokens with a per-channel
    softmax over Z + positional bias.

        C : (B, T, c), Z : (B, T, c), B : (m, c)
    Returns comp : (B, ceil(T/m), c).
    """
    Bsz, T, c = C.shape
    n_blocks = math.ceil(T / m)
    pad = n_blocks * m - T
    if pad:
        C = F.pad(C, (0, 0, 0, pad))
        Z = F.pad(Z, (0, 0, 0, pad), value=float("-inf"))
    C = C.view(Bsz, n_blocks, m, c)
    Z = Z.view(Bsz, n_blocks, m, c)
    S = torch.softmax(Z + B, dim=2)
    return (S * C).sum(dim=2)


# ── base attention module ────────────────────────────────────────────────────

class HybridCompressedAttentionBase(nn.Module):
    """
    Shared machinery for CSA and HCA. Subclasses implement `_compressed_entries`
    (return normalised compressed KV entries, pre-RoPE, plus a per-(token, block)
    allow mask). The base handles low-rank queries, the window branch, RoPE +
    inverse-RoPE, the single concatenated MQA softmax with attention sink, and
    grouped output projection.
    """

    def __init__(
        self,
        dim: int,
        n_head: int = 8,
        head_dim: int = 32,
        q_compress_dim: int = 64,
        window: int = 128,
        n_groups: int = 2,
        group_inter_dim: int = 64,
        compress_ratio: int = 4,
        rope_head_dim: int = 0,
        rope_theta: float = 10000.0,
        rope_original_seq_len: int = 0,
        rope_factor: float = 1.0,
        rope_max_seq_len: int = 4096,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert n_head % n_groups == 0, "n_head must be divisible by n_groups"
        assert rope_head_dim % 2 == 0 and rope_head_dim <= head_dim, "rope_head_dim must be even and <= head_dim"
        self.dim = dim
        self.n_head = n_head
        self.head_dim = head_dim                # c
        self.q_compress_dim = q_compress_dim    # d_c / q_lora_rank
        self.window = window                    # n_win
        self.n_groups = n_groups                # g
        self.group_inter_dim = group_inter_dim  # d_g / o_lora_rank
        self.m = compress_ratio                 # m (CSA) or m' (HCA)
        self.rope_head_dim = rope_head_dim
        self.rope_theta = rope_theta
        self.eps = norm_eps
        self.scale = head_dim ** -0.5
        # Precompute the RoPE rotation table once (positions 0..rope_max_seq_len-1)
        # and index it — avoids per-call `.item()` / dynamic sizing that thrash
        # torch.compile. Compressed-block positions (i*m) are always < seq_len.
        if rope_head_dim:
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(rope_head_dim, rope_max_seq_len, rope_theta,
                                     rope_original_seq_len, rope_factor),
                persistent=False,
            )

        # Low-rank query: wq_a -> RMSNorm(latent) -> wq_b ; latent shared w/ indexer.
        self.w_dq = nn.Linear(dim, q_compress_dim, bias=False)        # wq_a
        self.q_latent_norm = RMSNorm(q_compress_dim, norm_eps)        # q_norm on latent
        self.w_uq = nn.Linear(q_compress_dim, n_head * head_dim, bias=False)  # wq_b

        # Sliding-window KV (uncompressed, single shared KV head).
        self.w_win = nn.Linear(dim, head_dim, bias=False)
        self.kv_norm = RMSNorm(head_dim, norm_eps)         # window KV norm
        self.compress_norm = RMSNorm(head_dim, norm_eps)   # compressed KV norm

        self.sink_logit = nn.Parameter(torch.zeros(n_head))

        # Grouped output projection.
        in_per_group = (n_head // n_groups) * head_dim
        self.out_group = nn.Parameter(torch.empty(n_groups, in_per_group, group_inter_dim))
        nn.init.normal_(self.out_group, std=0.02)
        self.out_final = nn.Linear(n_groups * group_inter_dim, dim, bias=False)

    # -- query / window / output helpers --------------------------------------

    def _query_latent(self, H):
        """Normalised low-rank latent c^Q (shared with the CSA indexer)."""
        return self.q_latent_norm(self.w_dq(H))              # (B, T, d_c)

    def _queries(self, cq):
        B, T, _ = cq.shape
        q = self.w_uq(cq).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = rms_normalize(q, self.eps)                       # per-head, no gain
        if self.rope_head_dim:
            q = apply_rope(q, self.freqs_cis[:T], self.rope_head_dim)   # at query positions
        return q                                             # (B, n_head, T, c)

    def _window_kv(self, H):
        B, T, _ = H.shape
        wkv = self.kv_norm(self.w_win(H))                    # (B, T, c)
        if self.rope_head_dim:
            wkv = apply_rope(wkv, self.freqs_cis[:T], self.rope_head_dim)
        return wkv

    def _rope_compressed(self, ckv):
        """RoPE compressed entries at block-start positions i*m (reference convention)."""
        if not self.rope_head_dim:
            return ckv
        nb = ckv.shape[1]
        pos = torch.arange(nb, device=ckv.device) * self.m
        return apply_rope(ckv, self.freqs_cis[pos], self.rope_head_dim)

    def _inverse_rope_output(self, o):
        """De-rotate the output at the query position (relative-position trick)."""
        if not self.rope_head_dim:
            return o
        T = o.shape[2]
        return apply_rope(o, self.freqs_cis[:T], self.rope_head_dim, inverse=True)

    def _compressed_entries(self, H, cq):
        """Return (compressed_kv (B, nb, c) normalised, pre-RoPE; allow (B, T, nb) bool)."""
        raise NotImplementedError

    def _aux_loss(self, H):
        """Auxiliary training loss (0 by default; CSA overrides with indexer distillation)."""
        return H.new_zeros(())

    # -- core attention: single softmax over [compressed ; window ; sink] ------

    def _core_attention(self, q, ckv, c_allow, wkv, w_allow):
        B, Hh, T, c = q.shape
        n_blocks = ckv.shape[1]

        comp_logits = torch.einsum("bhtc,bsc->bhts", q, ckv) * self.scale
        comp_logits = comp_logits.masked_fill(~c_allow[:, None], float("-inf"))

        win_logits = torch.einsum("bhtc,bsc->bhts", q, wkv) * self.scale
        win_logits = win_logits.masked_fill(~w_allow[None, None], float("-inf"))

        sink = self.sink_logit.view(1, Hh, 1, 1).expand(B, Hh, T, 1)

        logits = torch.cat([comp_logits, win_logits, sink], dim=-1)
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)

        w_comp = weights[..., :n_blocks]
        w_win = weights[..., n_blocks:n_blocks + T]
        return torch.einsum("bhts,bsc->bhtc", w_comp, ckv) \
            + torch.einsum("bhts,bsc->bhtc", w_win, wkv)

    def _output(self, o):
        B, Hh, T, c = o.shape
        g = self.n_groups
        o = o.transpose(1, 2).reshape(B, T, Hh * c)
        o = o.view(B, T, g, (Hh // g) * c)
        o = torch.einsum("btgk,gkd->btgd", o, self.out_group)
        o = o.reshape(B, T, g * self.group_inter_dim)
        return self.out_final(o)

    # -- forward ---------------------------------------------------------------

    def forward(self, H, return_aux=False):
        """
        H : (B, T, dim). If return_aux, also returns the indexer training loss
        (0 for HCA). The hard top-k is non-differentiable, so the indexer cannot
        be trained by the main loss — see CSA.indexer_distillation_loss.
        """
        B, T, _ = H.shape
        cq = self._query_latent(H)
        q = self._queries(cq)
        ckv, c_allow = self._compressed_entries(H, cq)
        ckv = self._rope_compressed(ckv)
        wkv = self._window_kv(H)
        w_allow = causal_window_mask(T, self.window, H.device)
        o = self._core_attention(q, ckv, c_allow, wkv, w_allow)
        o = self._inverse_rope_output(o)
        out = self._output(o)
        if return_aux:
            return out, self._aux_loss(H)
        return out
