"""
Heavily Compressed Attention (HCA) — DeepSeek-V4 technical report, Sec. 2.3.2.

HCA = aggressive, non-overlapped KV compression (1 entry per m' tokens, m' >> m)
followed by *dense* MQA attention over all preceding compressed entries (no
indexer / no top-k), plus the same sliding-window branch and attention sink as CSA.

V4-Pro settings: m' = 128, n_head = 128, head_dim = 512, q_compress_dim = 1536,
n_win = 128.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # works both as a package and as standalone scripts on sys.path
    from .common import (
        HybridCompressedAttentionBase,
        blockwise_compress,
        preceding_block_mask,
    )
except ImportError:
    from common import (
        HybridCompressedAttentionBase,
        blockwise_compress,
        preceding_block_mask,
    )


class HeavilyCompressedAttention(HybridCompressedAttentionBase):
    def __init__(
        self,
        dim: int,
        n_head: int = 8,
        head_dim: int = 32,
        q_compress_dim: int = 64,
        window: int = 128,
        n_groups: int = 2,
        group_inter_dim: int = 64,
        compress_ratio: int = 128,        # m' (>> m)
        **kwargs,                         # rope_head_dim, rope_theta, ... -> base
    ):
        super().__init__(
            dim=dim, n_head=n_head, head_dim=head_dim, q_compress_dim=q_compress_dim,
            window=window, n_groups=n_groups, group_inter_dim=group_inter_dim,
            compress_ratio=compress_ratio, **kwargs,
        )
        c = head_dim
        # Single value series + single weight series (no overlap).
        self.w_kv = nn.Linear(dim, c, bias=False)
        self.w_z = nn.Linear(dim, c, bias=False)
        self.bias = nn.Parameter(torch.zeros(compress_ratio, c))            # B

    def _compressed_entries(self, H, cq):
        C = self.w_kv(H)
        Z = self.w_z(H)
        ckv = blockwise_compress(C, Z, self.bias, self.m)
        ckv = self.compress_norm(ckv)                                       # (B, nb, c), pre-RoPE
        # Dense over all preceding compressed blocks (no sparse selection).
        T, n_blocks = H.shape[1], ckv.shape[1]
        allow = preceding_block_mask(T, self.m, n_blocks, H.device)[None]   # (1, T, nb)
        allow = allow.expand(H.shape[0], -1, -1)
        return ckv, allow
