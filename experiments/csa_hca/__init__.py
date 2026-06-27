"""
Simple PyTorch implementations of DeepSeek-V4's hybrid attention:
Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA).

See README.md for the mapping to the technical report and the simplifications made.
"""

try:
    from .common import (
        HybridCompressedAttentionBase,
        RMSNorm,
        rms_normalize,
        apply_rope,
        precompute_freqs_cis,
        overlapped_compress,
        blockwise_compress,
        preceding_block_mask,
        causal_window_mask,
    )
    from .csa import CompressedSparseAttention
    from .hca import HeavilyCompressedAttention
except ImportError:
    from common import (
        HybridCompressedAttentionBase,
        RMSNorm,
        rms_normalize,
        apply_rope,
        precompute_freqs_cis,
        overlapped_compress,
        blockwise_compress,
        preceding_block_mask,
        causal_window_mask,
    )
    from csa import CompressedSparseAttention
    from hca import HeavilyCompressedAttention

__all__ = [
    "CompressedSparseAttention",
    "HeavilyCompressedAttention",
    "HybridCompressedAttentionBase",
    "RMSNorm",
    "overlapped_compress",
    "blockwise_compress",
    "preceding_block_mask",
    "causal_window_mask",
]
