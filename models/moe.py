"""
Mixture of Experts (MoE) implementation following nanoMOE.
https://github.com/hbfreed/nanoMOE

Uses MegaBlocks (stk + megablocks.ops) for block-sparse dispatch — no Python
for-loop over experts. All expert FFN weights are packed into single contiguous
w1 / w2 tensors and dispatched via token permutation + sparse matmul.

Install:
    pip install megablocks einops
    # megablocks ships stk (sparse toolkit) and the required CUDA/Triton kernels

Design choices (from nanoMOE / OLMoE):
  - Sigmoid routing (not softmax) — allows multi-label-style expert activation
  - LayerNorm before the router projection
  - norm_topk_prob: normalize selected top-k weights to sum to 1
  - Load-balance auxiliary loss (Switch Transformer style)
  - Router z-loss: penalizes large logit magnitudes for stability
  - Single contiguous w1/w2 — all experts packed into one tensor
  - Block-sparse dispatch: stk.ops.sdd / stk.ops.dsd (no per-expert loop)
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

import stk
import stk.ops
from megablocks import ops
from megablocks.layers.gelu import gelu


class MoERouter(nn.Module):
    """
    Sigmoid top-k router with LayerNorm before the projection (OLMoE / nanoMOE).

    Returns routing weights + indices for each token, plus a combined auxiliary
    loss (load-balance + z-loss).
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.load_balance_loss_weight = getattr(config, 'load_balance_loss_weight', 0.01)
        self.router_z_loss_weight = getattr(config, 'router_z_loss_weight', 0.001)

        # LN before projection (nanoMOE / OLMoE practice)
        self.norm = nn.LayerNorm(config.n_embd, bias=False)
        self.linear = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.sort_end_bit = max(int(math.ceil(math.log2(self.num_experts))), 1)

    def forward(self, x_flat):
        """
        Args:
            x_flat: (N, C) flattened token representations
        Returns:
            expert_weights:    (N, top_k)  routing weights
            expert_indices:    (N, top_k)  selected expert indices
            aux_loss:          scalar
            tokens_per_expert: (num_experts,) int32 — token counts for dispatch
        """
        router_logits = self.linear(self.norm(x_flat))   # (N, num_experts)

        # z-loss: penalizes large logit magnitude for routing stability
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)

        # Sigmoid routing — not softmax (nanoMOE style)
        router_probs = torch.sigmoid(router_logits)      # (N, num_experts)

        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)

        if self.norm_topk_prob:
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Token counts per expert (non-differentiable density f_i)
        tokens_per_expert = ops.histogram(expert_indices.flatten().int(), self.num_experts)

        # Load-balance loss: num_experts * sum(f_i * P_i)
        f_i = tokens_per_expert.float() / (x_flat.shape[0] * self.top_k)
        P_i = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * (f_i * P_i).sum()

        aux_loss = (
            self.load_balance_loss_weight * load_balance_loss
            + self.router_z_loss_weight * z_loss
        )

        return expert_weights, expert_indices, aux_loss, tokens_per_expert


class MoELayer(nn.Module):
    """
    Block-sparse Mixture of Experts FFN using MegaBlocks (stk).

    All expert weights live in two flat tensors:
        w1: (n_embd, num_experts * hidden_dim)  — input projection
        w2: (num_experts * hidden_dim, n_embd)  — output projection

    Dispatch is done via token permutation + stk.ops.sdd / stk.ops.dsd,
    matching nanoMOE's kernel path with no Python loop over experts.

    Additional config fields:
        moe_hidden_dim  — per-expert FFN width (default 4 * n_embd, rounded up
                          to a multiple of moe_block_size)
        moe_block_size  — Triton tile size for block-sparse kernels (default 128)
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.n_embd = config.n_embd
        self.block_size = getattr(config, 'moe_block_size', 128)

        hidden_dim = getattr(config, 'moe_hidden_dim', 0) or (4 * config.n_embd)
        # Round up to a block_size multiple (required by stk sparse kernels)
        hidden_dim = self.block_size * ((hidden_dim + self.block_size - 1) // self.block_size)
        self.hidden_dim = hidden_dim
        self.blocks_per_expert = hidden_dim // self.block_size
        self.total_expert_width = hidden_dim * self.num_experts

        self.sort_end_bit = max(int(math.ceil(math.log2(self.num_experts))), 1)
        max_col = self.total_expert_width // self.block_size
        self.transpose_sort_end_bit = max(int(math.ceil(math.log2(max(max_col, 2)))), 1)

        # Packed weight tensors — nanoMOE style (single tensor covers all experts)
        self.w1 = nn.Parameter(torch.empty(self.n_embd, self.total_expert_width))
        self.w2 = nn.Parameter(torch.empty(self.total_expert_width, self.n_embd))
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)

        self.router = MoERouter(config)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out:      (B, T, C)
            aux_loss: scalar
        """
        B, T, C = x.shape
        x_flat = rearrange(x, 'b t c -> (b t) c')   # (N, C)

        expert_weights, expert_indices, aux_loss, tokens_per_expert = self.router(x_flat)

        expert_weights_flat = expert_weights.to(x.dtype).flatten()   # (N * top_k,)
        expert_indices_flat = expert_indices.flatten().int()          # (N * top_k,)

        # --- Sort tokens into expert order ---
        bin_ids, indices = ops.sort(expert_indices_flat, self.sort_end_bit)

        # --- Pad each expert's bin to a block_size multiple ---
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.block_size)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0).contiguous()
        bins = ops.inclusive_cumsum(tokens_per_expert, 0).contiguous()

        # --- Permute tokens into expert-sorted order (with padding) ---
        x_permuted = ops.padded_gather(x_flat, indices, bin_ids, bins, padded_bins, self.top_k)

        # --- Build block-sparse topology ---
        topology = self._build_topology(x_flat, padded_bins, padded_tokens_per_expert)

        # --- Sparse FFN: w1 → GELU → w2 (no per-expert loop) ---
        x_permuted = stk.ops.sdd(x_permuted, self.w1, topology)   # sparse output
        x_permuted = gelu(x_permuted)
        x_permuted = stk.ops.dsd(x_permuted, self.w2)             # dense output

        # --- Scatter back and apply routing weights ---
        out_flat = ops.padded_scatter(
            x_permuted, indices, bin_ids, expert_weights_flat, bins, padded_bins, self.top_k
        )

        return rearrange(out_flat, '(b t) c -> b t c', b=B, t=T), aux_loss

    def _build_topology(self, x, padded_bins, padded_tokens_per_expert):
        """
        Build the stk.Matrix describing the block-sparse pattern for uniform experts.

        For expert i:
          - Row blocks:    [cumsum_padded[i-1] // bs,  cumsum_padded[i] // bs)
          - Column blocks: [i * bpe,  (i+1) * bpe)
        where bpe = blocks_per_expert = hidden_dim // block_size.
        """
        padded_tokens = padded_bins[-1].clamp_min(self.block_size)
        block_rows = int((padded_tokens // self.block_size).item())
        bpe = self.blocks_per_expert
        device = x.device
        dtype = x.dtype

        # Number of row blocks assigned to each expert
        expert_token_blocks = (padded_tokens_per_expert // self.block_size).long()  # (E,)

        # For each row block r owned by expert e: column blocks = [e*bpe, ..., (e+1)*bpe - 1]
        row_block_expert = torch.repeat_interleave(
            torch.arange(self.num_experts, dtype=torch.int32, device=device),
            expert_token_blocks,
        )  # (block_rows,)

        k = torch.arange(bpe, dtype=torch.int32, device=device)
        column_indices = (row_block_expert.unsqueeze(1) * bpe + k.unsqueeze(0)).reshape(-1)
        # (block_rows * bpe,) — one entry per nonzero block

        # CSR row offsets: each row block has exactly bpe nonzero column blocks
        offsets = torch.arange(block_rows + 1, dtype=torch.int32, device=device) * bpe

        shape = (int(padded_tokens.item()), self.total_expert_width)
        num_blocks = column_indices.numel()
        data = torch.empty(num_blocks, self.block_size, self.block_size, dtype=dtype, device='meta')

        column_indices = column_indices.to(torch.int32)
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices).to(torch.int32)
        col_t, off_t, blk_t = self._sparse_transpose(shape, row_indices, column_indices)

        return stk.Matrix(shape, data, row_indices, column_indices, offsets, col_t, off_t, blk_t)

    def _sparse_transpose(self, size, row_indices, column_indices):
        """Compute transposed CSR indices needed by stk.ops.dsd."""
        block_columns = self.total_expert_width // self.block_size
        _, gather_indices = ops.sort(column_indices.int(), self.transpose_sort_end_bit)
        column_indices_t = row_indices.gather(0, gather_indices.long()).to(torch.int32)
        block_offsets_t = gather_indices.to(torch.int32)
        zero = torch.zeros(1, dtype=torch.int32, device=row_indices.device)
        nnz_per_col = ops.histogram(column_indices, block_columns)
        nnz_per_col = ops.inclusive_cumsum(nnz_per_col, 0)
        if nnz_per_col.dim() == 0:
            nnz_per_col = nnz_per_col.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_col]).to(torch.int32)
        return column_indices_t, offsets_t, block_offsets_t
