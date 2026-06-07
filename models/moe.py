"""
Mixture of Experts (MoE) — Unsloth grouped-GEMM backend.

Uses unsloth's fused Triton grouped-GEMM kernels when available (requires the
`kernel` conda environment with PyTorch 2.7+ / Triton 3.3+).  Falls back to a
plain PyTorch loop over experts for development / CPU / older CUDA environments.

Weight layout (both paths):
    w1  (num_experts, hidden_dim, n_embd)   — input projection per expert
    w2  (num_experts, n_embd,    hidden_dim) — output projection per expert

Routing:
    Sigmoid top-k (OLMoE / nanoMOE style) with optional topk-prob normalisation.
    Auxiliary loss = load-balance loss + router z-loss.

Install for kernel path:
    conda activate kernel            # environment with PyTorch 2.7+, Triton 3.3
    pip install git+https://github.com/unslothai/unsloth   # or local clone

References:
    https://github.com/unslothai/unsloth/tree/main/unsloth/kernels/moe
    https://arxiv.org/abs/2409.02060  (OLMoE)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Unsloth grouped-GEMM (optional) ──────────────────────────────────────────
# Requires: conda activate kernel  (PyTorch 2.7+ / Triton 3.3)
#           pip install git+https://github.com/unslothai/unsloth
try:
    from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm as _grouped_gemm
    from unsloth.kernels.moe.autotune_cache import get_or_autotune_moe_kernels as _autotune
    _HAS_UNSLOTH = True
except ImportError:
    _grouped_gemm = None
    _autotune     = None
    _HAS_UNSLOTH  = False


# ── Routing helpers (pure PyTorch, no megablocks) ────────────────────────────

def _get_routing_indices(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute token-counts-per-expert and the stable-sort gather indices.

    Args:
        topk_ids : (N * top_k,) int64 — flat expert assignments
        num_experts : int

    Returns:
        m_sizes       : (num_experts,) int32 — token count per expert
        gather_indices: (N * top_k,) int32  — argsort(topk_ids, stable=True)
    """
    m_sizes = torch.histc(
        topk_ids.float(), bins=num_experts, min=0, max=num_experts - 1
    ).to(torch.int32)
    gather_indices = torch.argsort(topk_ids, stable=True).to(torch.int32)
    return m_sizes, gather_indices


# ── Pure-PyTorch fallback dispatch ───────────────────────────────────────────

def _moe_forward_pytorch(
    x_flat:         torch.Tensor,   # (N, C)
    w1:             torch.Tensor,   # (E, H, C)
    w2:             torch.Tensor,   # (E, C, H)
    expert_weights: torch.Tensor,   # (N, top_k)
    expert_indices: torch.Tensor,   # (N, top_k) int64
    num_experts:    int,
) -> torch.Tensor:
    """
    Loop-based reference dispatch.  Correct but O(E) Python overhead.
    """
    N, C = x_flat.shape
    top_k = expert_indices.shape[1]
    out = torch.zeros_like(x_flat)

    # Expand tokens for each expert assignment
    x_exp     = x_flat.unsqueeze(1).expand(N, top_k, C)          # (N, K, C)
    w_flat    = expert_weights.reshape(N * top_k)                 # (N*K,)
    idx_flat  = expert_indices.reshape(N * top_k)                 # (N*K,)
    x_flat_k  = x_exp.reshape(N * top_k, C)                      # (N*K, C)

    for e in range(num_experts):
        mask = (idx_flat == e).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            continue
        xe  = x_flat_k[mask]                                      # (m, C)
        h   = F.gelu(xe @ w1[e].T)                                # (m, H)
        ye  = h @ w2[e].T                                         # (m, C)
        # Scatter-add with routing weights
        src_tokens = mask // top_k                                 # which original token
        out.index_add_(0, src_tokens, ye * w_flat[mask].unsqueeze(-1))

    return out


# ── Router ───────────────────────────────────────────────────────────────────

class MoERouter(nn.Module):
    """
    Sigmoid top-k router with LayerNorm before projection (OLMoE / nanoMOE).

    Returns routing weights + indices for each token, plus a combined auxiliary
    loss (load-balance + z-loss).
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts            = config.num_experts
        self.top_k                  = config.num_experts_per_tok
        self.norm_topk_prob         = getattr(config, 'norm_topk_prob', True)
        self.load_balance_loss_weight = getattr(config, 'load_balance_loss_weight', 0.01)
        self.router_z_loss_weight   = getattr(config, 'router_z_loss_weight', 0.001)

        self.norm   = nn.LayerNorm(config.n_embd, bias=False)
        self.linear = nn.Linear(config.n_embd, self.num_experts, bias=False)

    def forward(self, x_flat: torch.Tensor):
        """
        Args:
            x_flat: (N, C)
        Returns:
            expert_weights : (N, top_k)  normalized routing weights
            expert_indices : (N, top_k)  int64 expert indices
            aux_loss       : scalar
        """
        router_logits = self.linear(self.norm(x_flat))             # (N, E)

        # z-loss: penalizes large logit magnitude
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)

        # Sigmoid routing (not softmax)
        router_probs = torch.sigmoid(router_logits)

        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)

        if self.norm_topk_prob:
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Load-balance loss: num_experts * sum(f_i * P_i)
        N = x_flat.shape[0]
        token_counts = torch.bincount(
            expert_indices.flatten().to(torch.long),
            minlength=self.num_experts,
        ).float()
        f_i = token_counts / (N * self.top_k)
        P_i = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * (f_i * P_i).sum()

        aux_loss = (
            self.load_balance_loss_weight * load_balance_loss
            + self.router_z_loss_weight * z_loss
        )

        return expert_weights, expert_indices.to(torch.int64), aux_loss


# ── MoE Layer ────────────────────────────────────────────────────────────────

class MoELayer(nn.Module):
    """
    MoE FFN block.

    When `unsloth.kernels.moe` is importable (kernel conda env), dispatches via
    fused Triton grouped-GEMM kernels.  Otherwise falls back to a plain PyTorch
    loop over experts.

    Weight shapes:
        w1: (num_experts, hidden_dim, n_embd)   — up-projection
        w2: (num_experts, n_embd,    hidden_dim) — down-projection

    Forward:
        h   = GELU(x_expert @ w1_expert.T)
        out = sum_k( weight_k * h_k @ w2_expert.T )
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k       = config.num_experts_per_tok
        self.n_embd      = config.n_embd

        hidden_dim = getattr(config, 'moe_hidden_dim', 0) or (4 * config.n_embd)
        # Align to 128 for kernel efficiency
        hidden_dim = 128 * math.ceil(hidden_dim / 128)
        self.hidden_dim = hidden_dim

        E, H, C = self.num_experts, hidden_dim, config.n_embd
        self.w1 = nn.Parameter(torch.empty(E, H, C))
        self.w2 = nn.Parameter(torch.empty(E, C, H))
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)

        self.router = MoERouter(config)

        # Kernel configs (populated lazily on first unsloth forward pass)
        self._kernel_config_fwd   = None
        self._kernel_config_bwd_dX = None
        self._kernel_config_bwd_dW = None

    @property
    def use_unsloth(self) -> bool:
        return _HAS_UNSLOTH and self.w1.is_cuda

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C)
        Returns:
            out:      (B, T, C)
            aux_loss: scalar
        """
        B, T, C = x.shape
        N = B * T
        x_flat = x.reshape(N, C)                                   # (N, C)

        expert_weights, expert_indices, aux_loss = self.router(x_flat)

        if self.use_unsloth:
            out_flat = self._forward_unsloth(x_flat, expert_weights, expert_indices)
        else:
            out_flat = _moe_forward_pytorch(
                x_flat, self.w1, self.w2, expert_weights, expert_indices, self.num_experts
            )

        return out_flat.reshape(B, T, C), aux_loss

    def _get_kernel_configs(self, dtype: torch.dtype):
        """Lazily fetch (and cache) autotuned kernel configs for this layer shape."""
        if self._kernel_config_fwd is None and _autotune is not None:
            try:
                (
                    self._kernel_config_fwd,
                    self._kernel_config_bwd_dX,
                    self._kernel_config_bwd_dW,
                ) = _autotune(
                    num_experts      = self.num_experts,
                    hidden_dim       = self.n_embd,       # input dim (C)
                    intermediate_dim = self.hidden_dim,   # expert hidden dim (H)
                    top_k            = self.top_k,
                    dtype            = dtype,
                )
            except Exception:
                pass   # fall through to autotune=True per-call default
        return self._kernel_config_fwd, self._kernel_config_bwd_dX, self._kernel_config_bwd_dW

    def _forward_unsloth(
        self,
        x_flat:         torch.Tensor,   # (N, C)
        expert_weights: torch.Tensor,   # (N, top_k)
        expert_indices: torch.Tensor,   # (N, top_k) int64
    ) -> torch.Tensor:
        """
        Fused Triton dispatch via unsloth grouped_gemm.

        Flow:
            1. get_routing_indices → m_sizes, gather_indices
            2. 1st GEMM (permute_x=True):  (N, C) → (N*K, H) in expert order
            3. GELU
            4. 2nd GEMM (permute_y=True, fuse_mul_post=True):
                         (N*K, H) → (N, C) unpermuted and weighted in one pass
        """
        topk_ids = expert_indices.flatten().to(torch.int64)
        m_sizes, gather_indices = _get_routing_indices(topk_ids, self.num_experts)
        topk_weights_flat = expert_weights.flatten()   # (N*top_k,)

        cfg_fwd, cfg_bwd_dX, cfg_bwd_dW = self._get_kernel_configs(x_flat.dtype)

        # 1st GEMM: (N, C) → (N*K, H), permuting tokens into expert order
        intermediate = _grouped_gemm(
            X              = x_flat,
            W              = self.w1,
            m_sizes        = m_sizes,
            topk           = self.top_k,
            gather_indices = gather_indices,
            permute_x      = True,
            permute_y      = False,
            is_first_gemm  = True,
            kernel_config_fwd  = cfg_fwd,
            kernel_config_bwd_dX = cfg_bwd_dX,
            kernel_config_bwd_dW = cfg_bwd_dW,
        )   # (N * top_k, H) in expert-sorted order

        intermediate = F.gelu(intermediate)

        # 2nd GEMM: (N*K, H) → (N, C), unpermuting + weighted scatter-add fused
        out = _grouped_gemm(
            X              = intermediate,
            W              = self.w2,
            m_sizes        = m_sizes,
            topk           = self.top_k,
            gather_indices = gather_indices,
            permute_x      = False,
            permute_y      = True,
            topk_weights   = topk_weights_flat,
            fuse_mul_post  = True,
            is_first_gemm  = False,
            kernel_config_fwd  = cfg_fwd,
            kernel_config_bwd_dX = cfg_bwd_dX,
            kernel_config_bwd_dW = cfg_bwd_dW,
        )   # (N, C)

        return out
