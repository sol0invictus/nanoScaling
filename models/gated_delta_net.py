"""
Gated Delta Net layer using the chunked gated delta rule from the
flash-linear-attention (FLA) package.

Paper: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
       https://arxiv.org/abs/2412.06464

Requirements:
    pip install flash-linear-attention

The recurrent state update rule is:
    S_t = exp(g_t) ⊙ S_{t-1}  +  β_t * (v_t - S_{t-1} k_t) ⊗ k_t
    o_t = S_t q_t

where:
    g_t  = log_sigmoid(g_proj(x_t))   — per-head log forget gate
    β_t  = sigmoid(β_proj(x_t))       — per-head delta update rate
    Q, K are L2-normalised (per-head RMSNorm) for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    chunk_gated_delta_rule = None


class GatedDeltaNetLayer(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention that uses the Gated Delta Rule
    (linear attention with forgetting and selective writing).

    Compatible with the Block.forward() interface:
        out = layer(x, attn_mask=None)   ->  (B, T, C)
    The `attn_mask` argument is accepted but ignored — GDN is inherently causal
    via its recurrent formulation.

    Config fields consumed (all optional with defaults):
        n_embd              — model dimension (required, inherited from GPTConfig)
        n_head              — number of heads (required, inherited from GPTConfig)
        bias                — use bias in linear projections (inherited)
        dropout             — residual dropout probability (inherited)
        delta_net_chunk_size — chunk size for the FLA chunked scan (default: 64)
    """

    def __init__(self, config):
        super().__init__()
        if not HAS_FLA:
            raise ImportError(
                "flash-linear-attention (fla) is required for GatedDeltaNetLayer.\n"
                "Install with:  pip install flash-linear-attention"
            )

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        self.head_dim = self.n_embd // self.n_head
        self.chunk_size = getattr(config, 'delta_net_chunk_size', 64)

        bias = config.bias

        # Standard Q / K / V projections (same parameter budget as MHA)
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)

        # Per-head scalar projections — one scalar per head per token
        self.beta_proj = nn.Linear(self.n_embd, self.n_head, bias=bias)  # update rate
        self.g_proj    = nn.Linear(self.n_embd, self.n_head, bias=bias)  # forget gate

        # Output projection
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Sub-norms on Q and K improve training stability in linear attention
        # (equivalent to the qk_norm option in many modern implementations)
        from models.gpt import RMSNorm as _RMSNorm
        self.q_norm = _RMSNorm(self.head_dim)
        self.k_norm = _RMSNorm(self.head_dim)

    # ------------------------------------------------------------------
    def forward(self, x, attn_mask=None):
        """
        Args:
            x         : (B, T, C)  — input tensor
            attn_mask : ignored    — GDN is inherently causal

        Returns:
            (B, T, C)
        """
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        # ── Projections ────────────────────────────────────────────────
        # Reshape to (B, H, T, D) as expected by FLA with head_first=True
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # L2-normalise Q and K across the head dimension for stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ── Scalar gates ───────────────────────────────────────────────
        # beta: per-head update rate ∈ (0, 1), shape (B, H, T)
        beta = torch.sigmoid(self.beta_proj(x).transpose(1, 2))

        # g: per-head log forget gate ∈ (-∞, 0], shape (B, H, T)
        # FLA expects log-space so that it can use exp(g) = sigmoid(raw)
        g = F.logsigmoid(self.g_proj(x).transpose(1, 2))

        # ── Chunked gated delta rule (FLA kernel) ──────────────────────
        o, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            scale=D ** -0.5,
            initial_state=None,
            output_final_state=False,
            chunk_size=self.chunk_size,
            head_first=True,
        )  # o: (B, H, T, D)

        # ── Merge heads and project ────────────────────────────────────
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(o))
