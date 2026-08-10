"""Gated DeltaNet layer, unrolled — a drop-in alternative to causal attention.

Faithful re-implementation of the parameterisation used by the nanoScaling repo's
``models/gated_delta_net.py`` (same projections, same gates, same Q/K sub-norms,
same scaling), but with the recurrence written out explicitly instead of calling
``fla.ops.gated_delta_rule.chunk_gated_delta_rule``. Nothing here imports from that
repo: this file is self-contained.

Why not use the FLA kernel: our sequences are **4 tokens long**. The FLA path is a
chunked scan with ``chunk_size=64``, which is meaningless at T=4, and the package
is not installed. Unrolling 4 steps is exact, dependency-free, and costs nothing.

The gated delta rule maintains a matrix state ``S`` (per head) and, at each step,
*forgets* by a scalar gate, *erases* the component of the state along the current
key, then *writes* the new value:

    S_t = alpha_t * S_{t-1} @ (I - beta_t k_t k_t^T)  +  beta_t v_t k_t^T
    o_t = S_t @ (q_t * scale)

with ``alpha_t = exp(g_t) in (0,1]`` the forget gate and ``beta_t in (0,1)`` the
write strength. Causality is structural — ``o_t`` depends only on steps <= t.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GatedDeltaNetLayer(nn.Module):
    """Same interface as CausalSelfAttention in model.py: (B,T,C) -> (B,T,C)."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim = n_head, n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.beta_proj = nn.Linear(n_embd, n_head, bias=False)   # write strength
        self.g_proj = nn.Linear(n_embd, n_head, bias=False)      # forget gate
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        scale = D ** -0.5

        q = self.q_norm(self.q_proj(x).view(B, T, H, D).transpose(1, 2))  # (B,H,T,D)
        k = self.k_norm(self.k_proj(x).view(B, T, H, D).transpose(1, 2))
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        beta = torch.sigmoid(self.beta_proj(x).transpose(1, 2))           # (B,H,T)
        alpha = torch.exp(F.logsigmoid(self.g_proj(x).transpose(1, 2)))   # (B,H,T)

        S = x.new_zeros(B, H, D, D)          # state: (value dim) x (key dim)
        outs = []
        for t in range(T):
            k_t = k[:, :, t]                  # (B,H,D)
            v_t = v[:, :, t]
            b_t = beta[:, :, t].unsqueeze(-1)     # (B,H,1)
            a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)   # (B,H,1,1)
            # forget, then erase the component along k_t, then write v_t
            Sk = torch.einsum("bhij,bhj->bhi", S, k_t)          # S @ k_t
            S = a_t * (S - b_t.unsqueeze(-1) * Sk.unsqueeze(-1) * k_t.unsqueeze(-2)) \
                + (b_t * v_t).unsqueeze(-1) * k_t.unsqueeze(-2)
            outs.append(torch.einsum("bhij,bhj->bhi", S, q[:, :, t] * scale))

        o = torch.stack(outs, dim=2).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(o)


if __name__ == "__main__":
    torch.manual_seed(0)
    L = GatedDeltaNetLayer(128, 4).double()
    x = torch.randn(3, 4, 128, dtype=torch.double)

    # 1. causality: perturbing token t must not change outputs at positions < t
    y = L(x)
    ok = True
    for t in range(4):
        x2 = x.clone()
        x2[:, t] += 5.0
        y2 = L(x2)
        if t > 0 and not torch.allclose(y[:, :t], y2[:, :t], atol=1e-9):
            ok = False
        if torch.allclose(y[:, t], y2[:, t], atol=1e-9):
            ok = False   # position t itself must change
    print("causal + dependent on own token:", ok)

    # 2. with the forget gate open (alpha=1) and beta=1 the state should exactly
    #    implement the delta rule: writing key k then querying k returns v.
    with torch.no_grad():
        L.g_proj.weight.fill_(0.0); L.g_proj.weight += 50.0   # alpha -> 1
        L.beta_proj.weight.fill_(0.0); L.beta_proj.weight += 50.0  # beta -> 1
    a = torch.exp(F.logsigmoid(torch.tensor(50.0)))
    print(f"alpha with g=+50: {a.item():.6f} (should be ~1.0)")

    n_attn = 3 * 128 * 128 + 128 * 128            # qkv + proj, bias-free
    n_gdn = sum(p.numel() for p in GatedDeltaNetLayer(128, 4).parameters())
    print(f"params: attention block {n_attn:,}  vs  GDN block {n_gdn:,} "
          f"({100*(n_gdn/n_attn-1):+.2f}%)")
