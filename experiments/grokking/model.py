"""The paper's transformer, as specified in Power et al. 2022 Appendix A.1.2:

    "We trained a standard decoder-only transformer (Vaswani et al. 2017) with
     causal attention masking, and calculated loss and accuracy only on the
     answer part of the equation. For all experiments we used a transformer
     with 2 layers, width 128, and 4 attention heads, with a total of about
     4 * 10^5 non-embedding parameters."

``GrokTransformer(...).n_params_non_embedding`` reproduces 393,216 for the
default configuration, matching that figure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        self.act = F.gelu(self.fc(x))  # retained for the sparsity probe
        return self.proj(self.act)


class Block(nn.Module):
    """Pre-LN block. ``kind`` selects the token-mixing primitive:
    ``attn`` = softmax causal attention, ``gdn`` = Gated DeltaNet (see gdn.py)."""

    def __init__(self, n_embd: int, n_head: int, kind: str = "attn"):
        super().__init__()
        self.kind = kind
        self.ln1 = nn.LayerNorm(n_embd)
        if kind == "attn":
            self.attn = CausalSelfAttention(n_embd, n_head)
        elif kind == "gdn":
            from gdn import GatedDeltaNetLayer
            self.attn = GatedDeltaNetLayer(n_embd, n_head)
        else:
            raise ValueError(f"unknown layer kind {kind!r}")
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GrokTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int = 2, n_embd: int = 128,
                 n_head: int = 4, seq_len: int = 4, layer_types: str = ""):
        super().__init__()
        self.seq_len = seq_len
        # layer_types: comma-separated per-layer kinds, e.g. "attn,gdn".
        # Empty string = all attention (the paper's architecture).
        kinds = [k.strip() for k in layer_types.split(",") if k.strip()] or \
                ["attn"] * n_layer
        assert len(kinds) == n_layer, \
            f"layer_types has {len(kinds)} entries but n_layer={n_layer}"
        self.layer_types = kinds
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, k) for k in kinds)
        self.ln_f = nn.LayerNorm(n_embd)
        self.unembed = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """idx: (B, T) -> logits (B, vocab). Only the final position is scored,
        which is the paper's "loss on the answer part of the equation only"."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.unembed(self.ln_f(x[:, -1]))

    @property
    def n_params_non_embedding(self) -> int:
        emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        emb += self.unembed.weight.numel()
        return sum(p.numel() for p in self.parameters()) - emb

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = GrokTransformer(vocab_size=99)
    print(f"non-embedding params: {m.n_params_non_embedding:,} "
          f"(paper: ~4e5)   total: {m.n_params:,}")
    out = m(torch.randint(0, 99, (8, 4)))
    print("logits", tuple(out.shape))
