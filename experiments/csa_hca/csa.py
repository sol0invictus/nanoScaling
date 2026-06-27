"""
Compressed Sparse Attention (CSA) — DeepSeek-V4 technical report, Sec. 2.3.1.

CSA = overlapped KV compression (1 entry per m tokens) + a Lightning Indexer that
selects the top-k compressed blocks per query (DeepSeek Sparse Attention), plus a
sliding-window branch for local detail, all fused into one MQA softmax.

V4-Pro settings: m = 4, top-k = 1024, n_idx_head = 64, idx_head_dim = 128,
n_head = 128, head_dim = 512, q_compress_dim = 1536, n_win = 128.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # works both as a package and as standalone scripts on sys.path
    from .common import (
        HybridCompressedAttentionBase,
        RMSNorm,
        apply_rope,
        overlapped_compress,
        preceding_block_mask,
    )
except ImportError:
    from common import (
        HybridCompressedAttentionBase,
        RMSNorm,
        apply_rope,
        overlapped_compress,
        preceding_block_mask,
    )


class CompressedSparseAttention(HybridCompressedAttentionBase):
    def __init__(
        self,
        dim: int,
        n_head: int = 8,
        head_dim: int = 32,
        q_compress_dim: int = 64,
        window: int = 128,
        n_groups: int = 2,
        group_inter_dim: int = 64,
        compress_ratio: int = 4,          # m
        top_k: int = 16,                  # number of compressed blocks kept per query
        n_idx_head: int = 4,              # n^I_h
        idx_head_dim: int = 16,           # c_I
        **kwargs,                         # rope_head_dim, rope_theta, ... -> base
    ):
        super().__init__(
            dim=dim, n_head=n_head, head_dim=head_dim, q_compress_dim=q_compress_dim,
            window=window, n_groups=n_groups, group_inter_dim=group_inter_dim,
            compress_ratio=compress_ratio, **kwargs,
        )
        assert self.rope_head_dim <= idx_head_dim, "rope_head_dim must be <= idx_head_dim"
        self.top_k = top_k
        self.n_idx_head = n_idx_head
        self.idx_head_dim = idx_head_dim
        c = head_dim

        # --- overlapped KV compression (two value series + two weight series) ---
        self.w_kv_a = nn.Linear(dim, c, bias=False)
        self.w_kv_b = nn.Linear(dim, c, bias=False)
        self.w_z_a = nn.Linear(dim, c, bias=False)
        self.w_z_b = nn.Linear(dim, c, bias=False)
        self.bias_a = nn.Parameter(torch.zeros(compress_ratio, c))   # B^a
        self.bias_b = nn.Parameter(torch.zeros(compress_ratio, c))   # B^b

        # --- Lightning Indexer ---
        # Indexer keys are produced by the SAME overlapped-compression procedure,
        # but on their own (low-dim) feature/weight projections.
        ci = idx_head_dim
        self.w_ik_a = nn.Linear(dim, ci, bias=False)
        self.w_ik_b = nn.Linear(dim, ci, bias=False)
        self.w_iz_a = nn.Linear(dim, ci, bias=False)
        self.w_iz_b = nn.Linear(dim, ci, bias=False)
        self.ibias_a = nn.Parameter(torch.zeros(compress_ratio, ci))
        self.ibias_b = nn.Parameter(torch.zeros(compress_ratio, ci))
        # indexer queries (low-rank, from the shared latent c^Q) + per-head weights
        self.w_iuq = nn.Linear(q_compress_dim, n_idx_head * idx_head_dim, bias=False)
        self.w_w = nn.Linear(dim, n_idx_head, bias=False)
        self.idx_norm = RMSNorm(idx_head_dim, self.eps)   # indexer's compressor norm
        self.idx_scale = idx_head_dim ** -0.5

    # -- indexer scoring -------------------------------------------------------

    def _indexer_keys(self, H):
        """Normalised + RoPE'd compressed indexer keys (B, n_blocks, c_I)."""
        Ia = self.w_ik_a(H); Ib = self.w_ik_b(H)
        Za = self.w_iz_a(H); Zb = self.w_iz_b(H)
        kI = self.idx_norm(overlapped_compress(Ia, Ib, Za, Zb, self.ibias_a, self.ibias_b, self.m))
        if self.rope_head_dim:                                                # rope at block-start positions
            nb = kI.shape[1]
            pos = torch.arange(nb, device=kI.device) * self.m
            kI = apply_rope(kI, self.freqs_cis[pos], self.rope_head_dim)
        return kI

    def _indexer_scores(self, H, cq):
        """I[b, t, s] = sum_h w^I_{t,h} * ReLU(q^I_{t,h} . K^IComp_s)  -> (B, T, n_blocks)."""
        B, T, _ = H.shape
        kI = self._indexer_keys(H)                                            # (B, nb, c_I)
        qI = self.w_iuq(cq).view(B, T, self.n_idx_head, self.idx_head_dim)
        qI = qI.transpose(1, 2)                                               # (B, n_ih, T, c_I)
        if self.rope_head_dim:                                                # rope at query positions
            qI = apply_rope(qI, self.freqs_cis[:T], self.rope_head_dim)
        wI = self.w_w(H).transpose(1, 2)                                      # (B, n_ih, T)
        dots = F.relu(torch.einsum("bhtc,bsc->bhts", qI, kI) * self.idx_scale)
        return torch.einsum("bht,bhts->bts", wI, dots)                       # (B, T, nb)

    def _select(self, scores, T):
        """Top-k over *preceding* compressed blocks. Returns (B, T, n_blocks) bool."""
        n_blocks = scores.shape[-1]
        pre = preceding_block_mask(T, self.m, n_blocks, scores.device)        # (T, nb)
        masked = scores.masked_fill(~pre[None], float("-inf"))
        k = min(self.top_k, n_blocks)
        idx = masked.topk(k, dim=-1).indices                                 # (B, T, k)
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, idx, True)
        return keep & pre[None]            # drop any -inf picks when fewer than k are preceding

    # -- compressed entries (overlapped) + sparse allow-mask -------------------

    def _compress_kv(self, H):
        """Overlapped, normalised compressed KV entries (B, n_blocks, c), pre-RoPE."""
        Ca = self.w_kv_a(H); Cb = self.w_kv_b(H)
        Za = self.w_z_a(H); Zb = self.w_z_b(H)
        ckv = overlapped_compress(Ca, Cb, Za, Zb, self.bias_a, self.bias_b, self.m)
        return self.compress_norm(ckv)

    def _compressed_entries(self, H, cq):
        ckv = self._compress_kv(H)                                          # (B, nb, c)
        scores = self._indexer_scores(H, cq)
        allow = self._select(scores, H.shape[1])
        return ckv, allow

    # -- Lightning Indexer training (the top-k is non-differentiable) ----------

    def indexer_distillation_loss(self, H, eps: float = 1e-9):
        """
        Train the Lightning Indexer to mimic the main attention's choice of which
        compressed blocks matter (DeepSeek Sparse Attention, V3.2 -> V4).

        Target  : the *dense* main-attention distribution over all preceding
                  compressed blocks, averaged over heads (detached). Computed as
                  if attention were dense — matching DeepSeek's recipe of warming
                  up the indexer under dense attention before going sparse.
        Student : softmax over the indexer scores for the same preceding blocks.
        Loss    : KL(target || student), mean over query tokens that have at least
                  one preceding block.

        Gradients are isolated to the indexer's own parameters: the inputs to the
        indexer (hidden states and the shared query latent) are detached, so this
        loss does not perturb the main model. Add it to your training objective as
            loss = main_loss + lambda * csa.indexer_distillation_loss(H)
        """
        B, T, _ = H.shape

        # --- target: dense attention distribution over preceding blocks ---
        with torch.no_grad():
            cq = self._query_latent(H)
            q = self._queries(cq)                                  # (B, Hh, T, c) RoPE'd
            ckv = self._rope_compressed(self._compress_kv(H))      # (B, nb, c) RoPE'd
            n_blocks = ckv.shape[1]
            pre = preceding_block_mask(T, self.m, n_blocks, H.device)  # (T, nb)
            logits = torch.einsum("bhtc,bsc->bhts", q, ckv) * self.scale
            logits = logits.masked_fill(~pre[None, None], float("-inf"))
            attn = torch.softmax(logits, dim=-1)                  # per-head dist over blocks
            target = torch.nan_to_num(attn.mean(dim=1), nan=0.0)  # (B, T, nb), invalid rows -> 0

        # --- student: indexer distribution; detach inputs so only indexer trains ---
        cq_d = self._query_latent(H.detach()).detach()
        scores = self._indexer_scores(H.detach(), cq_d)           # (B, T, nb)
        scores = scores.masked_fill(~pre[None], float("-inf"))
        logp = torch.log_softmax(scores, dim=-1)                  # (B, T, nb)

        # KL(target || student), guarding 0 * (-inf) at masked / zero-prob blocks
        term = target * (torch.log(target + eps) - logp)
        term = torch.where(pre[None] & (target > 0), term, torch.zeros_like(term))
        kl = term.sum(dim=-1)                                     # (B, T)

        valid = pre.any(dim=-1)                                   # (T,) tokens with >=1 preceding
        if not bool(valid.any()):
            return H.new_zeros(())
        return kl[:, valid].mean()

    def _aux_loss(self, H):
        return self.indexer_distillation_loss(H)
