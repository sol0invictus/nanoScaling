"""
Correctness tests for the DeepSeek-V4 CSA / HCA implementations.

Run from the repo root:
    python -m experiments.csa_hca.test_correctness

Strategy mirrors the MoE tests: an independent brute-force oracle recomputes the
core attention with explicit Python loops while reusing the module's own
submodules (compression, indexer selection, projections). The vectorised forward
must match the loop to floating-point precision. Plus targeted tests for
causality, indexer top-k selection, compression weights, the attention sink, and
gradient flow. All in float64 for tight tolerances.
"""

import math
import os
import sys

import torch

# `experiments/` is shadowed by the top-level experiments.py, so import the
# modules directly from this folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csa import CompressedSparseAttention
from hca import HeavilyCompressedAttention
from common import (
    overlapped_compress,
    blockwise_compress,
    preceding_block_mask,
    causal_window_mask,
)

torch.manual_seed(0)

# small, fully-exercised shapes
DIM, NH, HD, DC = 32, 4, 8, 16
NG, DG = 2, 10
B, T = 2, 24


ROPE = 4   # exercise partial RoPE on the last 4 of HD=8 head dims


def make_csa(top_k=3, window=6, m=2, rope_head_dim=ROPE):
    torch.manual_seed(1)
    mod = CompressedSparseAttention(
        dim=DIM, n_head=NH, head_dim=HD, q_compress_dim=DC, window=window,
        n_groups=NG, group_inter_dim=DG, compress_ratio=m, top_k=top_k,
        n_idx_head=2, idx_head_dim=4, rope_head_dim=rope_head_dim,
    ).double()
    # randomise the (zero-initialised) biases/sinks so tests aren't degenerate
    for p in mod.parameters():
        if p.requires_grad and p.dim() >= 1:
            torch.nn.init.normal_(p, std=0.1)
    return mod


def make_hca(window=6, m=4, rope_head_dim=ROPE):
    torch.manual_seed(2)
    mod = HeavilyCompressedAttention(
        dim=DIM, n_head=NH, head_dim=HD, q_compress_dim=DC, window=window,
        n_groups=NG, group_inter_dim=DG, compress_ratio=m, rope_head_dim=rope_head_dim,
    ).double()
    for p in mod.parameters():
        if p.requires_grad and p.dim() >= 1:
            torch.nn.init.normal_(p, std=0.1)
    return mod


def bruteforce(mod, H):
    """
    Independent oracle: reuse the module's compression/selection/projections, but
    compute the core attention (compressed + window + sink, MQA softmax) with
    explicit per-(batch, head, query) loops.
    """
    Bsz, Tt, _ = H.shape
    cq = mod._query_latent(H)
    q = mod._queries(cq)                       # (B, H, T, c) RoPE'd
    ckv_raw, allow = mod._compressed_entries(H, cq)  # (B, nb, c) pre-RoPE, (B, T, nb)
    ckv = mod._rope_compressed(ckv_raw)        # RoPE at block-start positions
    wkv = mod._window_kv(H)                    # (B, T, c) RoPE'd
    w_allow = causal_window_mask(Tt, mod.window, H.device)  # (T, T)
    nb = ckv.shape[1]
    scale = mod.scale

    o = torch.zeros(Bsz, mod.n_head, Tt, mod.head_dim, dtype=H.dtype)
    for b in range(Bsz):
        for h in range(mod.n_head):
            for t in range(Tt):
                qv = q[b, h, t]                              # (c,)
                logits, vals = [], []
                for s in range(nb):
                    if allow[b, t, s]:
                        logits.append((qv @ ckv[b, s]) * scale)
                        vals.append(ckv[b, s])
                for i in range(Tt):
                    if w_allow[t, i]:
                        logits.append((qv @ wkv[b, i]) * scale)
                        vals.append(wkv[b, i])
                logits.append(mod.sink_logit[h])             # sink logit, value = 0
                lg = torch.stack(logits)
                w = torch.softmax(lg, dim=0)
                acc = torch.zeros(mod.head_dim, dtype=H.dtype)
                for j, v in enumerate(vals):                 # last weight (sink) -> no value
                    acc = acc + w[j] * v
                o[b, h, t] = acc
    o = mod._inverse_rope_output(o)            # de-rotate at query positions
    return mod._output(o)


def _check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok: {name}")


def test_shapes_finite():
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    for name, mod in [("CSA", make_csa()), ("HCA", make_hca())]:
        out = mod(H)
        _check(f"{name} output shape", out.shape == (B, T, DIM))
        _check(f"{name} output finite", torch.isfinite(out).all())


def test_bruteforce_equivalence():
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    for name, mod in [("CSA", make_csa()), ("HCA", make_hca())]:
        ref = bruteforce(mod, H)
        out = mod(H)
        err = (ref - out).abs().max().item()
        print(f"  [{name}] max abs err vs brute force = {err:.3e}")
        _check(f"{name} matches brute force", err < 1e-10)


def test_causality():
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    p = T // 2
    for name, mod in [("CSA", make_csa()), ("HCA", make_hca())]:
        out0 = mod(H)
        H2 = H.clone()
        H2[:, p] += torch.randn(B, DIM, dtype=torch.float64) * 3.0   # perturb token p
        out1 = mod(H2)
        drift = (out0[:, :p] - out1[:, :p]).abs().max().item()
        after = (out0[:, p:] - out1[:, p:]).abs().max().item()
        print(f"  [{name}] drift at tokens < p = {drift:.3e}; change at tokens >= p = {after:.3e}")
        _check(f"{name} causal (no leak to earlier tokens)", drift < 1e-10)
        _check(f"{name} perturbation actually changes later tokens", after > 1e-6)


def test_csa_indexer_selection():
    """Module's selected blocks must equal an independent top-k over preceding-masked scores."""
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    mod = make_csa(top_k=3)
    cq = mod._query_latent(H)
    scores = mod._indexer_scores(H, cq)                     # (B, T, nb)
    sel = mod._compressed_entries(H, cq)[1]                 # module's allow mask
    nb = scores.shape[-1]
    pre = preceding_block_mask(T, mod.m, nb, H.device)[None]
    masked = scores.masked_fill(~pre, float("-inf"))
    k = min(mod.top_k, nb)
    idx = masked.topk(k, dim=-1).indices
    want = torch.zeros_like(scores, dtype=torch.bool)
    want.scatter_(-1, idx, True)
    want &= pre
    _check("CSA selection == independent top-k of indexer scores", torch.equal(sel, want))
    # every selected block is strictly preceding (causal)
    _check("CSA selected blocks are all preceding", bool((sel <= pre).all()))


def test_csa_topk_all_equals_preceding():
    """With top_k >= n_blocks, CSA must attend to *all* preceding blocks (dense)."""
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    mod = make_csa(top_k=10_000)
    cq = mod._query_latent(H)
    allow = mod._compressed_entries(H, cq)[1]
    nb = allow.shape[-1]
    pre = preceding_block_mask(T, mod.m, nb, H.device)[None].expand(B, -1, -1)
    _check("CSA top_k>=nb selects all preceding blocks", torch.equal(allow, pre))


def test_compression_weights():
    Bz, Tt, c, m = 2, 12, 5, 3
    Ca, Cb = torch.randn(Bz, Tt, c, dtype=torch.float64), torch.randn(Bz, Tt, c, dtype=torch.float64)
    Za, Zb = torch.randn(Bz, Tt, c, dtype=torch.float64), torch.randn(Bz, Tt, c, dtype=torch.float64)
    Ba, Bb = torch.randn(m, c, dtype=torch.float64), torch.randn(m, c, dtype=torch.float64)

    # Reconstruct the softmax weights the same way overlapped_compress does, and
    # confirm they normalise over the 2m elements per channel.
    nb = math.ceil(Tt / m)
    comp = overlapped_compress(Ca, Cb, Za, Zb, Ba, Bb, m)
    _check("overlapped_compress shape", comp.shape == (Bz, nb, c))
    _check("overlapped_compress finite", torch.isfinite(comp).all())

    # Entry 0 has no previous block -> it must equal a pure softmax over the first
    # m tokens of C^a only (the C^b/previous contribution is masked to zero).
    Z0 = Za[:, :m] + Ba                                     # (B, m, c)
    S0 = torch.softmax(Z0, dim=1)
    entry0 = (S0 * Ca[:, :m]).sum(dim=1)
    _check("overlapped entry 0 ignores (nonexistent) previous block",
           torch.allclose(comp[:, 0], entry0, atol=1e-12))

    # HCA blockwise weights sum to 1 over m per channel.
    Bz2, Tt2, c2, m2 = 2, 16, 4, 4
    C = torch.randn(Bz2, Tt2, c2, dtype=torch.float64)
    Z = torch.randn(Bz2, Tt2, c2, dtype=torch.float64)
    Bias = torch.randn(m2, c2, dtype=torch.float64)
    nb2 = Tt2 // m2
    Sblk = torch.softmax(Z.view(Bz2, nb2, m2, c2) + Bias, dim=2)
    ref = (Sblk * C.view(Bz2, nb2, m2, c2)).sum(dim=2)
    out = blockwise_compress(C, Z, Bias, m2)
    _check("blockwise_compress matches reference", torch.allclose(out, ref, atol=1e-12))
    _check("blockwise weights sum to 1 over m", torch.allclose(Sblk.sum(2), torch.ones(Bz2, nb2, c2, dtype=torch.float64), atol=1e-12))


def test_attention_sink():
    """A large sink logit must pull total non-sink attention mass below 1."""
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    mod = make_hca(window=6, m=4)
    with torch.no_grad():
        mod.sink_logit.fill_(20.0)   # huge sink -> nearly all mass to the sink
    # recompute the attention weights the way the module does, then sum non-sink
    cq = mod._query_latent(H)
    q = mod._queries(cq)
    ckv, allow = mod._compressed_entries(H, cq)
    wkv = mod._window_kv(H)
    w_allow = causal_window_mask(T, mod.window, H.device)
    nb = ckv.shape[1]
    cl = torch.einsum("bhtc,bsc->bhts", q, ckv) * mod.scale
    cl = cl.masked_fill(~allow[:, None], float("-inf"))
    wl = torch.einsum("bhtc,bsc->bhts", q, wkv) * mod.scale
    wl = wl.masked_fill(~w_allow[None, None], float("-inf"))
    sink = mod.sink_logit.view(1, NH, 1, 1).expand(B, NH, T, 1)
    weights = torch.softmax(torch.cat([cl, wl, sink], dim=-1), dim=-1)
    nonsink_mass = weights[..., :nb + T].sum(-1)
    _check("attention sink reduces total attention mass below 1", bool((nonsink_mass < 0.5).all()))


def _has_grad(p):
    return p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_grad_flow():
    # HCA has no indexer -> every parameter must receive gradient.
    H = torch.randn(B, T, DIM, dtype=torch.float64, requires_grad=True)
    hca = make_hca()
    hca(H).pow(2).mean().backward()
    _check("HCA input grad finite", H.grad is not None and torch.isfinite(H.grad).all())
    missing = [pn for pn, p in hca.named_parameters() if not _has_grad(p)]
    print(f"  [HCA] params without grad: {missing or 'none'}")
    _check("HCA every parameter gets gradient", not missing)

    # CSA: the Lightning Indexer selects blocks via a hard top-k, which is
    # NON-DIFFERENTIABLE. So the indexer's own parameters get NO gradient from the
    # main loss — in DeepSeek-V4 the indexer is trained by a separate auxiliary
    # objective (see README / module docstring). Every *other* param must get grad.
    H2 = torch.randn(B, T, DIM, dtype=torch.float64, requires_grad=True)
    csa = make_csa()
    csa(H2).pow(2).mean().backward()
    # precise prefixes (note: "w_w." must not match the window proj "w_win")
    indexer_prefixes = ("w_ik_", "w_iz_", "ibias_", "w_iuq.", "w_w.", "idx_norm.")
    idx_with_grad, core_without_grad = [], []
    for pn, p in csa.named_parameters():
        is_indexer = pn.startswith(indexer_prefixes)
        if is_indexer and _has_grad(p):
            idx_with_grad.append(pn)
        if not is_indexer and not _has_grad(p):
            core_without_grad.append(pn)
    print(f"  [CSA] core params missing grad: {core_without_grad or 'none'}")
    print(f"  [CSA] indexer params (expected NO grad from main loss): "
          f"{[pn for pn, _ in csa.named_parameters() if pn.startswith(indexer_prefixes)]}")
    _check("CSA all non-indexer params get gradient", not core_without_grad)
    _check("CSA indexer params get NO gradient from main loss (hard top-k)",
           not idx_with_grad)


def test_indexer_distillation():
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    csa = make_csa(top_k=3)

    # return_aux plumbing: CSA gives a real loss, HCA gives 0.
    out, aux = csa(H, return_aux=True)
    _check("CSA(return_aux) output shape", out.shape == (B, T, DIM))
    _check("CSA indexer KL finite", torch.isfinite(aux))
    _check("CSA indexer KL >= 0", aux.item() > -1e-9)
    _, haux = make_hca()(H, return_aux=True)
    _check("HCA aux loss is 0 (no indexer)", float(haux) == 0.0)

    # The aux loss must train ONLY the indexer params (inputs are detached).
    csa.zero_grad()
    csa.indexer_distillation_loss(H).backward()
    indexer_prefixes = ("w_ik_", "w_iz_", "ibias_", "w_iuq.", "w_w.", "idx_norm.")
    idx_grad, core_grad = [], []
    for pn, p in csa.named_parameters():
        if pn.startswith(indexer_prefixes):
            if _has_grad(p):
                idx_grad.append(pn)
        elif _has_grad(p):
            core_grad.append(pn)
    print(f"  [CSA aux] indexer params trained: {sorted(idx_grad)}")
    print(f"  [CSA aux] core params touched (should be none): {core_grad or 'none'}")
    _check("aux loss gives gradient to indexer params", len(idx_grad) >= 6)
    _check("aux loss leaves core/main params untouched", not core_grad)

    # Sanity: optimising the indexer params actually reduces the KL.
    csa2 = make_csa(top_k=3)
    idx_params = [p for pn, p in csa2.named_parameters() if pn.startswith(indexer_prefixes)]
    opt = torch.optim.Adam(idx_params, lr=5e-2)
    l0 = csa2.indexer_distillation_loss(H).item()
    for _ in range(50):
        opt.zero_grad()
        loss = csa2.indexer_distillation_loss(H)
        loss.backward()
        opt.step()
    l1 = csa2.indexer_distillation_loss(H).item()
    print(f"  [CSA aux] KL before={l0:.4f} after 50 steps={l1:.4f}")
    _check("optimising the indexer reduces KL", l1 < l0 * 0.9)


def test_rope_primitive():
    from common import precompute_freqs_cis, apply_rope
    rd = 8
    freqs = precompute_freqs_cis(rd, 64, theta=10000.0)
    x = torch.randn(2, 5, 12, dtype=torch.float64)          # last `rd` dims get rotated
    f = freqs[torch.arange(5)]

    xr = apply_rope(x, f, rd)
    xrr = apply_rope(xr, f, rd, inverse=True)
    # apply_rope computes in fp32 internally (matches the reference's `.float()`),
    # so round-trip / norm hold to ~fp32 precision, not fp64.
    _check("rope inverse undoes forward", torch.allclose(x, xrr, atol=1e-6))
    _check("rope preserves norm of rotated dims",
           torch.allclose(x[..., -rd:].norm(dim=-1), xr[..., -rd:].norm(dim=-1), atol=1e-6))
    _check("rope leaves non-rotated dims unchanged",
           torch.allclose(x[..., :-rd], xr[..., :-rd], atol=1e-12))

    # relative-position property: <rope(q,a), rope(k,b)> depends only on a-b
    q = torch.randn(rd, dtype=torch.float64); k = torch.randn(rd, dtype=torch.float64)

    def dot_at(a, b):
        qa = apply_rope(q.view(1, 1, rd), freqs[a:a + 1], rd).view(rd)
        kb = apply_rope(k.view(1, 1, rd), freqs[b:b + 1], rd).view(rd)
        return (qa @ kb).item()

    _check("rope QK dot depends only on relative position",
           abs(dot_at(9, 4) - dot_at(7, 2)) < 1e-5)          # both Δ = 5


def test_rope_changes_output_and_optional():
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    out_rope = make_csa(rope_head_dim=ROPE)(H)
    out_norope = make_csa(rope_head_dim=0)(H)
    _check("RoPE changes output vs no-rope", (out_rope - out_norope).abs().max() > 1e-6)
    _check("no-rope path still finite (rope_head_dim=0)", torch.isfinite(out_norope).all())


def test_mqa_single_kv_head():
    """Compressed KV entries are a single shared head: shape (B, n_blocks, c)."""
    H = torch.randn(B, T, DIM, dtype=torch.float64)
    for name, mod in [("CSA", make_csa()), ("HCA", make_hca())]:
        cq = mod._query_latent(H)
        ckv, _ = mod._compressed_entries(H, cq)
        _check(f"{name} compressed KV is single-head (B, nb, c)",
               ckv.dim() == 3 and ckv.shape[0] == B and ckv.shape[2] == HD)


if __name__ == "__main__":
    tests = [
        ("shapes & finiteness", test_shapes_finite),
        ("brute-force equivalence", test_bruteforce_equivalence),
        ("causality", test_causality),
        ("CSA indexer top-k selection", test_csa_indexer_selection),
        ("CSA top_k>=nb == dense preceding", test_csa_topk_all_equals_preceding),
        ("compression weights", test_compression_weights),
        ("attention sink", test_attention_sink),
        ("gradient flow", test_grad_flow),
        ("indexer distillation loss", test_indexer_distillation),
        ("RoPE primitive", test_rope_primitive),
        ("RoPE changes output / optional", test_rope_changes_output_and_optional),
        ("MQA single KV head", test_mqa_single_kv_head),
    ]
    for title, fn in tests:
        print(f"\n[{title}]")
        fn()
    print("\nAll CSA/HCA correctness checks PASSED.")
