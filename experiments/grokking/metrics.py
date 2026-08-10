"""Internal statistics for dissecting the grokking phase transition.

Two cadences:

* **cheap** (default every 25 steps) -- per-module weight and gradient norms,
  grad/weight alignment, update geometry, logit margins. All O(#params).
* **expensive** (default every 250 steps) -- singular spectra, effective and
  stable rank, Fourier "circularity" of the embedding, linear probes for group
  structure, activation sparsity. Involves SVDs and an extra forward pass.

Everything is appended to line-buffered CSVs under ``{out_dir}/logs`` so the
analysis notebook can be developed against partial results while runs are live.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _module_group(name: str) -> str:
    """Map a parameter name onto a coarse module group."""
    if name.startswith("tok_emb"):
        return "tok_emb"
    if name.startswith("pos_emb"):
        return "pos_emb"
    if name.startswith("unembed"):
        return "unembed"
    if "attn.qkv" in name:
        return "attn_qkv"
    if "attn.proj" in name:
        return "attn_proj"
    # Gated DeltaNet layers (gdn.py) use different names for the same roles,
    # so they land in the same groups and stay comparable with attention runs.
    if any(t in name for t in ("attn.q_proj", "attn.k_proj", "attn.v_proj")):
        return "attn_qkv"
    if "attn.o_proj" in name:
        return "attn_proj"
    if any(t in name for t in ("attn.beta_proj", "attn.g_proj")):
        return "attn_gate"
    if any(t in name for t in ("attn.q_norm", "attn.k_norm")):
        return "layernorm"
    if "mlp.fc" in name:
        return "mlp_fc"
    if "mlp.proj" in name:
        return "mlp_proj"
    if name.startswith("ln") or ".ln" in name:
        return "layernorm"
    return "other"


GROUPS = ["tok_emb", "pos_emb", "attn_qkv", "attn_proj", "attn_gate",
          "mlp_fc", "mlp_proj", "unembed", "layernorm", "other"]


def _spectrum_stats(W: torch.Tensor, topk: int = 32):
    """Singular-value summary of a 2-D weight matrix.

    stable rank    = ||W||_F^2 / ||W||_2^2      (soft rank, insensitive to tail)
    effective rank = exp(H(s / sum s))          (Roy & Vetterli entropy rank)
    """
    s = torch.linalg.svdvals(W.detach().float())
    fro2 = (s ** 2).sum()
    p = s / s.sum().clamp_min(1e-12)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return {
        "stable_rank": (fro2 / s[0].clamp_min(1e-12) ** 2).item(),
        "effective_rank": torch.exp(ent).item(),
        "s_max": s[0].item(),
        "s_sum": s.sum().item(),
        "topk": s[:topk].cpu().numpy(),
    }


def _fourier_concentration(rows: np.ndarray, top: int = 5):
    """How circular / Fourier-sparse is this set of embedding rows?

    ``rows`` are the embedding vectors of one cycle of a cyclic subgroup, in
    group order. If the network has learned a circular (Fourier) representation
    -- the structure Power et al. visualise in Fig. 3 and that later work
    identifies as the mechanism behind modular-arithmetic grokking -- the power
    spectrum over the cyclic index concentrates on a handful of frequencies.

    Returns the fraction of non-DC spectral power in the top-k frequencies
    (1.0 = perfectly sparse/circular, ~2k/n = white/memorised) plus the
    normalised entropy of the spectrum (low = structured).
    """
    X = rows - rows.mean(axis=0, keepdims=True)
    F = np.fft.rfft(X, axis=0)
    power = (np.abs(F) ** 2).sum(axis=1)
    power[0] = 0.0  # DC removed by centering; guard against numerical residue
    tot = power.sum()
    if tot <= 0:
        return {"top_mass": float("nan"), "spec_entropy": float("nan")}
    p = power / tot
    order = np.sort(p)[::-1]
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p) - 1)
    return {"top_mass": float(order[:top].sum()), "spec_entropy": float(ent)}


def _linear_probe_acc(X: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of a ridge-regression linear probe (leave-nothing-out, in-sample).

    Used to ask *when* a binary structural factor (e.g. the reflection bit of
    D_n) becomes linearly decodable from the symbol embedding.
    """
    Xc = np.concatenate([X - X.mean(0, keepdims=True),
                         np.ones((len(X), 1))], axis=1)
    t = 2.0 * y - 1.0
    w, *_ = np.linalg.lstsq(Xc.T @ Xc + 1e-3 * np.eye(Xc.shape[1]), Xc.T @ t,
                            rcond=None)
    return float(((Xc @ w > 0).astype(np.int64) == y).mean())


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector: 0 = uniform, 1 = one-hot."""
    x = np.sort(np.abs(x).ravel())
    n = len(x)
    if x.sum() <= 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


class _CSV:
    """Append-only, line-buffered CSV with a schema fixed by the first row."""

    def __init__(self, path: str):
        self.path, self.f, self.w = path, None, None

    def write(self, row: dict):
        if self.f is None:
            new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
            self.f = open(self.path, "a", buffering=1)
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            if new:
                self.w.writeheader()
        self.w.writerow({k: row.get(k, "") for k in self.w.fieldnames})

    def close(self):
        if self.f is not None:
            self.f.close()


# --------------------------------------------------------------------------- #
# the monitor
# --------------------------------------------------------------------------- #

class GrokkingMonitor:
    def __init__(self, model, out_dir: str, meta: dict, topk: int = 32):
        self.model, self.meta, self.topk = model, meta, topk
        self.log_dir = os.path.join(out_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv = {k: _CSV(os.path.join(self.log_dir, f"{k}.csv"))
                    for k in ("train", "weights", "grads", "spectral", "structure")}

        with torch.no_grad():
            self.init_norm = {g: self._group_norm(g) for g in GROUPS}
            self.prev_flat = self._flat_params()
        self.prev_delta = None

        # symbol ids only (exclude the <op> / <=> tokens) for structure probes
        self.n_sym = meta["n_symbols"]

    # -- parameter bookkeeping ------------------------------------------------
    def _named(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def _flat_params(self):
        return torch.cat([p.detach().reshape(-1) for _, p in self._named()])

    def _group_norm(self, group: str) -> float:
        t = [p.detach().pow(2).sum() for n, p in self._named()
             if _module_group(n) == group]
        return float(torch.stack(t).sum().sqrt()) if t else 0.0

    # -- cheap ---------------------------------------------------------------
    def log_cheap(self, step: int, scalars: dict):
        """Per-module weight/grad norms, alignment, and update geometry."""
        wrow = {"step": step}
        grow = {"step": step}
        w_tot = g_tot = dot_tot = 0.0

        for g in GROUPS:
            ws, gs, dot = [], [], 0.0
            for n, p in self._named():
                if _module_group(n) != g:
                    continue
                ws.append(p.detach().pow(2).sum())
                if p.grad is not None:
                    gs.append(p.grad.detach().pow(2).sum())
                    dot += float((p.grad.detach() * p.detach()).sum())
            wn = float(torch.stack(ws).sum().sqrt()) if ws else 0.0
            gn = float(torch.stack(gs).sum().sqrt()) if gs else 0.0
            wrow[f"w_norm/{g}"] = wn
            wrow[f"w_ratio/{g}"] = wn / self.init_norm[g] if self.init_norm[g] else 0.0
            grow[f"g_norm/{g}"] = gn
            # cos(grad, W): >0 means the gradient pushes the weights outward
            grow[f"cos_gw/{g}"] = dot / (wn * gn) if wn > 0 and gn > 0 else 0.0
            w_tot += wn ** 2
            g_tot += gn ** 2
            dot_tot += dot

        w_tot, g_tot = float(np.sqrt(w_tot)), float(np.sqrt(g_tot))
        wrow["w_norm/global"] = w_tot
        grow["g_norm/global"] = g_tot
        grow["cos_gw/global"] = dot_tot / (w_tot * g_tot) if w_tot > 0 and g_tot > 0 else 0.0

        # update geometry: how far did we move, and how coherent are the steps?
        with torch.no_grad():
            flat = self._flat_params()
            delta = flat - self.prev_flat
            dn = float(delta.norm())
            grow["update/norm"] = dn
            if self.prev_delta is not None:
                pn = float(self.prev_delta.norm())
                grow["update/cos_prev"] = (float(delta.dot(self.prev_delta))
                                           / (dn * pn)) if dn > 0 and pn > 0 else 0.0
            self.prev_delta, self.prev_flat = delta, flat

        self.csv["weights"].write(wrow)
        self.csv["grads"].write(grow)
        self.csv["train"].write({"step": step, **scalars})

    # -- expensive -----------------------------------------------------------
    @torch.no_grad()
    def log_expensive(self, step: int, probe_batch=None):
        """Spectra, embedding geometry, group-structure probes, sparsity."""
        srow = {"step": step}
        mats = {
            "tok_emb": self.model.tok_emb.weight,
            "unembed": self.model.unembed.weight,
            "mlp_fc0": self.model.blocks[0].mlp.fc.weight,
            "mlp_proj0": self.model.blocks[0].mlp.proj.weight,
        }
        topk_dump = {}
        for name, W in mats.items():
            st = _spectrum_stats(W, self.topk)
            topk_dump[name] = st.pop("topk")
            for k, v in st.items():
                srow[f"{name}/{k}"] = v
        self.csv["spectral"].write(srow)
        np.savez_compressed(
            os.path.join(self.log_dir, f"svd_step_{step:07d}.npz"), **topk_dump)

        # ---- embedding structure ----
        E = self.model.tok_emb.weight.detach()[: self.n_sym].float().cpu().numpy()
        strow = {"step": step}

        Ec = E - E.mean(0, keepdims=True)
        sv = np.linalg.svd(Ec, compute_uv=False)
        strow["emb/pca_top2_var"] = float((sv[:2] ** 2).sum() / (sv ** 2).sum())

        # Fourier circularity over each cyclic subgroup of the task
        for label, order in self.meta["cyclic"].items():
            fc = _fourier_concentration(E[np.asarray(order)])
            strow[f"fourier/{label}/top5_mass"] = fc["top_mass"]
            strow[f"fourier/{label}/entropy"] = fc["spec_entropy"]

        # linear probes for binary group structure (e.g. the D_n reflection bit)
        for label, y in self.meta["probes"].items():
            strow[f"probe/{label}"] = _linear_probe_acc(E, np.asarray(y))

        # activation sparsity: does the MLP develop a few specialised neurons?
        if probe_batch is not None:
            self.model(probe_batch)
            for li, blk in enumerate(self.model.blocks):
                a = blk.mlp.act.detach()[:, -1].float().cpu().numpy()
                strow[f"act/gini_l{li}"] = float(np.mean([_gini(r) for r in a[:256]]))
                strow[f"act/frac_dead_l{li}"] = float((np.abs(a).max(0) < 1e-3).mean())

        self.csv["structure"].write(strow)

    def close(self):
        for c in self.csv.values():
            c.close()
