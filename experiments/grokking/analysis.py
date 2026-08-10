"""Figures for the grokking study.

Design rules applied throughout (see EXPERIMENT_LOG.md):
* **No dual-axis plots.** Quantities on different scales go in stacked panels that
  share the log-step x-axis, so the reader compares timing without a second y-scale.
* Fixed, CVD-validated categorical hue order (Okabe-Ito subset, validated at
  worst-adjacent dE 9.6 for deuteranopia).
* Transition markers (t_memorise, t_grok) are drawn on every panel of a figure, so
  "what happens at the jump" is read vertically.

Run ``python analysis.py`` to regenerate everything into ``figures/``.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# CVD-validated categorical order (worst adjacent dE 9.6 deutan / 20.0 normal)
C = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d4"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.alpha": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "lines.linewidth": 1.8, "figure.dpi": 130,
})


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

class Run:
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(RUNS, name)
        self.cfg = json.load(open(os.path.join(self.dir, "config.json")))
        L = os.path.join(self.dir, "logs")
        self.train = pd.read_csv(os.path.join(L, "train.csv"))
        self.w = self._try(os.path.join(L, "weights.csv"))
        self.g = self._try(os.path.join(L, "grads.csv"))
        self.sp = self._try(os.path.join(L, "spectral.csv"))
        self.st = self._try(os.path.join(L, "structure.csv"))

    @staticmethod
    def _try(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    def _first(self, col, thr=0.99):
        s = self.train[self.train[col] > thr]["step"]
        return int(s.iloc[0]) if len(s) else None

    @property
    def t_mem(self):
        return self._first("train_acc")

    @property
    def t_grok(self):
        return self._first("val_acc")

    @property
    def label(self):
        c = self.cfg
        return f"{c['task']} frac={c['train_frac']} wd={c['weight_decay']}"


def load(*names):
    out = []
    for n in names:
        try:
            out.append(Run(n))
        except Exception as e:
            print(f"  ! skip {n}: {type(e).__name__}")
    return out


def _x(df):
    """Step axis clipped to >=1 so it can be drawn on a log scale."""
    return np.maximum(df["step"].values, 1)


def mark(ax, run, show_label=False):
    """Draw the memorisation and generalisation markers on a panel."""
    for t, col, lab in ((run.t_mem, C[2], "memorised"), (run.t_grok, C[1], "grokked")):
        if t:
            ax.axvline(max(t, 1), color=col, ls="--", lw=1.2, alpha=0.9,
                       label=(f"{lab} (step {t})" if show_label else None))
    if run.t_mem and run.t_grok and run.t_grok > run.t_mem:
        ax.axvspan(max(run.t_mem, 1), run.t_grok, color=C[1], alpha=0.06, lw=0)


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def fig1_curves(runs):
    """The headline: train vs validation accuracy on a log step axis."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, runs):
        ax.plot(_x(r.train), r.train["train_acc"], color=C[0], label="train")
        ax.plot(_x(r.train), r.train["val_acc"], color=C[1], label="validation")
        mark(ax, r, show_label=True)
        ax.set_xscale("log")
        ax.set_xlabel("optimisation steps")
        ax.set_title(r.label, fontsize=9)
        ax.set_ylim(-0.03, 1.05)
        ax.legend(loc="center left", fontsize=8)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Grokking: validation accuracy rises long after the training set is memorised",
                 fontsize=11, y=1.04)
    _finish(fig, os.path.join(FIGS, "fig1_grokking_curves.png"))


def fig2_detail(r):
    """Accuracy and loss for one run; loss shows the characteristic val-loss hump."""
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    axes[0].plot(_x(r.train), r.train["train_acc"], color=C[0], label="train")
    axes[0].plot(_x(r.train), r.train["val_acc"], color=C[1], label="validation")
    axes[0].axhline(1 / r.cfg.get("n_symbols", 96), color=MUTED, ls=":", lw=1,
                    label="chance")
    axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8, loc="center left")
    axes[1].plot(_x(r.train), r.train["train_loss"], color=C[0], label="train")
    axes[1].plot(_x(r.train), r.train["val_loss"], color=C[1], label="validation")
    axes[1].set_yscale("log"); axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_xlabel("optimisation steps"); axes[1].legend(fontsize=8)
    for i, ax in enumerate(axes):
        mark(ax, r, show_label=(i == 1))
        ax.set_xscale("log")
    fig.suptitle(f"{r.label} — the transition in detail", fontsize=11, y=0.98)
    _finish(fig, os.path.join(FIGS, "fig2_transition_detail.png"))


def fig3_weight_norms(r):
    """Does the weight norm move before, with, or after generalisation?"""
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.0), sharex=True)
    axes[0].plot(_x(r.train), r.train["val_acc"], color=C[1], label="validation acc")
    axes[0].plot(_x(r.train), r.train["train_acc"], color=C[0], label="train acc")
    axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8, loc="center left")

    axes[1].plot(_x(r.w), r.w["w_norm/global"], color=C[0])
    axes[1].set_ylabel("global $\\|W\\|_2$")

    mods = ["tok_emb", "unembed", "attn_qkv", "attn_proj", "mlp_fc", "mlp_proj"]
    for i, m in enumerate(mods):
        c = f"w_ratio/{m}"
        if c in r.w:
            axes[2].plot(_x(r.w), r.w[c], color=C[i % len(C)], label=m)
    axes[2].axhline(1.0, color=MUTED, ls=":", lw=1)
    axes[2].set_ylabel("$\\|W\\| / \\|W_{init}\\|$")
    axes[2].set_xlabel("optimisation steps")
    axes[2].legend(fontsize=7, ncol=3, loc="upper left")
    for i, ax in enumerate(axes):
        mark(ax, r, show_label=(i == 0)); ax.set_xscale("log")
    fig.suptitle(f"{r.label} — weight norm through the transition", fontsize=11, y=0.995)
    _finish(fig, os.path.join(FIGS, "fig3_weight_norms.png"))


def fig4_wd(runs):
    """Weight-decay ablation: the paper's strongest intervention."""
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    for i, r in enumerate(sorted(runs, key=lambda r: -r.cfg["weight_decay"])):
        lab = f"wd = {r.cfg['weight_decay']}"
        axes[0].plot(_x(r.train), r.train["val_acc"], color=C[i], label=lab)
        if r.w is not None:
            axes[1].plot(_x(r.w), r.w["w_norm/global"], color=C[i], label=lab)
        if r.t_grok:
            axes[0].axvline(r.t_grok, color=C[i], ls="--", lw=1, alpha=0.7)
    axes[0].set_ylabel("validation accuracy"); axes[0].legend(fontsize=8)
    axes[1].set_ylabel("global $\\|W\\|_2$"); axes[1].set_xlabel("optimisation steps")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
    fig.suptitle("Weight decay controls whether grokking happens (D$_{48}$, 30% data)",
                 fontsize=11, y=0.98)
    _finish(fig, os.path.join(FIGS, "fig4_weight_decay_ablation.png"))


def fig5_grads(r):
    """Gradient geometry: which statistic moves first?"""
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 8.4), sharex=True)
    axes[0].plot(_x(r.train), r.train["val_acc"], color=C[1], label="validation acc")
    axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8, loc="center left")
    axes[1].plot(_x(r.g), r.g["g_norm/global"], color=C[0])
    axes[1].set_yscale("log"); axes[1].set_ylabel("$\\|\\nabla L\\|$")
    axes[2].plot(_x(r.g), r.g["cos_gw/global"], color=C[2])
    axes[2].axhline(0, color=MUTED, ls=":", lw=1)
    axes[2].set_ylabel(r"$\cos(\nabla L, W)$")
    if "update/cos_prev" in r.g:
        axes[3].plot(_x(r.g), r.g["update/cos_prev"], color=C[3])
        axes[3].axhline(0, color=MUTED, ls=":", lw=1)
    axes[3].set_ylabel("cos(update$_t$, update$_{t-1}$)")
    axes[3].set_xlabel("optimisation steps")
    for i, ax in enumerate(axes):
        mark(ax, r, show_label=(i == 0)); ax.set_xscale("log")
    fig.suptitle(f"{r.label} — gradient geometry across the phases", fontsize=11, y=0.995)
    _finish(fig, os.path.join(FIGS, "fig5_gradient_geometry.png"))


def fig6_spectral(r):
    """Rank collapse: memorisation is high-rank, the general rule is low-rank."""
    if r.sp is None or not len(r.sp):
        return
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.0), sharex=True)
    axes[0].plot(_x(r.train), r.train["val_acc"], color=C[1], label="validation acc")
    axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8, loc="center left")
    for i, m in enumerate(["tok_emb", "unembed", "mlp_fc0", "mlp_proj0"]):
        if f"{m}/effective_rank" in r.sp:
            axes[1].plot(_x(r.sp), r.sp[f"{m}/effective_rank"], color=C[i], label=m)
            axes[2].plot(_x(r.sp), r.sp[f"{m}/stable_rank"], color=C[i], label=m)
    axes[1].set_ylabel("effective rank"); axes[1].legend(fontsize=7, ncol=2)
    axes[2].set_ylabel("stable rank"); axes[2].set_xlabel("optimisation steps")
    for i, ax in enumerate(axes):
        mark(ax, r, show_label=(i == 0)); ax.set_xscale("log")
    fig.suptitle(f"{r.label} — spectral collapse at the transition", fontsize=11, y=0.995)
    _finish(fig, os.path.join(FIGS, "fig6_spectral_rank.png"))


def fig7_structure(r):
    """Task-specific progress measures: when does group structure appear?"""
    if r.st is None or not len(r.st):
        return
    fourier = [c for c in r.st.columns if c.startswith("fourier/") and c.endswith("top5_mass")]
    probes = [c for c in r.st.columns if c.startswith("probe/")]
    n = 2 + (1 if fourier else 0) + (1 if probes else 0)
    fig, axes = plt.subplots(n, 1, figsize=(6.4, 2.3 * n), sharex=True)
    k = 0
    axes[k].plot(_x(r.train), r.train["val_acc"], color=C[1], label="validation acc")
    axes[k].set_ylabel("accuracy"); axes[k].legend(fontsize=8, loc="center left"); k += 1
    if fourier:
        for i, c in enumerate(fourier):
            axes[k].plot(_x(r.st), r.st[c], color=C[i],
                         label=c.split("/")[1])
        axes[k].set_ylabel("top-5 Fourier mass")
        axes[k].legend(fontsize=8); k += 1
    if probes:
        for i, c in enumerate(probes):
            axes[k].plot(_x(r.st), r.st[c], color=C[i + 2], label=c.split("/")[1])
        axes[k].axhline(0.5, color=MUTED, ls=":", lw=1, label="chance")
        axes[k].set_ylabel("linear probe acc")
        axes[k].legend(fontsize=8); k += 1
    axes[k].plot(_x(r.st), r.st["emb/pca_top2_var"], color=C[0])
    axes[k].set_ylabel("emb. var in top-2 PCs")
    axes[k].set_xlabel("optimisation steps")
    for i, ax in enumerate(axes):
        mark(ax, r, show_label=(i == 0)); ax.set_xscale("log")
    fig.suptitle(f"{r.label} — emergence of group structure in the embedding",
                 fontsize=11, y=0.995)
    _finish(fig, os.path.join(FIGS, "fig7_structure_probes.png"))


def fig8_data_efficiency(runs):
    """Steps-to-generalise vs training fraction (the paper's Fig. 1-centre trend)."""
    pts = []
    for r in runs:
        if r.cfg["task"] != "dihedral_48":
            continue
        if r.cfg["weight_decay"] != 1.0:
            continue
        pts.append((r.cfg["train_frac"], r.t_mem, r.t_grok, r.train["step"].max()))
    if not pts:
        return
    pts.sort()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fr = [p[0] for p in pts]
    ax.plot(fr, [p[1] or np.nan for p in pts], "o-", color=C[2], label="memorisation (train acc > 99%)")
    gr = [p[2] for p in pts]
    ok = [(f, g) for f, g in zip(fr, gr) if g]
    if ok:
        ax.plot([o[0] for o in ok], [o[1] for o in ok], "o-", color=C[1],
                label="generalisation (val acc > 99%)")
    for f, m, g, last in pts:
        if not g:  # censored: did not grok within budget
            ax.annotate("", xy=(f, last * 1.9), xytext=(f, last),
                        arrowprops=dict(arrowstyle="-|>", color=C[1], lw=1.4))
            ax.plot([f], [last], "^", color=C[1], ms=7, mfc="none")
    ax.set_yscale("log")
    ax.set_xlabel("training data fraction")
    ax.set_ylabel("optimisation steps")
    ax.set_title("Less data $\\Rightarrow$ far more optimisation to generalise\n"
                 "(D$_{48}$; open triangles = did not grok within budget)", fontsize=9)
    ax.legend(fontsize=8)
    _finish(fig, os.path.join(FIGS, "fig8_data_efficiency.png"))


def _load_emb(run, step):
    p = os.path.join(run.dir, "checkpoints", f"ckpt_{step:07d}.pt")
    sd = torch.load(p, map_location="cpu")["model"]
    return sd["tok_emb.weight"].float().numpy()


def fig9_embeddings(r_dih, r_add):
    """Embedding geometry before vs after grokking."""
    def ckpts(r):
        d = os.path.join(r.dir, "checkpoints")
        return sorted(int(f[5:-3]) for f in os.listdir(d) if f.startswith("ckpt_"))

    panels = []
    if r_dih is not None:
        cs = ckpts(r_dih)
        pre = max([c for c in cs if r_dih.t_mem and c >= r_dih.t_mem] or [cs[0]])
        pre = min([c for c in cs if r_dih.t_mem and c >= r_dih.t_mem] or [cs[-1]])
        panels += [(r_dih, pre, "D$_{48}$ just after memorising"),
                   (r_dih, cs[-1], "D$_{48}$ after grokking")]
    if r_add is not None:
        cs = ckpts(r_add)
        panels += [(r_add, cs[-1], "mod-add 97 after grokking")]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.5))
    axes = np.atleast_1d(axes)
    for ax, (r, step, title) in zip(axes, panels):
        E = _load_emb(r, step)
        n_sym = 96 if r.cfg["task"] == "dihedral_48" else 97
        E = E[:n_sym]
        E = E - E.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        P = E @ Vt[:2].T
        if r.cfg["task"] == "dihedral_48":
            ax.scatter(P[:48, 0], P[:48, 1], s=26, color=C[0], label="rotations $r^i$")
            ax.scatter(P[48:, 0], P[48:, 1], s=26, color=C[1], marker="^",
                       label="reflections $r^i s$")
            ax.legend(fontsize=7)
        else:
            sc = ax.scatter(P[:, 0], P[:, 1], s=26, c=np.arange(n_sym), cmap="twilight")
            plt.colorbar(sc, ax=ax, label="residue", fraction=0.046)
        ax.set_title(f"{title}\n(step {step})", fontsize=9)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Symbol embeddings: memorisation is unstructured, generalisation is geometric",
                 fontsize=11, y=1.03)
    _finish(fig, os.path.join(FIGS, "fig9_embeddings.png"))


def fig10_optimizers(runs):
    """Does the choice of optimiser change whether (and when) grokking happens?"""
    if len(runs) < 2:
        return
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.2), sharex=True)
    order = ["adamw (wd=1, decoupled)", "muon", "adam_l2", "rmsprop", "sgd"]
    runs = sorted(runs, key=lambda r: order.index(_optlabel(r))
                  if _optlabel(r) in order else 99)
    for i, r in enumerate(runs):
        lab = _optlabel(r)
        axes[0].plot(_x(r.train), r.train["train_acc"], color=C[i % len(C)], label=lab)
        axes[1].plot(_x(r.train), r.train["val_acc"], color=C[i % len(C)], label=lab)
        if r.w is not None:
            axes[2].plot(_x(r.w), r.w["w_norm/global"], color=C[i % len(C)], label=lab)
        if r.t_grok:
            axes[1].axvline(r.t_grok, color=C[i % len(C)], ls="--", lw=1, alpha=0.6)
    axes[0].set_ylabel("train accuracy"); axes[0].legend(fontsize=7, loc="lower right")
    axes[1].set_ylabel("validation accuracy")
    axes[2].set_ylabel("global $\\|W\\|_2$"); axes[2].set_yscale("log")
    axes[2].set_xlabel("optimisation steps")
    for ax in axes:
        ax.set_xscale("log")
    fig.suptitle("Optimiser choice controls the size of the grokking gap "
                 "(D$_{48}$, 30% data)", fontsize=11, y=0.995)
    _finish(fig, os.path.join(FIGS, "fig10_optimizers.png"))


def fig11_stability(r_all, r_excl):
    """Post-grokking instability is caused by decaying 1-D parameters.

    Both runs are identical except that the right-hand one excludes LayerNorm
    gains and biases from weight decay. Grokking survives; the oscillation does not.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.4), sharex=True, sharey="row")
    for j, (r, title) in enumerate(((r_all, "weight decay on ALL params (as-written)"),
                                    (r_excl, "weight decay excludes 1-D params"))):
        axes[0, j].plot(_x(r.train), r.train["train_acc"], color=C[0], label="train")
        axes[0, j].plot(_x(r.train), r.train["val_acc"], color=C[1], label="validation")
        axes[0, j].set_title(title, fontsize=9)
        axes[0, j].legend(fontsize=8, loc="center left")
        if "w_norm/layernorm" in r.w:
            axes[1, j].plot(_x(r.w), r.w["w_norm/layernorm"], color=C[2],
                            label="LayerNorm params")
        axes[1, j].plot(_x(r.w), r.w["w_norm/tok_emb"], color=C[0], label="token emb")
        axes[1, j].legend(fontsize=8)
        axes[1, j].set_xlabel("optimisation steps")
        for i in (0, 1):
            mark(axes[i, j], r)
            axes[i, j].set_xscale("log")
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("$\\|W\\|_2$")
    axes[1, 0].set_yscale("log")
    fig.suptitle("Post-grokking collapse is an artefact of decaying LayerNorm gains",
                 fontsize=11, y=0.99)
    _finish(fig, os.path.join(FIGS, "fig11_stability.png"))


def fig12_lr_sweep(runs):
    """Grokking gap vs learning rate, for AdamW and Muon.

    Muon and AdamW learning rates are not on a common scale: Muon's Newton-Schulz
    update has ~unit *spectral* norm while AdamW's has ~unit *max-abs* norm, which
    is why Muon papers use 10-50x larger values. The fair comparison is therefore
    each optimiser swept over its own range, which is what this plots.
    """
    pts = {"adamw": [], "muon (hidden)": [], "muon (all 2-D)": []}
    for r in runs:
        o = r.cfg.get("optimizer", "adamw")
        # only the dedicated LR-sweep runs (plus the wd=1 baseline), so that the
        # weight-decay study's runs -- which also use adamw at wd=1 -- stay out.
        if o == "adamw" and (r.name.startswith("lr_adamw") or
                             r.name == "r3_dihedral_main"):
            pts["adamw"].append((r.cfg["lr"], r.t_mem, r.t_grok,
                                 int(r.train["step"].max())))
        elif o == "muon":
            k = ("muon (all 2-D)" if r.cfg.get("muon_scope") == "all_2d"
                 else "muon (hidden)")
            pts[k].append((r.cfg["muon_lr"], r.t_mem, r.t_grok,
                           int(r.train["step"].max())))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for i, (k, v) in enumerate(pts.items()):
        if not v:
            continue
        v.sort(key=lambda p: p[0])
        lrs = [p[0] for p in v]
        axes[0].plot(lrs, [p[1] for p in v], "o--", color=C[i], alpha=0.55,
                     label=f"{k}: memorise")
        gt = [(p[0], p[2]) for p in v if p[2]]
        if gt:
            axes[0].plot([g[0] for g in gt], [g[1] for g in gt], "o-", color=C[i],
                         label=f"{k}: generalise")
        for lr, tm, tg, last in v:
            if not tg:
                axes[0].plot([lr], [last], "^", color=C[i], ms=8, mfc="none")
        gaps = [(p[0], p[2] / p[1]) for p in v if p[1] and p[2]]
        if gaps:
            axes[1].plot([g[0] for g in gaps], [g[1] for g in gaps], "o-",
                         color=C[i], label=k)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("learning rate (each optimiser's own scale)")
    axes[0].set_ylabel("optimisation steps")
    axes[0].set_title("when it memorises vs. generalises", fontsize=9)
    axes[0].legend(fontsize=6.5, ncol=1)
    axes[1].axhline(1.0, color=MUTED, ls=":", lw=1)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("learning rate (each optimiser's own scale)")
    axes[1].set_ylabel("grokking gap  $t_{grok}/t_{mem}$")
    axes[1].set_title("gap = 1 means no grokking", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.suptitle("The grokking gap is largely a learning-rate effect; "
                 "Muon is insensitive to it", fontsize=11, y=1.02)
    _finish(fig, os.path.join(FIGS, "fig12_lr_sweep.png"))


def _optlabel(r):
    o = r.cfg.get("optimizer", "adamw")
    wd = r.cfg["weight_decay"]
    if o == "adamw" and wd == 1.0 and not r.cfg.get("wd_exclude_1d"):
        return "adamw (wd=1, decoupled)"
    return o if o != "adamw" else f"adamw (wd={wd})"


def summary_table(runs):
    rows = []
    for r in runs:
        rows.append({
            "run": r.name, "task": r.cfg["task"], "frac": r.cfg["train_frac"],
            "wd": r.cfg["weight_decay"], "seed": r.cfg["seed"],
            "opt": r.cfg.get("optimizer", "adamw"), "lr": r.cfg["lr"],
            "muon_lr": r.cfg.get("muon_lr") if r.cfg.get("optimizer") == "muon" else None,
            "muon_scope": r.cfg.get("muon_scope") if r.cfg.get("optimizer") == "muon" else None,
            "steps_done": int(r.train["step"].max()),
            "t_memorise": r.t_mem, "t_grok": r.t_grok,
            "grok_gap": round(r.t_grok / r.t_mem, 1) if (r.t_mem and r.t_grok) else None,
            "final_train_acc": round(float(r.train["train_acc"].iloc[-1]), 4),
            "final_val_acc": round(float(r.train["val_acc"].iloc[-1]), 4),
            "best_val_acc": round(float(r.train["val_acc"].max()), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results_summary.csv"), index=False)
    return df


ALL = ["r1_mod_div_control", "r2_mod_add_ref", "r3_dihedral_main",
       "r4_dihedral_seed1", "r5_dihedral_wd0", "r6_dihedral_wd01",
       "r7_dihedral_frac020", "r8_dihedral_frac040", "r9_dihedral_wdexcl",
       "o_muon", "o_adam_l2", "o_sgd", "o_rmsprop",
       "lr_adamw_0.0003", "lr_adamw_0.003", "lr_adamw_0.01",
       "lr_muon_0.005_hidden", "lr_muon_0.02_hidden", "lr_muon_0.05_hidden",
       "lr_muon_0.02_all2d",
       "wd_prod1e-2_lr1e-3", "wd_prod1e-3_lr1e-2", "wd_prod1e-3_lr3e-3",
       "wd_prod1e-4_lr1e-2", "wd_late2000", "wd_late10000", "wd_only_attn",
       "wd_only_mlp", "wd_only_embed", "wd_off8000",
       "g_attn_gdn", "g_gdn_attn", "g_gdn_gdn"]


def main():
    runs = load(*ALL)
    by = {r.name: r for r in runs}
    df = summary_table(runs)
    print(df.to_string(index=False))

    head = [by[n] for n in ("r1_mod_div_control", "r2_mod_add_ref", "r3_dihedral_main")
            if n in by]
    if head:
        fig1_curves(head)
    main_run = by.get("r3_dihedral_main")
    if main_run:
        fig2_detail(main_run)
        fig3_weight_norms(main_run)
        fig5_grads(main_run)
        fig6_spectral(main_run)
        fig7_structure(main_run)
    wd = [by[n] for n in ("r3_dihedral_main", "r5_dihedral_wd0", "r6_dihedral_wd01")
          if n in by]
    if len(wd) > 1:
        fig4_wd(wd)
    opt_runs = [by[n] for n in ("r3_dihedral_main", "o_muon", "o_adam_l2",
                                "o_rmsprop", "o_sgd") if n in by]
    fig10_optimizers(opt_runs)
    if "r9_dihedral_wdexcl" in by:
        fig11_stability(by["r3_dihedral_main"], by["r9_dihedral_wdexcl"])
    fig8_data_efficiency([r for r in runs if r.name.startswith("r")])
    fig12_lr_sweep(runs)
    aligned = [by[n] for n in ("r8_dihedral_frac040", "lr_adamw_0.003",
                               "r3_dihedral_main", "r6_dihedral_wd01") if n in by]
    fig13_transition_aligned(aligned)
    if main_run:
        fig14_lead_lag(main_run)
        fig15_zoom(main_run)
    prog = [by[n] for n in ("r3_dihedral_main", "r5_dihedral_wd0") if n in by]
    if prog:
        fig16_irrep_progress(prog)
    sol = [by[n] for n in ("r3_dihedral_main", "lr_adamw_0.003", "lr_adamw_0.01",
                           "o_muon", "lr_muon_0.02_hidden", "lr_muon_0.05_hidden",
                           "r6_dihedral_wd01", "r5_dihedral_wd0") if n in by]
    fig17_irrep_solutions(sol)
    fig18_weight_decay_study(by)
    fig19_token_mixer(by)
    try:
        fig9_embeddings(main_run, by.get("r2_mod_add_ref"))
    except Exception as e:
        print("  ! fig9 skipped:", type(e).__name__, e)




# --------------------------------------------------------------------------- #
# transition-focused figures
# --------------------------------------------------------------------------- #

# (statistic key, source frame, display label, how to normalise)
TRANSITION_STATS = [
    ("val_acc",                    "train",     "validation accuracy",        "raw"),
    ("w_norm/global",              "w",         "global $\\|W\\|_2$",         "at_grok"),
    ("g_norm/global",              "g",         "gradient norm",              "at_grok_log"),
    ("cos_gw/global",              "g",         r"$\cos(\nabla L, W)$",       "raw"),
    ("tok_emb/effective_rank",     "sp",        "embedding effective rank",   "at_init"),
    ("fourier/rot_Z48/top5_mass",  "st",        "Fourier concentration",      "raw"),
    ("emb/pca_top2_var",           "st",        "emb. var in top-2 PCs",      "raw"),
    ("act/gini_l0",                "st",        "MLP activation Gini",        "raw"),
]


def _series(r, key, src):
    df = getattr(r, src)
    if df is None or key not in df.columns:
        return None, None
    return df["step"].values.astype(float), df[key].values.astype(float)


def fig13_transition_aligned(runs):
    """Do different runs share one internal signature at their transition?

    Each run's step axis is divided by its own ``t_grok``, so all transitions land
    at x = 1. The runs span a 15x range in when they grok (2100 to 31550 steps).
    If the curves collapse, the statistical signature of grokking is universal and
    independent of *when* it happens.
    """
    runs = [r for r in runs if r.t_grok and r.t_mem]
    if len(runs) < 2:
        return
    n = len(TRANSITION_STATS)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(10.4, 2.3 * ((n + 1) // 2)),
                             sharex=True)
    axes = axes.ravel()
    for ax, (key, src, label, norm) in zip(axes, TRANSITION_STATS):
        for i, r in enumerate(runs):
            x, y = _series(r, key, src)
            if x is None:
                continue
            m = x > 0
            x, y = x[m], y[m]
            if norm.startswith("at_grok"):
                ref = np.interp(r.t_grok, x, y)
                y = y / ref if ref else y
            elif norm == "at_init":
                y = y / y[0] if y[0] else y
            ax.plot(x / r.t_grok, y, color=C[i % len(C)], lw=1.5,
                    label=f"{_shortlabel(r)} ($t_g$={r.t_grok})")
        ax.axvline(1.0, color=MUTED, ls="--", lw=1.2)
        ax.set_xscale("log")
        if norm.endswith("_log"):
            ax.set_yscale("log")
        ax.set_ylabel(label + ("\n(rel. to $t_{grok}$)" if norm.startswith("at_grok") else
                               "\n(rel. to init)" if norm == "at_init" else ""),
                      fontsize=8)
    for ax in axes[-2:]:
        ax.set_xlabel("step / $t_{grok}$   (1.0 = the transition)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    for ax in axes[len(TRANSITION_STATS):]:
        ax.set_visible(False)
    fig.suptitle("Transition-aligned statistics: the same internal signature at every "
                 "grokking event\n(runs differ 15$\\times$ in when they grok)",
                 fontsize=11, y=1.0)
    _finish(fig, os.path.join(FIGS, "fig13_transition_aligned.png"))


def _shortlabel(r):
    c = r.cfg
    if c.get("optimizer") == "muon":
        return f"muon lr={c['muon_lr']:g}"
    bits = [f"frac={c['train_frac']:g}", f"wd={c['weight_decay']:g}"]
    if c["lr"] != 1e-3:
        bits.append(f"lr={c['lr']:g}")
    return " ".join(bits)


def fig14_lead_lag(r):
    """Which statistics move *before* validation accuracy, and which lag it?

    For each statistic we take its value at t_memorise as the baseline and its
    value at the end as the settled level, then record the step at which it first
    covers half that distance. Plotting those steps against t_grok shows the
    ordering of events through the transition.
    """
    events = []
    for key, src, label, _ in TRANSITION_STATS:
        x, y = _series(r, key, src)
        if x is None:
            continue
        m = x >= r.t_mem
        x, y = x[m], y[m]
        if len(x) < 3:
            continue
        base, final = y[0], np.median(y[-5:])
        if not np.isfinite(base) or not np.isfinite(final) or abs(final - base) < 1e-9:
            continue
        half = base + 0.5 * (final - base)
        crossed = (y >= half) if final > base else (y <= half)
        idx = np.argmax(crossed) if crossed.any() else None
        if idx is None or not crossed.any():
            continue
        events.append((label, float(x[idx]), base, final))
    if not events:
        return
    events.sort(key=lambda e: e[1])
    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(events) + 2.0))
    ys = np.arange(len(events))
    for i, (label, step, base, final) in enumerate(events):
        lead = step < r.t_grok
        ax.plot([step], [i], "o", ms=9, color=C[0] if lead else C[1],
                zorder=3)
        ax.hlines(i, min(step, r.t_grok), max(step, r.t_grok),
                  color=C[0] if lead else C[1], lw=2, alpha=0.35)
        ax.annotate(f"{base:.2f}→{final:.2f}", (step, i), fontsize=7,
                    color=MUTED, xytext=(9, -9), textcoords="offset points")
    ax.axvline(r.t_grok, color=C[1], ls="--", lw=1.5,
               label=f"val acc > 99% (step {r.t_grok})")
    ax.axvline(r.t_mem, color=C[2], ls="--", lw=1.5,
               label=f"train acc > 99% (step {r.t_mem})")
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in events], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("step at which the statistic completes half of its change")
    ax.set_ylim(-0.8, len(events) - 0.2)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"{r.label} — when each statistic completes half its post-memorisation change\n"
                 "CAVEAT: monotonically drifting quantities (grad norm, $\\|W\\|$) reach their\n"
                 "midpoint early by construction — that is drift, not anticipation. Only\n"
                 "step-like statistics can be compared against the jump.", fontsize=8)
    _finish(fig, os.path.join(FIGS, "fig14_lead_lag.png"))


def fig15_zoom(r):
    """Linear-axis zoom on the emergence itself, with per-module weight norms."""
    lo, hi = r.t_mem * 0.5, r.t_grok * 2.0
    fig, axes = plt.subplots(4, 1, figsize=(6.8, 9.0), sharex=True)

    def win(df):
        return df[(df["step"] >= lo) & (df["step"] <= hi)]

    t = win(r.train)
    axes[0].plot(t["step"], t["train_acc"], color=C[0], label="train")
    axes[0].plot(t["step"], t["val_acc"], color=C[1], label="validation")
    axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8, loc="center left")

    w = win(r.w)
    axes[1].plot(w["step"], w["w_norm/global"], color=C[0])
    axes[1].set_ylabel("global $\\|W\\|_2$")

    for i, m in enumerate(["tok_emb", "unembed", "attn_qkv", "mlp_fc", "mlp_proj"]):
        c = f"w_norm/{m}"
        if c in w:
            axes[2].plot(w["step"], w[c], color=C[i % len(C)], label=m)
    axes[2].set_ylabel("per-module $\\|W\\|_2$")
    axes[2].legend(fontsize=7, ncol=3)

    g = win(r.g)
    axes[3].plot(g["step"], g["g_norm/global"], color=C[2], label="grad norm")
    axes[3].set_yscale("log")
    axes[3].set_ylabel("$\\|\\nabla L\\|$")
    axes[3].set_xlabel("optimisation steps (linear)")
    axes[3].legend(fontsize=8)

    for ax in axes:
        mark(ax, r)
    fig.suptitle(f"{r.label} — zoom on the emergence (linear axis)", fontsize=11, y=0.995)
    _finish(fig, os.path.join(FIGS, "fig15_emergence_zoom.png"))




# --------------------------------------------------------------------------- #
# representation-theoretic analysis (irreps of D_48)
# --------------------------------------------------------------------------- #

def fig16_irrep_progress(runs):
    """Does the collapse onto a few irreps happen gradually, or at the jump?

    The 'effective number of irreps' is the inverse Simpson index of the irrep
    power spectrum: 27 means power is spread uniformly over all irreps of D_48
    (no group structure), 1 means a single irrep carries everything.
    """
    import irreps as IR
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.4), sharex=True)
    for i, r in enumerate(runs):
        rows = IR.sweep_run(r.dir)
        if not rows:
            continue
        st = np.array([x["step"] for x in rows], dtype=float)
        st = np.maximum(st, 1)
        lab = _shortlabel(r) + (f"  ($t_g$={r.t_grok})" if r.t_grok else "  (never groks)")
        axes[0].plot(np.maximum(r.train["step"], 1), r.train["val_acc"],
                     color=C[i % len(C)], label=lab)
        axes[1].plot(st, [x["emb_eff_n_irreps"] for x in rows], "o-",
                     color=C[i % len(C)], ms=3, label=lab)
        axes[2].plot(st, [x["emb_top3_mass"] for x in rows], "o-",
                     color=C[i % len(C)], ms=3, label=lab)
        if r.t_grok:
            for ax in axes:
                ax.axvline(r.t_grok, color=C[i % len(C)], ls="--", lw=1, alpha=0.6)
    axes[0].set_ylabel("validation accuracy"); axes[0].legend(fontsize=7)
    axes[1].axhline(27, color=MUTED, ls=":", lw=1)
    axes[1].set_ylabel("effective # irreps\n(27 = unstructured)")
    axes[2].set_ylabel("power in top-3 irreps")
    axes[2].set_xlabel("optimisation steps")
    for ax in axes:
        ax.set_xscale("log")
    fig.suptitle("Group structure emerges AT the transition, and only with weight decay",
                 fontsize=11, y=0.995)
    _finish(fig, os.path.join(FIGS, "fig16_irrep_progress.png"))


def fig17_irrep_solutions(runs):
    """Do different optimisers / learning rates find the SAME solution?"""
    import irreps as IR
    data = []
    for r in runs:
        steps = IR._ckpt_steps(r.dir)
        if not steps:
            continue
        E, _ = IR.load_matrices(r.dir, steps[-1])
        c = IR.concentration(E)
        data.append((_shortlabel(r), c, r.t_grok))
    if not data:
        return
    # keep the irreps that matter for any run
    frac = np.array([d[1]["frac"] for d in data])
    keep = np.argsort(frac.max(axis=0))[::-1][:10]
    keep = keep[np.argsort(-frac[:, keep].mean(axis=0))]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 0.46 * len(data) + 2.6),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    M = frac[:, keep]
    im = axes[0].imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=M.max())
    axes[0].set_xticks(range(len(keep)))
    axes[0].set_xticklabels([IR.IRREP_NAMES[k] for k in keep], rotation=45,
                            ha="right", fontsize=8)
    axes[0].set_yticks(range(len(data)))
    axes[0].set_yticklabels([d[0] for d in data], fontsize=8)
    axes[0].set_title("share of embedding power per irrep", fontsize=9)
    axes[0].grid(False)
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            if M[a, b] > 0.04:
                axes[0].text(b, a, f"{M[a,b]:.2f}", ha="center", va="center",
                             fontsize=6.5,
                             color="white" if M[a, b] > 0.55 * M.max() else INK)
    plt.colorbar(im, ax=axes[0], fraction=0.03, label="power fraction")

    ys = np.arange(len(data))
    axes[1].barh(ys, [d[1]["eff_n_irreps"] for d in data], color=C[0], height=0.6)
    axes[1].axvline(27, color=MUTED, ls=":", lw=1)
    axes[1].set_yticks(ys); axes[1].set_yticklabels([])
    axes[0].invert_yaxis()   # match barh, which puts index 0 at the bottom
    axes[1].set_xlabel("effective # irreps (27 = unstructured)", fontsize=8)
    axes[1].set_title("sparsity of the solution", fontsize=9)
    for i, d in enumerate(data):
        axes[1].text(d[1]["eff_n_irreps"] + 0.3, i, f"{d[1]['eff_n_irreps']:.1f}",
                     va="center", fontsize=7, color=MUTED)
    fig.suptitle("Every grokked run finds an equally SPARSE solution — but a "
                 "DIFFERENT set of irreps", fontsize=11, y=1.0)
    _finish(fig, os.path.join(FIGS, "fig17_irrep_solutions.png"))




def fig18_weight_decay_study(by):
    """Where, when, and how much weight decay is needed."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), sharey=True)

    # (a) decay switched on late -- the clock starts when decay starts
    for i, (n, onset) in enumerate([("r3_dihedral_main", 0), ("wd_late2000", 2000),
                                    ("wd_late10000", 10000), ("r5_dihedral_wd0", None)]):
        if n not in by:
            continue
        r = by[n]
        lab = "wd on from step 0" if onset == 0 else (
            "wd never on" if onset is None else f"wd on from step {onset}")
        axes[0].plot(np.maximum(r.train["step"], 1), r.train["val_acc"],
                     color=C[i], label=lab)
        if onset:
            axes[0].axvline(onset, color=C[i], ls=":", lw=1.4)
        if r.t_grok:
            axes[0].axvline(r.t_grok, color=C[i], ls="--", lw=1, alpha=0.7)
    axes[0].set_title("(a) decay switched on late:\nthe clock starts when decay does", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("validation accuracy")

    # (b) module-selective decay -- all partial variants fail
    for i, (n, lab) in enumerate([("r3_dihedral_main", "all params"),
                                  ("wd_only_attn", "attention only"),
                                  ("wd_only_mlp", "MLP only"),
                                  ("wd_only_embed", "embeddings only")]):
        if n not in by:
            continue
        r = by[n]
        axes[1].plot(np.maximum(r.train["step"], 1), r.train["val_acc"],
                     color=C[i], label=lab)
    axes[1].set_title("(b) which modules must be decayed?\nonly decaying everything works", fontsize=9)
    axes[1].legend(fontsize=7, loc="upper left")

    # (c) decay removed after the transition -- solution persists
    for i, (n, lab) in enumerate([("r3_dihedral_main", "wd on throughout"),
                                  ("wd_off8000", "wd OFF from step 8000")]):
        if n not in by:
            continue
        r = by[n]
        axes[2].plot(np.maximum(r.train["step"], 1), r.train["val_acc"],
                     color=C[i], label=lab)
    if "wd_off8000" in by:
        axes[2].axvline(8000, color=C[1], ls=":", lw=1.4)
    axes[2].set_title("(c) decay removed after grokking:\nthe solution persists", fontsize=9)
    axes[2].legend(fontsize=7, loc="lower right")

    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("optimisation steps"); ax.set_ylim(-0.03, 1.05)
    fig.suptitle("Weight decay: it must act everywhere, it sets the clock, "
                 "and it is only needed to *reach* the solution", fontsize=11, y=1.02)
    _finish(fig, os.path.join(FIGS, "fig18_weight_decay_study.png"))


def fig19_token_mixer(by):
    """Is grokking specific to softmax attention?"""
    import irreps as IR
    order = [("r3_dihedral_main", "attn, attn  (baseline)"),
             ("g_attn_gdn", "attn, GDN"),
             ("g_gdn_attn", "GDN, attn"),
             ("g_gdn_gdn", "GDN, GDN")]
    order = [(n, l) for n, l in order if n in by]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    effs, labs = [], []
    for i, (n, lab) in enumerate(order):
        r = by[n]
        axes[0].plot(np.maximum(r.train["step"], 1), r.train["val_acc"],
                     color=C[i], label=f"{lab}" + (f"  ($t_g$={r.t_grok})" if r.t_grok
                                                   else "  (never groks)"))
        steps = IR._ckpt_steps(r.dir)
        if steps:
            E, _ = IR.load_matrices(r.dir, steps[-1])
            effs.append(IR.concentration(E)["eff_n_irreps"]); labs.append(lab)
    axes[0].set_xscale("log"); axes[0].set_xlabel("optimisation steps")
    axes[0].set_ylabel("validation accuracy"); axes[0].legend(fontsize=7.5)
    axes[0].set_title("grokking survives replacing attention with a "
                      "gated linear recurrence", fontsize=9)

    ys = np.arange(len(effs))
    cols = [C[2] if e < 15 else C[1] for e in effs]
    axes[1].barh(ys, effs, color=cols, height=0.6)
    axes[1].axvline(27, color=MUTED, ls=":", lw=1)
    axes[1].set_yticks(ys); axes[1].set_yticklabels(labs, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("effective # irreps (27 = unstructured)", fontsize=8)
    axes[1].set_title("the solution found", fontsize=9)
    for i, e in enumerate(effs):
        axes[1].text(e + 0.4, i, f"{e:.1f}", va="center", fontsize=8, color=MUTED)
    fig.suptitle("Token-mixer ablation (sequence length 4)", fontsize=11, y=1.03)
    _finish(fig, os.path.join(FIGS, "fig19_token_mixer.png"))


if __name__ == "__main__":
    main()
