"""Single grokking run: train a small transformer on one algorithmic dataset and
record the internal statistics of the memorisation -> generalisation transition.

Usage
-----
    python train.py configs/r3_dihedral.yaml
    python train.py configs/r3_dihedral.yaml --max_steps=50000 --weight_decay=0.0

Optimisation follows Power et al. 2022 Appendix A.1.2: AdamW, lr 1e-3,
betas (0.9, 0.98), weight decay 1.0, linear warmup over the first 10 updates,
no annealing, minibatch min(512, |train|/2).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data import build_task, make_splits
from metrics import GrokkingMonitor
from model import GrokTransformer


@dataclass
class Config:
    task: str = "dihedral_48"
    train_frac: float = 0.5
    seed: int = 0

    n_layer: int = 2
    n_embd: int = 128
    n_head: int = 4
    layer_types: str = ""   # e.g. 'attn,gdn'; empty = all attention

    optimizer: str = "adamw"   # adamw | adam_l2 | sgd | rmsprop | muon
    lr: float = 1e-3
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 1.0
    wd_exclude_1d: bool = False
    wd_start_step: int = 0      # weight decay is OFF before this step
    wd_stop_step: int = -1      # weight decay is OFF from this step on (-1 = never)
    wd_modules: str = "all"     # all | attn | mlp | embed  (which modules are decayed)
    muon_lr: float = 0.02      # muon only: LR for 2D weights (1D use AdamW at `lr`)
    muon_momentum: float = 0.95
    muon_scope: str = "hidden"   # hidden | all_2d  (which matrices Muon owns)
    warmup_steps: int = 10
    batch_size: int = 512
    max_steps: int = 100_000

    cheap_every: int = 25
    expensive_every: int = 250
    n_checkpoints: int = 40

    out_dir: str = "runs/debug"
    device: str = "cuda"
    run_name: str = ""
    tags: dict = field(default_factory=dict)


def load_config() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=None)
    args, extra = ap.parse_known_args()

    cfg = Config()
    if args.config:
        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if not hasattr(cfg, k):
                    raise KeyError(f"unknown config key {k!r}")
                setattr(cfg, k, v)
    for a in extra:                                   # --key=value overrides
        assert a.startswith("--") and "=" in a, f"bad override {a!r}"
        k, v = a[2:].split("=", 1)
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key {k!r}")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v == "True")
    if not cfg.run_name:
        cfg.run_name = os.path.basename(cfg.out_dir)
    return cfg


def build_optimizer(model, cfg, dev):
    """Dispatch on ``cfg.optimizer``.

    All variants keep the paper's decoupled weight decay where the optimiser
    supports it, so the comparison isolates the *update rule* rather than the
    regularisation. ``adam_l2`` is the deliberate exception: it applies the same
    coefficient as coupled L2 (added to the gradient) instead of decoupled decay,
    which isolates the AdamW-vs-Adam distinction the paper leans on.

    Each param group records ``initial_lr`` so warmup can be applied as a
    *multiplier*, keeping Muon's 2D LR and AdamW's 1D LR in proportion.
    """
    name = cfg.optimizer.lower()
    fused = dev.type == "cuda"
    params = list(model.parameters())

    if cfg.wd_modules != "all":
        # Decay only one module family, to ask *where* weight decay has to act.
        # E.g. the norm rise during memorisation is carried almost entirely by
        # attn_qkv (see WRITEUP 3.4b) -- is decaying attention alone sufficient?
        sel = {"attn": ("attn",), "mlp": ("mlp",),
               "embed": ("tok_emb", "pos_emb", "unembed")}[cfg.wd_modules]
        hit = lambda n: any(t in n for t in sel)
        groups = [{"params": [p for n, p in model.named_parameters() if hit(n)],
                   "weight_decay": cfg.weight_decay, "wd_on": True},
                  {"params": [p for n, p in model.named_parameters() if not hit(n)],
                   "weight_decay": 0.0, "wd_on": False}]
        opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), fused=fused)
        for pg in opt.param_groups:
            pg["initial_lr"] = pg["lr"]
        print(f"  wd_modules={cfg.wd_modules}: {len(groups[0]['params'])} decayed, "
              f"{len(groups[1]['params'])} not", flush=True)
        return opt

    if cfg.wd_exclude_1d and name in ("adamw", "adam_l2"):
        # Standard practice: decay only matrices, never LayerNorm gains/biases.
        # Decaying 1-D parameters drives the LayerNorm scale toward zero, which
        # we found destabilises training badly after the network has fit the
        # data (see EXPERIMENT_LOG.md E8). The paper does not state whether it
        # excludes them; this flag makes the choice explicit and testable.
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        groups = [{"params": decay, "weight_decay": cfg.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        opt = cls(groups, lr=cfg.lr, betas=tuple(cfg.betas), fused=fused)
        for pg in opt.param_groups:
            pg["initial_lr"] = pg["lr"]
        return opt

    if name == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.lr, betas=tuple(cfg.betas),
                                weight_decay=cfg.weight_decay, fused=fused)
    elif name == "adam_l2":
        opt = torch.optim.Adam(params, lr=cfg.lr, betas=tuple(cfg.betas),
                               weight_decay=cfg.weight_decay, fused=fused)
    elif name == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, nesterov=True,
                              weight_decay=cfg.weight_decay)
    elif name == "rmsprop":
        opt = torch.optim.RMSprop(params, lr=cfg.lr, alpha=0.99,
                                  weight_decay=cfg.weight_decay)
    elif name == "muon":
        from muon import Muon   # vendored; see muon.py
        # Muon orthogonalises 2D gradients; 1D params (embeddings, norms) go to AdamW.
        # `muon_scope` controls which matrices Muon owns:
        #   "hidden"  - standard practice: attention/MLP weight matrices only.
        #               Embeddings, the output head, and any matrix with a
        #               dimension < 16 go to AdamW.
        #   "all_2d"  - every 2-D tensor, including tok_emb/pos_emb/unembed.
        # This matters for our analysis: orthogonalising the *embedding* update
        # could by itself manufacture the spectral structure we measure, so the
        # two scopes must be distinguished rather than conflated.
        named = list(model.named_parameters())
        if cfg.muon_scope == "hidden":
            excl = ("tok_emb", "pos_emb", "unembed")
            is_muon = lambda n, p: (p.ndim == 2 and min(p.shape) >= 16
                                    and not n.startswith(excl))
        elif cfg.muon_scope == "all_2d":
            is_muon = lambda n, p: p.ndim == 2
        else:
            raise ValueError(f"unknown muon_scope {cfg.muon_scope!r}")
        muon_p = [p for n, p in named if is_muon(n, p)]
        adam_p = [p for n, p in named if not is_muon(n, p)]
        print(f"  muon_scope={cfg.muon_scope}: {len(muon_p)} matrices to Muon, "
              f"{len(adam_p)} tensors to AdamW", flush=True)
        m = Muon(muon_p, lr=cfg.muon_lr, momentum=cfg.muon_momentum,
                 weight_decay=cfg.weight_decay)
        a = torch.optim.AdamW(adam_p, lr=cfg.lr, betas=tuple(cfg.betas),
                              weight_decay=cfg.weight_decay, fused=fused)

        class Combined:
            """Minimal combined optimiser exposing the interface train.py uses."""
            def __init__(self, opts):
                self.opts = opts
                self.param_groups = [g for o in opts for g in o.param_groups]

            def zero_grad(self, set_to_none=True):
                for o in self.opts:
                    o.zero_grad(set_to_none=set_to_none)

            def step(self):
                for o in self.opts:
                    o.step()

        opt = Combined([m, a])
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

    for pg in opt.param_groups:
        pg["initial_lr"] = pg["lr"]
    return opt


@torch.no_grad()
def evaluate(model, X, Y, chunk: int = 8192):
    """Exact loss / accuracy / logit margin over a whole split."""
    model.eval()
    tot_loss = tot_correct = tot_margin = tot_maxlogit = 0.0
    for i in range(0, len(X), chunk):
        xb, yb = X[i:i + chunk], Y[i:i + chunk]
        logits = model(xb)
        tot_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        tot_correct += (logits.argmax(-1) == yb).sum().item()
        correct = logits.gather(1, yb[:, None]).squeeze(1)
        other = logits.scatter(1, yb[:, None], -float("inf")).max(-1).values
        tot_margin += (correct - other).sum().item()
        tot_maxlogit += logits.abs().max(-1).values.sum().item()
    model.train()
    n = len(X)
    return {"loss": tot_loss / n, "acc": tot_correct / n,
            "margin": tot_margin / n, "logit_absmax": tot_maxlogit / n}


def main():
    cfg = load_config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---- data ----
    inputs, answers, meta = build_task(cfg.task)
    (xtr, ytr), (xva, yva) = make_splits(inputs, answers, cfg.train_frac, cfg.seed)
    Xtr = torch.from_numpy(xtr).to(dev); Ytr = torch.from_numpy(ytr).to(dev)
    Xva = torch.from_numpy(xva).to(dev); Yva = torch.from_numpy(yva).to(dev)
    bs = min(cfg.batch_size, max(1, len(Xtr) // 2))   # paper's rule

    # ---- model / optimiser ----
    model = GrokTransformer(meta["vocab_size"], cfg.n_layer, cfg.n_embd,
                            cfg.n_head, meta["seq_len"], cfg.layer_types).to(dev)
    opt = build_optimizer(model, cfg, dev)
    mon = GrokkingMonitor(model, cfg.out_dir, meta)

    ckpt_steps = sorted(set(
        np.unique(np.round(np.logspace(0, np.log10(cfg.max_steps),
                                       cfg.n_checkpoints)).astype(int)).tolist() + [0]))
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    probe_batch = Xtr[:512]

    print(f"[{cfg.run_name}] task={cfg.task} |train|={len(Xtr)} |val|={len(Xva)} "
          f"batch={bs} wd={cfg.weight_decay} params={model.n_params:,} "
          f"(non-emb {model.n_params_non_embedding:,})", flush=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "n_train": len(Xtr), "n_val": len(Xva),
                   "batch_size_used": bs}, f, indent=2)

    t_mem = t_grok = None
    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    t0 = time.time()

    for step in range(cfg.max_steps + 1):
        # ---- evaluation / logging (before the update at this step) ----
        if step % cfg.cheap_every == 0:
            tr, va = evaluate(model, Xtr, Ytr), evaluate(model, Xva, Yva)
            if t_mem is None and tr["acc"] > 0.99:
                t_mem = step
            if t_grok is None and va["acc"] > 0.99:
                t_grok = step
            scal = {f"train_{k}": v for k, v in tr.items()}
            scal.update({f"val_{k}": v for k, v in va.items()})
            scal["lr"] = opt.param_groups[0]["lr"]
            scal["wall_s"] = time.time() - t0
            mon.log_cheap(step, scal)
            if step % (cfg.cheap_every * 40) == 0:
                print(f"[{cfg.run_name}] {step:>7d} "
                      f"train {tr['loss']:.4f}/{tr['acc']:.3f}  "
                      f"val {va['loss']:.4f}/{va['acc']:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if step % cfg.expensive_every == 0:
            mon.log_expensive(step, probe_batch)
        if step in ckpt_steps:
            torch.save({"step": step, "model": model.state_dict()},
                       os.path.join(ckpt_dir, f"ckpt_{step:07d}.pt"))
        if step == cfg.max_steps:
            break

        # ---- optimisation step ----
        # linear warmup over the first 10 updates, then constant (the paper does
        # not anneal). Applied as a multiplier on each group's own initial LR.
        mult = min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        wd_on = (step >= cfg.wd_start_step) and \
                (cfg.wd_stop_step < 0 or step < cfg.wd_stop_step)
        for pg in opt.param_groups:
            pg["lr"] = pg["initial_lr"] * mult
            if cfg.wd_start_step > 0 or cfg.wd_stop_step >= 0:
                base = cfg.weight_decay if pg.get("wd_on", True) else 0.0
                pg["weight_decay"] = base if wd_on else 0.0
        idx = torch.randint(0, len(Xtr), (bs,), device=dev, generator=g)
        loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    tr, va = evaluate(model, Xtr, Ytr), evaluate(model, Xva, Yva)
    summary = {
        "config": asdict(cfg), "n_train": len(Xtr), "n_val": len(Xva),
        "final_train": tr, "final_val": va,
        "t_memorise": t_mem, "t_grok": t_grok,
        "grokking_gap": (t_grok / t_mem) if (t_mem and t_grok and t_mem > 0) else None,
        "wall_time_s": time.time() - t0,
        "steps_per_s": cfg.max_steps / max(1e-9, time.time() - t0),
    }
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    mon.close()
    print(f"[{cfg.run_name}] DONE t_mem={t_mem} t_grok={t_grok} "
          f"val_acc={va['acc']:.4f} ({summary['wall_time_s']:.0f}s, "
          f"{summary['steps_per_s']:.0f} steps/s)", flush=True)


if __name__ == "__main__":
    main()
