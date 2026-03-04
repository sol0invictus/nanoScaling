"""
Spectral Dynamics Logger — measurement infrastructure for RQ1 / RQ2 / RQ4.

Implements the full measurement toolkit from the Muon research proposal:

All metrics fire together every time log_step() is called (no internal freq gating).
Call it from the training loop at log_interval cadence. The only separate gate is
sharpness_freq, since sharpness requires N full forward passes.

Per-call metrics (weight matrices):
  - Stable rank:              ||W||_F² / ||W||_2²
  - Per-layer gradient Frobenius norm, spectral norm, direction cosine
  - Nuclear / Frobenius ratio: ||W||_* / ||W||_F
  - Spectral entropy H(W) = -Σ p_i log p_i  where p_i = σ_i² / Σ σ_j²
  - Effective rank = exp(H(W))

Per-call metrics (activations, via forward hooks):
  - Mean, variance, kurtosis (excess)
  - Dead neuron fraction  (|x| < 1e-3)
  - Representation isotropy (1 − mean pairwise cosine similarity)

Checkpoint-level metrics (full SVD, call log_checkpoint()):
  - Complete singular value spectrum (stored to JSON)
  - Stable rank, nuc/frob, entropy, effective rank, threshold rank

Sharpness measurement (only when step % sharpness_freq == 0, 0 = disabled):
  - Mean loss increase over n_samples random ε-perturbations of weights

Stability monitoring (for RQ2):
  - NaN / Inf detection in weights and activations
  - Gradient norm spike detection (>100× rolling median)
"""

import csv
import math
import json
import os
from collections import deque
from typing import Dict, Optional, List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CSV Logger — raw data for offline analysis
# ---------------------------------------------------------------------------

class CSVLogger:
    """
    Writes per-category metrics to CSV files under a given directory.

    One file per category; rows are appended incrementally during training so
    the files are readable even if a run is interrupted.  Each file is opened
    in append mode, so resuming a run continues from where it left off.

    Files created:
        logs/weight.csv      — per-parameter weight-geometry metrics
        logs/grad.csv        — per-parameter gradient norms + direction
        logs/activations.csv — per-block activation statistics
        logs/sharpness.csv   — loss-landscape sharpness samples

    Load with pandas after the run:
        import pandas as pd
        df = pd.read_csv('out/logs/weight.csv')
    """

    SCHEMA: Dict[str, List[str]] = {
        'weight':      ['step', 'param', 'stable_rank', 'nuc_frob_ratio',
                        'spectral_entropy', 'effective_rank'],
        'grad':        ['step', 'param', 'fro_norm', 'spec_norm', 'direction_cosine'],
        'activations': ['step', 'layer', 'mean', 'var', 'std', 'kurtosis',
                        'dead_fraction', 'isotropy'],
        'sharpness':   ['step', 'value'],
    }

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._files = {}
        self._writers: Dict[str, csv.DictWriter] = {}
        for name, cols in self.SCHEMA.items():
            path = os.path.join(log_dir, f'{name}.csv')
            new_file = not os.path.exists(path)
            f = open(path, 'a', newline='', buffering=1)   # line-buffered
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            if new_file:
                writer.writeheader()
            self._files[name] = f
            self._writers[name] = writer

    def write(self, category: str, row: dict) -> None:
        if category in self._writers:
            self._writers[category].writerow(row)

    def flush(self) -> None:
        for f in self._files.values():
            f.flush()

    def close(self) -> None:
        for f in self._files.values():
            f.close()


# ---------------------------------------------------------------------------
# Scalar metric primitives
# ---------------------------------------------------------------------------

@torch.no_grad()
def stable_rank(W: torch.Tensor) -> float:
    """Stable rank = ||W||_F² / ||W||_2²  (spectral norm squared).

    Equals 1 for rank-1 matrices; equals min(m, n) for orthogonal matrices.
    Cheap: Frobenius norm is O(mn), spectral norm uses one matrix-norm call.
    """
    W = W.float()
    fro_sq = W.norm('fro').pow(2).item()
    spec = torch.linalg.matrix_norm(W, ord=2).item()
    return fro_sq / (spec ** 2 + 1e-12)


@torch.no_grad()
def topk_svd_metrics(W: torch.Tensor, k: int = 32) -> Dict[str, float]:
    """Nuclear/Frobenius ratio + spectral entropy from full SVD (top-k used for entropy).

    Returns dict with keys: nuc_frob_ratio, spectral_entropy, effective_rank.
    """
    try:
        sv = torch.linalg.svdvals(W.float())  # descending, no U/V
    except Exception:
        return {'nuc_frob_ratio': float('nan'),
                'spectral_entropy': float('nan'),
                'effective_rank': float('nan')}

    # Nuclear / Frobenius uses all singular values
    nuc = sv.sum().item()
    fro = sv.pow(2).sum().sqrt().item()
    nuc_frob = nuc / (fro + 1e-12)

    # Spectral entropy from top-k
    sv_k = sv[:min(k, sv.shape[0])]
    sv_k_sq = sv_k.pow(2)
    p = sv_k_sq / (sv_k_sq.sum() + 1e-12)
    p = p.clamp(min=1e-12)
    entropy = -(p * p.log()).sum().item()
    effective_rank = math.exp(entropy)

    return {
        'nuc_frob_ratio': nuc_frob,
        'spectral_entropy': entropy,
        'effective_rank': effective_rank,
    }


@torch.no_grad()
def full_svd_stats(W: torch.Tensor) -> Dict:
    """Full SVD analysis for checkpoint snapshots. Returns JSON-serializable dict."""
    try:
        sv = torch.linalg.svdvals(W.float())
    except Exception:
        return {}

    sv_sq = sv.pow(2)
    fro_sq = sv_sq.sum().item()
    spec = sv[0].item()
    nuc = sv.sum().item()
    fro = math.sqrt(max(fro_sq, 0.0))

    sr = fro_sq / (spec ** 2 + 1e-12)
    nfr = nuc / (fro + 1e-12)

    p = (sv_sq / (fro_sq + 1e-12)).clamp(min=1e-12)
    entropy = -(p * p.log()).sum().item()

    threshold = spec * 1e-3
    rank_thresh = int((sv > threshold).sum().item())

    return {
        'stable_rank': sr,
        'nuc_frob_ratio': nfr,
        'spectral_entropy': entropy,
        'effective_rank': math.exp(entropy),
        'rank_1e3': rank_thresh,
        'top1_sv': spec,
        'top5_sv': sv[:min(5, len(sv))].tolist(),
        'fro_norm': fro,
        'nuc_norm': nuc,
        # Full spectrum truncated to 256 for storage efficiency
        'singular_values': sv[:256].tolist(),
    }


@torch.no_grad()
def grad_norms(g: torch.Tensor) -> Dict[str, float]:
    """Frobenius and spectral norm of a gradient matrix."""
    g_f = g.float()
    fro = g_f.norm('fro').item()
    try:
        spec = torch.linalg.matrix_norm(g_f, ord=2).item()
    except Exception:
        spec = float('nan')
    return {'fro': fro, 'spec': spec}


@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened)."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (a_f @ b_f / (a_f.norm() * b_f.norm() + 1e-12)).item()


@torch.no_grad()
def activation_stats(act: torch.Tensor) -> Dict[str, float]:
    """Comprehensive statistics of an activation tensor.

    Supports shapes (B, T, C), (N, C), or any tensor.
    """
    a = act.float()
    mean = a.mean().item()
    var = a.var().item()
    std = math.sqrt(max(var, 0.0))

    # Excess kurtosis
    if std > 1e-8:
        kurt = ((a - mean).pow(4).mean().item() / (std ** 4)) - 3.0
    else:
        kurt = 0.0

    dead = (a.abs() < 1e-3).float().mean().item()

    # Representation isotropy: 1 - mean pairwise cosine similarity
    # Project to 2D (tokens/samples × features)
    if a.dim() == 3:
        a2d = a.reshape(-1, a.shape[-1])
    elif a.dim() == 2:
        a2d = a
    else:
        a2d = a.flatten(0, -2) if a.dim() > 1 else a.unsqueeze(0)

    n = min(a2d.shape[0], 64)
    sample = a2d[:n]
    norms = sample.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = sample / norms
    cos_mat = normed @ normed.T
    if n > 1:
        off_diag = (cos_mat.sum() - cos_mat.trace()).item() / (n * (n - 1))
    else:
        off_diag = 0.0
    isotropy = 1.0 - off_diag

    return {
        'mean': mean,
        'var': var,
        'std': std,
        'kurtosis': kurt,
        'dead_fraction': dead,
        'isotropy': isotropy,
    }


@torch.no_grad()
def measure_sharpness(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    ctx,
    n_samples: int = 32,
    epsilon: float = 1e-3,
) -> float:
    """Sharpness via random perturbation: mean increase in loss over n_samples trials."""
    model.eval()
    with ctx:
        _, base_loss = model(X, Y)
    base = base_loss.item()

    saved = {n: p.data.clone() for n, p in model.named_parameters()}
    deltas: List[float] = []

    for _ in range(n_samples):
        for p in model.parameters():
            p.data.add_(torch.randn_like(p.data) * epsilon)
        with ctx:
            _, perturbed = model(X, Y)
        deltas.append(perturbed.item() - base)
        for n, p in model.named_parameters():
            p.data.copy_(saved[n])

    model.train()
    return sum(deltas) / max(len(deltas), 1)


# ---------------------------------------------------------------------------
# Stability monitor (for RQ2 convergence criterion)
# ---------------------------------------------------------------------------

class StabilityMonitor:
    """
    Tracks gradient norm history and loss history to apply the RQ2 stability
    criterion:
      1. val loss decreases monotonically over any 2000-step window
      2. per-layer gradient norms bounded (no spike > 100× rolling median over 500 steps)
      3. no NaN/Inf in weights or activations

    Call update_grad_norm(step, name, norm) after each backward pass.
    Call check(step, val_loss, model, act_cache) at eval intervals.
    """

    def __init__(self, spike_window: int = 500, spike_factor: float = 100.0,
                 loss_window: int = 2000):
        self.spike_window = spike_window
        self.spike_factor = spike_factor
        self.loss_window = loss_window
        self._grad_history: Dict[str, deque] = {}
        self._loss_history: deque = deque(maxlen=loss_window + 1)

    def update_grad_norm(self, name: str, norm: float):
        if name not in self._grad_history:
            self._grad_history[name] = deque(maxlen=self.spike_window)
        self._grad_history[name].append(norm)

    def check(
        self,
        step: int,
        val_loss: float,
        model: nn.Module,
        act_cache: Dict[str, torch.Tensor],
        writer=None,
    ) -> Dict:
        issues = []

        # --- NaN/Inf in weights ---
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                issues.append(f'non_finite_weight:{name}')

        # --- NaN/Inf in activations ---
        for name, act in act_cache.items():
            if not torch.isfinite(act).all():
                issues.append(f'non_finite_activation:{name}')

        # --- Gradient norm spikes ---
        spike_layers = []
        for name, history in self._grad_history.items():
            if len(history) < 10:
                continue
            vals = sorted(history)
            median = vals[len(vals) // 2]
            if median > 0 and history[-1] > self.spike_factor * median:
                spike_layers.append(name)
                issues.append(f'grad_spike:{name}')

        # --- Loss monotonicity (over window) ---
        self._loss_history.append(val_loss)
        loss_increasing = False
        if len(self._loss_history) >= 2:
            recent = list(self._loss_history)[-min(self.loss_window, len(self._loss_history)):]
            # Count how many steps loss increased relative to minimum seen so far
            min_loss = recent[0]
            for l in recent[1:]:
                if l > min_loss * 1.05:  # 5% tolerance
                    loss_increasing = True
                min_loss = min(min_loss, l)

        is_stable = len(issues) == 0 and math.isfinite(val_loss)

        result = {
            'is_stable': is_stable,
            'issues': issues,
            'spike_layers': spike_layers,
            'loss_increasing': loss_increasing,
        }

        if writer is not None:
            writer.add_scalar('stability/is_stable', float(is_stable), step)
            writer.add_scalar('stability/num_issues', len(issues), step)
            writer.add_scalar('stability/grad_spikes', len(spike_layers), step)
            writer.add_scalar('stability/loss_increasing', float(loss_increasing), step)

        return result


# ---------------------------------------------------------------------------
# SpectralLogger — composable callback
# ---------------------------------------------------------------------------

class SpectralLogger:
    """
    Composable logging callback for spectral and dynamics metrics.

    Registers forward hooks on transformer blocks (block_0 ... block_N)
    to capture activation tensors. Call log_step() at whatever cadence
    you want (typically every log_interval steps from the training loop) —
    it logs *all* metrics every time it is called.  Call log_checkpoint()
    at major checkpoints for full SVD dumps.

    The only separate frequency is sharpness_freq because sharpness requires
    N full forward passes and is fundamentally more expensive. Set to 0 to
    disable entirely (default).

    Args:
        model:             Raw (non-DDP-wrapped) model.
        tb_writer:         TensorBoard SummaryWriter, or None to skip TB.
        out_dir:           Directory for JSON checkpoint logs.
        csv_dir:           Directory for CSV raw-data logs, or None to skip.
        sharpness_freq:    Steps between sharpness measurements (0 = disabled).
        topk:              k for top-k SVD entropy computation.
        stability_monitor: Optional StabilityMonitor for RQ2 checks.
    """

    def __init__(
        self,
        model: nn.Module,
        tb_writer,
        out_dir: str = 'out',
        csv_dir: Optional[str] = None,
        sharpness_freq: int = 0,
        topk: int = 32,
        stability_monitor: Optional[StabilityMonitor] = None,
    ):
        self.model = model
        self.writer = tb_writer
        self.out_dir = out_dir
        self.sharpness_freq = sharpness_freq
        self.topk = topk
        self.stability = stability_monitor

        # CSV logger — None if csv_dir not given
        self.csv = CSVLogger(csv_dir) if csv_dir is not None else None

        # Previous gradients for direction cosine similarity
        self._prev_grads: Dict[str, torch.Tensor] = {}
        # Activation cache populated by forward hooks
        self._act_cache: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self):
        m = self.model
        if not (hasattr(m, 'transformer') and hasattr(m.transformer, 'h')):
            return
        for i, block in enumerate(m.transformer.h):
            handle = block.register_forward_hook(self._make_hook(f'block_{i}'))
            self._hooks.append(handle)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            # Block.forward returns (x, aux_loss)
            tensor = out[0] if isinstance(out, tuple) else out
            if isinstance(tensor, torch.Tensor):
                self._act_cache[name] = tensor.detach()
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Per-step logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        X: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        ctx=None,
    ):
        """
        Log all metrics for this step. Call from the training loop at whatever
        cadence you want (e.g. every log_interval steps). All metrics fire
        every call — no internal frequency gating.

        X, Y, ctx are only needed if sharpness_freq > 0.
        """
        w = self.writer
        model = self.model

        do_sharp = (self.sharpness_freq > 0
                    and step % self.sharpness_freq == 0
                    and X is not None and ctx is not None)

        for name, p in model.named_parameters():
            if p.dim() != 2:
                continue

            W = p.detach()
            g = p.grad.detach() if p.grad is not None else None

            # --- Gradient metrics ---
            if g is not None:
                gn = grad_norms(g)
                if w:
                    w.add_scalar(f'grad/fro_norm/{name}', gn['fro'], step)
                    w.add_scalar(f'grad/spec_norm/{name}', gn['spec'], step)
                if self.stability is not None:
                    self.stability.update_grad_norm(name, gn['fro'])
                sim = None
                if name in self._prev_grads:
                    sim = cosine_sim(g, self._prev_grads[name].to(g.device))
                    if w:
                        w.add_scalar(f'grad/direction_cosine/{name}', sim, step)
                self._prev_grads[name] = g.float().cpu()
                if self.csv:
                    self.csv.write('grad', {
                        'step': step, 'param': name,
                        'fro_norm': gn['fro'], 'spec_norm': gn['spec'],
                        'direction_cosine': sim if sim is not None else '',
                    })

            # --- Weight geometry: stable rank + top-k SVD ---
            sr = stable_rank(W)
            svd_m = topk_svd_metrics(W, k=self.topk)
            if w:
                w.add_scalar(f'weight/stable_rank/{name}', sr, step)
                for k_name, v in svd_m.items():
                    w.add_scalar(f'weight/{k_name}/{name}', v, step)
            if self.csv:
                self.csv.write('weight', {
                    'step': step, 'param': name,
                    'stable_rank': sr,
                    **svd_m,
                })

        # --- Activation statistics ---
        for name, act in self._act_cache.items():
            stats = activation_stats(act)
            if w:
                for stat_key, v in stats.items():
                    w.add_scalar(f'activations/{stat_key}/{name}', v, step)
            if self.csv:
                self.csv.write('activations', {'step': step, 'layer': name, **stats})
        self._act_cache.clear()

        # --- Sharpness (gated separately — expensive) ---
        if do_sharp:
            sh = measure_sharpness(model, X, Y, ctx)
            if w:
                w.add_scalar('sharpness/value', sh, step)
            if self.csv:
                self.csv.write('sharpness', {'step': step, 'value': sh})

    # ------------------------------------------------------------------
    # Checkpoint-level full SVD
    # ------------------------------------------------------------------

    def log_checkpoint(self, step: int) -> Dict:
        """
        Full SVD analysis for all 2D weight matrices. Writes JSON to out_dir.
        Returns dict keyed by parameter name.
        """
        results = {}
        for name, p in self.model.named_parameters():
            if p.dim() != 2:
                continue
            stats = full_svd_stats(p.detach())
            results[name] = stats
            if self.writer:
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        # TB: checkpoint/weight/{metric}/{param}
                        self.writer.add_scalar(f'checkpoint/weight/{k}/{name}', v, step)

        path = os.path.join(self.out_dir, f'svd_step_{step:07d}.json')
        try:
            with open(path, 'w') as f:
                json.dump({'step': step, 'svd': results}, f)
            print(f"Saved SVD checkpoint to {path}")
        except Exception as e:
            print(f"Warning: could not save SVD checkpoint: {e}")

        return results

    def close(self) -> None:
        """Flush and close CSV files. Call at the end of training."""
        if self.csv is not None:
            self.csv.close()

    # ------------------------------------------------------------------
    # RQ2 stability check
    # ------------------------------------------------------------------

    def check_stability(self, step: int, val_loss: float, writer=None) -> Dict:
        """Delegate to StabilityMonitor. Returns stability verdict dict."""
        if self.stability is None:
            return {'is_stable': True, 'issues': []}
        return self.stability.check(
            step, val_loss, self.model, self._act_cache, writer or self.writer
        )
