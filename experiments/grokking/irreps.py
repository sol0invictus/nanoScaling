"""Representation-theoretic analysis of what the network learns on D_48.

For an abelian group the natural basis for a function on the group is the Fourier
basis, which is what `metrics.py` uses on the cyclic subgroup. D_48 is *non*-abelian,
so the correct generalisation is the decomposition into **irreducible
representations** (irreps): every function f: G -> R decomposes as

    power_rho(f) = (d_rho / |G|) * || sum_g f(g) rho(g) ||_F^2 ,
    sum_rho power_rho(f) = sum_g f(g)^2                        (Parseval)

D_48 has 27 irreps: four 1-dimensional and twenty-three 2-dimensional
(4*1 + 23*4 = 96 = |G|). The 2-dimensional ones carry the genuinely non-abelian
structure, which a cyclic-subgroup Fourier measure cannot see.

We treat each of the 128 embedding dimensions as a function on the group and ask how
concentrated the total power is across irreps. A network that has learned the group
should use a *sparse* set; a memorising network should look near-uniform.
"""

from __future__ import annotations

import os

import numpy as np
import torch

N_ROT = 48
N_ELEM = 2 * N_ROT


def build_irreps(n: int = N_ROT):
    """All irreps of D_n (n even), indexed by element id e = f*n + i."""
    N = 2 * n
    elem = lambda e: (e % n, e // n)
    out = []
    ones = [("triv", lambda i, f: 1.0),
            ("sgn_s", lambda i, f: (-1.0) ** f),
            ("sgn_r", lambda i, f: (-1.0) ** i),
            ("sgn_rs", lambda i, f: (-1.0) ** (i + f))]
    for name, chi in ones:
        M = np.array([[[chi(*elem(e))]] for e in range(N)])
        out.append((name, 1, M))
    S = np.array([[1.0, 0.0], [0.0, -1.0]])
    for k in range(1, n // 2):
        M = np.zeros((N, 2, 2))
        for e in range(N):
            i, f = elem(e)
            th = 2 * np.pi * k * i / n
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            M[e] = R @ (S if f else np.eye(2))
        out.append((f"rho_{k}", 2, M))
    assert sum(d * d for _, d, _ in out) == N, "irrep dimensions must sum to |G|"
    return out


IRREPS = build_irreps()
IRREP_NAMES = [n for n, _, _ in IRREPS]


def irrep_power(F: np.ndarray, irreps=IRREPS) -> np.ndarray:
    """F: (|G|, C) columns are functions on the group -> power per irrep."""
    N = F.shape[0]
    return np.array([d / N * (np.einsum("gc,gij->cij", F, M) ** 2).sum()
                     for _, d, M in irreps])


def concentration(F: np.ndarray) -> dict:
    """Sparsity summary of the irrep spectrum of a matrix of group functions."""
    F = F - F.mean(axis=0, keepdims=True)
    p = irrep_power(F)
    total = p.sum()
    assert abs(total - (F ** 2).sum()) < 1e-4 * max(1.0, (F ** 2).sum()), "Parseval failed"
    frac = p / total if total > 0 else p
    order = np.argsort(frac)[::-1]
    cum = np.cumsum(frac[order])
    return {
        # inverse Simpson index: the *effective* number of irreps in use.
        # 27 = perfectly uniform (no structure), 1 = a single irrep.
        "eff_n_irreps": float(1.0 / (frac ** 2).sum()),
        "n_for_90pct": int((cum < 0.9).sum() + 1),
        "top1_mass": float(frac[order[0]]),
        "top3_mass": float(frac[order[:3]].sum()),
        "mass_1dim": float(frac[:4].sum()),
        "mass_2dim": float(frac[4:].sum()),
        "top_irrep": IRREP_NAMES[order[0]],
        "frac": frac,
    }


# --------------------------------------------------------------------------- #
# checkpoint sweeps
# --------------------------------------------------------------------------- #

def _ckpt_steps(run_dir):
    d = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(d):
        return []
    return sorted(int(f[5:-3]) for f in os.listdir(d) if f.startswith("ckpt_"))


def load_matrices(run_dir, step):
    sd = torch.load(os.path.join(run_dir, "checkpoints", f"ckpt_{step:07d}.pt"),
                    map_location="cpu")["model"]
    return (sd["tok_emb.weight"].float().numpy()[:N_ELEM],
            sd["unembed.weight"].float().numpy()[:N_ELEM])


def sweep_run(run_dir):
    """Irrep concentration at every checkpoint of a run."""
    rows = []
    for s in _ckpt_steps(run_dir):
        try:
            E, U = load_matrices(run_dir, s)
        except Exception:
            continue
        ce, cu = concentration(E), concentration(U)
        rows.append({"step": s,
                     **{f"emb_{k}": v for k, v in ce.items() if k != "frac"},
                     **{f"unemb_{k}": v for k, v in cu.items() if k != "frac"},
                     "_emb_frac": ce["frac"]})
    return rows
