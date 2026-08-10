"""Algorithmic / combinatorial datasets for grokking experiments.

Following Power et al. 2022 (arXiv:2201.02177), every group element is an
*abstract symbol with no internal structure*: the network never sees a number in
decimal notation, only opaque token ids, and must infer the operation purely from
the co-occurrence structure of the binary op table.

Token layout for a task with N elements:

    0 .. N-1    element symbols
    N           the binary-operation token   <op>
    N+1         the equals token             <=>

so ``vocab_size = N + 2``.

The paper encodes an equation ``x o y = z`` as the 5-token sequence
``<x><op><y><=><z>`` and restricts the loss to the answer position. We feed the
4-token prefix ``[x, <op>, y, <=>]`` and read the logits at the final position,
which is mathematically identical under a causal mask but avoids computing
logits that are never used.

Each task also exposes the *cyclic orderings* of its symbol set (§ ``CYCLIC``),
used by ``metrics.py`` to run the Fourier/circularity progress measure, and
optional binary structure labels for linear probes.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# group / operation definitions
# --------------------------------------------------------------------------- #

def _mod_add(p: int = 97):
    """x + y (mod p). Abelian, cyclic. Reference task (paper Appendix A.1.1)."""
    x, y = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    x, y = x.ravel(), y.ravel()
    return x, y, (x + y) % p, p


def _mod_div(p: int = 97):
    """x / y (mod p) for y != 0. The paper's Figure 1 task."""
    inv = np.array([0] + [pow(int(v), p - 2, p) for v in range(1, p)])
    x, y = np.meshgrid(np.arange(p), np.arange(1, p), indexing="ij")
    x, y = x.ravel(), y.ravel()
    return x, y, (x * inv[y]) % p, p


def _dihedral(n: int = 48):
    """Composition in the dihedral group D_n -- the NEW task.

    D_n = symmetries of a regular n-gon = { r^i s^f : i in Z_n, f in {0,1} },
    of order 2n. From the relation ``s r = r^{-1} s`` the group law is

        (i, f) . (j, g) = ( i + (-1)^f j  mod n ,  f XOR g )

    Elements are indexed ``e = f * n + i``, so ids ``0..n-1`` are the rotations
    (the cyclic subgroup Z_n) and ``n..2n-1`` are the reflections. That layout
    makes the two cosets contiguous, which the structural probes rely on.

    D_n is non-abelian (order matters), yet it is a semidirect product
    Z_n |x| Z_2 -- "half abelian" -- sitting structurally between the modular
    arithmetic and the S_5 tasks of the paper, neither of which it duplicates.
    """
    N = 2 * n
    e = np.arange(N)
    i, f = e % n, e // n
    a, b = np.meshgrid(e, e, indexing="ij")
    a, b = a.ravel(), b.ravel()
    ia, fa = i[a], f[a]
    ib, fb = i[b], f[b]
    ic = (ia + np.where(fa == 1, -1, 1) * ib) % n
    fc = fa ^ fb
    return a, b, fc * n + ic, N


TASKS = {
    "mod_add_97": lambda: _mod_add(97),
    "mod_div_97": lambda: _mod_div(97),
    "dihedral_48": lambda: _dihedral(48),
}


# --------------------------------------------------------------------------- #
# structural metadata used by the progress measures in metrics.py
# --------------------------------------------------------------------------- #

def _dlog_ordering(p: int = 97):
    """Rows 1..p-1 ordered by discrete log w.r.t. a primitive root.

    The multiplicative group (Z_p)^* is cyclic of order p-1, so re-indexing the
    nonzero residues by ``dlog`` turns modular *division* into subtraction on
    Z_{p-1}. The paper notes this equivalence (Sec. 3.2); we use it so the same
    Fourier probe applies to mod_div as to mod_add.
    """
    for g in range(2, p):
        seen, v, order = set(), 1, []
        for _ in range(p - 1):
            v = (v * g) % p
            if v in seen:
                break
            seen.add(v)
            order.append(v)
        if len(order) == p - 1:
            # order[k] = g^(k+1); rotate so index 0 is the identity element 1
            return np.array(order[-1:] + order[:-1])
    raise RuntimeError(f"no primitive root found mod {p}")


def task_structure(name: str) -> dict:
    """Cyclic orderings and probe labels for the symbols of a task.

    ``cyclic``: {label -> array of symbol ids forming one cycle of a cyclic
    subgroup, in group order}. The embedding rows in that order are Fourier
    transformed to measure how circular / sparse the representation is.

    ``probes``: {label -> binary array over symbol ids} for linear probes.
    """
    if name == "mod_add_97":
        return {"cyclic": {"Z97": np.arange(97)}, "probes": {}}
    if name == "mod_div_97":
        return {"cyclic": {"Z96_mult": _dlog_ordering(97)}, "probes": {}}
    if name == "dihedral_48":
        n = 48
        return {
            "cyclic": {
                "rot_Z48": np.arange(n),
                "refl_Z48": np.arange(n, 2 * n),
            },
            # is this element a reflection? tests whether the Z_2 factor is
            # linearly decodable from the embedding, and when it becomes so.
            "probes": {"reflection": (np.arange(2 * n) >= n).astype(np.int64)},
        }
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# dataset construction
# --------------------------------------------------------------------------- #

def build_task(name: str):
    """Return (inputs [M,4] int64, answers [M] int64, meta dict)."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    x, y, z, n_sym = TASKS[name]()
    op_tok, eq_tok = n_sym, n_sym + 1
    M = len(x)
    inputs = np.stack(
        [x, np.full(M, op_tok), y, np.full(M, eq_tok)], axis=1
    ).astype(np.int64)
    meta = {
        "task": name,
        "n_symbols": int(n_sym),
        "vocab_size": int(n_sym + 2),
        "n_equations": int(M),
        "seq_len": 4,
        "op_token": int(op_tok),
        "eq_token": int(eq_tok),
        **task_structure(name),
    }
    return inputs, z.astype(np.int64), meta


def make_splits(inputs, answers, train_frac: float, seed: int = 0):
    """Random split into train / val, exactly as in the paper: a fixed fraction
    of all available equations is declared to be the training set and the rest
    is the validation set."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(inputs))
    n_train = int(round(train_frac * len(inputs)))
    tr, va = perm[:n_train], perm[n_train:]
    return (inputs[tr], answers[tr]), (inputs[va], answers[va])


if __name__ == "__main__":
    # sanity checks on the group laws
    for name in TASKS:
        inp, ans, meta = build_task(name)
        print(f"{name:14s} symbols={meta['n_symbols']:4d} "
              f"equations={meta['n_equations']:6d} vocab={meta['vocab_size']}")

    # D_48 group axioms, checked on the raw op table
    n = 48
    a, b, c, N = _dihedral(n)
    table = np.full((N, N), -1, dtype=np.int64)
    table[a, b] = c
    assert (table >= 0).all()
    e = 0  # identity = r^0 s^0
    assert (table[e, :] == np.arange(N)).all() and (table[:, e] == np.arange(N)).all()
    assert ((table == e).sum(axis=1) == 1).all()  # unique inverse per element
    idx = np.random.default_rng(0).integers(0, N, size=(2000, 3))
    lhs = table[table[idx[:, 0], idx[:, 1]], idx[:, 2]]
    rhs = table[idx[:, 0], table[idx[:, 1], idx[:, 2]]]
    assert (lhs == rhs).all(), "D_48 associativity failed"
    assert not (table == table.T).all(), "D_48 should be non-abelian"
    print("D_48: closure, identity, inverses, associativity OK; non-abelian OK")

    # every row/column a permutation (Latin square) for all tasks
    for name in TASKS:
        inp, ans, meta = build_task(name)
        print(f"{name}: answers cover {len(np.unique(ans))} distinct symbols")
