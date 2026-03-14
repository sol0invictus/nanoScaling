"""
Tensor Parallelism (TP) utilities for nanoScaling.

Tensor Parallelism splits individual layers across multiple GPUs using the
Megatron-LM column-then-row pattern:

  Input (replicated)
       │
  ColumnParallelLinear   →  each rank holds out_features // tp_size output rows
       │                    (no all-reduce needed; output stays sharded)
  [activation fn]
       │
  RowParallelLinear      →  each rank holds in_features // tp_size input cols
       │                    all-reduce(SUM) → full output (replicated again)
  Output (replicated)

Typical mappings:
  Attention:  Q/K/V projections = ColumnParallel, output projection = RowParallel
  MLP/SwiGLU: first linear(s)   = ColumnParallel, last linear       = RowParallel

Process group topology for combined TP + DDP (world_size = tp_size × dp_size):
  rank r:  tp_rank = r % tp_size,  dp_rank = r // tp_size
  TP group for dp_rank k : ranks [k*tp_size, …, k*tp_size + tp_size - 1]
  DP group for tp_rank j : ranks [j, j+tp_size, j+2*tp_size, …]

Example with 8 GPUs, tp_size=4, dp_size=2:
  TP groups: {0,1,2,3}  {4,5,6,7}
  DP groups: {0,4}  {1,5}  {2,6}  {3,7}

Usage:
    # Before creating the model:
    from utils.tensor_parallel import init_tensor_parallel
    init_tensor_parallel(tp_group)   # or init_tensor_parallel(None) for tp=1

    # In layer definitions:
    from utils.tensor_parallel import get_tp_world_size, ColumnParallelLinear, RowParallelLinear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Global tensor-parallel state (module-level singletons)
# ---------------------------------------------------------------------------

_TP_GROUP = None        # torch.distributed process group or None
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1


def init_tensor_parallel(tp_group) -> None:
    """
    Initialise the global TP state.  Must be called once before model creation.

    Args:
        tp_group: A ``torch.distributed`` process group containing all ranks
                  in the same tensor-parallel slice, or ``None`` for tp_size=1.
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE
    _TP_GROUP = tp_group
    if tp_group is not None and dist.is_initialized():
        _TP_RANK = dist.get_rank(tp_group)
        _TP_WORLD_SIZE = dist.get_world_size(tp_group)
    else:
        _TP_RANK = 0
        _TP_WORLD_SIZE = 1


def get_tp_rank() -> int:
    """Return this rank's position within its TP group (0-indexed)."""
    return _TP_RANK


def get_tp_world_size() -> int:
    """Return the number of ranks in the TP group (1 when TP is disabled)."""
    return _TP_WORLD_SIZE


def get_tp_group():
    """Return the TP process group (``None`` when TP is disabled)."""
    return _TP_GROUP


# ---------------------------------------------------------------------------
# Parallel linear layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The weight matrix W (out_features × in_features) is sharded along the
    output (row) dimension.  Rank *r* holds rows
    ``[r * out_local : (r+1) * out_local]``  where
    ``out_local = out_features // tp_world_size``.

    Forward pass:
        input  : ``(*, in_features)``  — **identical on all TP ranks**
        output : ``(*, out_local)``    — **different on each TP rank**
                 or ``(*, out_features)`` when ``gather_output=True``

    No all-reduce is required in the forward pass when ``gather_output=False``
    because the ``RowParallelLinear`` that immediately follows does the
    all-reduce after it has accumulated partial sums.

    Constraint: ``out_features`` must be divisible by ``tp_world_size``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        tp_size = get_tp_world_size()
        assert out_features % tp_size == 0, (
            f"ColumnParallelLinear: out_features ({out_features}) must be "
            f"divisible by tp_world_size ({tp_size})"
        )
        self.out_features_local = out_features // tp_size

        self.weight = nn.Parameter(torch.empty(self.out_features_local, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features_local)) if bias else None

        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output and _TP_WORLD_SIZE > 1:
            gathered = [torch.empty_like(out) for _ in range(_TP_WORLD_SIZE)]
            dist.all_gather(gathered, out.contiguous(), group=_TP_GROUP)
            out = torch.cat(gathered, dim=-1)
        return out

    def extra_repr(self) -> str:  # pragma: no cover
        return (
            f"in={self.in_features}, "
            f"out_local={self.out_features_local}/{self.out_features}, "
            f"bias={self.bias is not None}, "
            f"gather_output={self.gather_output}"
        )


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The weight matrix W (out_features × in_features) is sharded along the
    input (column) dimension.  Rank *r* holds columns
    ``[:, r * in_local : (r+1) * in_local]``  where
    ``in_local = in_features // tp_world_size``.

    Forward pass:
        input  : ``(*, in_local)``      — **already sharded** (output of
                                           ``ColumnParallelLinear``)
        output : ``(*, out_features)``  — **all-reduced** across TP ranks

    The bias is added **after** the all-reduce so that it contributes only once.

    Constraint: ``in_features`` must be divisible by ``tp_world_size``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        tp_size = get_tp_world_size()
        assert in_features % tp_size == 0, (
            f"RowParallelLinear: in_features ({in_features}) must be "
            f"divisible by tp_world_size ({tp_size})"
        )
        self.in_features_local = in_features // tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_local))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Partial matrix multiply — no bias yet
        out = F.linear(x, self.weight)
        # Sum partial results across the TP group
        if _TP_WORLD_SIZE > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=_TP_GROUP)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:  # pragma: no cover
        return (
            f"in_local={self.in_features_local}/{self.in_features}, "
            f"out={self.out_features}, "
            f"bias={self.bias is not None}"
        )
