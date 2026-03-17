import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class ParametrizationConfig:
    mode: str = 'SP' # 'SP', 'MuP', 'CompleteP'
    # Optional overrides for specific parameter types if needed in future
    # e.g., {'embedding': {'init_std': '1/sqrt(d)', 'lr_mult': 10.0}}
    # For now, the mode determines the logic in parametrization.py

@dataclass
class ExperimentConfig:
    # I/O
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1          # console print frequency (steps)
    metrics_log_interval: int = 0  # TensorBoard/CSV logging frequency (0 = same as log_interval)
    eval_iters: int = 200
    eval_only: bool = False
    checkpoint_interval: int = 0   # save checkpoint every N steps (0 = tie to eval_interval)
    always_save_checkpoint: bool = True
    keep_last_n_checkpoints: int = 3  # keep N most recent numbered checkpoints (0 = keep all)
    init_from: str = 'scratch'

    # Logging
    tensorboard_log: bool = True
    tensorboard_run_name: str = 'gpt2_muon'

    # Data
    dataset: str = 'openwebtext'
    val_splits: List[str] = field(default_factory=lambda: ['val'])
    # Validation dataset folders evaluated at each eval_interval alongside val_splits.
    # Each entry is a path to a dataset folder (e.g. 'data/wikitext103', 'data/pile_val').
    # Both parquet and bin formats are auto-detected per folder.
    val_datasets: List[str] = field(default_factory=list)
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    block_size: int = 1024
    data_format: str = 'auto'           # 'auto' (detect), 'bin' (memmap), 'parquet' (on-the-fly)
    dataloader_buffer_size: int = 1000  # parquet mode: number of docs in tokenization buffer
    tokenizer_batch_size: int = 128     # parquet mode: reserved for future threaded tokenization

    # Model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # default to False for modern architecture

    # Architecture Toggles
    use_rmsnorm: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    multiple_of: int = 256 # for SwiGLU

    # MoE
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_every: int = 2  # Apply MoE every N layers (2 means 0, 2, 4...)
    norm_topk_prob: bool = True
    load_balance_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    moe_hidden_dim: int = 0      # 0 = default 4*n_embd; set explicitly to override
    moe_block_size: int = 128    # Triton tile size for block-sparse kernels

    # Flash Attention 3 (requires: pip install flash-attn>=3.0, H100/Hopper GPU recommended)
    use_flash_attn3: bool = False

    # Hybrid (Gated Delta Net interleaved with standard attention)
    # Requires: pip install flash-linear-attention
    use_hybrid: bool = False
    delta_net_every: int = 2      # layers 0, delta_net_every, 2*delta_net_every, ... use GDN
    delta_net_chunk_size: int = 64  # chunk size for FLA chunked parallel scan

    # Normalization (for RQ2 ablations)
    norm_position: str = 'pre'   # 'pre' (Pre-LN), 'post' (Post-LN), 'none' (no normalization)
    norm_affine: bool = True     # whether norm layers have learnable gamma/beta
    norm_free_scaled_init: bool = False  # tighter residual init for norm-free training

    # Optimizer
    optimizer: str = 'adamw' # 'adamw' or 'muon' or 'scion'

    # Muon Optimizer Args
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.0

    # Scion Optimizer Args
    scion_norm: str = 'Auto'
    scion_scale: float = 1.0
    scion_momentum: float = 1.0
    scion_unconstrained: bool = False

    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR Scheduler
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # System
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = True
    backend: str = 'nccl'

    # Parametrization
    parametrization: ParametrizationConfig = field(default_factory=ParametrizationConfig)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Enforce types based on default values (dataclasses don't auto-convert)
        # This fixes issues where scientific notation like "6e-4" is loaded as str by PyYAML
        default_obj = cls()
        for k, v in data.items():
            if k == 'parametrization': continue
            if hasattr(default_obj, k):
                # Get the type of the default value
                expected_type = type(getattr(default_obj, k))
                # Only attempt cast for primitives
                if expected_type in (int, float, bool, str):
                    try:
                        if expected_type == bool and isinstance(v, str):
                            data[k] = v.lower() == 'true'
                        elif expected_type != type(v):
                            data[k] = expected_type(v)
                            # print(f"Casted {k} from {type(v)} to {expected_type}")
                    except Exception:
                        pass

        # Handle nested config
        if 'parametrization' in data:
            data['parametrization'] = ParametrizationConfig(**data['parametrization'])
        return cls(**data)

    def to_dict(self):
        # simple recursion for dataclasses
        d = vars(self).copy()
        if isinstance(d['parametrization'], ParametrizationConfig):
            d['parametrization'] = vars(d['parametrization'])
        return d
