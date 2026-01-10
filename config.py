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
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'
    
    # Logging
    wandb_log: bool = False
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2'
    tensorboard_log: bool = True
    tensorboard_run_name: str = 'gpt2_muon'
    
    # Data
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    block_size: int = 1024
    
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
    
    # Optimizer
    optimizer: str = 'adamw' # 'adamw' or 'muon' (or mixed)
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
