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
