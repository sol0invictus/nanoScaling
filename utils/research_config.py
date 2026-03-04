from dataclasses import dataclass


@dataclass
class ResearchConfig:
    """
    Optional research/experiment logging config.

    Load by adding a 'research:' nested block to any YAML config:

        research:
          enable_spectral_logging: true
          proportional_lr_decay: true
          svd_checkpoint_freq: 2000

    All fields default to off so the base training path is unaffected.
    """
    # RQ1: SpectralLogger — weight geometry, grad norms, activations, CSV files in out_dir/logs/
    enable_spectral_logging: bool = False

    # RQ1: Full SVD dump frequency in steps (0 = only at end of run)
    svd_checkpoint_freq: int = 0

    # RQ1: Sharpness measurement frequency (0 = disabled; expensive: N forward passes per step)
    sharpness_freq: int = 0

    # RQ1: k for top-k SVD entropy
    spectral_topk: int = 32

    # RQ2: StabilityMonitor — grad spike detection, NaN early stopping, is_stable in metrics.json
    enable_stability_monitor: bool = False

    # RQ1+RQ2: Per-optimizer-group cosine decay (required for Muon+AdamW CombinedOptimizer)
    proportional_lr_decay: bool = False
