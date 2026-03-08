#!/usr/bin/env bash
# env_setup.bash — nanoScaling environment setup
# Usage: bash env_setup.bash [--cuda <version>] [--moe] [--hybrid]
#
# Options:
#   --cuda <version>   Install a specific PyTorch CUDA build (e.g. cu121, cu118).
#                      Default: auto-detect from nvcc, or install CPU build.
#   --moe              Also install megablocks + einops (Mixture of Experts).
#   --hybrid           Also install flash-linear-attention (Hybrid GatedDeltaNet).
#   --dev              Install optional dev tools (tensorboard, jupyterlab).
#   -h / --help        Show this message and exit.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
CUDA_VERSION=""
INSTALL_MOE=0
INSTALL_HYBRID=0
INSTALL_DEV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)   CUDA_VERSION="$2"; shift 2 ;;
        --moe)    INSTALL_MOE=1;      shift   ;;
        --hybrid) INSTALL_HYBRID=1;   shift   ;;
        --dev)    INSTALL_DEV=1;      shift   ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Detect CUDA if not supplied
# ---------------------------------------------------------------------------
if [[ -z "$CUDA_VERSION" ]]; then
    if command -v nvcc &>/dev/null; then
        RAW=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        MAJOR=$(echo "$RAW" | cut -d. -f1)
        MINOR=$(echo "$RAW" | cut -d. -f2)
        if   [[ "$MAJOR" -ge 12 ]]; then CUDA_VERSION="cu121"
        elif [[ "$MAJOR" -eq 11 && "$MINOR" -ge 8 ]]; then CUDA_VERSION="cu118"
        elif [[ "$MAJOR" -eq 11 ]]; then CUDA_VERSION="cu117"
        else CUDA_VERSION="cpu"
        fi
        echo "Detected CUDA ${RAW} → using PyTorch index: ${CUDA_VERSION}"
    else
        CUDA_VERSION="cpu"
        echo "nvcc not found — installing CPU-only PyTorch"
    fi
fi

# ---------------------------------------------------------------------------
# Torch index URL
# ---------------------------------------------------------------------------
case "$CUDA_VERSION" in
    cu121) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    cu118) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    cu117) TORCH_INDEX="https://download.pytorch.org/whl/cu117" ;;
    cpu)   TORCH_INDEX="https://download.pytorch.org/whl/cpu"  ;;
    *)
        echo "Unrecognised CUDA version '${CUDA_VERSION}'. Falling back to default PyPI torch."
        TORCH_INDEX=""
        ;;
esac

# ---------------------------------------------------------------------------
# Core dependencies
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing core dependencies ==="

if [[ -n "$TORCH_INDEX" ]]; then
    pip install torch --index-url "$TORCH_INDEX"
else
    pip install "torch>=2.0.0"
fi

pip install \
    numpy \
    transformers \
    datasets \
    tiktoken \
    wandb \
    tqdm \
    matplotlib \
    pandas \
    seaborn \
    PyYAML \
    pyarrow \
    requests

# ---------------------------------------------------------------------------
# Optional: TensorBoard + JupyterLab
# ---------------------------------------------------------------------------
if [[ "$INSTALL_DEV" -eq 1 ]]; then
    echo ""
    echo "=== Installing dev tools (tensorboard, jupyterlab) ==="
    pip install tensorboard jupyterlab
fi

# ---------------------------------------------------------------------------
# Optional: Mixture of Experts
# ---------------------------------------------------------------------------
if [[ "$INSTALL_MOE" -eq 1 ]]; then
    echo ""
    echo "=== Installing MoE dependencies (megablocks, einops) ==="
    pip install megablocks einops
fi

# ---------------------------------------------------------------------------
# Optional: Hybrid GatedDeltaNet
# ---------------------------------------------------------------------------
if [[ "$INSTALL_HYBRID" -eq 1 ]]; then
    echo ""
    echo "=== Installing flash-linear-attention (Hybrid GatedDeltaNet) ==="
    pip install flash-linear-attention
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Environment setup complete ==="
python - <<'EOF'
import torch
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA ver  : {torch.version.cuda}")
EOF
