#!/usr/bin/env bash
# env_setup.bash — nanoScaling environment setup
#
# Creates a persistent Python venv at VENV_DIR (default: ~/envs/nanoscaling).
# Re-running is safe: skips venv creation if it already exists, only upgrades
# packages. Use --reinstall to wipe and recreate from scratch.
#
# Usage:
#   bash env_setup.bash [OPTIONS]
#
# Options:
#   --venv <path>      Venv location (default: ~/envs/nanoscaling)
#   --cuda <version>   PyTorch CUDA build: cu121, cu118, cu117, cpu
#                      Default: auto-detect from nvcc
#   --moe              Also install megablocks + einops (Mixture of Experts)
#   --hybrid           Also install flash-linear-attention (Hybrid GatedDeltaNet)
#   --dev              Also install tensorboard + jupyterlab
#   --reinstall        Delete existing venv and start fresh
#   -h / --help        Show this message and exit
#
# After first run, activate with:
#   source ~/envs/nanoscaling/bin/activate
#
# Add to ~/.bashrc for permanent activation:
#   echo 'source ~/envs/nanoscaling/bin/activate' >> ~/.bashrc

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VENV_DIR="${HOME}/envs/nanoscaling"
CUDA_VERSION=""
INSTALL_MOE=0
INSTALL_HYBRID=0
INSTALL_DEV=0
REINSTALL=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)      VENV_DIR="$2";   shift 2 ;;
        --cuda)      CUDA_VERSION="$2"; shift 2 ;;
        --moe)       INSTALL_MOE=1;    shift   ;;
        --hybrid)    INSTALL_HYBRID=1; shift   ;;
        --dev)       INSTALL_DEV=1;    shift   ;;
        --reinstall) REINSTALL=1;      shift   ;;
        -h|--help)
            sed -n '2,25p' "$0"
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
        if   [[ "$MAJOR" -ge 12 ]];                              then CUDA_VERSION="cu121"
        elif [[ "$MAJOR" -eq 11 && "$MINOR" -ge 8 ]];           then CUDA_VERSION="cu118"
        elif [[ "$MAJOR" -eq 11 ]];                              then CUDA_VERSION="cu117"
        else CUDA_VERSION="cpu"
        fi
        echo "Detected CUDA ${RAW} → PyTorch index: ${CUDA_VERSION}"
    else
        CUDA_VERSION="cpu"
        echo "nvcc not found — will install CPU-only PyTorch"
    fi
fi

# ---------------------------------------------------------------------------
# Torch index URL
# ---------------------------------------------------------------------------
case "$CUDA_VERSION" in
    cu121) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    cu118) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    cu117) TORCH_INDEX="https://download.pytorch.org/whl/cu117" ;;
    cpu)   TORCH_INDEX="https://download.pytorch.org/whl/cpu"   ;;
    *)
        echo "Unrecognised CUDA build '${CUDA_VERSION}' — using default PyPI torch."
        TORCH_INDEX=""
        ;;
esac

# ---------------------------------------------------------------------------
# Create / reuse venv
# ---------------------------------------------------------------------------
if [[ "$REINSTALL" -eq 1 && -d "$VENV_DIR" ]]; then
    echo ""
    echo "=== --reinstall: removing existing venv at ${VENV_DIR} ==="
    rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
    echo ""
    echo "=== Reusing existing venv at ${VENV_DIR} ==="
    echo "    (run with --reinstall to wipe and recreate)"
else
    echo ""
    echo "=== Creating venv at ${VENV_DIR} ==="
    python3 -m venv "$VENV_DIR"
fi

# Activate for the rest of this script
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "    Python : $(python --version)"

# Always upgrade pip/setuptools first
pip install --quiet --upgrade pip setuptools wheel

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
# Optional extras
# ---------------------------------------------------------------------------
if [[ "$INSTALL_DEV" -eq 1 ]]; then
    echo ""
    echo "=== Installing dev tools (tensorboard, jupyterlab) ==="
    pip install tensorboard jupyterlab
fi

if [[ "$INSTALL_MOE" -eq 1 ]]; then
    echo ""
    echo "=== Installing MoE dependencies (megablocks, einops) ==="
    pip install megablocks einops
fi

if [[ "$INSTALL_HYBRID" -eq 1 ]]; then
    echo ""
    echo "=== Installing flash-linear-attention (Hybrid GatedDeltaNet) ==="
    pip install flash-linear-attention
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
echo "=== Environment ready ==="
python - <<'EOF'
import torch
print(f"  PyTorch   : {torch.__version__}")
print(f"  CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA ver  : {torch.version.cuda}")
EOF

# ---------------------------------------------------------------------------
# Activation hint
# ---------------------------------------------------------------------------
ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
echo ""
echo "=== To activate this environment ==="
echo "    ${ACTIVATE_CMD}"
echo ""
echo "    To activate automatically on login, add to ~/.bashrc:"
echo "    echo '${ACTIVATE_CMD}' >> ~/.bashrc"
