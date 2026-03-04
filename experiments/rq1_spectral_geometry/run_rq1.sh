#!/usr/bin/env bash
# Run all RQ1 experiments sequentially.
# For parallel execution on multiple GPUs, use CUDA_VISIBLE_DEVICES and run in background.
#
# Usage:
#   bash experiments/rq1_spectral_geometry/run_rq1.sh
#
# Quick smoke test (shakespeare, 500 iters):
#   bash experiments/rq1_spectral_geometry/run_rq1.sh --quick

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN="$PROJECT_ROOT/train.py"

QUICK_ARGS=""
if [[ "$1" == "--quick" ]]; then
    QUICK_ARGS="--max_iters=500 --eval_interval=100 --log_interval=10"
    echo ">>> Running in QUICK mode (500 iters, shakespeare_char)"
fi

run() {
    local config="$1"
    shift
    echo ""
    echo "============================================================"
    echo "Starting: $config"
    echo "============================================================"
    python "$TRAIN" "$config" $QUICK_ARGS "$@"
}

# ---- Baselines ----
# run "$SCRIPT_DIR/configs/125m_adamw.yaml"
run "$SCRIPT_DIR/configs/125m_muon.yaml"

# # ---- Newton-Schulz ablations ----
# run "$SCRIPT_DIR/configs/125m_muon_ns3.yaml"
# run "$SCRIPT_DIR/configs/125m_muon_ns10.yaml"

# # ---- Momentum ablation ----
# run "$SCRIPT_DIR/configs/125m_muon_beta85.yaml"

# echo ""
# echo "All RQ1 runs complete. Results in out/rq1/"
# echo "Launch TensorBoard with: tensorboard --logdir out/rq1"
