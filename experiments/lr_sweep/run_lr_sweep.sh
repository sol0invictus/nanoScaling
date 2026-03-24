#!/bin/bash
# LR sweep for AdamW and Muon — 125M Pre-RMSNorm baseline
# Run before RQ2 to confirm optimal LR for each optimizer.
#
# Usage:
#   bash experiments/lr_sweep/run_lr_sweep.sh           # all 8 runs sequentially
#   bash experiments/lr_sweep/run_lr_sweep.sh --quick   # shakespeare_char smoke test (fast)
#
# After running, compare val loss across runs to pick optimal LR for each optimizer,
# then update the RQ2 configs accordingly.

set -e

SCRIPT="python train.py"
CONFIGS_DIR="experiments/lr_sweep/configs"

# Override dataset for quick smoke test
EXTRA_ARGS=""
if [[ "$1" == "--quick" ]]; then
    EXTRA_ARGS="--dataset=shakespeare_char --max_iters=500 --eval_interval=100 --eval_iters=20"
    echo "Running in quick mode (shakespeare_char, 500 iters)"
fi

ADAMW_CONFIGS=(
    "125m_adamw_lr1e-4.yaml"
    "125m_adamw_lr3e-4.yaml"
    "125m_adamw_lr1e-3.yaml"
    "125m_adamw_lr3e-3.yaml"
)

MUON_CONFIGS=(
    "125m_muon_lr3e-3.yaml"
    "125m_muon_lr1e-2.yaml"
    "125m_muon_lr2e-2.yaml"
    "125m_muon_lr5e-2.yaml"
)

echo "=== AdamW LR sweep ==="
for cfg in "${ADAMW_CONFIGS[@]}"; do
    echo "--- Running $cfg ---"
    $SCRIPT "$CONFIGS_DIR/$cfg" $EXTRA_ARGS
done

echo "=== Muon LR sweep ==="
for cfg in "${MUON_CONFIGS[@]}"; do
    echo "--- Running $cfg ---"
    $SCRIPT "$CONFIGS_DIR/$cfg" $EXTRA_ARGS
done

echo ""
echo "=== Sweep complete ==="
echo "Compare val loss across runs:"
echo "  AdamW candidates: 1e-4, 3e-4, 1e-3, 3e-3"
echo "  Muon candidates:  3e-3, 1e-2, 2e-2, 5e-2"
echo ""
echo "Update RQ2 configs with the winning LRs before running run_rq2.sh"
