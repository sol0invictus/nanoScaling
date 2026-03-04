#!/usr/bin/env bash
# Run all RQ2 normalization ablation experiments.
#
# Full experiment: 12 conditions (6 norm variants × 2 optimizers) to 10B tokens.
# Recommended order: run baselines first, then gate no-norm conditions on early stability.
#
# Usage:
#   bash experiments/rq2_normalization/run_rq2.sh           # full runs
#   bash experiments/rq2_normalization/run_rq2.sh --quick   # 500-iter smoke test

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

# ---- Group 1: Pre-RMSNorm baselines (should both train well) ----
run "$SCRIPT_DIR/configs/125m_adamw_pre_rmsnorm.yaml"
run "$SCRIPT_DIR/configs/125m_muon_pre_rmsnorm.yaml"

# ---- Group 2: Pre-LayerNorm (full affine) ----
run "$SCRIPT_DIR/configs/125m_adamw_pre_layernorm.yaml"
run "$SCRIPT_DIR/configs/125m_muon_pre_layernorm.yaml"

# ---- Group 3: Post-LayerNorm (H2b: Muon stable, AdamW potentially unstable) ----
run "$SCRIPT_DIR/configs/125m_adamw_post_layernorm.yaml"
run "$SCRIPT_DIR/configs/125m_muon_post_layernorm.yaml"

# ---- Group 4: No Normalization (H2a: core novelty) ----
# Decision point: if muon_no_norm is stable at ~5B tokens (iter 10000), scale to 370M.
run "$SCRIPT_DIR/configs/125m_adamw_no_norm.yaml"
run "$SCRIPT_DIR/configs/125m_muon_no_norm.yaml"

# ---- Group 5: No Norm + Scaled Init (controls for init effects) ----
run "$SCRIPT_DIR/configs/125m_adamw_no_norm_scaled_init.yaml"
run "$SCRIPT_DIR/configs/125m_muon_no_norm_scaled_init.yaml"

# ---- Group 6: Fixed-Scale Norm / no affine (H2c: are γ/β load-bearing?) ----
run "$SCRIPT_DIR/configs/125m_adamw_fixed_scale_norm.yaml"
run "$SCRIPT_DIR/configs/125m_muon_fixed_scale_norm.yaml"

echo ""
echo "All RQ2 runs complete. Results in out/rq2/"
echo "Launch TensorBoard with: tensorboard --logdir out/rq2"
echo ""
echo "Stability verdicts (check metrics.json in each out_dir):"
for d in "$PROJECT_ROOT"/out/rq2/*/; do
    if [ -f "$d/metrics.json" ]; then
        stable=$(python3 -c "import json; m=json.load(open('$d/metrics.json')); print('STABLE' if m.get('is_stable', True) else 'UNSTABLE')" 2>/dev/null || echo "N/A")
        echo "  $(basename $d): $stable"
    fi
done
