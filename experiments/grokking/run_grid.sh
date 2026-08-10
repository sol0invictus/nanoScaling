#!/usr/bin/env bash
# Launch the full grokking grid. The model is tiny and CPU launch-bound (the GPU
# sits at ~2% utilisation), so all runs execute concurrently across CPU cores
# rather than contending for GPU compute. See EXPERIMENT_LOG.md E3.
set -u
cd "$(dirname "$0")"
mkdir -p runs

CONFIGS=${*:-configs/r*.yaml}
for cfg in $CONFIGS; do
    name=$(basename "$cfg" .yaml)
    echo "launching $name"
    setsid nohup python train.py "$cfg" > "runs/${name}.log" 2>&1 &
done
wait
echo "all runs finished"
