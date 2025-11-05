#!/usr/bin/env bash
# Test batch: run ESN and RON on FordA, Adiac, OliveOil
# Runs BOTH modes per dataset: (1) standard 1-layer, (2) antisymmetric multi-layer
# Uses 200 units, 3 trials each, CPU device by default.
# No virtualenv activation required; uses whatever `python3` points to.

set -euo pipefail

PY=${PYTHON:-python3}
TRIALS=${TRIALS:-3}
UNITS=${UNITS:-200}
DEVICE=${DEVICE:-cpu}
BATCH_SIZE=${BATCH_SIZE:-128}

COMMON=(--n_units "$UNITS" --n_trials "$TRIALS" --device "$DEVICE" --batch_size "$BATCH_SIZE")

mkdir -p ucr_results

echo "Datasets: FordA, Adiac, OliveOil"
echo "Models: ESN, RON"
echo "Modes: standard(1-layer), antisymmetric"
echo "Units: $UNITS | Trials: $TRIALS | Device: $DEVICE | Batch size: $BATCH_SIZE"

for DATASET in FordA Adiac OliveOil; do
  for MODEL in esn ron; do
    echo "============================================================"
    echo "[STANDARD] $DATASET - $MODEL"
    echo "============================================================"
    "$PY" ucr_hyperparameter_search.py --dataset "$DATASET" --model "$MODEL" "${COMMON[@]}"

    echo "============================================================"
    echo "[ANTISYMMETRIC] $DATASET - $MODEL"
    echo "============================================================"
    "$PY" ucr_hyperparameter_search.py --dataset "$DATASET" --model "$MODEL" "${COMMON[@]}" --antisymmetric
  done
done

echo "\nAll tests finished. Results in ./ucr_results"