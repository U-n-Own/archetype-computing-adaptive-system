#!/usr/bin/env bash
# Small test batch: runs ESN and RON searches on FordA, Adiac, OliveOil
# Uses 200 units, 3 trials each, CPU device, antisymmetric coupling.
# No virtualenv activation needed; uses whatever `python3` points to.

set -euo pipefail

PY=${PYTHON:-python3}
BATCH_SIZE=${BATCH_SIZE:-128}
COMMON_ARGS=(--n_units 200 --n_trials 3 --device cpu --antisymmetric --batch_size "$BATCH_SIZE")

# Create results dir if not present
mkdir -p ucr_results

# Datasets to run
for DATASET in FordA Adiac OliveOil; do
  for MODEL in esn ron; do
    echo "============================================================"
  echo "Running $DATASET - $MODEL (trials=3, batch_size=$BATCH_SIZE)"
    echo "============================================================"
    $PY ucr_hyperparameter_search.py --dataset "$DATASET" --model "$MODEL" "${COMMON_ARGS[@]}"
  done
done

echo "\nAll test searches finished. Check results in ./ucr_results"