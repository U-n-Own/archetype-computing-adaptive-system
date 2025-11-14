#!/bin/bash

# Script to run all three test scripts sequentially
# Results are logged to their respective directories:
# - results_smnist_all_models/results_summary.txt
# - results_npcifar10_all_models/results_summary.txt
# - results_psmnist_all_models/results_summary.txt

echo "=================================="
echo "Running All Model Tests"
echo "=================================="
echo "Parameters: --n_hid 500 --batch_size 1000"
echo ""

# Run test_smnist_all_models.py
echo "Starting test_smnist_all_models.py..."
python antisymmetric_experiments/test_smnist_all_models.py --n_hid 500 --batch_size 1000
if [ $? -eq 0 ]; then
    echo "✓ test_smnist_all_models.py completed successfully"
    echo "  Results: results_smnist_all_models/results_summary.txt"
else
    echo "✗ test_smnist_all_models.py failed with exit code $?"
    exit 1
fi
echo ""

# Run test_npcifar10_all_models.py
echo "Starting test_npcifar10_all_models.py..."
python antisymmetric_experiments/test_npcifar10_all_models.py --n_hid 500 --batch_size 1000
if [ $? -eq 0 ]; then
    echo "✓ test_npcifar10_all_models.py completed successfully"
    echo "  Results: results_npcifar10_all_models/results_summary.txt"
else
    echo "✗ test_npcifar10_all_models.py failed with exit code $?"
    exit 1
fi
echo ""

# Run test_psmnist_all_models.py
echo "Starting test_psmnist_all_models.py..."
python antisymmetric_experiments/test_psmnist_all_models.py --n_hid 500 --batch_size 1000
if [ $? -eq 0 ]; then
    echo "✓ test_psmnist_all_models.py completed successfully"
    echo "  Results: results_psmnist_all_models/results_summary.txt"
else
    echo "✗ test_psmnist_all_models.py failed with exit code $?"
    exit 1
fi
echo ""

echo "=================================="
echo "All tests completed successfully!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - results_smnist_all_models/results_summary.txt"
echo "  - results_npcifar10_all_models/results_summary.txt"
echo "  - results_psmnist_all_models/results_summary.txt"
