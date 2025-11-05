#!/usr/bin/env python3
"""
Batch script to run hyperparameter search on multiple UCR datasets.
Runs searches for FordA, Adiac, and OliveOil datasets with both ESN and RON models.
"""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration
DATASETS = ['FordA', 'Adiac', 'OliveOil']
MODELS = ['esn', 'ron']
N_UNITS = 200  # Number of reservoir units
N_TRIALS = 20  # Number of optimization trials
ANTISYMMETRIC = True  # Use antisymmetric coupling
DEVICE = 'cpu'  # Change to 'cuda' if GPU available
DATAROOT = './data/UCR'
RESULTROOT = './ucr_results'
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))

def run_search(dataset: str, model: str):
    """Run hyperparameter search for a dataset and model combination."""
    print("\n" + "=" * 80)
    print(f"STARTING SEARCH: {dataset} - {model.upper()}")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build command - use current Python interpreter (no venv activation required)
    python_exec = sys.executable
    cmd = [
        python_exec,
        'ucr_hyperparameter_search.py',
        '--dataset', dataset,
        '--model', model,
        '--n_units', str(N_UNITS),
        '--n_trials', str(N_TRIALS),
        '--device', DEVICE,
        '--dataroot', DATAROOT,
        '--resultroot', RESULTROOT,
        '--batch_size', str(BATCH_SIZE),
    ]
    
    if ANTISYMMETRIC:
        cmd.append('--antisymmetric')
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the search
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False,  # Show output in real-time
        )
        
        print("\n" + "=" * 80)
        print(f"✓ COMPLETED: {dataset} - {model.upper()}")
        print("=" * 80)
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"✗ FAILED: {dataset} - {model.upper()}")
        print(f"Error: {e}")
        print("=" * 80)
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return False


def main():
    """Run all searches."""
    print("=" * 80)
    print("UCR HYPERPARAMETER SEARCH BATCH")
    print("=" * 80)
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Models: {', '.join([m.upper() for m in MODELS])}")
    print(f"Units: {N_UNITS}")
    print(f"Trials per search: {N_TRIALS}")
    print(f"Antisymmetric: {ANTISYMMETRIC}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    # Create results directory
    Path(RESULTROOT).mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = {}
    total_searches = len(DATASETS) * len(MODELS)
    completed = 0
    
    # Run all combinations
    for dataset in DATASETS:
        for model in MODELS:
            key = f"{dataset}_{model}"
            success = run_search(dataset, model)
            results[key] = success
            
            if success:
                completed += 1
            
            print(f"\nProgress: {completed}/{total_searches} searches completed")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count
    
    print(f"Total searches: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()
    
    for key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {key}")
    
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Check results in:", RESULTROOT)
    print("  2. Run final training with best configs using ucr_train_final.py")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Batch interrupted by user")
        sys.exit(1)
