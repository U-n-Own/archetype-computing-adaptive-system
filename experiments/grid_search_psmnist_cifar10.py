#!/usr/bin/env python3
"""
Grid Search for psMNIST and CIFAR-10
Compares RON, ESN, and their antisymmetric variants with 500 units.
Antisymmetric models use 5 layers.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from itertools import product
import numpy as np

# Configuration
DATASETS = ["psmnist", "cifar10"]
N_HID = 500
N_LAYERS_ANTISYM = 5
N_TRIALS = 3

# Model configurations
MODELS = {
    "ESN": {
        "flag": "--esn",
        "antisymmetric": False,
        "n_layers": 1,
        "params": {
            "rho": [0.9, 0.95, 0.99],
            "inp_scaling": [0.1, 0.5, 1.0],
            "leaky": [0.8, 1.0],
        }
    },
    "ESN_Antisym": {
        "flag": "--esn",
        "antisymmetric": True,
        "n_layers": N_LAYERS_ANTISYM,
        "params": {
            "rho": [0.9, 0.95, 0.99],
            "inp_scaling": [0.1, 0.5, 1.0],
            "leaky": [0.8, 1.0],
            "coupling_epsilon": [0.1, 0.4, 0.8],
        }
    },
    "RON": {
        "flag": "--ron",
        "antisymmetric": False,
        "n_layers": 1,
        "params": {
            "dt": [0.01, 0.042, 0.1],
            "gamma": [1.5, 2.7, 4.0],
            "epsilon": [2.0, 4.7, 8.0],
            "inp_scaling": [0.1, 0.5, 1.0],
            "topology": ["full", "ring"],
        }
    },
    "RON_Antisym": {
        "flag": "--ron",
        "antisymmetric": True,
        "n_layers": N_LAYERS_ANTISYM,
        "params": {
            "dt": [0.01, 0.042, 0.1],
            "gamma": [1.5, 2.7, 4.0],
            "epsilon": [2.0, 4.7, 8.0],
            "inp_scaling": [0.1, 0.5, 1.0],
            "topology": ["full", "ring"],
            "coupling_epsilon": [0.1, 0.4, 0.8],
        }
    },
}


def run_experiment(dataset, model_name, model_config, param_combo, result_dir):
    """Run a single experiment with given parameters."""
    
    # Build command
    script = f"experiments/{dataset}.py"
    cmd = [
        "python3", script,
        model_config["flag"],
        "--n_hid", str(N_HID),
        "--n_layers", str(model_config["n_layers"]),
        "--trials", str(N_TRIALS),
        "--resultroot", result_dir,
        "--resultsuffix", f"_gridsearch_{model_name}",
    ]
    
    # Add antisymmetric flag if needed
    if model_config["antisymmetric"]:
        cmd.append("--antisymmetric")
    
    # Add parameters
    for param_name, param_value in param_combo.items():
        if param_name == "gamma" or param_name == "epsilon":
            # For RON, these need special handling
            cmd.extend([f"--{param_name}", str(param_value)])
            cmd.extend([f"--{param_name}_range", "0.0"])
        else:
            cmd.extend([f"--{param_name}", str(param_value)])
    
    # Run experiment
    print(f"\n{'='*80}")
    print(f"Running: {dataset} - {model_name}")
    print(f"Parameters: {param_combo}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ Success")
            return {
                "dataset": dataset,
                "model": model_name,
                "params": param_combo,
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        else:
            print(f"❌ Failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return {
                "dataset": dataset,
                "model": model_name,
                "params": param_combo,
                "status": "failed",
                "error": result.stderr,
            }
    except subprocess.TimeoutExpired:
        print("❌ Timeout (>1 hour)")
        return {
            "dataset": dataset,
            "model": model_name,
            "params": param_combo,
            "status": "timeout",
        }
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {
            "dataset": dataset,
            "model": model_name,
            "params": param_combo,
            "status": "error",
            "error": str(e),
        }


def generate_param_combinations(params_dict):
    """Generate all combinations of parameters."""
    param_names = list(params_dict.keys())
    param_values = list(params_dict.values())
    
    for combo in product(*param_values):
        yield dict(zip(param_names, combo))


def main():
    print("="*80)
    print("Grid Search for psMNIST and CIFAR-10")
    print("="*80)
    print(f"Datasets: {DATASETS}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Hidden units: {N_HID}")
    print(f"Antisymmetric layers: {N_LAYERS_ANTISYM}")
    print(f"Trials per config: {N_TRIALS}")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"experiments/results_gridsearch_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    print(f"\nResults will be saved to: {result_dir}")
    
    # Track all results
    all_results = []
    total_experiments = 0
    
    # Count total experiments
    for dataset in DATASETS:
        for model_name, model_config in MODELS.items():
            n_combos = np.prod([len(v) for v in model_config["params"].values()])
            total_experiments += n_combos
    
    print(f"\nTotal experiments to run: {total_experiments}")
    print("="*80)
    
    # Run grid search
    experiment_count = 0
    for dataset in DATASETS:
        for model_name, model_config in MODELS.items():
            print(f"\n{'#'*80}")
            print(f"# Dataset: {dataset.upper()} | Model: {model_name}")
            print(f"{'#'*80}")
            
            # Generate parameter combinations
            for param_combo in generate_param_combinations(model_config["params"]):
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}")
                
                result = run_experiment(
                    dataset, model_name, model_config, param_combo, result_dir
                )
                all_results.append(result)
                
                # Save intermediate results
                results_file = os.path.join(result_dir, "grid_search_results.json")
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2)
    
    # Final summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    
    success_count = sum(1 for r in all_results if r["status"] == "success")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    timeout_count = sum(1 for r in all_results if r["status"] == "timeout")
    error_count = sum(1 for r in all_results if r["status"] == "error")
    
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  ⏱️  Timeout: {timeout_count}")
    print(f"  🔴 Error: {error_count}")
    
    print(f"\nResults saved to:")
    print(f"  - {result_dir}/")
    print(f"  - {results_file}")
    
    # Summary by dataset and model
    print("\n" + "="*80)
    print("SUMMARY BY DATASET AND MODEL")
    print("="*80)
    
    for dataset in DATASETS:
        print(f"\n{dataset.upper()}:")
        for model_name in MODELS.keys():
            model_results = [r for r in all_results 
                           if r["dataset"] == dataset and r["model"] == model_name]
            successes = sum(1 for r in model_results if r["status"] == "success")
            total = len(model_results)
            print(f"  {model_name:20s}: {successes}/{total} successful")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
