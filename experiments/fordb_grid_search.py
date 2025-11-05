#!/usr/bin/env python3
"""
Grid search for FordB dataset hyperparameters
Tests both ESN and RON models with and without antisymmetric coupling
"""

import subprocess
import itertools
import json
import os
import re
from datetime import datetime
from pathlib import Path

# Grid search parameters
PARAM_GRID = {
    'esn': {
        'leaky': [0.1, 0.5, 0.9],
        'rho': [0.9, 0.99],
        'inp_scaling': [1, 10, 50],
    },
    'ron': {
        'dt': [0.01, 0.05],
        'rho': [0.9, 0.99],
        'inp_scaling': [1, 10, 50],
        'epsilon': [5, 10],
        'epsilon_range': [0.5, 2.5],
        'gamma': [1, 3],
        'gamma_range': [0.5, 1],
    }
}

# Fixed parameters
FIXED_PARAMS = {
    'n_hid': 100,
    'trials': 5,  # Reduced for faster grid search
    'dataroot': 'data/UCR/FordB',  # Corrected path for FordB
    'cpu': True,
    'use_test': True,
}

# Results directory
RESULTS_DIR = Path('experiments/fordb_grid_search_results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(model_type, params, antisymmetric=False, coupling_epsilon=None):
    """
    Run a single experiment with given parameters
    
    Parameters:
    -----------
    model_type : str
        'esn' or 'ron'
    params : dict
        Hyperparameters for the model
    antisymmetric : bool
        Whether to use antisymmetric coupling
    coupling_epsilon : float or None
        Coupling strength (only used if antisymmetric=True)
    
    Returns:
    --------
    dict : Results containing accuracy and parameters
    """
    # Build command
    cmd = ['python3', 'experiments/fordb.py']
    
    # Add model flag
    if model_type == 'esn':
        cmd.append('--esn')
    elif model_type == 'ron':
        cmd.append('--ron')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Add fixed parameters
    for key, value in FIXED_PARAMS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    # Add hyperparameters
    for key, value in params.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Add antisymmetric coupling if requested
    if antisymmetric:
        cmd.append('--antisymmetric')
        if coupling_epsilon is not None:
            cmd.extend(['--coupling_epsilon', str(coupling_epsilon)])
    
    # Run experiment
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    # Determine expected log file name
    ac_suffix = "_AC" if antisymmetric else ""
    if model_type == 'ron':
        # RON uses topology in filename (default is 'full')
        topology = params.get('topology', 'full')
        log_filename = f"FordB_log_RON_{topology}{ac_suffix}.txt"
    elif model_type == 'esn':
        log_filename = f"FordB_log_ESN{ac_suffix}.txt"
    else:
        log_filename = f"FordB_log_{model_type.upper()}{ac_suffix}.txt"
    
    log_path = Path(log_filename)
    
    # Remove old log file if it exists
    if log_path.exists():
        log_path.unlink()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Parse log file to extract accuracy
        train_acc = None
        test_acc = None
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            # Parse the log file format: mean/std train: (np.float64(0.6174), ...) mean/std test: (np.float64(0.5807), ...)
            # Find train accuracy (mean)
            train_match = re.search(r"mean/std train:\s*\(np\.float64\(([\d.]+)\)", log_content)
            if train_match:
                train_acc = float(train_match.group(1)) * 100  # Convert to percentage
            
            # Find test accuracy (mean)
            test_match = re.search(r"mean/std test:\s*\(np\.float64\(([\d.]+)\)", log_content)
            if test_match:
                test_acc = float(test_match.group(1)) * 100  # Convert to percentage
            
            if train_acc is not None and test_acc is not None:
                print(f"✓ Train: {train_acc:.2f}%, Test: {test_acc:.2f}%")
            else:
                print(f"⚠ Could not parse accuracies from log file")
        else:
            print(f"✗ Log file not found: {log_filename}")
        
        return {
            'model': model_type,
            'params': params,
            'antisymmetric': antisymmetric,
            'coupling_epsilon': coupling_epsilon,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'success': result.returncode == 0 and train_acc is not None,
            'log_file': str(log_path) if log_path.exists() else None,
        }
    
    except subprocess.TimeoutExpired:
        print(f"ERROR: Experiment timed out!")
        return {
            'model': model_type,
            'params': params,
            'antisymmetric': antisymmetric,
            'coupling_epsilon': coupling_epsilon,
            'train_accuracy': None,
            'test_accuracy': None,
            'success': False,
            'error': 'Timeout',
        }
    
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'model': model_type,
            'params': params,
            'antisymmetric': antisymmetric,
            'coupling_epsilon': coupling_epsilon,
            'train_accuracy': None,
            'test_accuracy': None,
            'success': False,
            'error': str(e),
        }


def grid_search_esn(antisymmetric=False, coupling_epsilon=None):
    """Grid search for ESN model"""
    print(f"\n{'#'*80}")
    print(f"# ESN Grid Search (antisymmetric={antisymmetric})")
    print(f"{'#'*80}\n")
    
    results = []
    
    # Generate all parameter combinations
    param_names = list(PARAM_GRID['esn'].keys())
    param_values = [PARAM_GRID['esn'][name] for name in param_names]
    
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"Total combinations to test: {total_combinations}\n")
    
    for i, combination in enumerate(itertools.product(*param_values), 1):
        params = dict(zip(param_names, combination))
        
        print(f"\n[{i}/{total_combinations}] Testing ESN with params:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        result = run_experiment('esn', params, antisymmetric, coupling_epsilon)
        results.append(result)
        
        if result['success'] and result['test_accuracy'] is not None:
            print(f"✓ Test Accuracy: {result['test_accuracy']:.2f}%")
        else:
            print(f"✗ Experiment failed")
    
    return results


def grid_search_ron(antisymmetric=False, coupling_epsilon=None):
    """Grid search for RON model"""
    print(f"\n{'#'*80}")
    print(f"# RON Grid Search (antisymmetric={antisymmetric})")
    print(f"{'#'*80}\n")
    
    results = []
    
    # Generate all parameter combinations
    param_names = list(PARAM_GRID['ron'].keys())
    param_values = [PARAM_GRID['ron'][name] for name in param_names]
    
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"Total combinations to test: {total_combinations}\n")
    
    for i, combination in enumerate(itertools.product(*param_values), 1):
        params = dict(zip(param_names, combination))
        
        print(f"\n[{i}/{total_combinations}] Testing RON with params:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        result = run_experiment('ron', params, antisymmetric, coupling_epsilon)
        results.append(result)
        
        if result['success'] and result['test_accuracy'] is not None:
            print(f"✓ Test Accuracy: {result['test_accuracy']:.2f}%")
        else:
            print(f"✗ Experiment failed")
    
    return results


def save_results(results, filename):
    """Save results to JSON file"""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")


def print_summary(results, title):
    """Print summary of best results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    # Filter successful experiments
    successful = [r for r in results if r['success'] and r['test_accuracy'] is not None]
    
    if not successful:
        print("No successful experiments!")
        return
    
    # Sort by test accuracy
    successful.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}\n")
    
    # Top 5 results
    print("Top 5 Results:")
    print("-" * 80)
    for i, result in enumerate(successful[:5], 1):
        print(f"\n{i}. Test Accuracy: {result['test_accuracy']:.2f}% (Train: {result['train_accuracy']:.2f}%)")
        print(f"   Model: {result['model'].upper()}")
        print(f"   Parameters:")
        for key, value in sorted(result['params'].items()):
            print(f"     {key}: {value}")
        if result['antisymmetric']:
            print(f"     antisymmetric: True")
            print(f"     coupling_epsilon: {result['coupling_epsilon']}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run complete grid search"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {}
    
    # 1. ESN without antisymmetric coupling
    print("\n" + "="*80)
    print("PHASE 1: ESN Grid Search (No Antisymmetric Coupling)")
    print("="*80)
    esn_results = grid_search_esn(antisymmetric=False)
    all_results['esn_no_coupling'] = esn_results
    save_results(esn_results, f'esn_no_coupling_{timestamp}.json')
    print_summary(esn_results, "ESN Results (No Coupling)")
    
    # 2. ESN with antisymmetric coupling
    print("\n" + "="*80)
    print("PHASE 2: ESN Grid Search (With Antisymmetric Coupling)")
    print("="*80)
    esn_ac_results = grid_search_esn(antisymmetric=True, coupling_epsilon=0.1)
    all_results['esn_with_coupling'] = esn_ac_results
    save_results(esn_ac_results, f'esn_with_coupling_{timestamp}.json')
    print_summary(esn_ac_results, "ESN Results (With Coupling)")
    
    # 3. RON without antisymmetric coupling
    print("\n" + "="*80)
    print("PHASE 3: RON Grid Search (No Antisymmetric Coupling)")
    print("="*80)
    ron_results = grid_search_ron(antisymmetric=False)
    all_results['ron_no_coupling'] = ron_results
    save_results(ron_results, f'ron_no_coupling_{timestamp}.json')
    print_summary(ron_results, "RON Results (No Coupling)")
    
    # 4. RON with antisymmetric coupling
    print("\n" + "="*80)
    print("PHASE 4: RON Grid Search (With Antisymmetric Coupling)")
    print("="*80)
    ron_ac_results = grid_search_ron(antisymmetric=True, coupling_epsilon=0.1)
    all_results['ron_with_coupling'] = ron_ac_results
    save_results(ron_ac_results, f'ron_with_coupling_{timestamp}.json')
    print_summary(ron_ac_results, "RON Results (With Coupling)")
    
    # Save all results
    save_results(all_results, f'all_results_{timestamp}.json')
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print(f"Timestamp: {timestamp}")
    
    # Compare best results across all configurations
    all_successful = []
    for config_name, config_results in all_results.items():
        for result in config_results:
            if result['success'] and result['test_accuracy'] is not None:
                result['config'] = config_name
                all_successful.append(result)
    
    if all_successful:
        all_successful.sort(key=lambda x: x['test_accuracy'], reverse=True)
        best = all_successful[0]
        
        print(f"\n🏆 BEST OVERALL RESULT:")
        print(f"   Configuration: {best['config']}")
        print(f"   Model: {best['model'].upper()}")
        print(f"   Test Accuracy: {best['test_accuracy']:.2f}%")
        print(f"   Train Accuracy: {best['train_accuracy']:.2f}%")
        print(f"   Parameters:")
        for key, value in sorted(best['params'].items()):
            print(f"     {key}: {value}")
        if best['antisymmetric']:
            print(f"     antisymmetric: True")
            print(f"     coupling_epsilon: {best['coupling_epsilon']}")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   FordB Hyperparameter Grid Search                           ║
║                                                                              ║
║  This script will test all combinations of hyperparameters for:             ║
║    - ESN (with and without antisymmetric coupling)                           ║
║    - RON (with and without antisymmetric coupling)                           ║
║                                                                              ║
║  Dataset: FordB (data/UCR/FordB)                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    main()
