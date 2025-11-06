#!/usr/bin/env python3
"""
Unified Grid Search for FordB and OliveOil datasets
Selects best model based on validation accuracy (no --use_test)
"""

import subprocess
import itertools
import json
import os
import re
from datetime import datetime
from pathlib import Path
import argparse

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
    'batch': 30,  # Batch size for dataloaders
    'trials': 1,  # Use 1 trial for validation-based selection (no --use_test)
    'cpu': True,
}

# Dataset configurations
DATASETS = {
    'fordb': {
        'script': 'experiments/fordb.py',
        'dataroot': 'data/UCR/FordB',
        'log_prefix': 'FordB_log',
    },
    'oliveoil': {
        'script': 'experiments/olive_oil.py',
        'dataroot': 'data/UCR/OliveOil',  # Update this path if needed
        'log_prefix': 'OliveOil_log',
    }
}


def run_experiment(dataset_name, model_type, params, antisymmetric=False, coupling_epsilon=None):
    """
    Run a single experiment with given parameters
    
    Parameters:
    -----------
    dataset_name : str
        'fordb' or 'oliveoil'
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
    dataset_config = DATASETS[dataset_name]
    
    # Build command
    cmd = ['python3', dataset_config['script']]
    
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
    
    # Add dataset dataroot
    cmd.extend(['--dataroot', dataset_config['dataroot']])
    
    # Add hyperparameters
    for key, value in params.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Add antisymmetric coupling if requested
    if antisymmetric:
        cmd.append('--antisymmetric')
        if coupling_epsilon is not None:
            cmd.extend(['--coupling_epsilon', str(coupling_epsilon)])
    
    # Determine expected log file name
    ac_suffix = "_AC" if antisymmetric else ""
    if model_type == 'ron':
        topology = params.get('topology', 'full')
        log_filename = f"{dataset_config['log_prefix']}_RON_{topology}{ac_suffix}.txt"
    elif model_type == 'esn':
        log_filename = f"{dataset_config['log_prefix']}_ESN{ac_suffix}.txt"
    else:
        log_filename = f"{dataset_config['log_prefix']}_{model_type.upper()}{ac_suffix}.txt"
    
    log_path = Path(log_filename)
    
    # Remove old log file if it exists
    if log_path.exists():
        log_path.unlink()
    
    # Run experiment
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Parse log file to extract accuracy
        train_acc = None
        valid_acc = None
        test_acc = None
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            # Parse the log file format: mean/std train: (np.float64(0.6174), ...) mean/std valid: (np.float64(0.5807), ...)
            # Find train accuracy (mean)
            train_match = re.search(r"mean/std train:\s*\(np\.float64\(([\d.]+)\)", log_content)
            if train_match:
                train_acc = float(train_match.group(1)) * 100  # Convert to percentage
            
            # Find valid accuracy (mean)
            valid_match = re.search(r"mean/std valid:\s*\(np\.float64\(([\d.]+)\)", log_content)
            if valid_match:
                valid_acc = float(valid_match.group(1)) * 100  # Convert to percentage
            
            # Find test accuracy (mean)
            test_match = re.search(r"mean/std test:\s*\(np\.float64\(([\d.]+)\)", log_content)
            if test_match:
                test_acc = float(test_match.group(1)) * 100  # Convert to percentage
            
            if train_acc is not None and valid_acc is not None:
                print(f"✓ Train: {train_acc:.2f}%, Valid: {valid_acc:.2f}%, Test: {test_acc:.2f}%")
            else:
                print(f"⚠ Could not parse accuracies from log file")
        else:
            print(f"✗ Log file not found: {log_filename}")
        
        return {
            'dataset': dataset_name,
            'model': model_type,
            'params': params,
            'antisymmetric': antisymmetric,
            'coupling_epsilon': coupling_epsilon,
            'train_accuracy': train_acc,
            'valid_accuracy': valid_acc,
            'test_accuracy': test_acc,
            'success': result.returncode == 0 and train_acc is not None,
            'log_file': str(log_path) if log_path.exists() else None,
        }
    
    except subprocess.TimeoutExpired:
        print(f"ERROR: Experiment timed out!")
        return {
            'dataset': dataset_name,
            'model': model_type,
            'params': params,
            'antisymmetric': antisymmetric,
            'coupling_epsilon': coupling_epsilon,
            'train_accuracy': None,
            'valid_accuracy': None,
            'test_accuracy': None,
            'success': False,
            'error': 'Timeout',
        }
    
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'dataset': dataset_name,
            'model': model_type,
            'params': params,
            'antisymmetric': antisymmetric,
            'coupling_epsilon': coupling_epsilon,
            'train_accuracy': None,
            'valid_accuracy': None,
            'test_accuracy': None,
            'success': False,
            'error': str(e),
        }


def grid_search_model(dataset_name, model_type, antisymmetric=False, coupling_epsilon=None):
    """Grid search for a specific model on a specific dataset"""
    print(f"\n{'#'*80}")
    print(f"# {dataset_name.upper()} - {model_type.upper()} Grid Search (antisymmetric={antisymmetric})")
    print(f"{'#'*80}\n")
    
    results = []
    
    # Generate all parameter combinations
    param_names = list(PARAM_GRID[model_type].keys())
    param_values = [PARAM_GRID[model_type][name] for name in param_names]
    
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"Total combinations to test: {total_combinations}\n")
    
    for i, combination in enumerate(itertools.product(*param_values), 1):
        params = dict(zip(param_names, combination))
        
        print(f"\n[{i}/{total_combinations}] Testing {model_type.upper()} with params:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        result = run_experiment(dataset_name, model_type, params, antisymmetric, coupling_epsilon)
        results.append(result)
        
        if result['success'] and result['valid_accuracy'] is not None:
            print(f"✓ Valid Accuracy: {result['valid_accuracy']:.2f}%")
        else:
            print(f"✗ Experiment failed")
    
    return results


def save_results(results, filename, results_dir):
    """Save results to JSON file"""
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")


def print_summary(results, title):
    """Print summary of best results based on validation accuracy"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    # Filter successful experiments
    successful = [r for r in results if r['success'] and r['valid_accuracy'] is not None]
    
    if not successful:
        print("No successful experiments!")
        return None
    
    # Sort by validation accuracy
    successful.sort(key=lambda x: x['valid_accuracy'], reverse=True)
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}\n")
    
    # Top 5 results
    print("Top 5 Results (by Validation Accuracy):")
    print("-" * 80)
    for i, result in enumerate(successful[:5], 1):
        print(f"\n{i}. Valid Accuracy: {result['valid_accuracy']:.2f}% (Train: {result['train_accuracy']:.2f}%, Test: {result['test_accuracy']:.2f}%)")
        print(f"   Dataset: {result['dataset'].upper()}")
        print(f"   Model: {result['model'].upper()}")
        print(f"   Parameters:")
        for key, value in sorted(result['params'].items()):
            print(f"     {key}: {value}")
        if result['antisymmetric']:
            print(f"     antisymmetric: True")
            print(f"     coupling_epsilon: {result['coupling_epsilon']}")
    
    print("\n" + "="*80 + "\n")
    
    return successful[0] if successful else None


def main():
    """Run complete grid search"""
    parser = argparse.ArgumentParser(description='Unified grid search for FordB and OliveOil')
    parser.add_argument('--datasets', nargs='+', choices=['fordb', 'oliveoil', 'both'], 
                        default=['both'], help='Which datasets to run grid search on')
    parser.add_argument('--models', nargs='+', choices=['esn', 'ron', 'both'], 
                        default=['both'], help='Which models to test')
    parser.add_argument('--skip-coupling', action='store_true', 
                        help='Skip antisymmetric coupling experiments')
    args = parser.parse_args()
    
    # Determine which datasets to use
    if 'both' in args.datasets:
        datasets_to_run = ['fordb', 'oliveoil']
    else:
        datasets_to_run = args.datasets
    
    # Determine which models to use
    if 'both' in args.models:
        models_to_run = ['esn', 'ron']
    else:
        models_to_run = args.models
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('experiments/unified_grid_search_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    best_models = {}
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              Unified Hyperparameter Grid Search                              ║
║                                                                              ║
║  Datasets: FordB, OliveOil                                                   ║
║  Models: ESN, RON                                                            ║
║  Selection Criterion: Validation Accuracy (no --use_test)                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    for dataset_name in datasets_to_run:
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        dataset_results = {}
        
        for model_type in models_to_run:
            # Without antisymmetric coupling
            config_name = f"{dataset_name}_{model_type}_no_coupling"
            print(f"\n{'-'*80}")
            print(f"Configuration: {config_name}")
            print(f"{'-'*80}")
            
            results = grid_search_model(dataset_name, model_type, antisymmetric=False)
            dataset_results[config_name] = results
            save_results(results, f'{config_name}_{timestamp}.json', results_dir)
            best = print_summary(results, f"{dataset_name.upper()} - {model_type.upper()} (No Coupling)")
            if best:
                best_models[config_name] = best
            
            # With antisymmetric coupling (if not skipped)
            if not args.skip_coupling:
                config_name = f"{dataset_name}_{model_type}_with_coupling"
                print(f"\n{'-'*80}")
                print(f"Configuration: {config_name}")
                print(f"{'-'*80}")
                
                results = grid_search_model(dataset_name, model_type, antisymmetric=True, coupling_epsilon=0.1)
                dataset_results[config_name] = results
                save_results(results, f'{config_name}_{timestamp}.json', results_dir)
                best = print_summary(results, f"{dataset_name.upper()} - {model_type.upper()} (With Coupling)")
                if best:
                    best_models[config_name] = best
        
        all_results[dataset_name] = dataset_results
    
    # Save all results
    save_results(all_results, f'all_results_{timestamp}.json', results_dir)
    save_results(best_models, f'best_models_{timestamp}.json', results_dir)
    
    # Final summary - Best model per dataset
    print("\n" + "="*80)
    print("FINAL SUMMARY - BEST MODELS PER DATASET")
    print("="*80)
    
    for dataset_name in datasets_to_run:
        print(f"\n{'─'*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'─'*80}")
        
        # Find best model for this dataset across all configurations
        dataset_best = None
        dataset_best_config = None
        
        for config_name, best_model in best_models.items():
            if best_model['dataset'] == dataset_name:
                if dataset_best is None or best_model['valid_accuracy'] > dataset_best['valid_accuracy']:
                    dataset_best = best_model
                    dataset_best_config = config_name
        
        if dataset_best:
            print(f"\n🏆 BEST MODEL: {dataset_best_config}")
            print(f"   Model: {dataset_best['model'].upper()}")
            print(f"   Valid Accuracy: {dataset_best['valid_accuracy']:.2f}%")
            print(f"   Train Accuracy: {dataset_best['train_accuracy']:.2f}%")
            print(f"   Test Accuracy: {dataset_best['test_accuracy']:.2f}%")
            print(f"   Parameters:")
            for key, value in sorted(dataset_best['params'].items()):
                print(f"     {key}: {value}")
            if dataset_best['antisymmetric']:
                print(f"     antisymmetric: True")
                print(f"     coupling_epsilon: {dataset_best['coupling_epsilon']}")
        else:
            print("   No successful experiments!")
    
    print("\n" + "="*80)
    print(f"All results saved to: {results_dir}")
    print(f"Timestamp: {timestamp}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
