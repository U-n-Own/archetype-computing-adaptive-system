"""
Batch Experiment Runner for UCR Datasets
Runs hyperparameter search for all datasets and model configurations.

Usage:
    python run_all_ucr_experiments.py --phase search
    python run_all_ucr_experiments.py --phase train
"""
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time

from ucr_experiments.ucr_data_loader import UCR_DATASETS


# Experiment configurations
EXPERIMENTS = [
    # ESN Standard
    {'model': 'esn', 'antisymmetric': False, 'topology': 'full'},
    # ESN Antisymmetric
    {'model': 'esn', 'antisymmetric': True, 'topology': 'full'},
    # RON Standard (Full topology)
    {'model': 'ron', 'antisymmetric': False, 'topology': 'full'},
    # RON Antisymmetric (Full topology)
    {'model': 'ron', 'antisymmetric': True, 'topology': 'full'},
    # RON Antisymmetric (Antisymmetric topology)
    {'model': 'ron', 'antisymmetric': True, 'topology': 'antisymmetric'},
    # RON Antisymmetric (Orthogonal topology)
    {'model': 'ron', 'antisymmetric': True, 'topology': 'orthogonal'},
]


def run_search(
    dataset: str,
    experiment: Dict,
    args: argparse.Namespace,
) -> Dict:
    """Run hyperparameter search for a single experiment.
    
    Args:
        dataset: Dataset name
        experiment: Experiment configuration
        args: Command line arguments
        
    Returns:
        Result dictionary with status and paths
    """
    model_desc = f"{experiment['model']}_antisym_{experiment['topology']}" if experiment['antisymmetric'] else experiment['model']
    
    print("\n" + "=" * 80)
    print(f"SEARCH: {dataset} - {model_desc}")
    print("=" * 80)
    
    # Build command
    cmd = [
        'python', 'ucr_hyperparameter_search.py',
        '--dataset', dataset,
        '--model', experiment['model'],
        '--topology', experiment['topology'],
        '--n_trials', str(args.n_trials),
        '--batch_size', str(args.batch_size),
        '--device', args.device,
        '--dataroot', args.dataroot,
        '--resultroot', args.resultroot,
        '--seed', str(args.seed),
    ]
    
    if experiment['antisymmetric']:
        cmd.append('--antisymmetric')
    
    if args.timeout:
        cmd.extend(['--timeout', str(args.timeout)])
    
    # Run search
    start_time = time.time()
    try:
        # Show output in real-time instead of capturing it
        result = subprocess.run(
            cmd,
            timeout=args.max_time_per_experiment,
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Search completed in {elapsed:.1f}s")
            
            # Find the results directory
            results_base = Path(args.resultroot)
            pattern = f"{dataset}_{model_desc}_*"
            matching_dirs = sorted(results_base.glob(pattern))
            
            if matching_dirs:
                latest_dir = matching_dirs[-1]
                config_file = latest_dir / 'best_config.json'
                
                return {
                    'status': 'success',
                    'dataset': dataset,
                    'model': model_desc,
                    'elapsed': elapsed,
                    'results_dir': str(latest_dir),
                    'config_file': str(config_file) if config_file.exists() else None,
                }
            else:
                print(f"⚠ Results directory not found")
                return {
                    'status': 'no_results',
                    'dataset': dataset,
                    'model': model_desc,
                    'elapsed': elapsed,
                }
        else:
            print(f"✗ Search failed with return code {result.returncode}")
            return {
                'status': 'failed',
                'dataset': dataset,
                'model': model_desc,
                'elapsed': elapsed,
                'error': f"Return code {result.returncode}",
            }
    
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ Search timed out after {elapsed:.1f}s")
        return {
            'status': 'timeout',
            'dataset': dataset,
            'model': model_desc,
            'elapsed': elapsed,
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Search failed with exception: {e}")
        return {
            'status': 'exception',
            'dataset': dataset,
            'model': model_desc,
            'elapsed': elapsed,
            'error': str(e),
        }


def run_training(
    dataset: str,
    config_file: str,
    args: argparse.Namespace,
) -> Dict:
    """Run final training for a single experiment.
    
    Args:
        dataset: Dataset name
        config_file: Path to best config file
        args: Command line arguments
        
    Returns:
        Result dictionary with status and metrics
    """
    print("\n" + "=" * 80)
    print(f"TRAINING: {dataset} - {config_file}")
    print("=" * 80)
    
    # Build command
    cmd = [
        'python', 'ucr_train_final.py',
        '--config', config_file,
        '--dataset', dataset,
        '--trials', str(args.final_trials),
        '--batch_size', str(args.batch_size),
        '--device', args.device,
        '--dataroot', args.dataroot,
        '--resultroot', args.resultroot,
        '--seed', str(args.seed),
        '--max_iter', str(args.max_iter),
    ]
    
    # Run training
    start_time = time.time()
    try:
        # Show output in real-time instead of capturing it
        result = subprocess.run(
            cmd,
            timeout=args.max_time_per_experiment,
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Training completed in {elapsed:.1f}s")
            
            # Try to parse results
            results_base = Path(args.resultroot)
            # Find latest results directory for this dataset
            pattern = f"{dataset}_*_final_*"
            matching_dirs = sorted(results_base.glob(pattern))
            
            if matching_dirs:
                latest_dir = matching_dirs[-1]
                results_file = latest_dir / 'final_results.json'
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results_data = json.load(f)
                    
                    return {
                        'status': 'success',
                        'dataset': dataset,
                        'config_file': config_file,
                        'elapsed': elapsed,
                        'results_dir': str(latest_dir),
                        'test_acc_mean': results_data['test_accuracy']['mean'],
                        'test_acc_std': results_data['test_accuracy']['std'],
                    }
            
            return {
                'status': 'success_no_parse',
                'dataset': dataset,
                'config_file': config_file,
                'elapsed': elapsed,
            }
        else:
            print(f"✗ Training failed with return code {result.returncode}")
            return {
                'status': 'failed',
                'dataset': dataset,
                'config_file': config_file,
                'elapsed': elapsed,
                'error': f"Return code {result.returncode}",
            }
    
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ Training timed out after {elapsed:.1f}s")
        return {
            'status': 'timeout',
            'dataset': dataset,
            'config_file': config_file,
            'elapsed': elapsed,
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Training failed with exception: {e}")
        return {
            'status': 'exception',
            'dataset': dataset,
            'config_file': config_file,
            'elapsed': elapsed,
            'error': str(e),
        }


def run_all_searches(args: argparse.Namespace):
    """Run hyperparameter search for all datasets and configurations."""
    print("=" * 80)
    print("UCR BATCH HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Experiments per dataset: {len(EXPERIMENTS)}")
    print(f"Total searches: {len(args.datasets) * len(EXPERIMENTS)}")
    print("=" * 80)
    
    all_results = []
    
    for dataset in args.datasets:
        for experiment in EXPERIMENTS:
            if args.skip_standard and not experiment['antisymmetric']:
                print(f"\nSkipping standard {experiment['model']} for {dataset}")
                continue
            
            result = run_search(dataset, experiment, args)
            all_results.append(result)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(args.resultroot) / f'search_summary_{timestamp}.json'
    
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'args': vars(args),
            'results': all_results,
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SEARCH SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    print(f"Successful: {success_count}/{len(all_results)}")
    print(f"Summary saved to: {summary_file}")
    
    # Print table
    print("\nResults:")
    for result in all_results:
        status_icon = "✓" if result['status'] == 'success' else "✗"
        print(f"  {status_icon} {result['dataset']:15s} {result['model']:25s} {result['status']:10s} ({result['elapsed']:.1f}s)")
    
    print("=" * 80)


def run_all_training(args: argparse.Namespace):
    """Run final training for all found configurations."""
    print("=" * 80)
    print("UCR BATCH FINAL TRAINING")
    print("=" * 80)
    
    # Find all config files
    results_base = Path(args.resultroot)
    config_files = list(results_base.glob("*/best_config.json"))
    
    print(f"Found {len(config_files)} config files")
    print("=" * 80)
    
    all_results = []
    
    for config_file in config_files:
        # Extract dataset name from parent directory
        parent_name = config_file.parent.name
        dataset = None
        for ds in UCR_DATASETS.keys():
            if parent_name.startswith(ds):
                dataset = ds
                break
        
        if dataset is None:
            print(f"⚠ Could not determine dataset for {config_file}")
            continue
        
        result = run_training(dataset, str(config_file), args)
        all_results.append(result)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(args.resultroot) / f'training_summary_{timestamp}.json'
    
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'args': vars(args),
            'results': all_results,
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    print(f"Successful: {success_count}/{len(all_results)}")
    print(f"Summary saved to: {summary_file}")
    
    # Print table
    print("\nResults:")
    for result in all_results:
        status_icon = "✓" if result['status'] == 'success' else "✗"
        acc_str = f"{result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}" if 'test_acc_mean' in result else "N/A"
        print(f"  {status_icon} {result['dataset']:15s} {acc_str:20s} ({result['elapsed']:.1f}s)")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Batch UCR Experiments")
    
    # Phase
    parser.add_argument(
        '--phase',
        type=str,
        required=True,
        choices=['search', 'train', 'both'],
        help='Experiment phase'
    )
    
    # Datasets
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=list(UCR_DATASETS.keys()),
        choices=list(UCR_DATASETS.keys()),
        help='Datasets to run experiments on'
    )
    
    # Search parameters
    parser.add_argument(
        '--n_trials',
        type=int,
        default=100,
        help='Number of search trials per experiment'
    )
    parser.add_argument(
        '--skip_standard',
        action='store_true',
        help='Skip standard (non-antisymmetric) models'
    )
    
    # Training parameters
    parser.add_argument(
        '--final_trials',
        type=int,
        default=10,
        help='Number of trials for final training'
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=1000,
        help='Max iterations for readout training'
    )
    
    # System
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu or cuda)'
    )
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/UCR',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--resultroot',
        type=str,
        default='./ucr_results',
        help='Root directory for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout for individual search (seconds)'
    )
    parser.add_argument(
        '--max_time_per_experiment',
        type=int,
        default=7200,  # 2 hours
        help='Maximum time per experiment (seconds)'
    )
    
    args = parser.parse_args()
    
    # Run experiments
    if args.phase in ['search', 'both']:
        run_all_searches(args)
    
    if args.phase in ['train', 'both']:
        run_all_training(args)


if __name__ == '__main__':
    main()
