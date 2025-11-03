"""
Final Training Script for UCR Time Series Classification
Trains models with best hyperparameters from search and evaluates on test set.

Usage:
    python ucr_train_final.py --config ./ucr_results/FordA_esn_antisym_*/best_config.json --dataset FordA --trials 10
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import torch
import numpy as np
from tqdm import tqdm

from ucr_experiments.ucr_data_loader import get_ucr_data, UCR_DATASETS
from ucr_experiments.model_factory import create_model
from ucr_experiments.training import train_and_evaluate


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to config JSON file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_final_training(args: argparse.Namespace):
    """Run final training with multiple trials on test set.
    
    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print(f"UCR FINAL TRAINING: {args.dataset}")
    print("=" * 80)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")
    print(f"Model: {config['model'].upper()}")
    print(f"Antisymmetric: {config['antisymmetric']}")
    if config['model'] == 'ron':
        print(f"Topology: {config['topology']}")
    print(f"Trials: {args.trials}")
    print("=" * 80)
    
    # Setup device
    device = torch.device(args.device)
    
    # Load data (use whole training set for final training)
    print("\nLoading data...")
    data_root = Path(args.dataroot)
    train_loader, _, test_loader, metadata = get_ucr_data(
        args.dataset,
        data_root,
        bs_train=args.batch_size,
        bs_test=args.batch_size,
        valid_split=0.0,
        whole_train=True,  # Use all training data
        download=True,
    )
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_desc = f"{config['model']}_antisym" if config['antisymmetric'] else config['model']
    if config['model'] == 'ron' and config['antisymmetric']:
        model_desc += f"_{config['topology']}"
    
    results_dir = Path(args.resultroot) / f"{args.dataset}_{model_desc}_final_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {results_dir}")
    
    # Run multiple trials
    print(f"\nRunning {args.trials} trials...")
    train_accs: List[float] = []
    test_accs: List[float] = []
    stable_trials = 0
    
    for trial in tqdm(range(args.trials), desc="Training trials"):
        # Set different seed for each trial
        set_seed(args.seed + trial)
        
        # Create model
        try:
            model = create_model(
                config['model'],
                config['params'],
                device,
                antisymmetric=config['antisymmetric'],
                topology=config.get('topology', 'full'),
            )
        except Exception as e:
            print(f"\n⚠ Trial {trial}: Failed to create model: {e}")
            continue
        
        # Train and evaluate
        train_acc, _, test_acc, is_stable = train_and_evaluate(
            model,
            train_loader,
            train_loader,  # No validation for final training
            test_loader,
            device,
            max_iter=args.max_iter,
            return_test=True,
        )
        
        if is_stable:
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            stable_trials += 1
            
            if (trial + 1) % 10 == 0 or trial == args.trials - 1:
                print(f"\nTrial {trial + 1}/{args.trials}:")
                print(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}")
        else:
            print(f"\n⚠ Trial {trial}: Model unstable, skipping")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute statistics
    if len(test_accs) == 0:
        print("\n✗ No stable trials! Check hyperparameters.")
        return
    
    train_mean = np.mean(train_accs)
    train_std = np.std(train_accs)
    test_mean = np.mean(test_accs)
    test_std = np.std(test_accs)
    test_max = np.max(test_accs)
    test_min = np.min(test_accs)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Stable trials: {stable_trials}/{args.trials}")
    print(f"\nTrain Accuracy: {train_mean:.4f} ± {train_std:.4f}")
    print(f"Test Accuracy:  {test_mean:.4f} ± {test_std:.4f}")
    print(f"Test Max:       {test_max:.4f}")
    print(f"Test Min:       {test_min:.4f}")
    print("=" * 80)
    
    # Save results
    results = {
        'dataset': args.dataset,
        'model': config['model'],
        'antisymmetric': config['antisymmetric'],
        'topology': config.get('topology'),
        'config': config,
        'trials': {
            'total': args.trials,
            'stable': stable_trials,
        },
        'train_accuracy': {
            'mean': float(train_mean),
            'std': float(train_std),
            'all': [float(x) for x in train_accs],
        },
        'test_accuracy': {
            'mean': float(test_mean),
            'std': float(test_std),
            'max': float(test_max),
            'min': float(test_min),
            'all': [float(x) for x in test_accs],
        },
        'metadata': {
            'dataset_name': metadata['dataset_name'],
            'n_classes': metadata['n_classes'],
            'seq_length': metadata['seq_length'],
            'n_train': metadata['n_train'],
            'n_test': metadata['n_test'],
        },
        'timestamp': timestamp,
    }
    
    # Save JSON
    json_path = results_dir / 'final_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {json_path}")
    
    # Save text summary
    summary_path = results_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {config['model'].upper()}\n")
        f.write(f"Antisymmetric: {config['antisymmetric']}\n")
        if config['model'] == 'ron':
            f.write(f"Topology: {config['topology']}\n")
        f.write(f"\nStable trials: {stable_trials}/{args.trials}\n")
        f.write(f"\nTrain Accuracy: {train_mean:.4f} ± {train_std:.4f}\n")
        f.write(f"Test Accuracy:  {test_mean:.4f} ± {test_std:.4f}\n")
        f.write(f"Test Max:       {test_max:.4f}\n")
        f.write(f"Test Min:       {test_min:.4f}\n")
        f.write(f"\nBest Hyperparameters:\n")
        for key, value in config['params'].items():
            f.write(f"  {key}: {value}\n")
    print(f"✓ Summary saved to: {summary_path}")
    
    # Generate plot
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram of test accuracies
        ax1.hist(test_accs, bins=min(20, len(test_accs)), edgecolor='black', alpha=0.7)
        ax1.axvline(test_mean, color='red', linestyle='--', label=f'Mean: {test_mean:.4f}')
        ax1.set_xlabel('Test Accuracy')
        ax1.set_ylabel('Count')
        ax1.set_title(f'{args.dataset} - Test Accuracy Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Trial progression
        ax2.plot(range(1, len(test_accs) + 1), test_accs, 'o-', alpha=0.6)
        ax2.axhline(test_mean, color='red', linestyle='--', label=f'Mean: {test_mean:.4f}')
        ax2.fill_between(range(1, len(test_accs) + 1), 
                         test_mean - test_std, 
                         test_mean + test_std, 
                         alpha=0.2, color='red')
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title(f'{args.dataset} - Trial Progression')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved to: {results_dir / 'results.png'}")
    except Exception as e:
        print(f"⚠ Could not generate plot: {e}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="UCR Final Training with Best Config")
    
    # Config
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to best_config.json from hyperparameter search'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(UCR_DATASETS.keys()),
        help='UCR dataset name'
    )
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/UCR',
        help='Root directory for datasets'
    )
    
    # Training
    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='Number of training trials'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=1000,
        help='Maximum iterations for readout training'
    )
    
    # System
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (base seed, each trial uses seed+trial_num)'
    )
    parser.add_argument(
        '--resultroot',
        type=str,
        default='./ucr_results',
        help='Root directory for results'
    )
    
    args = parser.parse_args()
    
    # Run training
    run_final_training(args)


if __name__ == '__main__':
    main()
