"""
Hyperparameter Search for UCR Time Series Classification
Uses Optuna for Bayesian optimization of RON and ESN models.

Usage:
    python ucr_hyperparameter_search.py --dataset FordA --model esn --antisymmetric
    python ucr_hyperparameter_search.py --dataset Adiac --model ron --topology antisymmetric
"""
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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
    # Note: deterministic algorithms can slow down training
    # torch.use_deterministic_algorithms(True)


def create_esn_search_space(trial: optuna.Trial, antisymmetric: bool) -> Dict[str, Any]:
    """Create search space for ESN hyperparameters.
    
    Args:
        trial: Optuna trial object
        antisymmetric: Whether using antisymmetric coupling
        
    Returns:
        Configuration dictionary
    """
    config = {
        'n_inp': 1,
        'rho': trial.suggest_float('rho', 0.1, 0.99),
        'input_scaling': trial.suggest_float('input_scaling', 0.1, 2.0),
        'leaky': trial.suggest_float('leaky', 0.0001, 0.1, log=True),
    }
    
    if antisymmetric:
        config['coupling_epsilon'] = trial.suggest_float('coupling_epsilon', 0.001, 1.0, log=True)
        config['inter_scaling'] = trial.suggest_float('inter_scaling', 0.1, 2.0)
    
    return config


def create_ron_search_space(trial: optuna.Trial, antisymmetric: bool) -> Dict[str, Any]:
    """Create search space for RON hyperparameters.
    
    Args:
        trial: Optuna trial object
        antisymmetric: Whether using antisymmetric coupling
        
    Returns:
        Configuration dictionary
    """
    config = {
        'n_inp': 1,
        'dt': trial.suggest_float('dt', 0.001, 0.1, log=True),
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),
        'epsilon': trial.suggest_float('epsilon', 0.1, 5.0),
        'gamma_range': trial.suggest_float('gamma_range', 0.0, 2.0),
        'epsilon_range': trial.suggest_float('epsilon_range', 0.0, 2.0),
        'rho': trial.suggest_float('rho', 0.1, 0.99),
        'input_scaling': trial.suggest_float('input_scaling', 0.1, 10.0),
        'reservoir_scaler': trial.suggest_float('reservoir_scaler', 0.0, 2.0),
        'diffusive_gamma': trial.suggest_float('diffusive_gamma', 0.0, 0.1),
    }
    
    if antisymmetric:
        config['coupling_epsilon'] = trial.suggest_float('coupling_epsilon', 0.001, 1.0, log=True)
        config['inter_scaling'] = trial.suggest_float('inter_scaling', 0.1, 2.0)
    
    return config


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    train_loader,
    valid_loader,
    test_loader,
    device: torch.device,
) -> float:
    """Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial
        args: Command line arguments
        train_loader: Training data loader
        valid_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Validation accuracy to maximize
    """
    # Create search space
    if args.model == 'esn':
        config = create_esn_search_space(trial, args.antisymmetric)
    elif args.model == 'ron':
        config = create_ron_search_space(trial, args.antisymmetric)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Create model
    try:
        model = create_model(
            args.model,
            config,
            device,
            antisymmetric=args.antisymmetric,
            topology=args.topology,
        )
    except Exception as e:
        print(f"⚠ Failed to create model: {e}")
        return 0.0
    
    # Train and evaluate
    train_acc, valid_acc, test_acc, is_stable = train_and_evaluate(
        model,
        train_loader,
        valid_loader,
        test_loader,
        device,
        max_iter=1000,
        return_test=False,
    )
    
    if not is_stable:
        # Penalize unstable configurations
        return 0.0
    
    # Store additional metrics
    trial.set_user_attr('train_acc', train_acc)
    trial.set_user_attr('valid_acc', valid_acc)
    trial.set_user_attr('is_stable', is_stable)
    
    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return valid_acc


def run_hyperparameter_search(args: argparse.Namespace):
    """Run hyperparameter search with Optuna.
    
    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print(f"UCR HYPERPARAMETER SEARCH: {args.dataset}")
    print("=" * 80)
    print(f"Model: {args.model.upper()}")
    print(f"Antisymmetric: {args.antisymmetric}")
    if args.model == 'ron':
        print(f"Topology: {args.topology}")
    print(f"Trials: {args.n_trials}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device)
    
    # Load data
    print("\nLoading data...")
    data_root = Path(args.dataroot)
    train_loader, valid_loader, test_loader, metadata = get_ucr_data(
        args.dataset,
        data_root,
        bs_train=args.batch_size,
        bs_test=args.batch_size,
        valid_split=0.2,
        whole_train=False,
        download=True,
    )
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_desc = f"{args.model}_antisym" if args.antisymmetric else args.model
    if args.model == 'ron' and args.antisymmetric:
        model_desc += f"_{args.topology}"
    
    results_dir = Path(args.resultroot) / f"{args.dataset}_{model_desc}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {results_dir}")
    
    # Create study
    study_name = f"{args.dataset}_{model_desc}"
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )
    
    # Run optimization
    print("\nStarting optimization...")
    study.optimize(
        lambda trial: objective(trial, args, train_loader, valid_loader, test_loader, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,  # Sequential execution for GPU
        show_progress_bar=True,
    )
    
    # Get best trial
    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {best_trial.value:.4f}")
    print(f"Best trial number: {best_trial.number}")
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    # Create a JSON-serializable version of metadata (exclude non-serializable objects)
    metadata_for_json = {
        'dataset_name': metadata['dataset_name'],
        'n_classes': int(metadata['n_classes']),
        'seq_length': int(metadata['seq_length']),
        'n_train': int(metadata['n_train']),
        'n_valid': int(metadata['n_valid']),
        'n_test': int(metadata['n_test']),
        'label_classes': metadata['label_encoder'].classes_.tolist() if 'label_encoder' in metadata else None,
    }
    
    results = {
        'dataset': args.dataset,
        'model': args.model,
        'antisymmetric': args.antisymmetric,
        'topology': args.topology if args.model == 'ron' else None,
        'n_trials': args.n_trials,
        'best_trial': {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params,
            'user_attrs': best_trial.user_attrs,
        },
        'metadata': metadata_for_json,
        'timestamp': timestamp,
    }
    
    # Save JSON
    json_path = results_dir / 'search_results.json'
    with open(json_path, 'w') as f:
        # Convert numpy types to python types for JSON serialization
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            return o
        
        json.dump(results, f, indent=2, default=convert)
    print(f"\n✓ Results saved to: {json_path}")
    
    # Save best config for training
    best_config_path = results_dir / 'best_config.json'
    with open(best_config_path, 'w') as f:
        json.dump({
            'model': args.model,
            'antisymmetric': args.antisymmetric,
            'topology': args.topology if args.model == 'ron' else None,
            'params': best_trial.params,
            'n_inp': 1,
        }, f, indent=2)
    print(f"✓ Best config saved to: {best_config_path}")
    
    # Save study for further analysis
    study_path = results_dir / 'optuna_study.pkl'
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"✓ Optuna study saved to: {study_path}")
    
    # Generate optimization history plot
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        
        fig = plot_optimization_history(study)
        fig.savefig(results_dir / 'optimization_history.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        if len(best_trial.params) > 1:
            fig = plot_param_importances(study)
            fig.savefig(results_dir / 'param_importances.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"✓ Plots saved to: {results_dir}")
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")
    
    print("\n" + "=" * 80)
    print(f"Next step: Run training with best config using ucr_train_final.py")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="UCR Time Series Hyperparameter Search")
    
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
    
    # Model
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['esn', 'ron'],
        help='Model type'
    )
    parser.add_argument(
        '--antisymmetric',
        action='store_true',
        help='Use antisymmetric coupling (5 layers with 100 units each)'
    )
    parser.add_argument(
        '--topology',
        type=str,
        default='full',
        choices=['full', 'antisymmetric', 'orthogonal'],
        help='Topology for RON models'
    )
    
    # Search
    parser.add_argument(
        '--n_trials',
        type=int,
        default=30,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds (None for no timeout)'
    )
    
    # Training
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
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
        help='Random seed'
    )
    parser.add_argument(
        '--resultroot',
        type=str,
        default='./ucr_results',
        help='Root directory for results'
    )
    
    args = parser.parse_args()
    
    # Run search
    run_hyperparameter_search(args)


if __name__ == '__main__':
    main()
