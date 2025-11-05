"""
Hyperparameter Search with K-Fold Cross-Validation for Small UCR Datasets
Uses Optuna for Bayesian optimization with k-fold CV.

Usage:
    python ucr_hyperparameter_search_kfold.py --dataset OliveOil --model esn --n_folds 5
    python ucr_hyperparameter_search_kfold.py --dataset CinCECGTorso --model ron --antisymmetric
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold

from ucr_experiments.ucr_data_loader import get_ucr_data, UCR_DATASETS
from ucr_experiments.model_factory import create_model
from ucr_experiments.training import train_and_evaluate
from torch.utils.data import DataLoader, Subset


# Dataset-specific configurations
SMALL_DATASETS = {
    'OliveOil': {'reservoir_size': 150, 'default_folds': 5},
    'CinCECGTorso': {'reservoir_size': 150, 'default_folds': 5},
}


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_esn_search_space(trial: optuna.Trial, antisymmetric: bool) -> Dict[str, Any]:
    """Create search space for ESN hyperparameters."""
    config = {
        'n_inp': 1,
        'rho': trial.suggest_float('rho', 0.9, 0.99),
        'input_scaling': trial.suggest_float('input_scaling', 0.1, 2.0),
        'leaky': trial.suggest_float('leaky', 0.0001, 0.1, log=True),
    }
    
    if antisymmetric:
        config['coupling_epsilon'] = trial.suggest_float('coupling_epsilon', 0.001, 20.0, log=True)
    
    return config


def create_ron_search_space(trial: optuna.Trial, antisymmetric: bool) -> Dict[str, Any]:
    """Create search space for RON hyperparameters."""
    config = {
        'n_inp': 1,
        'dt': trial.suggest_float('dt', 0.001, 1, log=True),
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),
        'epsilon': trial.suggest_float('epsilon', 0.1, 5.0),
        'gamma_range': trial.suggest_float('gamma_range', 0.0, 2.0),
        'epsilon_range': trial.suggest_float('epsilon_range', 0.0, 2.0),
        'rho': trial.suggest_float('rho', 0.9, 0.99),
        'input_scaling': trial.suggest_float('input_scaling', 0.1, 10.0),
        'reservoir_scaler': trial.suggest_float('reservoir_scaler', 0.0, 2.0),
        'diffusive_gamma': trial.suggest_float('diffusive_gamma', 0.0, 0.1),
    }
    
    if antisymmetric:
        config['coupling_epsilon'] = trial.suggest_float('coupling_epsilon', 0.001, 20.0, log=True)
    
    return config


def objective_kfold(
    trial: optuna.Trial,
    args: argparse.Namespace,
    full_dataset,
    full_labels,
    test_loader,
    device: torch.device,
    reservoir_size: int,
    n_folds: int = 5,
) -> float:
    """Objective function with k-fold cross-validation."""
    # Create search space
    if args.model == 'esn':
        config = create_esn_search_space(trial, args.antisymmetric)
    elif args.model == 'ron':
        config = create_ron_search_space(trial, args.antisymmetric)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    config['reservoir_size'] = reservoir_size
    
    # K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), full_labels)):
        # Create dataloaders for this fold
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        # Create model for this fold
        try:
            model = create_model(
                args.model,
                config,
                device,
                antisymmetric=args.antisymmetric,
            )
        except Exception as e:
            print(f"⚠ Failed to create model for fold {fold}: {e}")
            return 0.0
        
        # Train and evaluate
        train_acc, valid_acc, _, is_stable = train_and_evaluate(
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
            max_iter=1000,
            return_test=False,
        )
        
        if not is_stable:
            return 0.0
        
        fold_scores.append(valid_acc)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return mean validation accuracy across folds
    mean_cv_score = np.mean(fold_scores)
    std_cv_score = np.std(fold_scores)
    
    # Store fold results
    trial.set_user_attr('cv_scores', fold_scores)
    trial.set_user_attr('cv_mean', mean_cv_score)
    trial.set_user_attr('cv_std', std_cv_score)
    
    return mean_cv_score


def run_kfold_search(args: argparse.Namespace):
    """Run hyperparameter search with k-fold CV."""
    if args.dataset not in SMALL_DATASETS:
        print(f"⚠️  Warning: {args.dataset} is not configured as a small dataset.")
        print(f"   Consider using 'ucr_hyperparameter_search.py' instead.")
        reservoir_size = 100
        n_folds = args.n_folds
    else:
        dataset_config = SMALL_DATASETS[args.dataset]
        reservoir_size = dataset_config['reservoir_size']
        n_folds = args.n_folds if args.n_folds else dataset_config['default_folds']
    
    print("=" * 80)
    print(f"UCR K-FOLD HYPERPARAMETER SEARCH: {args.dataset}")
    print("=" * 80)
    print(f"Model: {args.model.upper()}")
    print(f"Antisymmetric: {args.antisymmetric}")
    print(f"Reservoir Size: {reservoir_size} units")
    print(f"K-Folds: {n_folds}")
    print(f"Trials: {args.n_trials}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device)
    
    # Load data - use whole_train=True since we'll do CV manually
    print("\nLoading data...")
    data_root = Path(args.dataroot)
    train_loader, _, test_loader, metadata = get_ucr_data(
        args.dataset,
        data_root,
        bs_train=args.batch_size,
        bs_test=args.batch_size,
        valid_split=0.0,
        whole_train=True,
        download=True,
    )
    
    # Extract full dataset and labels for CV
    full_dataset = train_loader.dataset
    full_labels = np.array([label.item() if torch.is_tensor(label) else label 
                           for _, label in full_dataset])
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_desc = f"{args.model}_antisym" if args.antisymmetric else args.model
    
    results_dir = Path(args.resultroot) / f"{args.dataset}_{model_desc}_kfold_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {results_dir}")
    
    # Create study
    study_name = f"{args.dataset}_{model_desc}_kfold"
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )
    
    # Run optimization
    print("\nStarting k-fold cross-validation optimization...")
    study.optimize(
        lambda trial: objective_kfold(
            trial, args, full_dataset, full_labels, test_loader, 
            device, reservoir_size, n_folds
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        show_progress_bar=True,
    )
    
    # Get best trial
    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("K-FOLD OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best CV accuracy: {best_trial.value:.4f}")
    print(f"CV std: {best_trial.user_attrs.get('cv_std', 0.0):.4f}")
    print(f"Best trial number: {best_trial.number}")
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    metadata_for_json = {
        'dataset_name': metadata['dataset_name'],
        'n_classes': int(metadata['n_classes']),
        'seq_length': int(metadata['seq_length']),
        'n_train': int(metadata['n_train']),
        'n_test': int(metadata['n_test']),
        'label_classes': metadata['label_encoder'].classes_.tolist() if 'label_encoder' in metadata else None,
    }
    
    results = {
        'dataset': args.dataset,
        'model': args.model,
        'antisymmetric': args.antisymmetric,
        'n_trials': args.n_trials,
        'n_folds': n_folds,
        'reservoir_size': reservoir_size,
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
    json_path = results_dir / 'search_results_kfold.json'
    with open(json_path, 'w') as f:
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
    
    # Save best config
    best_config_path = results_dir / 'best_config.json'
    with open(best_config_path, 'w') as f:
        json.dump({
            'model': args.model,
            'antisymmetric': args.antisymmetric,
            'reservoir_size': reservoir_size,
            'params': best_trial.params,
            'n_inp': 1,
            'cv_mean': best_trial.user_attrs.get('cv_mean'),
            'cv_std': best_trial.user_attrs.get('cv_std'),
        }, f, indent=2)
    print(f"✓ Best config saved to: {best_config_path}")
    
    # Save study
    study_path = results_dir / 'optuna_study.pkl'
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"✓ Optuna study saved to: {study_path}")
    
    print("\n" + "=" * 80)
    print(f"Next: Run final training with ucr_train_final.py")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="UCR K-Fold Hyperparameter Search")
    
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
        help='Use antisymmetric coupling'
    )
    
    # Cross-validation
    parser.add_argument(
        '--n_folds',
        type=int,
        default=None,
        help='Number of CV folds (default: 5 for small datasets)'
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
        help='Timeout in seconds'
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
    run_kfold_search(args)


if __name__ == '__main__':
    main()
