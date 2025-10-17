"""
Search utilities for hyperparameter optimization
"""

import torch
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
import time
from typing import Dict, List, Tuple, Optional, Callable


def test_model(
    model,
    data_loader_or_data,
    classifier,
    scaler,
    device: torch.device,
    is_loader: bool = True,
    use_last_state: bool = True,
    batch_size: int = 64
) -> float:
    """
    Test model on data
    
    Args:
        model: ESN model
        data_loader_or_data: Either DataLoader or tuple (X, y) as numpy arrays
        classifier: Trained classifier
        scaler: Fitted scaler
        device: torch device
        is_loader: Whether data_loader_or_data is a DataLoader
        use_last_state: Whether to use last state or average
        batch_size: Batch size for processing numpy arrays
        
    Returns:
        Accuracy score
    """
    activations = []
    
    if is_loader:
        # DataLoader input
        ys = []
        for batch_data in data_loader_or_data:
            if len(batch_data) == 2:
                X_batch, y_batch = batch_data
            else:
                X_batch = batch_data
                y_batch = None
            
            X_batch = X_batch.to(device)
            
            with torch.no_grad():
                output = model(X_batch)
                if isinstance(output, tuple):
                    states = output[0]
                else:
                    states = output
                
                if use_last_state:
                    if len(states.shape) == 3:
                        final_state = states[:, -1, :].cpu().numpy()
                    else:
                        final_state = states.cpu().numpy()
                else:
                    final_state = states.mean(dim=1).cpu().numpy()
                
                activations.append(final_state)
            
            if y_batch is not None:
                ys.append(y_batch.cpu().numpy() if torch.is_tensor(y_batch) else y_batch)
        
        activations = np.concatenate(activations, axis=0)
        y_true = np.concatenate(ys, axis=0)
        
    else:
        # Numpy array input
        X_data, y_true = data_loader_or_data
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        
        for i in range(0, X_data.shape[0], batch_size):
            batch_X = X_tensor[i:i+batch_size].to(device)
            
            with torch.no_grad():
                output = model(batch_X)
                if isinstance(output, tuple):
                    states = output[0]
                else:
                    states = output
                
                if use_last_state:
                    if len(states.shape) == 3:
                        final_state = states[:, -1, :].cpu().numpy()
                    else:
                        final_state = states.cpu().numpy()
                else:
                    final_state = states.mean(dim=1).cpu().numpy()
                
                activations.append(final_state)
        
        activations = np.concatenate(activations, axis=0)
    
    # Scale and predict
    activations_scaled = scaler.transform(activations)
    return classifier.score(activations_scaled, y_true)


def connection_mode_search(
    train_data,
    valid_data,
    device: torch.device,
    model_constructor: Callable,
    mode: str,
    n_configs: int = 15,
    n_layers: int = 5,
    base_grid: Optional[Dict] = None,
    is_loader: bool = True,
    train_batches: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Hyperparameter search for a specific connection mode
    
    Args:
        train_data: Training data (DataLoader or (X, y) tuple)
        valid_data: Validation data (DataLoader or (X, y) tuple)
        device: torch device
        model_constructor: Function that creates model given params
        mode: Connection mode name (for display)
        n_configs: Number of configurations to test
        n_layers: Number of layers
        base_grid: Base hyperparameter grid (if None, uses default)
        is_loader: Whether data is DataLoader or numpy arrays
        train_batches: Limit training to N batches (for speed)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_config, all_results)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"{mode.upper()} HYPERPARAMETER SEARCH")
        print(f"Layers: {n_layers} | Configurations: {n_configs}")
        print(f"{'='*70}\n")
    
    # Default grid if not provided
    if base_grid is None:
        base_grid = {
            'spectral_radius': [0.9, 0.95, 0.99, 0.999],
            'leaky': [0.001, 0.01, 0.1, 0.5],
            'input_scaling': [0.5, 1.0, 2.0],
            'tot_units': [200, 500, 1000],
        }
    
    # Create parameter combinations
    all_params = list(ParameterGrid(base_grid))
    
    if verbose:
        print(f"Total possible combinations: {len(all_params)}")
    
    # Sample configurations
    if len(all_params) > n_configs:
        import random
        random.seed(42)
        sampled_params = random.sample(all_params, n_configs)
    else:
        sampled_params = all_params
    
    if verbose:
        print(f"Testing {len(sampled_params)} configurations...\n")
    
    best_config = None
    best_valid_acc = 0.0
    all_results = []
    
    for idx, params in enumerate(sampled_params):
        if verbose:
            print(f"\n[{idx+1}/{len(sampled_params)}] {mode} Configuration:")
            for k, v in params.items():
                print(f"  {k:20s}: {v}")
        
        try:
            # Create model
            if verbose:
                print(f"  Creating model...", end='', flush=True)
            
            model = model_constructor(params, n_layers)
            model = model.to(device)
            
            if verbose:
                print(" ✓", flush=True)
            
            # Generate training activations
            if verbose:
                print(f"  Processing training data...", end='', flush=True)
            
            start_time = time.time()
            activations = []
            ys = []
            
            if is_loader:
                for batch_idx, batch_data in enumerate(train_data):
                    if train_batches and batch_idx >= train_batches:
                        break
                    
                    if len(batch_data) == 2:
                        X_batch, y_batch = batch_data
                    else:
                        X_batch = batch_data
                        y_batch = None
                    
                    X_batch = X_batch.to(device)
                    
                    with torch.no_grad():
                        output = model(X_batch)
                        if isinstance(output, tuple):
                            states = output[0]
                        else:
                            states = output
                        
                        if len(states.shape) == 3:
                            final_state = states[:, -1, :].cpu().numpy()
                        else:
                            final_state = states.cpu().numpy()
                        
                        activations.append(final_state)
                    
                    if y_batch is not None:
                        ys.append(y_batch.cpu().numpy() if torch.is_tensor(y_batch) else y_batch)
                
                activations = np.concatenate(activations, axis=0)
                y_train = np.concatenate(ys, axis=0)
                
            else:
                # Numpy array input
                X_train, y_train = train_data
                X_tensor = torch.tensor(X_train, dtype=torch.float32)
                batch_size = 64
                
                n_batches = (len(X_train) + batch_size - 1) // batch_size
                if train_batches:
                    n_batches = min(n_batches, train_batches)
                
                for i in range(0, n_batches * batch_size, batch_size):
                    if i >= len(X_train):
                        break
                    batch_X = X_tensor[i:i+batch_size].to(device)
                    
                    with torch.no_grad():
                        output = model(batch_X)
                        if isinstance(output, tuple):
                            states = output[0]
                        else:
                            states = output
                        
                        if len(states.shape) == 3:
                            final_state = states[:, -1, :].cpu().numpy()
                        else:
                            final_state = states.cpu().numpy()
                        
                        activations.append(final_state)
                
                activations = np.concatenate(activations, axis=0)
                if train_batches:
                    max_samples = train_batches * batch_size
                    y_train = y_train[:max_samples]
            
            train_time = time.time() - start_time
            
            if verbose:
                print(f" ✓ ({activations.shape[0]} samples, {train_time:.2f}s)", flush=True)
            
            # Train classifier
            if verbose:
                print(f"  Training classifier...", end='', flush=True)
            
            scaler = preprocessing.StandardScaler().fit(activations)
            activations_scaled = scaler.transform(activations)
            classifier = LogisticRegression(max_iter=1000, verbose=0).fit(
                activations_scaled, y_train
            )
            
            if verbose:
                print(" ✓", flush=True)
            
            # Evaluate on validation
            if verbose:
                print(f"  Evaluating...", end='', flush=True)
            
            valid_acc = test_model(
                model, valid_data, classifier, scaler, device, 
                is_loader=is_loader
            )
            
            if verbose:
                print(f" ✓", flush=True)
            
            result = {
                'params': params,
                'valid_acc': valid_acc,
                'train_time': train_time,
                'mode': mode
            }
            all_results.append(result)
            
            if verbose:
                print(f"  ➤ Validation Accuracy: {valid_acc:.4f} ({valid_acc*100:.2f}%)")
            
            # Track best
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_config = params.copy()
                if verbose:
                    print(f"  ✨ NEW BEST for {mode}! ✨")
            
        except Exception as e:
            if verbose:
                print(f"  ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"{mode} SEARCH COMPLETE")
        print(f"{'='*70}")
        print(f"Tested: {len(all_results)} configurations")
        print(f"Best validation accuracy: {best_valid_acc:.4f} ({best_valid_acc*100:.2f}%)")
        print(f"\nBest configuration:")
        for key, value in best_config.items():
            print(f"  {key:20s}: {value}")
        
        all_results.sort(key=lambda x: x['valid_acc'], reverse=True)
        print(f"\nTop 3 Configurations:")
        for i, result in enumerate(all_results[:3], 1):
            print(f"{i}. Valid Acc: {result['valid_acc']:.4f} | Time: {result['train_time']:.2f}s")
    
    return best_config, all_results
