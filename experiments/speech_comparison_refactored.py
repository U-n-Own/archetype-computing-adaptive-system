#!/usr/bin/env python3
"""
Refactored Speech Dataset Comparison Script
Compare Antisymmetric vs Cycle ESN on Speech Commands Dataset using modular utilities
"""

import torch
import numpy as np
import sys
import os
import json
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acds.archetypes.esn import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# Import modular utilities
from utils import (
    plot_comparison,
    plot_predictions,
    plot_state_dynamics,
    plot_eigenspectrum_analysis,
    connection_mode_search,
    test_model
)


def load_speech_data(dataroot: str, seed: int = 42):
    """
    Load speech data from .npy files in the specified directory.
    
    Args:
        dataroot: Path to directory containing .npy files
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test, class_names)
    """
    TIME_STEPS = 101
    FEATURE_DIM = 40
    
    speech_files = [f for f in os.listdir(dataroot) if f.endswith('.npy')]
    if not speech_files:
        raise ValueError(f"No .npy files found in {dataroot}")
    
    all_data = []
    all_labels = []
    class_names = []
    
    print("Loading speech data files...")
    for class_idx, file_name in enumerate(tqdm(speech_files, desc="Loading")):
        filepath = os.path.join(dataroot, file_name)
        class_name = file_name.split('.')[0]
        class_names.append(class_name)
        
        try:
            data = np.load(filepath, allow_pickle=True)
            if data.ndim != 3 or data.shape[1] != TIME_STEPS or data.shape[2] != FEATURE_DIM:
                print(f"Skipping {file_name}: unexpected shape {data.shape}")
                continue
            if data.shape[0] == 0:
                print(f"Skipping {file_name}: no samples found")
                continue

            all_data.append(data)
            all_labels.append(np.full(data.shape[0], class_idx))
            print(f"  {file_name}: {data.shape[0]} samples")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data loaded")
    
    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    # Split into train, validation, and test sets
    np.random.seed(seed)
    indices = np.random.permutation(combined_data.shape[0])
    train_size = int(combined_data.shape[0] * 0.7)  # 70% train
    valid_size = int(combined_data.shape[0] * 0.15)  # 15% valid
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]
    
    X_train = combined_data[train_indices]
    y_train = combined_labels[train_indices]
    X_valid = combined_data[valid_indices]
    y_valid = combined_labels[valid_indices]
    X_test = combined_data[test_indices]
    y_test = combined_labels[test_indices]
    
    # Normalize input features
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_valid_reshaped = X_valid.reshape(-1, X_valid.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    input_scaler = preprocessing.StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train_reshaped)
    X_valid_scaled = input_scaler.transform(X_valid_reshaped)
    X_test_scaled = input_scaler.transform(X_test_reshaped)
    
    X_train = X_train_scaled.reshape(X_train.shape)
    X_valid = X_valid_scaled.reshape(X_valid.shape)
    X_test = X_test_scaled.reshape(X_test.shape)
    
    print(f"\nData split: Train={X_train.shape[0]}, Valid={X_valid.shape[0]}, Test={X_test.shape[0]}")
    print(f"Classes: {len(class_names)}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, class_names


def create_model_antisymmetric(params: dict, n_layers: int, n_inp: int) -> DeepReservoir:
    """Create antisymmetric ESN model for speech data"""
    units_per_layer = params['tot_units'] // n_layers
    
    return DeepReservoir(
        input_size=n_inp,
        tot_units=params['tot_units'],
        n_layers=n_layers,
        concat=True,
        spectral_radius=params['spectral_radius'],
        input_scaling=params['input_scaling'],
        leaky=params['leaky'],
        connectivity_recurrent=units_per_layer,
        connectivity_input=units_per_layer,
        connectivity_inter=units_per_layer,
        antisymmetric=True,
        epsilon=params.get('epsilon', 0.1),
        cycle=False,
        linear=False,
    )


def create_model_cycle(params: dict, n_layers: int, n_inp: int) -> DeepReservoir:
    """Create cycle ESN model for speech data"""
    units_per_layer = params['tot_units'] // n_layers
    
    return DeepReservoir(
        input_size=n_inp,
        tot_units=params['tot_units'],
        n_layers=n_layers,
        concat=True,
        spectral_radius=params['spectral_radius'],
        input_scaling=params['input_scaling'],
        leaky=params['leaky'],
        connectivity_recurrent=units_per_layer,
        connectivity_input=units_per_layer,
        connectivity_inter=units_per_layer,
        antisymmetric=False,
        cycle=True,
        linear=False,
    )


def build_final_model(params: dict, constructor, n_layers: int, n_inp: int,
                      train_data: tuple, device: torch.device):
    """
    Build and train final model with best configuration
    
    Args:
        params: Model hyperparameters
        constructor: Model constructor function
        n_layers: Number of layers
        n_inp: Input dimension
        train_data: Tuple of (X_train, y_train)
        device: torch device
        
    Returns:
        Dict with model, classifier, scaler
    """
    print(f"  Building model...", end='', flush=True)
    model = constructor(params, n_layers, n_inp).to(device)
    print(" ✓", flush=True)
    
    print(f"  Training readout...", end='', flush=True)
    X_train, y_train = train_data
    
    activations = []
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    batch_size = 64
    
    for i in range(0, X_train.shape[0], batch_size):
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
    
    scaler = preprocessing.StandardScaler().fit(activations)
    activations_scaled = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations_scaled, y_train)
    print(" ✓", flush=True)
    
    return {
        'model': model,
        'classifier': classifier,
        'scaler': scaler
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare Antisymmetric vs Cycle ESN on Speech Dataset (Refactored)'
    )
    parser.add_argument('--dataroot', type=str, default='./speech',
                       help='Path to speech data directory')
    parser.add_argument('--n_configs', type=int, default=15,
                       help='Number of configurations to test per mode')
    parser.add_argument('--n_layers', type=int, default=5,
                       help='Number of layers for both modes')
    parser.add_argument('--save_dir', type=str, default='./experiments/plots/speech_comparison_refactored',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    print(f"\n{'='*70}")
    print("ANTISYMMETRIC vs CYCLE ESN COMPARISON - SPEECH DATASET (Refactored)")
    print(f"{'='*70}")
    print(f"Data directory: {args.dataroot}")
    print(f"Layers: {args.n_layers}")
    print(f"Configurations per mode: {args.n_configs}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading speech data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names = load_speech_data(
        args.dataroot, seed=args.seed
    )
    n_inp = X_train.shape[2]  # Feature dimension
    print(f"✓ Data loaded (input_dim={n_inp}, n_classes={len(class_names)})\n")
    
    # Define hyperparameter grid
    base_grid = {
        'spectral_radius': [0.9, 0.95, 0.99, 0.999],
        'leaky': [0.001, 0.01, 0.1, 0.5],
        'input_scaling': [0.5, 1.0, 2.0],
        'tot_units': [200, 500],
    }
    
    # Search antisymmetric mode
    print("\n" + "="*70)
    print("PHASE 1: ANTISYMMETRIC MODE SEARCH")
    print("="*70)
    
    grid_anti = base_grid.copy()
    grid_anti['epsilon'] = [0.05, 0.1, 0.2, 0.4]
    
    # Wrapper for model constructor with n_inp
    def create_antisymmetric_wrapper(params, n_layers):
        return create_model_antisymmetric(params, n_layers, n_inp)
    
    best_anti_config, anti_results = connection_mode_search(
        train_data=(X_train, y_train),
        valid_data=(X_valid, y_valid),
        device=device,
        model_constructor=create_antisymmetric_wrapper,
        mode='Antisymmetric',
        n_configs=args.n_configs,
        n_layers=args.n_layers,
        base_grid=grid_anti,
        is_loader=False,
        verbose=True
    )
    
    # Search cycle mode
    print("\n" + "="*70)
    print("PHASE 2: CYCLE MODE SEARCH")
    print("="*70)
    
    grid_cycle = base_grid.copy()
    
    # Wrapper for model constructor with n_inp
    def create_cycle_wrapper(params, n_layers):
        return create_model_cycle(params, n_layers, n_inp)
    
    best_cycle_config, cycle_results = connection_mode_search(
        train_data=(X_train, y_train),
        valid_data=(X_valid, y_valid),
        device=device,
        model_constructor=create_cycle_wrapper,
        mode='Cycle',
        n_configs=args.n_configs,
        n_layers=args.n_layers,
        base_grid=grid_cycle,
        is_loader=False,
        verbose=True
    )
    
    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    best_anti_acc = max([r['valid_acc'] for r in anti_results])
    best_cycle_acc = max([r['valid_acc'] for r in cycle_results])
    mean_anti_acc = np.mean([r['valid_acc'] for r in anti_results])
    mean_cycle_acc = np.mean([r['valid_acc'] for r in cycle_results])
    
    print(f"\nAntisymmetric Mode:")
    print(f"  Best: {best_anti_acc:.4f} ({best_anti_acc*100:.2f}%)")
    print(f"  Mean: {mean_anti_acc:.4f} ({mean_anti_acc*100:.2f}%)")
    print(f"  Std:  {np.std([r['valid_acc'] for r in anti_results]):.4f}")
    
    print(f"\nCycle Mode:")
    print(f"  Best: {best_cycle_acc:.4f} ({best_cycle_acc*100:.2f}%)")
    print(f"  Mean: {mean_cycle_acc:.4f} ({mean_cycle_acc*100:.2f}%)")
    print(f"  Std:  {np.std([r['valid_acc'] for r in cycle_results]):.4f}")
    
    print(f"\nDifference (Antisymmetric - Cycle):")
    print(f"  Best: {(best_anti_acc - best_cycle_acc)*100:.2f}%")
    print(f"  Mean: {(mean_anti_acc - mean_cycle_acc)*100:.2f}%")
    
    if best_anti_acc > best_cycle_acc:
        print(f"\n🏆 Winner: ANTISYMMETRIC by {(best_anti_acc - best_cycle_acc)*100:.2f}%")
    elif best_cycle_acc > best_anti_acc:
        print(f"\n🏆 Winner: CYCLE by {(best_cycle_acc - best_anti_acc)*100:.2f}%")
    else:
        print(f"\n🤝 Tie!")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    results_data = {
        'n_layers': args.n_layers,
        'dataset': 'speech',
        'n_classes': len(class_names),
        'class_names': class_names,
        'antisymmetric': {
            'best_config': best_anti_config,
            'best_accuracy': best_anti_acc,
            'mean_accuracy': mean_anti_acc,
            'all_results': anti_results
        },
        'cycle': {
            'best_config': best_cycle_config,
            'best_accuracy': best_cycle_acc,
            'mean_accuracy': mean_cycle_acc,
            'all_results': cycle_results
        }
    }
    
    with open(f"{args.save_dir}/comparison_results_{args.n_layers}layers.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(
        anti_results, cycle_results,
        mode1_name="Antisymmetric",
        mode2_name="Cycle",
        n_layers=args.n_layers,
        save_dir=args.save_dir,
        dataset_name="Speech Commands"
    )
    
    # Build final models
    print("\n" + "="*70)
    print("BUILDING BEST MODELS FOR VISUALIZATION")
    print("="*70)
    
    print("\nAntisymmetric model:")
    anti_model_dict = build_final_model(
        best_anti_config, create_model_antisymmetric,
        args.n_layers, n_inp, (X_train, y_train), device
    )
    
    print("\nCycle model:")
    cycle_model_dict = build_final_model(
        best_cycle_config, create_model_cycle,
        args.n_layers, n_inp, (X_train, y_train), device
    )
    
    models_dict = {
        'antisymmetric': anti_model_dict,
        'cycle': cycle_model_dict
    }
    
    # Test accuracies
    print("\nTest Set Performance:")
    for mode_name, model_dict in [('Antisymmetric', anti_model_dict),
                                    ('Cycle', cycle_model_dict)]:
        test_acc = test_model(
            model_dict['model'], (X_test, y_test),
            model_dict['classifier'], model_dict['scaler'],
            device, is_loader=False
        )
        model_dict['test_acc'] = test_acc
        print(f"  {mode_name}: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\nPrediction examples...")
    plot_predictions(
        models_dict, X_test, y_test,
        class_names=class_names,
        device=device, save_dir=args.save_dir,
        n_examples=6,
        mode1_name="Antisymmetric",
        mode2_name="Cycle"
    )
    
    print("State dynamics...")
    plot_state_dynamics(
        models_dict, X_test, y_test,
        device=device, save_dir=args.save_dir,
        n_samples=100,
        mode1_name="Antisymmetric",
        mode2_name="Cycle"
    )
    
    print("Eigenspectrum analysis...")
    # Get sample input for Jacobian computation
    sample_input = torch.tensor(X_test[:1], dtype=torch.float32).to(device)
    plot_eigenspectrum_analysis(
        models_dict, sample_input,
        device=device, save_dir=args.save_dir,
        mode1_name="Antisymmetric",
        mode2_name="Cycle"
    )
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
