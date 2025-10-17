#!/usr/bin/env python3
"""
Refactored ESN comparison script using modular utilities
Compare Antisymmetric vs Simple ESN on sMNIST
"""

import torch
import numpy as np
import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acds.archetypes.esn import DeepReservoir
from acds.benchmarks.mnist import get_mnist_data
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# Import utilities
from utils import (
    plot_comparison,
    plot_predictions,
    plot_state_dynamics,
    plot_eigenspectrum_analysis,
    connection_mode_search,
    test_model
)


def create_model_antisymmetric(params: dict, n_layers: int) -> DeepReservoir:
    """Create antisymmetric ESN model"""
    unit_per_layer = params['tot_units']
    return DeepReservoir(
        input_size=1,
        tot_units=params['tot_units'],
        n_layers=n_layers,
        concat=True,
        spectral_radius=params['spectral_radius'],
        input_scaling=params['input_scaling'],
        leaky=params['leaky'],
        connectivity_recurrent=unit_per_layer,
        connectivity_input=unit_per_layer,
        connectivity_inter=unit_per_layer,
        antisymmetric=True,
        epsilon=params.get('epsilon', 0.1),
        cycle=False,
        linear=False,
    )


def create_model_simple(params: dict, n_layers: int = 1) -> DeepReservoir:
    """Create simple single-layer ESN model"""
    unit_per_layer = params['tot_units']
    return DeepReservoir(
        input_size=1,
        tot_units=params['tot_units'],
        n_layers=1,  # Always single layer
        concat=True,
        spectral_radius=params['spectral_radius'],
        input_scaling=params['input_scaling'],
        leaky=params['leaky'],
        connectivity_recurrent=unit_per_layer,
        connectivity_input=unit_per_layer,
        connectivity_inter=unit_per_layer,
        antisymmetric=False,
        cycle=False,
        linear=False,
    )


def preprocess_mnist_batch(batch_data):
    """Preprocess MNIST batch for ESN"""
    images, labels = batch_data
    # Reshape to sequence: (batch, 784, 1)
    images = images.view(images.shape[0], -1, 1)
    return images, labels


class MNISTDataWrapper:
    """Wrapper to preprocess MNIST data on-the-fly"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def __iter__(self):
        for batch in self.dataloader:
            images, labels = batch
            images = images.view(images.shape[0], -1, 1)
            yield images, labels
    
    def __len__(self):
        return len(self.dataloader)


def build_final_model(params: dict, constructor, n_layers: int, 
                      train_loader, device: torch.device):
    """
    Build and train final model with best config
    
    Returns:
        Dict with model, classifier, scaler, test_acc
    """
    print(f"  Building model...", end='', flush=True)
    model = constructor(params, n_layers).to(device)
    print(" ✓", flush=True)
    
    print(f"  Training readout...", end='', flush=True)
    activations = []
    ys = []
    
    for images, labels in train_loader:
        images = images.to(device).view(images.shape[0], -1, 1)
        with torch.no_grad():
            output = model(images)
            if isinstance(output, tuple):
                states = output[0]
            else:
                states = output
            
            if len(states.shape) == 3:
                final_state = states[:, -1, :].cpu().numpy()
            else:
                final_state = states.cpu().numpy()
            
            activations.append(final_state)
            ys.append(labels.numpy())
    
    activations = np.concatenate(activations, axis=0)
    ys = np.concatenate(ys, axis=0)
    
    scaler = preprocessing.StandardScaler().fit(activations)
    activations_scaled = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations_scaled, ys)
    print(" ✓", flush=True)
    
    return {
        'model': model,
        'classifier': classifier,
        'scaler': scaler
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare Antisymmetric vs Simple ESN on sMNIST (Refactored)'
    )
    parser.add_argument('--n_configs', type=int, default=15,
                       help='Number of configurations to test per mode')
    parser.add_argument('--n_layers', type=int, default=5,
                       help='Number of layers for antisymmetric (simple always 1)')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for data')
    parser.add_argument('--save_dir', type=str, default='./plots/antisym_vs_simple_refactored',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    print(f"\n{'='*70}")
    print("ANTISYMMETRIC vs SIMPLE ESN COMPARISON (Refactored)")
    print(f"{'='*70}")
    print(f"Antisymmetric layers: {args.n_layers}")
    print(f"Simple layers: 1")
    print(f"Configurations per mode: {args.n_configs}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading sMNIST data...")
    batch_size = 1000
    train_loader, valid_loader, test_loader = get_mnist_data(
        args.data_root, batch_size, batch_size
    )
    
    # Wrap loaders
    train_wrapped = MNISTDataWrapper(train_loader)
    valid_wrapped = MNISTDataWrapper(valid_loader)
    test_wrapped = MNISTDataWrapper(test_loader)
    
    print(f"✓ Data loaded (batch size: {batch_size})\n")
    
    # Define hyperparameter grid
    base_grid = {
        'spectral_radius': [0.9, 0.95, 0.99, 0.999],
        'leaky': [0.001, 0.01, 0.1, 0.5],
        'input_scaling': [0.5, 1.0, 2.0, 5.0],
        'tot_units': [1000],
    }
    
    # Search antisymmetric mode
    print("\n" + "="*70)
    print("PHASE 1: ANTISYMMETRIC MODE SEARCH")
    print("="*70)
    
    grid_anti = base_grid.copy()
    grid_anti['epsilon'] = [0.05, 0.1, 0.2, 0.4]
    
    best_anti_config, anti_results = connection_mode_search(
        train_data=train_wrapped,
        valid_data=valid_wrapped,
        device=device,
        model_constructor=create_model_antisymmetric,
        mode='Antisymmetric',
        n_configs=args.n_configs,
        n_layers=args.n_layers,
        base_grid=grid_anti,
        is_loader=True,
        train_batches=10,  # Limit for speed
        verbose=True
    )
    
    # Search simple mode
    print("\n" + "="*70)
    print("PHASE 2: SIMPLE MODE SEARCH")
    print("="*70)
    
    grid_simple = base_grid.copy()
    
    best_simple_config, simple_results = connection_mode_search(
        train_data=train_wrapped,
        valid_data=valid_wrapped,
        device=device,
        model_constructor=create_model_simple,
        mode='Simple',
        n_configs=args.n_configs,
        n_layers=1,
        base_grid=grid_simple,
        is_loader=True,
        train_batches=10,
        verbose=True
    )
    
    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    best_anti_acc = max([r['valid_acc'] for r in anti_results])
    best_simple_acc = max([r['valid_acc'] for r in simple_results])
    mean_anti_acc = np.mean([r['valid_acc'] for r in anti_results])
    mean_simple_acc = np.mean([r['valid_acc'] for r in simple_results])
    
    print(f"\nAntisymmetric Mode:")
    print(f"  Best: {best_anti_acc:.4f} | Mean: {mean_anti_acc:.4f}")
    print(f"\nSimple Mode:")
    print(f"  Best: {best_simple_acc:.4f} | Mean: {mean_simple_acc:.4f}")
    print(f"\nDifference: {(best_anti_acc - best_simple_acc)*100:.2f}%")
    
    if best_anti_acc > best_simple_acc:
        print(f"\n🏆 Winner: ANTISYMMETRIC")
    elif best_simple_acc > best_anti_acc:
        print(f"\n🏆 Winner: SIMPLE")
    else:
        print(f"\n🤝 Tie!")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    results_data = {
        'antisymmetric_layers': args.n_layers,
        'simple_layers': 1,
        'antisymmetric': {
            'best_config': best_anti_config,
            'best_accuracy': best_anti_acc,
            'mean_accuracy': mean_anti_acc,
            'all_results': anti_results
        },
        'simple': {
            'best_config': best_simple_config,
            'best_accuracy': best_simple_acc,
            'mean_accuracy': mean_simple_acc,
            'all_results': simple_results
        }
    }
    
    with open(f"{args.save_dir}/comparison_results_{args.n_layers}layers.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(
        anti_results, simple_results,
        mode1_name="Antisymmetric",
        mode2_name="Simple",
        n_layers=args.n_layers,
        save_dir=args.save_dir,
        dataset_name="sMNIST"
    )
    
    # Build final models
    print("\n" + "="*70)
    print("BUILDING BEST MODELS FOR VISUALIZATION")
    print("="*70)
    
    print("\nAntisymmetric model:")
    anti_model_dict = build_final_model(
        best_anti_config, create_model_antisymmetric, 
        args.n_layers, train_loader, device
    )
    
    print("\nSimple model:")
    simple_model_dict = build_final_model(
        best_simple_config, create_model_simple,
        1, train_loader, device
    )
    
    models_dict = {
        'antisymmetric': anti_model_dict,
        'simple': simple_model_dict
    }
    
    # Test accuracies
    print("\nTest Set Performance:")
    for mode_name, model_dict in [('Antisymmetric', anti_model_dict), 
                                    ('Simple', simple_model_dict)]:
        test_acc = test_model(
            model_dict['model'], test_wrapped,
            model_dict['classifier'], model_dict['scaler'],
            device, is_loader=True
        )
        model_dict['test_acc'] = test_acc
        print(f"  {mode_name}: {test_acc:.4f}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Get sample for visualizations
    for images, labels in test_loader:
        # Images from loader are (batch, 1, 28, 28)
        # Need to reshape to (batch, 784, 1) for ESN input
        sample_images = images[:1].view(1, -1, 1)
        sample_labels = labels[:1]
        # Reshape all test batch for plotting
        X_test_batch = images.view(images.shape[0], -1, 1).numpy()
        y_test_batch = labels.numpy()
        break
    
    print("\nPrediction examples...")
    plot_predictions(
        models_dict, X_test_batch, y_test_batch,
        class_names=[str(i) for i in range(10)],
        device=device, save_dir=args.save_dir,
        n_examples=6,
        mode1_name="Antisymmetric",
        mode2_name="Simple"
    )
    
    print("State dynamics...")
    plot_state_dynamics(
        models_dict, X_test_batch, y_test_batch,
        device=device, save_dir=args.save_dir,
        n_samples=100,
        mode1_name="Antisymmetric",
        mode2_name="Simple"
    )
    
    print("Eigenspectrum analysis...")
    plot_eigenspectrum_analysis(
        models_dict, sample_images.to(device),
        device=device, save_dir=args.save_dir,
        mode1_name="Antisymmetric",
        mode2_name="Simple"
    )
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
