#!/usr/bin/env python3
"""
Test antisymmetric coupling on sMNIST dataset
Following the same data processing as experiments/smnist.py
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acds.archetypes.esn import DeepReservoir
from acds.benchmarks.mnist import get_mnist_data
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def test(data_loader, model, classifier, scaler, device):
    """Test function following the original smnist.py pattern"""
    activations, ys = [], []
    for images, labels in tqdm(data_loader, desc="Testing"):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        with torch.no_grad():
            output = model(images)[0]  # Get states, take final timestep
            if len(output.shape) == 3:  # (batch, seq, features)
                output = output[:, -1, :]  # Take final timestep
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

def plot_training_comparison(results, save_dir="./plots"):
    """Create comparison plots between standard and antisymmetric ESN"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Accuracy comparison bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    modes = list(results.keys())
    metrics = ['train_acc', 'valid_acc', 'test_acc']
    metric_labels = ['Train', 'Validation', 'Test']
    
    # Bar plot for accuracies
    x = np.arange(len(metric_labels))
    width = 0.35
    
    for i, mode in enumerate(modes):
        values = [results[mode][m] * 100 for m in metrics]
        axes[0].bar(x + i*width, values, width, label=mode, alpha=0.8)
    
    axes[0].set_xlabel('Dataset Split', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(metric_labels)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Training time comparison
    times = [results[mode]['train_time'] for mode in modes]
    axes[1].bar(modes, times, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mode, time_val) in enumerate(zip(modes, times)):
        axes[1].text(i, time_val + 0.5, f'{time_val:.2f}s', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_time_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/accuracy_time_comparison.png")
    plt.close()

def plot_activation_analysis(all_activations, all_labels, results, save_dir="./plots"):
    """Plot activation analysis comparing standard and antisymmetric ESN"""
    os.makedirs(save_dir, exist_ok=True)
    
    modes = list(all_activations.keys())
    
    # 2. PCA visualization of activations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, mode in enumerate(modes):
        activations = all_activations[mode]
        labels = all_labels[mode]
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        activations_2d = pca.fit_transform(activations[:1000])  # Use first 1000 samples
        
        # Plot with different colors for each digit
        scatter = axes[idx].scatter(activations_2d[:, 0], activations_2d[:, 1], 
                                   c=labels[:1000], cmap='tab10', alpha=0.6, s=20)
        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11)
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11)
        axes[idx].set_title(f'{mode} ESN - PCA of Activations', fontsize=13, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[idx])
        cbar.set_label('Digit Class', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/activation_pca.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/activation_pca.png")
    plt.close()
    
    # 3. Activation statistics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, mode in enumerate(modes):
        activations = all_activations[mode]
        
        # Activation magnitude distribution
        ax = axes[0, idx]
        activation_norms = np.linalg.norm(activations, axis=1)
        ax.hist(activation_norms, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Activation Magnitude', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{mode} - Activation Magnitude Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Mean activation per feature
        ax = axes[1, idx]
        mean_activations = np.mean(activations, axis=0)
        ax.plot(mean_activations, alpha=0.7, linewidth=1)
        ax.set_xlabel('Feature Index', fontsize=11)
        ax.set_ylabel('Mean Activation', fontsize=11)
        ax.set_title(f'{mode} - Mean Activation per Feature', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/activation_statistics.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/activation_statistics.png")
    plt.close()

def plot_improvement_breakdown(results, save_dir="./plots"):
    """Plot detailed improvement breakdown"""
    os.makedirs(save_dir, exist_ok=True)
    
    modes = list(results.keys())
    if len(modes) != 2:
        return
    
    std_result = results[modes[0]]
    anti_result = results[modes[1]]
    
    # Calculate improvements
    metrics = ['train_acc', 'valid_acc', 'test_acc']
    metric_labels = ['Train', 'Validation', 'Test']
    improvements = []
    
    for metric in metrics:
        std_val = std_result[metric]
        anti_val = anti_result[metric]
        improvement = (anti_val - std_val) * 100  # Convert to percentage points
        improvements.append(improvement)
    
    # Create improvement plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.bar(metric_labels, improvements, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=12, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement (percentage points)', fontsize=12)
    ax.set_title('Antisymmetric ESN Improvement over Standard ESN', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/improvement_breakdown.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/improvement_breakdown.png")
    plt.close()

def hyperparameter_search(train_loader, valid_loader, device, antisymmetric=False, n_configs=10):
    """Perform hyperparameter search for ESN"""
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SEARCH ({'Antisymmetric' if antisymmetric else 'Standard'} ESN)")
    print(f"{'='*60}")
    
    # Define hyperparameter grid
    param_grid = {
        'spectral_radius': [0.9, 0.95, 0.99, 0.999],
        'leaky': [0.001, 0.01, 0.1, 0.5],
        'input_scaling': [0.1, 0.5, 1.0, 2.0],
        'tot_units': [100],
        'epsilon': [0.05, 0.1, 0.2, 0.4] if antisymmetric else [0.0],
    }
    
    # Create parameter combinations
    all_params = list(ParameterGrid(param_grid))
    
    # Randomly sample n_configs if there are too many
    if len(all_params) > n_configs:
        import random
        random.seed(42)
        sampled_params = random.sample(all_params, n_configs)
    else:
        sampled_params = all_params
    
    print(f"Testing {len(sampled_params)} hyperparameter configurations...")
    
    best_config = None
    best_valid_acc = 0.0
    all_results = []
    
    for idx, params in enumerate(sampled_params):
        print(f"\n[{idx+1}/{len(sampled_params)}] Testing config: {params}")
        
        try:
            # Create ESN with current hyperparameters
            unit_per_layer = params['tot_units']
            model = DeepReservoir(
                input_size=1,
                tot_units=params['tot_units'],
                n_layers=10,
                concat=True,
                spectral_radius=params['spectral_radius'],
                input_scaling=params['input_scaling'],
                leaky=params['leaky'],
                connectivity_recurrent=unit_per_layer,
                connectivity_input=unit_per_layer,
                connectivity_inter=unit_per_layer,
                antisymmetric=antisymmetric,
                epsilon=params.get('epsilon', 0.0),
                cycle=False,
                linear=False,
            ).to(device)
            
            # Generate training activations
            activations, ys = [], []
            start_time = time.time()
            
            for images, labels in train_loader:
                images = images.to(device)
                images = images.view(images.shape[0], -1).unsqueeze(-1)
                with torch.no_grad():
                    output = model(images)[0]
                    if len(output.shape) == 3:
                        output = output[:, -1, :]
                activations.append(output.cpu())
                ys.append(labels)
            
            train_time = time.time() - start_time
            
            activations = torch.cat(activations, dim=0).numpy()
            ys = torch.cat(ys, dim=0).squeeze().numpy()
            
            # Train classifier
            scaler = preprocessing.StandardScaler().fit(activations)
            activations_scaled = scaler.transform(activations)
            classifier = LogisticRegression(max_iter=1000, verbose=0).fit(activations_scaled, ys)
            
            # Evaluate on validation set
            valid_acc = test(valid_loader, model, classifier, scaler, device)
            
            result = {
                'params': params,
                'valid_acc': valid_acc,
                'train_time': train_time
            }
            all_results.append(result)
            
            print(f"  Valid Accuracy: {valid_acc:.4f} | Time: {train_time:.2f}s")
            
            # Track best configuration
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_config = params.copy()
                print(f"  ✨ New best config! Valid Acc: {best_valid_acc:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error with config: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_valid_acc:.4f}")
    print(f"Best Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # Sort results by validation accuracy
    all_results.sort(key=lambda x: x['valid_acc'], reverse=True)
    
    print(f"\nTop 5 Configurations:")
    for i, result in enumerate(all_results[:5], 1):
        print(f"{i}. Valid Acc: {result['valid_acc']:.4f} | Params: {result['params']}")
    
    return best_config, all_results

def plot_hyperparameter_results(results, antisymmetric, save_dir="./plots"):
    """Plot hyperparameter search results"""
    os.makedirs(save_dir, exist_ok=True)
    
    mode_name = "Antisymmetric" if antisymmetric else "Standard"
    
    # Extract data
    valid_accs = [r['valid_acc'] for r in results]
    train_times = [r['train_time'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Validation accuracy distribution
    axes[0, 0].hist(valid_accs, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(np.mean(valid_accs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_accs):.4f}')
    axes[0, 0].set_xlabel('Validation Accuracy', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title(f'{mode_name} ESN - Valid Acc Distribution', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Accuracy vs Training Time
    axes[0, 1].scatter(train_times, valid_accs, alpha=0.6, s=50, c=valid_accs, 
                       cmap='viridis', edgecolors='black')
    axes[0, 1].set_xlabel('Training Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('Validation Accuracy', fontsize=11)
    axes[0, 1].set_title(f'{mode_name} ESN - Accuracy vs Time', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Spectral radius effect
    spectral_radii = [r['params']['spectral_radius'] for r in results]
    axes[1, 0].scatter(spectral_radii, valid_accs, alpha=0.6, s=50, 
                       c=valid_accs, cmap='viridis', edgecolors='black')
    axes[1, 0].set_xlabel('Spectral Radius', fontsize=11)
    axes[1, 0].set_ylabel('Validation Accuracy', fontsize=11)
    axes[1, 0].set_title(f'{mode_name} ESN - Effect of Spectral Radius', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Leaky rate effect
    leaky_rates = [r['params']['leaky'] for r in results]
    axes[1, 1].scatter(leaky_rates, valid_accs, alpha=0.6, s=50, 
                       c=valid_accs, cmap='viridis', edgecolors='black')
    axes[1, 1].set_xlabel('Leaky Rate', fontsize=11)
    axes[1, 1].set_ylabel('Validation Accuracy', fontsize=11)
    axes[1, 1].set_title(f'{mode_name} ESN - Effect of Leaky Rate', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_dir}/hyperparam_search_{mode_name.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def test_smnist_antisymmetric(with_hyperparam_search=False, n_search_configs=10):
    """Test antisymmetric coupling on sMNIST dataset"""
    print("Testing Antisymmetric Coupling on sMNIST...")
    print("Following the same data processing as experiments/smnist.py")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create plots directory
    save_dir = "./plots/antisymmetric_comparison"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Plots will be saved to: {save_dir}")
    
    # Load sMNIST data using the same approach as the original experiment
    print("Loading sMNIST data...")
    data_root = "./data"
    
    # Use smaller batch size for faster processing during testing
    batch_size = 1000  # Original uses 1000, we use smaller for faster testing
    train_loader, valid_loader, test_loader = get_mnist_data(data_root, batch_size, batch_size)
    
    print(f"Data loaded with batch size: {batch_size}")
    
    # Perform hyperparameter search if requested
    best_configs = {}
    if with_hyperparam_search:
        print("\n" + "="*70)
        print("PHASE 1: HYPERPARAMETER SEARCH")
        print("="*70)
        
        # Search for standard ESN
        best_std_config, std_results = hyperparameter_search(
            train_loader, valid_loader, device, 
            antisymmetric=False, n_configs=n_search_configs
        )
        best_configs['Standard'] = best_std_config
        plot_hyperparameter_results(std_results, False, save_dir)
        
        # Search for antisymmetric ESN
        best_anti_config, anti_results = hyperparameter_search(
            train_loader, valid_loader, device, 
            antisymmetric=True, n_configs=n_search_configs
        )
        best_configs['Antisymmetric'] = best_anti_config
        plot_hyperparameter_results(anti_results, True, save_dir)
        
        # Save best configs to file
        with open(f"{save_dir}/best_configs.json", 'w') as f:
            json.dump(best_configs, f, indent=2)
        print(f"\nBest configurations saved to: {save_dir}/best_configs.json")
        
        print("\n" + "="*70)
        print("PHASE 2: FULL EVALUATION WITH BEST CONFIGS")
        print("="*70)
    else:
        # Use default configuration
        best_configs = {
            'Standard': {
                'spectral_radius': 0.999,
                'input_scaling': 1.0,
                'leaky': 0.001,
                'tot_units': 500,
                'epsilon': 0.0
            },
            'Antisymmetric': {
                'spectral_radius': 0.999,
                'input_scaling': 1.0,
                'leaky': 0.001,
                'tot_units': 500,
                'epsilon': 0.4
            }
        }
    
    # Test both standard and antisymmetric ESN
    results = {}
    all_activations = {}  # Store activations for visualization
    all_labels = {}  # Store labels for visualization
    
    for mode, antisymmetric in [("Standard", False), ("Antisymmetric", True)]:
        print(f"\n{'='*50}")
        print(f"Testing {mode} ESN")
        print(f"{'='*50}")
        
        # Get configuration for this mode
        config = best_configs[mode]
        tot_units = config['tot_units']
        unit_per_layer = tot_units
        
        print(f"Using configuration: {config}")
        
        # Create ESN with best hyperparameters
        model = DeepReservoir(
            input_size=1,
            tot_units=tot_units,
            n_layers=1,
            concat=True,
            spectral_radius=config['spectral_radius'],
            input_scaling=config['input_scaling'],
            leaky=config['leaky'],
            connectivity_recurrent=unit_per_layer,
            connectivity_input=unit_per_layer,
            connectivity_inter=unit_per_layer,
            antisymmetric=antisymmetric,
            epsilon=config['epsilon'],
            cycle=False,
            linear=False,
        ).to(device)
        
        print(f"ESN layers units: {[layer.net.units for layer in model.reservoir]}")
        
        # Generate training activations (following original smnist.py pattern)
        print("Generating training activations...")
        activations, ys = [], []
        start_time = time.time()
        
        for images, labels in tqdm(train_loader, desc="Training data"):
            images = images.to(device)
            images = images.view(images.shape[0], -1).unsqueeze(-1)
            with torch.no_grad():
                output = model(images)[0]  # Get states
                if len(output.shape) == 3:  # (batch, seq, features)
                    output = output[:, -1, :]  # Take final timestep
            activations.append(output.cpu())
            ys.append(labels)
        
        train_time = time.time() - start_time
        
        # Concatenate and process activations (following original pattern)
        activations = torch.cat(activations, dim=0).numpy()
        ys = torch.cat(ys, dim=0).squeeze().numpy()
        
        print(f"Training activations shape: {activations.shape}")
        print(f"Training labels shape: {ys.shape}")
        print(f"Training time: {train_time:.2f}s")
        
        # Store activations and labels for visualization
        all_activations[mode] = activations.copy()
        all_labels[mode] = ys.copy()
        
        # Standardize activations and train classifier (following original pattern)
        print("Standardizing activations and training classifier...")
        scaler = preprocessing.StandardScaler().fit(activations)
        activations_scaled = scaler.transform(activations)
        classifier = LogisticRegression(max_iter=1000).fit(activations_scaled, ys)
        
        # Evaluate on train, validation, and test sets
        print("Evaluating...")
        train_acc = test(train_loader, model, classifier, scaler, device)
        valid_acc = test(valid_loader, model, classifier, scaler, device)
        test_acc = test(test_loader, model, classifier, scaler, device)
        
        results[mode] = {
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'test_acc': test_acc,
            'train_time': train_time
        }
        
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Valid accuracy: {valid_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for mode, result in results.items():
        print(f"{mode} ESN:")
        print(f"  Train Accuracy: {result['train_acc']:.4f}")
        print(f"  Valid Accuracy: {result['valid_acc']:.4f}")
        print(f"  Test Accuracy:  {result['test_acc']:.4f}")
        print(f"  Train Time:     {result['train_time']:.2f}s")
        print()
    
    # Calculate improvement
    if len(results) == 2:
        modes = list(results.keys())
        std_result = results[modes[0]]
        anti_result = results[modes[1]]
        
        for metric in ['train_acc', 'valid_acc', 'test_acc']:
            std_val = std_result[metric]
            anti_val = anti_result[metric]
            improvement = anti_val - std_val
            improvement_pct = improvement / std_val * 100 if std_val > 0 else 0
            print(f"{metric.replace('_', ' ').title()} improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    # Generate comparison plots
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*60}")
    
    plot_training_comparison(results, save_dir)
    plot_activation_analysis(all_activations, all_labels, results, save_dir)
    plot_improvement_breakdown(results, save_dir)
    
    print(f"\n✅ All plots saved to: {save_dir}")
    print("✅ sMNIST antisymmetric coupling test completed!")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test antisymmetric ESN on sMNIST')
    parser.add_argument('--hypersearch', action='store_true', 
                       help='Perform hyperparameter search before evaluation')
    parser.add_argument('--n_configs', type=int, default=15,
                       help='Number of configurations to test in hyperparameter search')
    
    args = parser.parse_args()
    
    test_smnist_antisymmetric(
        with_hyperparam_search=args.hypersearch,
        n_search_configs=args.n_configs
    )
