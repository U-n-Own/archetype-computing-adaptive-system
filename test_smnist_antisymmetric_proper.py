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
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

def test_smnist_antisymmetric():
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
    
    # ESN configuration with perfectly divisible units
    esn_config = {
        'input_size': 1,           # Each pixel is fed sequentially
        'tot_units': 500,          # Total units (reduced for faster testing)
        'n_layers': 1,             # 1 layer for baseline comparison
        'concat': True,            # Concatenate all layer outputs
        'spectral_radius': 0.999,
        'input_scaling': 1,
        'leaky': 0.001,

    }
    
    # Test both standard and antisymmetric ESN
    results = {}
    all_activations = {}  # Store activations for visualization
    all_labels = {}  # Store labels for visualization
    
    for mode, antisymmetric in [("Standard", False), ("Antisymmetric", True)]:
        print(f"\n{'='*50}")
        print(f"Testing {mode} ESN")
        print(f"{'='*50}")
        
        unit_per_layer = esn_config['tot_units'] // esn_config['n_layers']
        
        # Create ESN
        model = DeepReservoir(
            **esn_config,
            antisymmetric=antisymmetric,
            epsilon=0.4 if antisymmetric else 0.0,
            cycle=False,
            connectivity_recurrent=unit_per_layer,
            connectivity_input=unit_per_layer,
            connectivity_inter=unit_per_layer,
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
    test_smnist_antisymmetric()
