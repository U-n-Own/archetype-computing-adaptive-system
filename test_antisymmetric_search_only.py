#!/usr/bin/env python3
"""
Focused hyperparameter search for Antisymmetric ESN with multiple layers
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
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")

def test(data_loader, model, classifier, scaler, device, max_batches=None):
    """Test function with optional batch limit"""
    activations, ys = [], []
    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        with torch.no_grad():
            output = model(images)[0]
            if len(output.shape) == 3:
                output = output[:, -1, :]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

def antisymmetric_search(train_loader, valid_loader, device, n_configs=15, n_layers=5):
    """Hyperparameter search focused on antisymmetric ESN"""
    print(f"\n{'='*70}")
    print(f"ANTISYMMETRIC ESN HYPERPARAMETER SEARCH")
    print(f"Layers: {n_layers} | Configurations to test: {n_configs}")
    print(f"{'='*70}\n")
    
    # Define hyperparameter grid for antisymmetric ESN
    param_grid = {
        'spectral_radius': [0.9, 0.95, 0.99, 0.999],
        'leaky': [0.001, 0.01, 0.1, 0.5],
        'input_scaling': [0.5, 1.0, 2.0],
        'tot_units': [50],
        'epsilon': [0.05, 0.1, 0.2, 0.4],  # Antisymmetric coupling strength
    }
    
    # Create parameter combinations
    all_params = list(ParameterGrid(param_grid))
    print(f"Total possible combinations: {len(all_params)}")
    
    # Randomly sample n_configs
    if len(all_params) > n_configs:
        import random
        random.seed(42)
        sampled_params = random.sample(all_params, n_configs)
    else:
        sampled_params = all_params
    
    print(f"Testing {len(sampled_params)} configurations...\n")
    
    best_config = None
    best_valid_acc = 0.0
    all_results = []
    
    for idx, params in enumerate(sampled_params):
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(sampled_params)}] Configuration:")
        for k, v in params.items():
            print(f"  {k:20s}: {v}")
        print(f"{'='*70}")
        
        try:
            # Create antisymmetric ESN
            unit_per_layer = params['tot_units']
            
            print(f"Creating {n_layers}-layer antisymmetric ESN with {params['tot_units']} total units...", flush=True)
            model = DeepReservoir(
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
                epsilon=params['epsilon'],
                cycle=False,
                linear=False,
            ).to(device)
            print(f"  ✓ Model created successfully")
            
            # Generate training activations (use subset for speed)
            print(f"Generating training activations (first 10 batches)...", flush=True)
            activations, ys = [], []
            start_time = time.time()
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx >= 10:  # Use first 10 batches
                    break
                images = images.to(device)
                images = images.view(images.shape[0], -1).unsqueeze(-1)
                with torch.no_grad():
                    output = model(images)[0]
                    if len(output.shape) == 3:
                        output = output[:, -1, :]
                activations.append(output.cpu())
                ys.append(labels)
                
                if batch_idx == 0:
                    print(f"  Batch shape: {images.shape}, Output shape: {output.shape}")
            
            train_time = time.time() - start_time
            
            activations = torch.cat(activations, dim=0).numpy()
            ys = torch.cat(ys, dim=0).squeeze().numpy()
            print(f"  ✓ Generated {activations.shape[0]} samples in {train_time:.2f}s")
            
            # Train classifier
            print(f"Training logistic regression classifier...", flush=True)
            scaler = preprocessing.StandardScaler().fit(activations)
            activations_scaled = scaler.transform(activations)
            classifier = LogisticRegression(max_iter=1000, verbose=0).fit(activations_scaled, ys)
            print(f"  ✓ Classifier trained")
            
            # Evaluate on validation set
            print(f"Evaluating on validation set...", flush=True)
            valid_acc = test(valid_loader, model, classifier, scaler, device)
            
            result = {
                'params': params,
                'valid_acc': valid_acc,
                'train_time': train_time
            }
            all_results.append(result)
            
            print(f"\n{'='*70}")
            print(f"RESULT: Validation Accuracy = {valid_acc:.4f} ({valid_acc*100:.2f}%)")
            print(f"        Training Time = {train_time:.2f}s")
            print(f"{'='*70}")
            
            # Track best configuration
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_config = params.copy()
                print(f"✨ NEW BEST CONFIG! Valid Acc: {best_valid_acc:.4f} ✨")
            
        except Exception as e:
            print(f"❌ ERROR with config: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final Summary
    print(f"\n\n{'='*70}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total configurations tested: {len(all_results)}")
    print(f"Best validation accuracy: {best_valid_acc:.4f} ({best_valid_acc*100:.2f}%)")
    print(f"\nBest configuration:")
    for key, value in best_config.items():
        print(f"  {key:20s}: {value}")
    
    # Sort results by validation accuracy
    all_results.sort(key=lambda x: x['valid_acc'], reverse=True)
    
    print(f"\n{'='*70}")
    print("TOP 5 CONFIGURATIONS:")
    print(f"{'='*70}")
    for i, result in enumerate(all_results[:5], 1):
        print(f"\n{i}. Validation Accuracy: {result['valid_acc']:.4f} ({result['valid_acc']*100:.2f}%)")
        print(f"   Time: {result['train_time']:.2f}s")
        for k, v in result['params'].items():
            print(f"   {k:18s}: {v}")
    
    return best_config, all_results

def plot_search_results(results, n_layers, save_dir="./plots/antisymmetric_search"):
    """Plot comprehensive search results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    valid_accs = [r['valid_acc'] for r in results]
    train_times = [r['train_time'] for r in results]
    spectral_radii = [r['params']['spectral_radius'] for r in results]
    leaky_rates = [r['params']['leaky'] for r in results]
    epsilons = [r['params']['epsilon'] for r in results]
    tot_units = [r['params']['tot_units'] for r in results]
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(valid_accs, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(valid_accs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(valid_accs):.4f}')
    ax1.axvline(np.max(valid_accs), color='green', linestyle='--', 
                label=f'Best: {np.max(valid_accs):.4f}')
    ax1.set_xlabel('Validation Accuracy', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Accuracy Distribution ({n_layers} layers)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Accuracy vs Time
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(train_times, valid_accs, alpha=0.6, s=100, c=valid_accs, 
                         cmap='viridis', edgecolors='black', linewidths=1.5)
    ax2.set_xlabel('Training Time (s)', fontsize=11)
    ax2.set_ylabel('Validation Accuracy', fontsize=11)
    ax2.set_title('Accuracy vs Training Time', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Valid Acc')
    
    # 3. Spectral radius effect
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(spectral_radii, valid_accs, alpha=0.6, s=100, c=valid_accs, 
                cmap='viridis', edgecolors='black', linewidths=1.5)
    ax3.set_xlabel('Spectral Radius', fontsize=11)
    ax3.set_ylabel('Validation Accuracy', fontsize=11)
    ax3.set_title('Effect of Spectral Radius', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Leaky rate effect
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(leaky_rates, valid_accs, alpha=0.6, s=100, c=valid_accs, 
                cmap='viridis', edgecolors='black', linewidths=1.5)
    ax4.set_xlabel('Leaky Rate', fontsize=11)
    ax4.set_ylabel('Validation Accuracy', fontsize=11)
    ax4.set_title('Effect of Leaky Rate', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(alpha=0.3)
    
    # 5. Epsilon (antisymmetric coupling) effect
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(epsilons, valid_accs, alpha=0.6, s=100, c=valid_accs, 
                cmap='viridis', edgecolors='black', linewidths=1.5)
    ax5.set_xlabel('Epsilon (Antisymmetric Coupling)', fontsize=11)
    ax5.set_ylabel('Validation Accuracy', fontsize=11)
    ax5.set_title('Effect of Epsilon', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # 6. Total units effect
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(tot_units, valid_accs, alpha=0.6, s=100, c=valid_accs, 
                cmap='viridis', edgecolors='black', linewidths=1.5)
    ax6.set_xlabel('Total Units', fontsize=11)
    ax6.set_ylabel('Validation Accuracy', fontsize=11)
    ax6.set_title('Effect of Total Units', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)
    
    # 7. Top configurations comparison
    ax7 = fig.add_subplot(gs[2, :])
    top_n = min(10, len(results))
    top_results = sorted(results, key=lambda x: x['valid_acc'], reverse=True)[:top_n]
    top_accs = [r['valid_acc'] for r in top_results]
    config_labels = [f"Config {i+1}" for i in range(top_n)]
    
    bars = ax7.barh(config_labels, top_accs, color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)), 
                    edgecolor='black', linewidth=1.5)
    ax7.set_xlabel('Validation Accuracy', fontsize=12)
    ax7.set_title(f'Top {top_n} Configurations', fontsize=13, fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, top_accs)):
        width = bar.get_width()
        ax7.text(width, bar.get_y() + bar.get_height()/2, 
                f' {acc:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Antisymmetric ESN Hyperparameter Search Results ({n_layers} layers)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    filename = f"{save_dir}/search_results_{n_layers}layers.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {filename}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Antisymmetric ESN Hyperparameter Search')
    parser.add_argument('--n_configs', type=int, default=15,
                       help='Number of configurations to test')
    parser.add_argument('--n_layers', type=int, default=5,
                       help='Number of ESN layers')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("ANTISYMMETRIC ESN HYPERPARAMETER SEARCH")
    print(f"{'='*70}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Configurations to test: {args.n_configs}")
    print(f"{'='*70}\n")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading sMNIST data...")
    data_root = "./data"
    batch_size = 1000
    train_loader, valid_loader, test_loader = get_mnist_data(data_root, batch_size, batch_size)
    print(f"✓ Data loaded (batch size: {batch_size})\n")
    
    # Run hyperparameter search
    best_config, all_results = antisymmetric_search(
        train_loader, valid_loader, device,
        n_configs=args.n_configs,
        n_layers=args.n_layers
    )
    
    # Save results
    save_dir = "./plots/antisymmetric_search"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/best_config_{args.n_layers}layers.json", 'w') as f:
        json.dump({
            'best_config': best_config,
            'n_layers': args.n_layers,
            'all_results': all_results
        }, f, indent=2)
    print(f"\n✓ Results saved: {save_dir}/best_config_{args.n_layers}layers.json")
    
    # Plot results
    plot_search_results(all_results, args.n_layers, save_dir)
    
    print(f"\n{'='*70}")
    print("✅ SEARCH COMPLETE!")
    print(f"{'='*70}\n")
