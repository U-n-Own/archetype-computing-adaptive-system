#!/usr/bin/env python3
"""
Compare Antisymmetric (multi-layer) vs Simple (single-layer) ESN
Search both modes and visualize differences
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

def connection_mode_search(train_loader, valid_loader, device, mode='antisymmetric', 
                          n_configs=15, n_layers=5):
    """
    Hyperparameter search for specific connection mode
    mode: 'antisymmetric' (multi-layer with antisymmetric connections) or 'simple' (single-layer ESN)
    """
    mode_name = "ANTISYMMETRIC (Multi-layer)" if mode == 'antisymmetric' else "SIMPLE (Single-layer)"
    print(f"\n{'='*70}")
    print(f"{mode_name} ESN HYPERPARAMETER SEARCH")
    print(f"Layers: {n_layers} | Configurations to test: {n_configs}")
    print(f"{'='*70}\n")
    
    # Define hyperparameter grid
    base_grid = {
        'spectral_radius': [0.9, 0.95, 0.99, 0.999],
        'leaky': [0.001, 0.01, 0.1, 0.5],
        'input_scaling': [0.5, 1.0, 2.0, 5.0],
        'tot_units': [1000],
    }
    
    if mode == 'antisymmetric':
        base_grid['epsilon'] = [0.05, 0.1, 0.2, 0.4]  # Antisymmetric coupling strength
    else:  # simple mode (no epsilon needed)
        base_grid['epsilon'] = [0.0]  # No epsilon for simple single-layer ESN
    
    # Create parameter combinations
    all_params = list(ParameterGrid(base_grid))
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
        print(f"\n[{idx+1}/{len(sampled_params)}] {mode_name} Configuration:")
        for k, v in params.items():
            if k != 'epsilon' or mode == 'antisymmetric':  # Only show epsilon for antisymmetric
                print(f"  {k:20s}: {v}")
        
        try:
            # Create ESN with specified connection mode
            unit_per_layer = params['tot_units']
            
            print(f"  Creating model...", end='', flush=True)
            
            if mode == 'antisymmetric':
                # Multi-layer ESN with antisymmetric connections
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
            else:  # simple mode - single layer ESN without special connections
                model = DeepReservoir(
                    input_size=1,
                    tot_units=params['tot_units'],
                    n_layers=1,  # Single layer only
                    concat=True,
                    spectral_radius=params['spectral_radius'],
                    input_scaling=params['input_scaling'],
                    leaky=params['leaky'],
                    connectivity_recurrent=unit_per_layer,
                    connectivity_input=unit_per_layer,
                    connectivity_inter=unit_per_layer,
                    antisymmetric=False,
                    cycle=False,  # No cycle mode
                    linear=False,
                ).to(device)
            
            print(" ✓", flush=True)
            
            # Generate training activations (use subset for speed)
            print(f"  Processing training data...", end='', flush=True)
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
            
            train_time = time.time() - start_time
            
            activations = torch.cat(activations, dim=0).numpy()
            ys = torch.cat(ys, dim=0).squeeze().numpy()
            print(f" ✓ ({activations.shape[0]} samples, {train_time:.2f}s)", flush=True)
            
            # Train classifier
            print(f"  Training classifier...", end='', flush=True)
            scaler = preprocessing.StandardScaler().fit(activations)
            activations_scaled = scaler.transform(activations)
            classifier = LogisticRegression(max_iter=1000, verbose=0).fit(activations_scaled, ys)
            print(" ✓", flush=True)
            
            # Evaluate on validation set
            print(f"  Evaluating...", end='', flush=True)
            valid_acc = test(valid_loader, model, classifier, scaler, device)
            print(f" ✓", flush=True)
            
            result = {
                'params': params,
                'valid_acc': valid_acc,
                'train_time': train_time,
                'mode': mode
            }
            all_results.append(result)
            
            print(f"  ➤ Validation Accuracy: {valid_acc:.4f} ({valid_acc*100:.2f}%)")
            
            # Track best configuration
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_config = params.copy()
                print(f"  ✨ NEW BEST for {mode_name}! ✨")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"{mode_name} SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Tested: {len(all_results)} configurations")
    print(f"Best validation accuracy: {best_valid_acc:.4f} ({best_valid_acc*100:.2f}%)")
    print(f"\nBest configuration:")
    for key, value in best_config.items():
        if key != 'epsilon' or mode == 'antisymmetric':
            print(f"  {key:20s}: {value}")
    
    # Sort results
    all_results.sort(key=lambda x: x['valid_acc'], reverse=True)
    
    print(f"\nTop 3 Configurations:")
    for i, result in enumerate(all_results[:3], 1):
        print(f"{i}. Valid Acc: {result['valid_acc']:.4f} | Time: {result['train_time']:.2f}s")
    
    return best_config, all_results

def plot_comparison(antisymmetric_results, simple_results, n_layers, save_dir="./plots/antisym_vs_simple"):
    """Plot comprehensive comparison between antisymmetric and simple (single-layer) modes"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for both modes
    anti_accs = [r['valid_acc'] for r in antisymmetric_results]
    simple_accs = [r['valid_acc'] for r in simple_results]
    
    anti_times = [r['train_time'] for r in antisymmetric_results]
    simple_times = [r['train_time'] for r in simple_results]
    
    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy Distribution Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(anti_accs, bins=15, alpha=0.6, label='Antisymmetric (Multi-layer)', color='coral', edgecolor='black')
    ax1.hist(simple_accs, bins=15, alpha=0.6, label='Simple (Single-layer)', color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(anti_accs), color='red', linestyle='--', linewidth=2, 
                label=f'Anti Mean: {np.mean(anti_accs):.4f}')
    ax1.axvline(np.mean(simple_accs), color='blue', linestyle='--', linewidth=2,
                label=f'Simple Mean: {np.mean(simple_accs):.4f}')
    ax1.set_xlabel('Validation Accuracy', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Accuracy Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Box Plot Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([anti_accs, simple_accs], labels=['Antisymmetric', 'Simple'],
                      patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('coral')
    bp['boxes'][1].set_facecolor('skyblue')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    ax2.set_ylabel('Validation Accuracy', fontsize=11)
    ax2.set_title('Accuracy Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Best Accuracy Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    best_anti = max(anti_accs)
    best_simple = max(simple_accs)
    mean_anti = np.mean(anti_accs)
    mean_simple = np.mean(simple_accs)
    
    x = np.arange(2)
    width = 0.35
    bars1 = ax3.bar(x - width/2, [best_anti, best_simple], width, 
                    label='Best', color=['coral', 'skyblue'], edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, [mean_anti, mean_simple], width,
                    label='Mean', color=['lightcoral', 'lightskyblue'], 
                    edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax3.set_ylabel('Validation Accuracy', fontsize=11)
    ax3.set_title('Best vs Mean Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Antisymmetric', 'Simple'])
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Training Time Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(anti_times, bins=15, alpha=0.6, label='Antisymmetric', color='coral', edgecolor='black')
    ax4.hist(simple_times, bins=15, alpha=0.6, label='Simple', color='skyblue', edgecolor='black')
    ax4.set_xlabel('Training Time (s)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Training Time Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Accuracy vs Time Scatter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(anti_times, anti_accs, alpha=0.6, s=80, label='Antisymmetric', 
                color='coral', edgecolors='black', linewidths=1)
    ax5.scatter(simple_times, simple_accs, alpha=0.6, s=80, label='Simple',
                color='skyblue', edgecolors='black', linewidths=1)
    ax5.set_xlabel('Training Time (s)', fontsize=11)
    ax5.set_ylabel('Validation Accuracy', fontsize=11)
    ax5.set_title('Accuracy vs Training Time', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Spectral Radius Effect Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    anti_sr = [r['params']['spectral_radius'] for r in antisymmetric_results]
    simple_sr = [r['params']['spectral_radius'] for r in simple_results]
    ax6.scatter(anti_sr, anti_accs, alpha=0.6, s=80, label='Antisymmetric',
                color='coral', edgecolors='black', linewidths=1)
    ax6.scatter(simple_sr, simple_accs, alpha=0.6, s=80, label='Simple',
                color='skyblue', edgecolors='black', linewidths=1)
    ax6.set_xlabel('Spectral Radius', fontsize=11)
    ax6.set_ylabel('Validation Accuracy', fontsize=11)
    ax6.set_title('Effect of Spectral Radius', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Leaky Rate Effect Comparison
    ax7 = fig.add_subplot(gs[2, 0])
    anti_leaky = [r['params']['leaky'] for r in antisymmetric_results]
    simple_leaky = [r['params']['leaky'] for r in simple_results]
    ax7.scatter(anti_leaky, anti_accs, alpha=0.6, s=80, label='Antisymmetric',
                color='coral', edgecolors='black', linewidths=1)
    ax7.scatter(simple_leaky, simple_accs, alpha=0.6, s=80, label='Simple',
                color='skyblue', edgecolors='black', linewidths=1)
    ax7.set_xlabel('Leaky Rate', fontsize=11)
    ax7.set_ylabel('Validation Accuracy', fontsize=11)
    ax7.set_title('Effect of Leaky Rate', fontsize=12, fontweight='bold')
    ax7.set_xscale('log')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # 8. Top Configurations Comparison
    ax8 = fig.add_subplot(gs[2, 1:])
    top_n = min(5, len(antisymmetric_results), len(simple_results))
    
    top_anti = sorted(antisymmetric_results, key=lambda x: x['valid_acc'], reverse=True)[:top_n]
    top_simple = sorted(simple_results, key=lambda x: x['valid_acc'], reverse=True)[:top_n]
    
    y_pos = np.arange(top_n * 2)
    accs = []
    labels = []
    colors = []
    
    for i in range(top_n):
        accs.append(top_anti[i]['valid_acc'])
        labels.append(f'Anti #{i+1}')
        colors.append('coral')
        accs.append(top_simple[i]['valid_acc'])
        labels.append(f'Simple #{i+1}')
        colors.append('skyblue')
    
    bars = ax8.barh(y_pos, accs, color=colors, edgecolor='black', linewidth=1.5)
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(labels)
    ax8.set_xlabel('Validation Accuracy', fontsize=11)
    ax8.set_title(f'Top {top_n} Configurations from Each Mode', fontsize=12, fontweight='bold')
    ax8.grid(alpha=0.3, axis='x')
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        width = bar.get_width()
        ax8.text(width, bar.get_y() + bar.get_height()/2,
                f' {acc:.4f}', ha='left', va='center', fontsize=9)
    
    # Overall title
    plt.suptitle(f'Antisymmetric vs Simple ESN Comparison ({n_layers} layers for antisymmetric)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    filename = f"{save_dir}/comparison_{n_layers}layers.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {filename}")
    plt.close()
    
    # Create summary statistics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stats = {
        'Mean Accuracy': [np.mean(anti_accs), np.mean(simple_accs)],
        'Best Accuracy': [max(anti_accs), max(simple_accs)],
        'Std Dev': [np.std(anti_accs), np.std(simple_accs)],
        'Mean Time (s)': [np.mean(anti_times), np.mean(simple_times)],
    }
    
    x = np.arange(len(stats))
    width = 0.35
    
    anti_vals = [stats[k][0] for k in stats.keys()]
    cycle_vals = [stats[k][1] for k in stats.keys()]
    
    # Normalize for visualization (except time)
    anti_vals_norm = anti_vals[:3] + [anti_vals[3] / 100]  # Scale time down
    cycle_vals_norm = cycle_vals[:3] + [cycle_vals[3] / 100]
    
    bars1 = ax.bar(x - width/2, anti_vals_norm, width, label='Antisymmetric',
                   color='coral', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, cycle_vals_norm, width, label='Cycle',
                   color='skyblue', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Summary Statistics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stats.keys(), rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add actual value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height(),
                f'{anti_vals[i]:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height(),
                f'{cycle_vals[i]:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = f"{save_dir}/summary_stats_{n_layers}layers.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Summary stats plot saved: {filename}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Antisymmetric vs Simple (Single-layer) ESN')
    parser.add_argument('--n_configs', type=int, default=15,
                       help='Number of configurations to test per mode')
    parser.add_argument('--n_layers', type=int, default=5,
                       help='Number of layers for antisymmetric ESN (simple always uses 1)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("ANTISYMMETRIC vs SIMPLE (SINGLE-LAYER) ESN COMPARISON")
    print(f"{'='*70}")
    print(f"Antisymmetric layers: {args.n_layers}")
    print(f"Simple layers: 1 (single-layer)")
    print(f"Configurations per mode: {args.n_configs}")
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
    
    # Search both modes
    print("\n" + "="*70)
    print("PHASE 1: ANTISYMMETRIC MODE SEARCH")
    print("="*70)
    best_anti_config, anti_results = connection_mode_search(
        train_loader, valid_loader, device,
        mode='antisymmetric',
        n_configs=args.n_configs,
        n_layers=args.n_layers
    )
    
    print("\n" + "="*70)
    print("PHASE 2: SIMPLE (SINGLE-LAYER) ESN SEARCH")
    print("="*70)
    best_simple_config, simple_results = connection_mode_search(
        train_loader, valid_loader, device,
        mode='simple',
        n_configs=args.n_configs,
        n_layers=1  # Simple mode always uses 1 layer
    )
    
    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    best_anti_acc = max([r['valid_acc'] for r in anti_results])
    best_simple_acc = max([r['valid_acc'] for r in simple_results])
    mean_anti_acc = np.mean([r['valid_acc'] for r in anti_results])
    mean_simple_acc = np.mean([r['valid_acc'] for r in simple_results])
    
    print(f"\nAntisymmetric Mode (Multi-layer):")
    print(f"  Layers:         {args.n_layers}")
    print(f"  Best Accuracy:  {best_anti_acc:.4f} ({best_anti_acc*100:.2f}%)")
    print(f"  Mean Accuracy:  {mean_anti_acc:.4f} ({mean_anti_acc*100:.2f}%)")
    print(f"  Std Dev:        {np.std([r['valid_acc'] for r in anti_results]):.4f}")
    
    print(f"\nSimple Mode (Single-layer):")
    print(f"  Layers:         1")
    print(f"  Best Accuracy:  {best_simple_acc:.4f} ({best_simple_acc*100:.2f}%)")
    print(f"  Mean Accuracy:  {mean_simple_acc:.4f} ({mean_simple_acc*100:.2f}%)")
    print(f"  Std Dev:        {np.std([r['valid_acc'] for r in simple_results]):.4f}")
    
    print(f"\nDifference (Antisymmetric - Simple):")
    print(f"  Best Accuracy:  {(best_anti_acc - best_simple_acc):.4f} ({(best_anti_acc - best_simple_acc)*100:.2f}%)")
    print(f"  Mean Accuracy:  {(mean_anti_acc - mean_simple_acc):.4f} ({(mean_anti_acc - mean_simple_acc)*100:.2f}%)")
    
    # Determine winner
    if best_anti_acc > best_simple_acc:
        print(f"\n🏆 Winner: ANTISYMMETRIC by {(best_anti_acc - best_simple_acc)*100:.2f}%")
    elif best_simple_acc > best_anti_acc:
        print(f"\n🏆 Winner: SIMPLE by {(best_simple_acc - best_anti_acc)*100:.2f}%")
    else:
        print(f"\n🤝 Tie!")
    
    # Save results
    save_dir = "./plots/antisym_vs_simple"
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    with open(f"{save_dir}/comparison_results_{args.n_layers}layers.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved: {save_dir}/comparison_results_{args.n_layers}layers.json")
    
    # Plot comparison
    plot_comparison(anti_results, simple_results, args.n_layers, save_dir)
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}\n")
