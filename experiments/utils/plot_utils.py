"""
Plotting utilities for ESN visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Optional
import torch

from .analysis_utils import (
    get_reservoir_matrices,
    compute_jacobian_eigenvalues
)

sns.set_style("whitegrid")


def plot_comparison(
    results1: List[Dict],
    results2: List[Dict],
    mode1_name: str,
    mode2_name: str,
    n_layers: int,
    save_dir: str,
    dataset_name: str = ""
):
    """
    Plot comprehensive comparison between two connection modes
    
    Args:
        results1: List of result dicts for mode 1
        results2: List of result dicts for mode 2
        mode1_name: Name of first mode (e.g., "Antisymmetric")
        mode2_name: Name of second mode (e.g., "Cycle")
        n_layers: Number of layers
        save_dir: Directory to save plots
        dataset_name: Optional dataset name for title
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    accs1 = [r['valid_acc'] for r in results1]
    accs2 = [r['valid_acc'] for r in results2]
    times1 = [r['train_time'] for r in results1]
    times2 = [r['train_time'] for r in results2]
    
    # Colors
    color1 = 'coral'
    color2 = 'skyblue'
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy Distribution Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(accs1, bins=15, alpha=0.6, label=mode1_name, color=color1, edgecolor='black')
    ax1.hist(accs2, bins=15, alpha=0.6, label=mode2_name, color=color2, edgecolor='black')
    ax1.axvline(np.mean(accs1), color='red', linestyle='--', linewidth=2,
                label=f'{mode1_name} Mean: {np.mean(accs1):.4f}')
    ax1.axvline(np.mean(accs2), color='blue', linestyle='--', linewidth=2,
                label=f'{mode2_name} Mean: {np.mean(accs2):.4f}')
    ax1.set_xlabel('Validation Accuracy', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Box Plot
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([accs1, accs2], labels=[mode1_name, mode2_name],
                      patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(color1)
    bp['boxes'][1].set_facecolor(color2)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    ax2.set_ylabel('Validation Accuracy', fontsize=11)
    ax2.set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Best vs Mean
    ax3 = fig.add_subplot(gs[0, 2])
    best1, best2 = max(accs1), max(accs2)
    mean1, mean2 = np.mean(accs1), np.mean(accs2)
    x = np.arange(2)
    width = 0.35
    bars1 = ax3.bar(x - width/2, [best1, best2], width, label='Best',
                    color=[color1, color2], edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, [mean1, mean2], width, label='Mean',
                    color=[color1, color2], edgecolor='black', linewidth=1.5, alpha=0.5)
    ax3.set_ylabel('Validation Accuracy', fontsize=11)
    ax3.set_title('Best vs Mean', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([mode1_name, mode2_name])
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Training Time
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(times1, bins=15, alpha=0.6, label=mode1_name, color=color1, edgecolor='black')
    ax4.hist(times2, bins=15, alpha=0.6, label=mode2_name, color=color2, edgecolor='black')
    ax4.set_xlabel('Training Time (s)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Training Time Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Accuracy vs Time
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(times1, accs1, alpha=0.6, s=80, label=mode1_name,
                color=color1, edgecolors='black', linewidths=1)
    ax5.scatter(times2, accs2, alpha=0.6, s=80, label=mode2_name,
                color=color2, edgecolors='black', linewidths=1)
    ax5.set_xlabel('Training Time (s)', fontsize=11)
    ax5.set_ylabel('Validation Accuracy', fontsize=11)
    ax5.set_title('Accuracy vs Training Time', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Spectral Radius Effect
    ax6 = fig.add_subplot(gs[1, 2])
    sr1 = [r['params']['spectral_radius'] for r in results1]
    sr2 = [r['params']['spectral_radius'] for r in results2]
    ax6.scatter(sr1, accs1, alpha=0.6, s=80, label=mode1_name,
                color=color1, edgecolors='black', linewidths=1)
    ax6.scatter(sr2, accs2, alpha=0.6, s=80, label=mode2_name,
                color=color2, edgecolors='black', linewidths=1)
    ax6.set_xlabel('Spectral Radius', fontsize=11)
    ax6.set_ylabel('Validation Accuracy', fontsize=11)
    ax6.set_title('Effect of Spectral Radius', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Leaky Rate Effect
    ax7 = fig.add_subplot(gs[2, 0])
    leaky1 = [r['params']['leaky'] for r in results1]
    leaky2 = [r['params']['leaky'] for r in results2]
    ax7.scatter(leaky1, accs1, alpha=0.6, s=80, label=mode1_name,
                color=color1, edgecolors='black', linewidths=1)
    ax7.scatter(leaky2, accs2, alpha=0.6, s=80, label=mode2_name,
                color=color2, edgecolors='black', linewidths=1)
    ax7.set_xlabel('Leaky Rate', fontsize=11)
    ax7.set_ylabel('Validation Accuracy', fontsize=11)
    ax7.set_title('Effect of Leaky Rate', fontsize=12, fontweight='bold')
    ax7.set_xscale('log')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # 8. Top Configurations
    ax8 = fig.add_subplot(gs[2, 1:])
    top_n = min(5, len(results1), len(results2))
    top1 = sorted(results1, key=lambda x: x['valid_acc'], reverse=True)[:top_n]
    top2 = sorted(results2, key=lambda x: x['valid_acc'], reverse=True)[:top_n]
    
    y_pos = np.arange(top_n * 2)
    accs_top = []
    labels_top = []
    colors_top = []
    
    for i in range(top_n):
        accs_top.append(top1[i]['valid_acc'])
        labels_top.append(f'{mode1_name} #{i+1}')
        colors_top.append(color1)
        accs_top.append(top2[i]['valid_acc'])
        labels_top.append(f'{mode2_name} #{i+1}')
        colors_top.append(color2)
    
    bars = ax8.barh(y_pos, accs_top, color=colors_top, edgecolor='black', linewidth=1.5)
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(labels_top)
    ax8.set_xlabel('Validation Accuracy', fontsize=11)
    ax8.set_title(f'Top {top_n} Configurations', fontsize=12, fontweight='bold')
    ax8.grid(alpha=0.3, axis='x')
    
    for bar, acc in zip(bars, accs_top):
        width = bar.get_width()
        ax8.text(width, bar.get_y() + bar.get_height()/2,
                f' {acc:.4f}', ha='left', va='center', fontsize=9)
    
    # Overall title
    title = f'{mode1_name} vs {mode2_name} Comparison'
    if dataset_name:
        title += f' - {dataset_name}'
    title += f' ({n_layers} layers)'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    filename = f"{save_dir}/comparison_{n_layers}layers.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved: {filename}")
    plt.close()


def plot_predictions(
    models_dict: Dict,
    X_test,
    y_test,
    class_names: Optional[List[str]],
    device: torch.device,
    save_dir: str,
    n_examples: int = 6,
    mode1_name: str = "Mode 1",
    mode2_name: str = "Mode 2"
):
    """
    Plot example predictions with reservoir states
    Works for both image and time series data
    
    Args:
        models_dict: Dict with keys for each mode containing model, classifier, scaler
        X_test: Test data (numpy array or tensor)
        y_test: Test labels
        class_names: List of class names (None for regression)
        device: torch device
        save_dir: Directory to save plots
        n_examples: Number of examples to show
        mode1_name: Name of first mode
        mode2_name: Name of second mode
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Select samples
    if class_names:
        # Classification: try to get diverse classes
        n_examples = min(n_examples, len(class_names), len(X_test))
        selected_indices = []
        selected_classes = np.random.choice(len(class_names), 
                                           min(n_examples, len(class_names)), 
                                           replace=False)
        for cls in selected_classes:
            cls_indices = np.where(y_test == cls)[0]
            if len(cls_indices) > 0:
                selected_indices.append(np.random.choice(cls_indices))
    else:
        # Regression: random samples
        selected_indices = np.random.choice(len(X_test), n_examples, replace=False)
    
    n_samples = len(selected_indices)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    mode_keys = list(models_dict.keys())
    
    for idx, sample_idx in enumerate(selected_indices):
        X_sample = X_test[sample_idx:sample_idx+1]
        y_true = y_test[sample_idx]
        
        # Convert to tensor
        if isinstance(X_sample, np.ndarray):
            X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
        else:
            X_tensor = X_sample.to(device)
        
        # Plot input
        ax = axes[idx, 0]
        if len(X_sample.shape) == 3:
            # Time series data: (1, time, features)
            ax.imshow(X_sample[0].T, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Feature Dim', fontsize=10)
        elif len(X_sample.shape) == 2:
            # Could be flattened image
            # Try to reshape to square
            total_pixels = X_sample.shape[1]
            side = int(np.sqrt(total_pixels))
            if side * side == total_pixels:
                ax.imshow(X_sample.reshape(side, side), cmap='gray')
                ax.axis('off')
            else:
                ax.plot(X_sample[0])
                ax.set_xlabel('Feature', fontsize=10)
        
        if class_names:
            true_label = class_names[int(y_true)]
            ax.set_title(f'True: "{true_label}"', fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'True: {y_true:.4f}', fontsize=11, fontweight='bold')
        
        # Process with both models
        for col, mode_key in enumerate(mode_keys[:2], start=1):
            model_info = models_dict[mode_key]
            model = model_info['model']
            classifier = model_info['classifier']
            scaler = model_info['scaler']
            mode_name = mode1_name if col == 1 else mode2_name
            
            with torch.no_grad():
                output = model(X_tensor)
                if isinstance(output, tuple):
                    states = output[0]
                else:
                    states = output
                
                # Get final state
                if len(states.shape) == 3:
                    final_state = states[:, -1, :].cpu().numpy()
                else:
                    final_state = states.cpu().numpy()
                
                # Predict
                final_state_scaled = scaler.transform(final_state)
                y_pred = classifier.predict(final_state_scaled)[0]
                
                if hasattr(classifier, 'predict_proba'):
                    y_proba = classifier.predict_proba(final_state_scaled)[0]
                    confidence = y_proba[int(y_pred)]
                else:
                    confidence = None
                
                # Plot reservoir states
                states_np = states.cpu().numpy()[0]
                
                # Downsample if needed
                if states_np.shape[0] > 100:
                    downsample_factor = states_np.shape[0] // 50
                    states_np = states_np[::downsample_factor, :]
                
                # Limit features shown
                if states_np.shape[1] > 100:
                    feature_indices = np.linspace(0, states_np.shape[1]-1, 100, dtype=int)
                    states_np = states_np[:, feature_indices]
                
                ax = axes[idx, col]
                im = ax.imshow(states_np.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                ax.set_xlabel('Time', fontsize=9)
                ax.set_ylabel('Units', fontsize=9)
                
                # Color code
                if class_names:
                    color = 'green' if y_pred == y_true else 'red'
                    pred_label = class_names[int(y_pred)]
                    if confidence:
                        title = f'{mode_name}\nPred: "{pred_label}" ({confidence:.2f})'
                    else:
                        title = f'{mode_name}\nPred: "{pred_label}"'
                else:
                    error = np.abs(y_pred - y_true)
                    color = 'green' if error < 0.1 else 'red'
                    title = f'{mode_name}\nPred: {y_pred:.4f}'
                
                ax.set_title(title, fontsize=10, fontweight='bold', color=color)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Prediction Examples: {mode1_name} vs {mode2_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_dir}/prediction_examples.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Prediction examples saved: {filename}")
    plt.close()


def plot_state_dynamics(
    models_dict: Dict,
    X_test,
    y_test,
    device: torch.device,
    save_dir: str,
    n_samples: int = 100,
    mode1_name: str = "Mode 1",
    mode2_name: str = "Mode 2"
):
    """
    Plot reservoir state dynamics and statistics
    
    Args:
        models_dict: Dict with model info for each mode
        X_test: Test data
        y_test: Test labels
        device: torch device
        save_dir: Directory to save plots
        n_samples: Number of samples to analyze
        mode1_name: Name of first mode
        mode2_name: Name of second mode
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples = min(n_samples, len(X_test))
    X_subset = X_test[:n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mode_keys = list(models_dict.keys())
    
    for row, mode_key in enumerate(mode_keys[:2]):
        model_info = models_dict[mode_key]
        model = model_info['model']
        mode_name = mode1_name if row == 0 else mode2_name
        color = 'coral' if row == 0 else 'skyblue'
        
        all_states = []
        all_final_states = []
        
        # Convert to tensor
        if isinstance(X_subset, np.ndarray):
            X_tensor = torch.tensor(X_subset, dtype=torch.float32).to(device)
        else:
            X_tensor = X_subset.to(device)
        
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(X_subset), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                output = model(batch_X)
                
                if isinstance(output, tuple):
                    states = output[0]
                else:
                    states = output
                
                all_states.append(states.cpu().numpy())
                
                if len(states.shape) == 3:
                    final_state = states[:, -1, :].cpu().numpy()
                else:
                    final_state = states.cpu().numpy()
                all_final_states.append(final_state)
        
        all_states = np.concatenate(all_states, axis=0)
        all_final_states = np.concatenate(all_final_states, axis=0)
        
        # 1. Mean activation over time
        mean_activation = np.mean(np.abs(all_states), axis=(0, 2))
        ax = axes[row, 0]
        ax.plot(mean_activation, linewidth=2, color=color)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Mean |Activation|', fontsize=11)
        ax.set_title(f'{mode_name} - Activation Evolution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 2. Final state distribution
        ax = axes[row, 1]
        ax.hist(all_final_states.flatten(), bins=50, alpha=0.7, edgecolor='black', color=color)
        ax.set_xlabel('Final State Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{mode_name} - Final State Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 3. State variance over time
        state_variance = np.var(all_states, axis=(0, 2))
        ax = axes[row, 2]
        ax.plot(state_variance, linewidth=2, color=color)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Variance', fontsize=11)
        ax.set_title(f'{mode_name} - State Variance Evolution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Reservoir State Dynamics: {mode1_name} vs {mode2_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_dir}/state_dynamics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ State dynamics saved: {filename}")
    plt.close()


def plot_eigenspectrum_analysis(
    models_dict: Dict,
    sample_input: torch.Tensor,
    device: torch.device,
    save_dir: str,
    mode1_name: str = "Mode 1",
    mode2_name: str = "Mode 2"
):
    """
    Plot eigenspectrum and Jacobian analysis
    
    Args:
        models_dict: Dict with model info for each mode
        sample_input: Sample input for Jacobian computation
        device: torch device
        save_dir: Directory to save plots
        mode1_name: Name of first mode
        mode2_name: Name of second mode
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    mode_keys = list(models_dict.keys())
    
    for row, mode_key in enumerate(mode_keys[:2]):
        model_info = models_dict[mode_key]
        model = model_info['model']
        mode_name = mode1_name if row == 0 else mode2_name
        color = 'coral' if row == 0 else 'skyblue'
        
        # 1. Recurrent matrix eigenspectrum
        ax1 = fig.add_subplot(gs[row, 0])
        matrices = get_reservoir_matrices(model)
        
        if matrices:
            all_eigenvalues = []
            for W_rec in matrices:
                eigs = np.linalg.eigvals(W_rec)
                all_eigenvalues.extend(eigs)
            
            all_eigenvalues = np.array(all_eigenvalues)
            
            ax1.scatter(all_eigenvalues.real, all_eigenvalues.imag,
                       alpha=0.5, s=10, color=color)
            
            # Unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            ax1.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5, label='Unit Circle')
            
            ax1.set_xlabel('Real Part', fontsize=11)
            ax1.set_ylabel('Imaginary Part', fontsize=11)
            ax1.set_title(f'{mode_name} - Recurrent Matrix Eigenvalues', 
                         fontsize=12, fontweight='bold')
            ax1.grid(alpha=0.3)
            ax1.legend()
            ax1.set_aspect('equal', adjustable='box')
            
            spectral_radius = np.max(np.abs(all_eigenvalues))
            ax1.text(0.05, 0.95, f'Spectral Radius: {spectral_radius:.4f}',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Eigenvalue magnitude distribution
        ax2 = fig.add_subplot(gs[row, 1])
        
        if matrices:
            eigenvalue_magnitudes = np.abs(all_eigenvalues)
            ax2.hist(eigenvalue_magnitudes, bins=50, alpha=0.7, edgecolor='black', color=color)
            ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Unit magnitude')
            ax2.set_xlabel('Eigenvalue Magnitude', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title(f'{mode_name} - Eigenvalue Distribution',
                         fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            stats_text = f'Mean: {np.mean(eigenvalue_magnitudes):.4f}\n'
            stats_text += f'Max: {np.max(eigenvalue_magnitudes):.4f}\n'
            stats_text += f'Std: {np.std(eigenvalue_magnitudes):.4f}'
            ax2.text(0.65, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Jacobian eigenspectrum
        ax3 = fig.add_subplot(gs[row, 2])
        
        print(f"  Computing Jacobian eigenvalues for {mode_name}...")
        try:
            jacobian_eigs = compute_jacobian_eigenvalues(model, sample_input, device, max_units=200)
            
            if np.iscomplexobj(jacobian_eigs):
                ax3.scatter(jacobian_eigs.real, jacobian_eigs.imag,
                           alpha=0.6, s=20, color=color, edgecolors='black', linewidths=0.5)
            else:
                ax3.scatter(jacobian_eigs.real, np.zeros_like(jacobian_eigs),
                           alpha=0.6, s=20, color=color, edgecolors='black', linewidths=0.5)
            
            theta = np.linspace(0, 2*np.pi, 100)
            ax3.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5, label='Unit Circle')
            
            ax3.set_xlabel('Real Part', fontsize=11)
            ax3.set_ylabel('Imaginary Part', fontsize=11)
            ax3.set_title(f'{mode_name} - Jacobian Eigenvalues',
                         fontsize=12, fontweight='bold')
            ax3.grid(alpha=0.3)
            ax3.legend()
            ax3.set_aspect('equal', adjustable='box')
            
            jac_spectral_radius = np.max(np.abs(jacobian_eigs))
            ax3.text(0.05, 0.95, f'Max |λ|: {jac_spectral_radius:.4f}',
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"    Warning: Could not compute Jacobian: {e}")
            ax3.text(0.5, 0.5, f'Jacobian computation\nnot available',
                    transform=ax3.transAxes, fontsize=12, ha='center', va='center')
            ax3.set_title(f'{mode_name} - Jacobian Eigenvalues',
                         fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Eigenspectrum Analysis: {mode1_name} vs {mode2_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_dir}/eigenspectrum_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Eigenspectrum analysis saved: {filename}")
    plt.close()
