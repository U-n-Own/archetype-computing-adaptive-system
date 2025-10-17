#!/usr/bin/env python3
"""
Compare Antisymmetric vs Cycle ESN on Speech Dataset
Search both modes and visualize differences
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acds.archetypes.esn import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")

def load_speech_data(dataroot: str, seed=42):
    """Load speech data from .npy files in the specified directory."""
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
    print(f"Classes: {len(class_names)} - {class_names}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, class_names


@torch.no_grad()
def test(model, X_data, y_data, scaler, classifier, device, use_last_state=True, batch_size=64):
    """Test function for speech data"""
    activations = []
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    
    for i in range(0, X_data.shape[0], batch_size):
        batch_X = X_tensor[i:i+batch_size].to(device)
        output = model(batch_X)
        
        if isinstance(output, tuple):
            states = output[0]
            last_hidden = output[1]
        else:
            states = output
            last_hidden = states[:, -1, :]
        
        if use_last_state:
            if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                final_state = last_hidden[-1].cpu().numpy()
            else:
                final_state = last_hidden.cpu().numpy()
            activations.append(final_state)
        else:
            pooled_states = states.mean(dim=1).cpu().numpy()
            activations.append(pooled_states)
    
    activations = np.concatenate(activations, axis=0)
    activations = scaler.transform(activations)
    
    return classifier.score(activations, y_data)


def connection_mode_search(X_train, y_train, X_valid, y_valid, device, n_inp, 
                          mode='antisymmetric', n_configs=15, n_layers=5):
    """
    Hyperparameter search for specific connection mode
    mode: 'antisymmetric' or 'cycle'
    """
    mode_name = "ANTISYMMETRIC" if mode == 'antisymmetric' else "CYCLE"
    print(f"\n{'='*70}")
    print(f"{mode_name} ESN HYPERPARAMETER SEARCH")
    print(f"Layers: {n_layers} | Configurations to test: {n_configs}")
    print(f"{'='*70}\n")
    
    # Define hyperparameter grid
    base_grid = {
        'spectral_radius': [0.9, 0.95, 0.99, 0.999],
        'leaky': [0.001, 0.01, 0.1, 0.5],
        'input_scaling': [0.5, 1.0, 2.0],
        'tot_units': [200, 500],
    }
    
    if mode == 'antisymmetric':
        base_grid['epsilon'] = [0.05, 0.1, 0.2, 0.4]
    else:  # cycle mode
        base_grid['epsilon'] = [0.0]
    
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
            if k != 'epsilon' or mode == 'antisymmetric':
                print(f"  {k:20s}: {v}")
        
        try:
            unit_per_layer = params['tot_units']
            
            print(f"  Creating model...", end='', flush=True)
            
            if mode == 'antisymmetric':
                model = DeepReservoir(
                    input_size=n_inp,
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
            else:  # cycle mode
                model = DeepReservoir(
                    input_size=n_inp,
                    tot_units=params['tot_units'],
                    n_layers=n_layers,
                    concat=True,
                    spectral_radius=params['spectral_radius'],
                    input_scaling=params['input_scaling'],
                    leaky=params['leaky'],
                    connectivity_recurrent=unit_per_layer,
                    connectivity_input=unit_per_layer,
                    connectivity_inter=unit_per_layer,
                    antisymmetric=False,
                    cycle=True,
                    linear=False,
                ).to(device)
            
            print(" ✓", flush=True)
            
            # Generate training activations
            print(f"  Processing training data...", end='', flush=True)
            activations = []
            start_time = time.time()
            
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            batch_size = 64
            
            for i in range(0, X_train.shape[0], batch_size):
                batch_X = X_train_tensor[i:i+batch_size].to(device)
                output = model(batch_X)
                
                if isinstance(output, tuple):
                    states = output[0]
                    last_hidden = output[1]
                else:
                    states = output
                    last_hidden = states[:, -1, :]
                
                # Use last state for speech data
                if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                    final_state = last_hidden[-1].cpu().numpy()
                else:
                    final_state = last_hidden.cpu().numpy()
                activations.append(final_state)
            
            train_time = time.time() - start_time
            
            activations = np.concatenate(activations, axis=0)
            print(f" ✓ ({activations.shape[0]} samples, {train_time:.2f}s)", flush=True)
            
            # Train classifier
            print(f"  Training classifier...", end='', flush=True)
            scaler = preprocessing.StandardScaler().fit(activations)
            activations_scaled = scaler.transform(activations)
            classifier = LogisticRegression(max_iter=1000, verbose=0).fit(activations_scaled, y_train)
            print(" ✓", flush=True)
            
            # Evaluate on validation set
            print(f"  Evaluating...", end='', flush=True)
            valid_acc = test(model, X_valid, y_valid, scaler, classifier, device)
            print(f" ✓", flush=True)
            
            result = {
                'params': params,
                'valid_acc': valid_acc,
                'train_time': train_time,
                'mode': mode
            }
            all_results.append(result)
            
            print(f"  ➤ Validation Accuracy: {valid_acc:.4f} ({valid_acc*100:.2f}%)")
            
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
    
    all_results.sort(key=lambda x: x['valid_acc'], reverse=True)
    
    print(f"\nTop 3 Configurations:")
    for i, result in enumerate(all_results[:3], 1):
        print(f"{i}. Valid Acc: {result['valid_acc']:.4f} | Time: {result['train_time']:.2f}s")
    
    return best_config, all_results


def plot_predictions(models_dict, X_test, y_test, class_names, device, save_dir="./plots/speech_antisym_vs_cycle"):
    """
    Plot example predictions showing reservoir states evolution over time
    Shows how both models process time series data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Select a few test samples from different classes
    n_examples = 6
    selected_indices = []
    selected_classes = np.random.choice(len(class_names), min(n_examples, len(class_names)), replace=False)
    
    for cls in selected_classes:
        cls_indices = np.where(y_test == cls)[0]
        if len(cls_indices) > 0:
            selected_indices.append(np.random.choice(cls_indices))
    
    fig, axes = plt.subplots(len(selected_indices), 3, figsize=(18, 4 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(selected_indices):
        X_sample = X_test[sample_idx:sample_idx+1]
        y_true = y_test[sample_idx]
        true_label = class_names[int(y_true)]
        
        X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
        
        # Plot input signal
        ax = axes[idx, 0]
        ax.imshow(X_sample[0].T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Feature Dim', fontsize=10)
        ax.set_title(f'Input: "{true_label}"', fontsize=11, fontweight='bold')
        ax.grid(False)
        
        # Process with both models
        for col, (mode_name, model_info) in enumerate([('Antisymmetric', models_dict['antisymmetric']), 
                                                         ('Cycle', models_dict['cycle'])], start=1):
            model = model_info['model']
            classifier = model_info['classifier']
            scaler = model_info['scaler']
            
            with torch.no_grad():
                output = model(X_tensor)
                if isinstance(output, tuple):
                    states = output[0]  # (1, time, features)
                    last_hidden = output[1]
                else:
                    states = output
                    last_hidden = states[:, -1, :]
                
                # Get final state for prediction
                if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                    final_state = last_hidden[-1].cpu().numpy()
                else:
                    final_state = last_hidden.cpu().numpy()
                
                # Predict
                final_state_scaled = scaler.transform(final_state)
                y_pred = classifier.predict(final_state_scaled)[0]
                y_proba = classifier.predict_proba(final_state_scaled)[0]
                pred_label = class_names[int(y_pred)]
                confidence = y_proba[int(y_pred)]
                
                # Plot reservoir states evolution
                states_np = states.cpu().numpy()[0]  # (time, features)
                
                ax = axes[idx, col]
                im = ax.imshow(states_np.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                ax.set_xlabel('Time Step', fontsize=10)
                ax.set_ylabel('Reservoir Units', fontsize=10)
                
                # Color code title based on correct/incorrect prediction
                color = 'green' if y_pred == y_true else 'red'
                ax.set_title(f'{mode_name}\nPred: "{pred_label}" ({confidence:.2f})', 
                           fontsize=11, fontweight='bold', color=color)
                ax.grid(False)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    filename = f"{save_dir}/prediction_examples.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Prediction examples saved: {filename}")
    plt.close()


def plot_state_statistics(models_dict, X_test, y_test, class_names, device, save_dir="./plots/speech_antisym_vs_cycle"):
    """
    Plot statistics of reservoir states for both models
    Shows activation patterns and dynamics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect states from a subset of test data
    n_samples = min(100, len(X_test))
    X_subset = X_test[:n_samples]
    y_subset = y_test[:n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for row, (mode_name, model_info) in enumerate([('Antisymmetric', models_dict['antisymmetric']),
                                                     ('Cycle', models_dict['cycle'])]):
        model = model_info['model']
        
        all_states = []
        all_final_states = []
        
        X_tensor = torch.tensor(X_subset, dtype=torch.float32).to(device)
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(X_subset), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                output = model(batch_X)
                
                if isinstance(output, tuple):
                    states = output[0]
                    last_hidden = output[1]
                else:
                    states = output
                    last_hidden = states[:, -1, :]
                
                all_states.append(states.cpu().numpy())
                
                if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                    final_state = last_hidden[-1].cpu().numpy()
                else:
                    final_state = last_hidden.cpu().numpy()
                all_final_states.append(final_state)
        
        all_states = np.concatenate(all_states, axis=0)  # (n_samples, time, features)
        all_final_states = np.concatenate(all_final_states, axis=0)  # (n_samples, features)
        
        # 1. Mean activation over time
        mean_activation = np.mean(np.abs(all_states), axis=(0, 2))  # Average over samples and features
        ax = axes[row, 0]
        ax.plot(mean_activation, linewidth=2, color='coral' if row == 0 else 'skyblue')
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Mean |Activation|', fontsize=11)
        ax.set_title(f'{mode_name} - Activation Evolution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 2. Final state distribution
        ax = axes[row, 1]
        ax.hist(all_final_states.flatten(), bins=50, alpha=0.7, edgecolor='black',
                color='coral' if row == 0 else 'skyblue')
        ax.set_xlabel('Final State Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{mode_name} - Final State Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 3. State variance over time
        state_variance = np.var(all_states, axis=(0, 2))  # Variance over samples and features
        ax = axes[row, 2]
        ax.plot(state_variance, linewidth=2, color='coral' if row == 0 else 'skyblue')
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Variance', fontsize=11)
        ax.set_title(f'{mode_name} - State Variance Evolution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Reservoir State Analysis: Antisymmetric vs Cycle', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_dir}/state_statistics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ State statistics saved: {filename}")
    plt.close()


def plot_comparison(antisymmetric_results, cycle_results, n_layers, save_dir="./plots/speech_antisym_vs_cycle"):
    """Plot comprehensive comparison between antisymmetric and cycle modes"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for both modes
    anti_accs = [r['valid_acc'] for r in antisymmetric_results]
    cycle_accs = [r['valid_acc'] for r in cycle_results]
    
    anti_times = [r['train_time'] for r in antisymmetric_results]
    cycle_times = [r['train_time'] for r in cycle_results]
    
    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy Distribution Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(anti_accs, bins=15, alpha=0.6, label='Antisymmetric', color='coral', edgecolor='black')
    ax1.hist(cycle_accs, bins=15, alpha=0.6, label='Cycle', color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(anti_accs), color='red', linestyle='--', linewidth=2, 
                label=f'Anti Mean: {np.mean(anti_accs):.4f}')
    ax1.axvline(np.mean(cycle_accs), color='blue', linestyle='--', linewidth=2,
                label=f'Cycle Mean: {np.mean(cycle_accs):.4f}')
    ax1.set_xlabel('Validation Accuracy', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Accuracy Distribution Comparison (Speech)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Box Plot Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([anti_accs, cycle_accs], labels=['Antisymmetric', 'Cycle'],
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
    best_cycle = max(cycle_accs)
    mean_anti = np.mean(anti_accs)
    mean_cycle = np.mean(cycle_accs)
    
    x = np.arange(2)
    width = 0.35
    bars1 = ax3.bar(x - width/2, [best_anti, best_cycle], width, 
                    label='Best', color=['coral', 'skyblue'], edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, [mean_anti, mean_cycle], width,
                    label='Mean', color=['lightcoral', 'lightskyblue'], 
                    edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax3.set_ylabel('Validation Accuracy', fontsize=11)
    ax3.set_title('Best vs Mean Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Antisymmetric', 'Cycle'])
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Training Time Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(anti_times, bins=15, alpha=0.6, label='Antisymmetric', color='coral', edgecolor='black')
    ax4.hist(cycle_times, bins=15, alpha=0.6, label='Cycle', color='skyblue', edgecolor='black')
    ax4.set_xlabel('Training Time (s)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Training Time Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Accuracy vs Time Scatter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(anti_times, anti_accs, alpha=0.6, s=80, label='Antisymmetric', 
                color='coral', edgecolors='black', linewidths=1)
    ax5.scatter(cycle_times, cycle_accs, alpha=0.6, s=80, label='Cycle',
                color='skyblue', edgecolors='black', linewidths=1)
    ax5.set_xlabel('Training Time (s)', fontsize=11)
    ax5.set_ylabel('Validation Accuracy', fontsize=11)
    ax5.set_title('Accuracy vs Training Time', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Spectral Radius Effect Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    anti_sr = [r['params']['spectral_radius'] for r in antisymmetric_results]
    cycle_sr = [r['params']['spectral_radius'] for r in cycle_results]
    ax6.scatter(anti_sr, anti_accs, alpha=0.6, s=80, label='Antisymmetric',
                color='coral', edgecolors='black', linewidths=1)
    ax6.scatter(cycle_sr, cycle_accs, alpha=0.6, s=80, label='Cycle',
                color='skyblue', edgecolors='black', linewidths=1)
    ax6.set_xlabel('Spectral Radius', fontsize=11)
    ax6.set_ylabel('Validation Accuracy', fontsize=11)
    ax6.set_title('Effect of Spectral Radius', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Leaky Rate Effect Comparison
    ax7 = fig.add_subplot(gs[2, 0])
    anti_leaky = [r['params']['leaky'] for r in antisymmetric_results]
    cycle_leaky = [r['params']['leaky'] for r in cycle_results]
    ax7.scatter(anti_leaky, anti_accs, alpha=0.6, s=80, label='Antisymmetric',
                color='coral', edgecolors='black', linewidths=1)
    ax7.scatter(cycle_leaky, cycle_accs, alpha=0.6, s=80, label='Cycle',
                color='skyblue', edgecolors='black', linewidths=1)
    ax7.set_xlabel('Leaky Rate', fontsize=11)
    ax7.set_ylabel('Validation Accuracy', fontsize=11)
    ax7.set_title('Effect of Leaky Rate', fontsize=12, fontweight='bold')
    ax7.set_xscale('log')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # 8. Top Configurations Comparison
    ax8 = fig.add_subplot(gs[2, 1:])
    top_n = min(5, len(antisymmetric_results), len(cycle_results))
    
    top_anti = sorted(antisymmetric_results, key=lambda x: x['valid_acc'], reverse=True)[:top_n]
    top_cycle = sorted(cycle_results, key=lambda x: x['valid_acc'], reverse=True)[:top_n]
    
    y_pos = np.arange(top_n * 2)
    accs = []
    labels = []
    colors = []
    
    for i in range(top_n):
        accs.append(top_anti[i]['valid_acc'])
        labels.append(f'Anti #{i+1}')
        colors.append('coral')
        accs.append(top_cycle[i]['valid_acc'])
        labels.append(f'Cycle #{i+1}')
        colors.append('skyblue')
    
    bars = ax8.barh(y_pos, accs, color=colors, edgecolor='black', linewidth=1.5)
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(labels)
    ax8.set_xlabel('Validation Accuracy', fontsize=11)
    ax8.set_title(f'Top {top_n} Configurations from Each Mode', fontsize=12, fontweight='bold')
    ax8.grid(alpha=0.3, axis='x')
    
    for bar, acc in zip(bars, accs):
        width = bar.get_width()
        ax8.text(width, bar.get_y() + bar.get_height()/2,
                f' {acc:.4f}', ha='left', va='center', fontsize=9)
    
    plt.suptitle(f'Antisymmetric vs Cycle Mode Comparison - Speech Dataset ({n_layers} layers)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    filename = f"{save_dir}/comparison_{n_layers}layers.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {filename}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Antisymmetric vs Cycle ESN on Speech Dataset')
    parser.add_argument('--dataroot', type=str, default='./speech',
                       help='Path to speech data directory')
    parser.add_argument('--n_configs', type=int, default=15,
                       help='Number of configurations to test per mode')
    parser.add_argument('--n_layers', type=int, default=5,
                       help='Number of ESN layers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("ANTISYMMETRIC vs CYCLE MODE COMPARISON - SPEECH DATASET")
    print(f"{'='*70}")
    print(f"Data directory: {args.dataroot}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Configurations per mode: {args.n_configs}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*70}\n")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names = load_speech_data(
        args.dataroot, seed=args.seed
    )
    n_inp = X_train.shape[2]  # Feature dimension
    
    # Search both modes
    print("\n" + "="*70)
    print("PHASE 1: ANTISYMMETRIC MODE SEARCH")
    print("="*70)
    best_anti_config, anti_results = connection_mode_search(
        X_train, y_train, X_valid, y_valid, device, n_inp,
        mode='antisymmetric',
        n_configs=args.n_configs,
        n_layers=args.n_layers
    )
    
    print("\n" + "="*70)
    print("PHASE 2: CYCLE MODE SEARCH")
    print("="*70)
    best_cycle_config, cycle_results = connection_mode_search(
        X_train, y_train, X_valid, y_valid, device, n_inp,
        mode='cycle',
        n_configs=args.n_configs,
        n_layers=args.n_layers
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
    print(f"  Best Accuracy:  {best_anti_acc:.4f} ({best_anti_acc*100:.2f}%)")
    print(f"  Mean Accuracy:  {mean_anti_acc:.4f} ({mean_anti_acc*100:.2f}%)")
    print(f"  Std Dev:        {np.std([r['valid_acc'] for r in anti_results]):.4f}")
    
    print(f"\nCycle Mode:")
    print(f"  Best Accuracy:  {best_cycle_acc:.4f} ({best_cycle_acc*100:.2f}%)")
    print(f"  Mean Accuracy:  {mean_cycle_acc:.4f} ({mean_cycle_acc*100:.2f}%)")
    print(f"  Std Dev:        {np.std([r['valid_acc'] for r in cycle_results]):.4f}")
    
    print(f"\nDifference (Antisymmetric - Cycle):")
    print(f"  Best Accuracy:  {(best_anti_acc - best_cycle_acc):.4f} ({(best_anti_acc - best_cycle_acc)*100:.2f}%)")
    print(f"  Mean Accuracy:  {(mean_anti_acc - mean_cycle_acc):.4f} ({(mean_anti_acc - mean_cycle_acc)*100:.2f}%)")
    
    # Determine winner
    if best_anti_acc > best_cycle_acc:
        print(f"\n🏆 Winner: ANTISYMMETRIC by {(best_anti_acc - best_cycle_acc)*100:.2f}%")
    elif best_cycle_acc > best_anti_acc:
        print(f"\n🏆 Winner: CYCLE by {(best_cycle_acc - best_anti_acc)*100:.2f}%")
    else:
        print(f"\n🤝 Tie!")
    
    # Save results
    save_dir = "./plots/speech_antisym_vs_cycle"
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    with open(f"{save_dir}/comparison_results_{args.n_layers}layers.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved: {save_dir}/comparison_results_{args.n_layers}layers.json")
    
    # Plot comparison
    plot_comparison(anti_results, cycle_results, args.n_layers, save_dir)
    
    # Test on test set with best configs and save models
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET WITH BEST CONFIGS")
    print("="*70)
    
    models_dict = {}
    
    for mode, config in [('Antisymmetric', best_anti_config), ('Cycle', best_cycle_config)]:
        print(f"\n{mode} mode:")
        unit_per_layer = config['tot_units']
        
        if mode == 'Antisymmetric':
            model = DeepReservoir(
                input_size=n_inp,
                tot_units=config['tot_units'],
                n_layers=args.n_layers,
                concat=True,
                spectral_radius=config['spectral_radius'],
                input_scaling=config['input_scaling'],
                leaky=config['leaky'],
                connectivity_recurrent=unit_per_layer,
                connectivity_input=unit_per_layer,
                connectivity_inter=unit_per_layer,
                antisymmetric=True,
                epsilon=config['epsilon'],
                cycle=False,
                linear=False,
            ).to(device)
        else:
            model = DeepReservoir(
                input_size=n_inp,
                tot_units=config['tot_units'],
                n_layers=args.n_layers,
                concat=True,
                spectral_radius=config['spectral_radius'],
                input_scaling=config['input_scaling'],
                leaky=config['leaky'],
                connectivity_recurrent=unit_per_layer,
                connectivity_input=unit_per_layer,
                connectivity_inter=unit_per_layer,
                antisymmetric=False,
                cycle=True,
                linear=False,
            ).to(device)
        
        # Train on full training set
        activations = []
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        batch_size = 64
        
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train_tensor[i:i+batch_size].to(device)
            output = model(batch_X)
            
            if isinstance(output, tuple):
                last_hidden = output[1]
            else:
                last_hidden = output[:, -1, :]
            
            if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                final_state = last_hidden[-1].cpu().numpy()
            else:
                final_state = last_hidden.cpu().numpy()
            activations.append(final_state)
        
        activations = np.concatenate(activations, axis=0)
        scaler = preprocessing.StandardScaler().fit(activations)
        activations_scaled = scaler.transform(activations)
        classifier = LogisticRegression(max_iter=1000).fit(activations_scaled, y_train)
        
        test_acc = test(model, X_test, y_test, scaler, classifier, device)
        print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Store model and components for visualization
        mode_key = 'antisymmetric' if mode == 'Antisymmetric' else 'cycle'
        models_dict[mode_key] = {
            'model': model,
            'classifier': classifier,
            'scaler': scaler,
            'test_acc': test_acc
        }
    
    # Generate prediction visualizations
    print("\n" + "="*70)
    print("GENERATING PREDICTION VISUALIZATIONS")
    print("="*70)
    
    print("\nCreating prediction examples...")
    plot_predictions(models_dict, X_test, y_test, class_names, device, save_dir)
    
    print("Creating state statistics plots...")
    plot_state_statistics(models_dict, X_test, y_test, class_names, device, save_dir)
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}\n")
