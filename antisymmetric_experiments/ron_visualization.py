"""
Visualization utilities for RON models.

This module provides functions for:
- Phase space trajectory visualization
- Temporal dynamics analysis
- Eigenvalue spectrum plotting
- Spectral radius comparison
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

from acds.archetypes import RandomizedOscillatorsNetwork, DeepRandomizedOscillatorsNetwork

# Set matplotlib backend for non-interactive plotting
matplotlib.use('Agg')


def collect_trajectories(model, data_loader, n_samples=5, device='cpu'):
    """
    Collect hidden state trajectories from the model for visualization.
    
    Args:
        model: RON or DeepRON model
        data_loader: data loader with samples
        n_samples: number of samples to collect trajectories for
        device: computation device
    
    Returns:
        Dictionary with trajectories and metadata
    """
    model.eval()
    trajectories = []
    labels_collected = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            if len(trajectories) >= n_samples:
                break
            
            images = images[:n_samples - len(trajectories)].to(device)
            images = images.view(images.shape[0], -1).unsqueeze(-1)
            labels = labels[:n_samples - len(trajectories)]
            
            # Get model output - handle both RON and DeepRON
            if isinstance(model, DeepRandomizedOscillatorsNetwork):
                batch_size, seq_len, _ = images.shape
                n_layers = len(model.ron_reservoir)
                
                # Initialize storage for trajectories: [batch, time, layer, hidden]
                batch_trajectories = [[[] for _ in range(n_layers)] for _ in range(batch_size)]
                
                # Run forward pass with state collection
                h_states = []
                hz_states = []
                for ron_layer in model.ron_reservoir:
                    h_states.append(torch.zeros(batch_size, ron_layer.n_hid, device=device))
                    hz_states.append(torch.zeros(batch_size, ron_layer.n_hid, device=device))
                
                for t in range(seq_len):
                    new_h_states = []
                    new_hz_states = []
                    
                    for i, ron_layer in enumerate(model.ron_reservoir):
                        # Prepare layer input
                        if i == 0:
                            layer_input = images[:, t, :]
                        else:
                            layer_input = new_h_states[i-1]
                        
                        # Prepare antisymmetric coupling inputs
                        h_prev_layer = h_states[i-1] if i > 0 else None
                        h_next_layer = h_states[i+1] if i < len(model.ron_reservoir) - 1 else None
                        
                        # Forward through the layer
                        new_h, new_hz = ron_layer.cell(
                            layer_input,
                            h_states[i],
                            hz_states[i],
                            first_layer=(i == 0),
                            h_last=None,
                            h_prev_layer=h_prev_layer,
                            h_next_layer=h_next_layer
                        )
                        
                        new_h_states.append(new_h)
                        new_hz_states.append(new_hz)
                        
                        # Store states for each batch sample
                        for b in range(batch_size):
                            batch_trajectories[b][i].append(new_h[b].cpu().numpy())
                    
                    # Update hidden states for next timestep
                    h_states = new_h_states
                    hz_states = new_hz_states
                
                # Convert to proper format
                for b in range(batch_size):
                    sample_traj = []
                    for t in range(seq_len):
                        timestep_states = [batch_trajectories[b][layer_idx][t] for layer_idx in range(n_layers)]
                        sample_traj.append(timestep_states)
                    trajectories.append(sample_traj)
                    labels_collected.append(labels[b].item())
                    
            else:
                # For standard RON
                batch_size, seq_len, _ = images.shape
                
                h = torch.zeros(batch_size, model.n_hid, device=device)
                hz = torch.zeros(batch_size, model.n_hid, device=device)
                
                all_hidden = []
                for t in range(seq_len):
                    inp = images[:, t, :]
                    h, hz = model.cell(inp, h, hz, first_layer=True, h_last=None)
                    all_hidden.append(h.cpu().numpy())
                
                # Stack: [time, batch, hidden]
                all_hidden = np.stack(all_hidden, axis=0)
                
                # Reorganize: [batch][time, hidden]
                for b in range(batch_size):
                    trajectories.append(all_hidden[:, b, :])  # [time, hidden]
                    labels_collected.append(labels[b].item())
    
    return {
        'trajectories': trajectories,
        'labels': labels_collected,
        'is_multilayer': isinstance(model, DeepRandomizedOscillatorsNetwork)
    }


def plot_phase_space_trajectories(trajectory_data, model_name, filename, results_dir, n_dims=3):
    """
    Visualize phase space trajectories using PCA projection.
    
    Args:
        trajectory_data: dict with trajectories and metadata
        model_name: name for the plot title
        filename: output filename
        results_dir: directory to save the plot
        n_dims: number of dimensions for PCA projection (2 or 3)
    """
    trajectories = trajectory_data['trajectories']
    labels = trajectory_data['labels']
    is_multilayer = trajectory_data['is_multilayer']
    
    if is_multilayer:
        # Multi-layer visualization
        n_samples = len(trajectories)
        n_timesteps = len(trajectories[0])
        n_layers = len(trajectories[0][0])
        
        # Create subplots for each layer
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        if n_dims == 3:
            fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
            
            for layer_idx in range(n_layers):
                # Collect all states for this layer
                layer_states = []
                for sample_idx in range(n_samples):
                    for t in range(n_timesteps):
                        if len(trajectories[sample_idx][t]) > layer_idx:
                            layer_states.append(trajectories[sample_idx][t][layer_idx])
                
                if not layer_states:
                    continue
                
                layer_states = np.array(layer_states)
                
                # Apply PCA
                pca = PCA(n_components=3)
                states_pca = pca.fit_transform(layer_states)
                var_explained = pca.explained_variance_ratio_
                
                # Reshape back to trajectories
                states_pca = states_pca.reshape(n_samples, n_timesteps, 3)
                
                # Create 3D subplot
                ax = fig.add_subplot(n_rows, n_cols, layer_idx + 1, projection='3d')
                
                # Plot each trajectory with different color per class
                colors = plt.cm.tab10(np.array(labels))
                for i in range(n_samples):
                    traj = states_pca[i]
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                           alpha=0.6, linewidth=2, color=colors[i])
                    # Mark start and end
                    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                              c=[colors[i]], marker='o', s=100, edgecolors='black', linewidth=2)
                    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                              c=[colors[i]], marker='s', s=100, edgecolors='black', linewidth=2)
                
                ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=10)
                ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=10)
                ax.set_zlabel(f'PC3 ({var_explained[2]*100:.1f}%)', fontsize=10)
                ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
        else:
            # 2D visualization
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_layers == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for layer_idx in range(n_layers):
                # Collect all states for this layer
                layer_states = []
                for sample_idx in range(n_samples):
                    for t in range(n_timesteps):
                        if len(trajectories[sample_idx][t]) > layer_idx:
                            layer_states.append(trajectories[sample_idx][t][layer_idx])
                
                if not layer_states:
                    continue
                
                layer_states = np.array(layer_states)
                
                # Apply PCA
                pca = PCA(n_components=2)
                states_pca = pca.fit_transform(layer_states)
                var_explained = pca.explained_variance_ratio_
                
                # Reshape back to trajectories
                states_pca = states_pca.reshape(n_samples, n_timesteps, 2)
                
                ax = axes[layer_idx]
                
                # Plot each trajectory with different color per class
                colors = plt.cm.tab10(np.array(labels))
                for i in range(n_samples):
                    traj = states_pca[i]
                    ax.plot(traj[:, 0], traj[:, 1], 
                           alpha=0.6, linewidth=2, color=colors[i], label=f'Digit {labels[i]}')
                    # Mark start and end
                    ax.scatter(traj[0, 0], traj[0, 1], 
                              c=[colors[i]], marker='o', s=100, edgecolors='black', linewidth=2)
                    ax.scatter(traj[-1, 0], traj[-1, 1], 
                              c=[colors[i]], marker='s', s=100, edgecolors='black', linewidth=2)
                
                ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=10)
                ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=10)
                ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_layers, len(axes)):
                axes[idx].axis('off')
        
        fig.suptitle(f'Phase Space Trajectories: {model_name}\n(○ = start, □ = end, colors = digit classes)', 
                     fontsize=14, fontweight='bold')
    else:
        # Single layer visualization
        n_samples = len(trajectories)
        n_timesteps = trajectories[0].shape[0]
        
        # Stack all states
        all_states = np.vstack(trajectories)  # [n_samples * n_timesteps, hidden_dim]
        
        # Apply PCA
        if n_dims == 3:
            pca = PCA(n_components=3)
            states_pca = pca.fit_transform(all_states)
            var_explained = pca.explained_variance_ratio_
            states_pca = states_pca.reshape(n_samples, n_timesteps, 3)
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            colors = plt.cm.tab10(np.array(labels))
            for i in range(n_samples):
                traj = states_pca[i]
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       alpha=0.7, linewidth=2.5, color=colors[i], label=f'Digit {labels[i]}')
                # Mark start and end
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                          c=[colors[i]], marker='o', s=150, edgecolors='black', linewidth=2)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                          c=[colors[i]], marker='s', s=150, edgecolors='black', linewidth=2)
            
            ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=12)
            ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=12)
            ax.set_zlabel(f'PC3 ({var_explained[2]*100:.1f}%)', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            pca = PCA(n_components=2)
            states_pca = pca.fit_transform(all_states)
            var_explained = pca.explained_variance_ratio_
            states_pca = states_pca.reshape(n_samples, n_timesteps, 2)
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            colors = plt.cm.tab10(np.array(labels))
            for i in range(n_samples):
                traj = states_pca[i]
                ax.plot(traj[:, 0], traj[:, 1], 
                       alpha=0.7, linewidth=2.5, color=colors[i], label=f'Digit {labels[i]}')
                # Mark start and end
                ax.scatter(traj[0, 0], traj[0, 1], 
                          c=[colors[i]], marker='o', s=150, edgecolors='black', linewidth=2)
                ax.scatter(traj[-1, 0], traj[-1, 1], 
                          c=[colors[i]], marker='s', s=150, edgecolors='black', linewidth=2)
            
            ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=12)
            ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Phase Space Trajectories: {model_name}\n(○ = start, □ = end)', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved phase space trajectory plot to: {filepath}")


def plot_temporal_evolution(trajectory_data, model_name, filename, results_dir):
    """
    Plot temporal evolution of hidden state norms and activity.
    
    Args:
        trajectory_data: dict with trajectories and metadata
        model_name: name for the plot title
        filename: output filename
        results_dir: directory to save the plot
    """
    trajectories = trajectory_data['trajectories']
    labels = trajectory_data['labels']
    is_multilayer = trajectory_data['is_multilayer']
    
    if is_multilayer:
        n_samples = len(trajectories)
        n_timesteps = len(trajectories[0])
        n_layers = len(trajectories[0][0])
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: State norm over time for each layer
        ax = axes[0]
        colors_layers = plt.cm.viridis(np.linspace(0, 1, n_layers))
        
        for layer_idx in range(n_layers):
            norms_over_time = []
            for t in range(n_timesteps):
                timestep_norms = []
                for sample_idx in range(n_samples):
                    if len(trajectories[sample_idx][t]) > layer_idx:
                        state = trajectories[sample_idx][t][layer_idx]
                        timestep_norms.append(np.linalg.norm(state))
                norms_over_time.append(timestep_norms)
            
            # Compute mean and std across samples
            mean_norms = [np.mean(norms) for norms in norms_over_time]
            std_norms = [np.std(norms) for norms in norms_over_time]
            
            timesteps = np.arange(n_timesteps)
            ax.plot(timesteps, mean_norms, color=colors_layers[layer_idx], 
                   linewidth=2, label=f'Layer {layer_idx + 1}')
            ax.fill_between(timesteps, 
                           np.array(mean_norms) - np.array(std_norms),
                           np.array(mean_norms) + np.array(std_norms),
                           color=colors_layers[layer_idx], alpha=0.2)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Hidden State Norm', fontsize=12)
        ax.set_title('State Magnitude Evolution Over Time', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean activity per layer
        ax = axes[1]
        layer_activities = []
        for layer_idx in range(n_layers):
            activities = []
            for sample_idx in range(n_samples):
                for t in range(n_timesteps):
                    if len(trajectories[sample_idx][t]) > layer_idx:
                        state = trajectories[sample_idx][t][layer_idx]
                        activities.append(np.mean(np.abs(state)))
            layer_activities.append(np.mean(activities))
        
        ax.bar(range(1, n_layers + 1), layer_activities, color=colors_layers, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Mean Absolute Activity', fontsize=12)
        ax.set_title('Average Activity per Layer', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
    else:
        # Single layer visualization
        n_samples = len(trajectories)
        n_timesteps = trajectories[0].shape[0]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: State norm over time
        ax = axes[0]
        colors_samples = plt.cm.tab10(np.array(labels))
        
        for i in range(n_samples):
            norms = [np.linalg.norm(trajectories[i][t]) for t in range(n_timesteps)]
            ax.plot(range(n_timesteps), norms, color=colors_samples[i], 
                   linewidth=2, alpha=0.7, label=f'Digit {labels[i]}')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Hidden State Norm', fontsize=12)
        ax.set_title('State Magnitude Evolution Over Time', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean activity across hidden dimensions
        ax = axes[1]
        mean_activity = []
        for t in range(n_timesteps):
            activities = [np.mean(np.abs(trajectories[i][t])) for i in range(n_samples)]
            mean_activity.append(np.mean(activities))
        
        ax.plot(range(n_timesteps), mean_activity, linewidth=2.5, color='blue')
        ax.fill_between(range(n_timesteps), 
                        np.array(mean_activity) * 0.9,
                        np.array(mean_activity) * 1.1,
                        alpha=0.2, color='blue')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Mean Absolute Activity', fontsize=12)
        ax.set_title('Average Activity Over Time', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Temporal Dynamics: {model_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved temporal evolution plot to: {filepath}")


def plot_eigenvalue_spectrum(eigenvalues, title, filename, results_dir, spectral_radius=None):
    """
    Plot the eigenvalue spectrum in the complex plane.
    
    Args:
        eigenvalues: array of complex eigenvalues
        title: plot title
        filename: output filename
        results_dir: directory to save the plot
        spectral_radius: if provided, draw a circle with this radius
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Eigenvalues in complex plane
    ax1.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.6, s=20, c='blue', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=1.5, alpha=0.5, label='Unit circle')
    
    # Draw spectral radius circle if provided
    if spectral_radius is not None:
        ax1.plot(spectral_radius * np.cos(theta), spectral_radius * np.sin(theta), 
                'g--', linewidth=2, alpha=0.7, label=f'ρ = {spectral_radius:.4f}')
    
    ax1.set_xlabel('Real part', fontsize=12)
    ax1.set_ylabel('Imaginary part', fontsize=12)
    ax1.set_title(f'Eigenvalue Spectrum in Complex Plane', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Magnitude histogram
    magnitudes = np.abs(eigenvalues)
    ax2.hist(magnitudes, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=1.0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Unit circle (ρ=1)')
    if spectral_radius is not None:
        ax2.axvline(x=spectral_radius, color='g', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'ρ = {spectral_radius:.4f}')
    ax2.set_xlabel('|λ| (Eigenvalue magnitude)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Eigenvalue Magnitudes', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved eigenvalue spectrum plot to: {filepath}")


def plot_spectral_radius_comparison(results, results_dir, baseline_result=None):
    """
    Create comparison plots of spectral radius vs accuracy for all models.
    
    Args:
        results: list of result dictionaries
        results_dir: directory to save the plot
        baseline_result: baseline result for reference line
    """
    # Filter results with spectral radius
    results_with_rho = [r for r in results if 'spectral_radius' in r and not r.get('failed', False)]
    
    if not results_with_rho:
        print("No results with spectral radius to plot.")
        return
    
    # Separate baseline and antisymmetric results
    antisym_results = [r for r in results_with_rho if 'coupling_epsilon' in r]
    
    if not antisym_results:
        print("No antisymmetric results to plot.")
        return
    
    # Extract data for antisymmetric models
    coupling_eps = np.array([r['coupling_epsilon'] for r in antisym_results])
    spectral_rho = np.array([r['spectral_radius'] for r in antisym_results])
    test_acc = np.array([r['test_acc'] * 100 for r in antisym_results])
    rho_rec = np.array([r['rho'] for r in antisym_results])
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Spectral radius vs coupling strength (colored by RHO)
    unique_rhos = np.unique(rho_rec)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_rhos)))
    
    for i, rho_val in enumerate(unique_rhos):
        mask = rho_rec == rho_val
        ax1.plot(coupling_eps[mask], spectral_rho[mask], 'o-', 
                color=colors[i], label=f'ρ_rec = {rho_val}', 
                markersize=8, linewidth=2, alpha=0.7)
    
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='ρ = 1 (stability boundary)')
    
    if baseline_result and 'spectral_radius' in baseline_result:
        ax1.axhline(y=baseline_result['spectral_radius'], color='green', 
                   linestyle='--', linewidth=2, alpha=0.5, 
                   label=f"Baseline ρ = {baseline_result['spectral_radius']:.4f}")
    
    ax1.set_xlabel('Coupling Strength (ε_c)', fontsize=13)
    ax1.set_ylabel('Spectral Radius (ρ) of W_total', fontsize=13)
    ax1.set_title('Spectral Radius vs Coupling Strength', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test accuracy vs spectral radius (colored by coupling strength)
    sc = ax2.scatter(spectral_rho, test_acc, c=coupling_eps, 
                    cmap='plasma', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add baseline reference if available
    if baseline_result and 'spectral_radius' in baseline_result:
        ax2.scatter(baseline_result['spectral_radius'], baseline_result['test_acc'] * 100,
                   marker='*', s=400, c='green', edgecolors='black', linewidth=2,
                   label='Baseline (1-layer RON)', zorder=10)
    
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='ρ = 1')
    
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Coupling Strength (ε_c)', fontsize=11)
    
    ax2.set_xlabel('Spectral Radius (ρ) of W_total', fontsize=13)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax2.set_title('Test Accuracy vs Spectral Radius', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(results_dir, 'spectral_radius_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved spectral radius comparison plot to: {filepath}")
