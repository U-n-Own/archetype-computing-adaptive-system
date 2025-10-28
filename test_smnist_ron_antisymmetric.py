"""
Test comparing standard 1-layer RON vs 5-layer antisymmetric RON on sMNIST
with coupling strength search.

Based on smnist.py experiment.
"""
import os
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from acds.archetypes import RandomizedOscillatorsNetwork, DeepRandomizedOscillatorsNetwork
from acds.benchmarks import get_mnist_data

# Set matplotlib backend for non-interactive plotting
matplotlib.use('Agg')

# Create results directory
RESULTS_DIR = "results_smnist_ron_antisymmetric"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_eigenvalue_spectrum(eigenvalues, title, filename, spectral_radius=None):
    """
    Plot the eigenvalue spectrum in the complex plane.
    
    Args:
        eigenvalues: array of complex eigenvalues
        title: plot title
        filename: output filename (will be saved in RESULTS_DIR)
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
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved eigenvalue spectrum plot to: {filepath}")


def plot_spectral_radius_comparison(results, baseline_result=None):
    """
    Create comparison plots of spectral radius vs accuracy for all models.
    
    Args:
        results: list of result dictionaries
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
    filepath = os.path.join(RESULTS_DIR, 'spectral_radius_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved spectral radius comparison plot to: {filepath}")


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
                # For DeepRON, call forward which returns (concat_states, layer_states_list)
                output, layer_states_final = model(images)
                
                # We need to extract per-timestep, per-layer states
                # The model stores them in layer_states_all during forward pass
                # Let's collect them by re-running with manual extraction
                batch_size, seq_len, _ = images.shape
                n_layers = len(model.ron_reservoir)
                
                # Initialize storage for trajectories: [batch, time, layer, hidden]
                batch_trajectories = [[[] for _ in range(n_layers)] for _ in range(batch_size)]
                
                # Re-run forward pass with state collection
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
                
                # Convert to proper format: [batch][time][layer] = numpy_array
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
                output, final_states = model(images)
                
                # Re-run to collect all timesteps
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


def plot_phase_space_trajectories(trajectory_data, model_name, filename, n_dims=3):
    """
    Visualize phase space trajectories using PCA projection.
    
    Args:
        trajectory_data: dict with trajectories and metadata
        model_name: name for the plot title
        filename: output filename
        n_dims: number of dimensions for PCA projection (2 or 3)
    """
    from sklearn.decomposition import PCA
    
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
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved phase space trajectory plot to: {filepath}")


def plot_temporal_evolution(trajectory_data, model_name, filename):
    """
    Plot temporal evolution of hidden state norms and activity.
    
    Args:
        trajectory_data: dict with trajectories and metadata
        model_name: name for the plot title
        filename: output filename
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
    
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved temporal evolution plot to: {filepath}")


def compute_total_weight_matrix_1layer(model):
    """
    Compute the total recurrent weight matrix for a 1-layer RON.
    
    Args:
        model: RandomizedOscillatorsNetwork
    
    Returns:
        W_total: numpy array of the recurrent weight matrix
        singular_values: singular values of W_total
        spectral_radius: spectral radius (largest absolute eigenvalue)
    """
    W_rec = model.h2h.detach().cpu().numpy()
    
    # Compute singular values
    singular_values = np.linalg.svd(W_rec, compute_uv=False)
    
    # Compute spectral radius (largest absolute eigenvalue)
    eigenvalues = np.linalg.eigvals(W_rec)
    spectral_radius = np.max(np.abs(eigenvalues))
    
    return W_rec, singular_values, spectral_radius


def compute_total_weight_matrix_antisymmetric(model, coupling_epsilon):
    """
    Compute the total recurrent weight matrix for antisymmetric coupled DeepRON.
    
    The total weight matrix is:
    W_total = [
        [W_rec^(1),     ε*A,         0,       ...,  0,           0        ],
        [-ε*A^T,        W_rec^(2),   ε*B,     ...,  0,           0        ],
        [0,             -ε*B^T,      W_rec^(3),...,  0,           0        ],
        [...,           ...,         ...,     ...,  ...,         ...      ],
        [0,             0,           0,       ...,  W_rec^(L-1), ε*Z      ],
        [0,             0,           0,       ...,  -ε*Z^T,      W_rec^(L)]
    ]
    
    Args:
        model: DeepRandomizedOscillatorsNetwork with antisymmetric coupling
        coupling_epsilon: coupling strength
    
    Returns:
        W_total: numpy array of the total weight matrix
        singular_values: singular values of W_total
        spectral_radius: spectral radius (largest absolute eigenvalue)
    """
    n_layers = len(model.ron_reservoir)
    layer_sizes = [layer.n_hid for layer in model.ron_reservoir]
    total_size = sum(layer_sizes)
    
    # Initialize total weight matrix
    W_total = np.zeros((total_size, total_size))
    
    # Fill in the block diagonal and coupling matrices
    row_offset = 0
    for i, layer in enumerate(model.ron_reservoir):
        layer_size = layer.n_hid
        
        # Add recurrent weight matrix W_rec^(i)
        W_rec = layer.h2h.detach().cpu().numpy()
        W_total[row_offset:row_offset+layer_size, row_offset:row_offset+layer_size] = W_rec
        
        # Add coupling matrices if not the last layer
        if i < n_layers - 1 and layer.antisymmetric_coupling:
            next_layer_size = model.ron_reservoir[i+1].n_hid
            
            # Get coupling matrix C
            C = layer.C_coupling.detach().cpu().numpy()
            
            # Add ε*C (forward coupling to next layer)
            W_total[row_offset:row_offset+layer_size, 
                   row_offset+layer_size:row_offset+layer_size+next_layer_size] = coupling_epsilon * C
            
            # Add -ε*C^T (backward coupling from next layer)
            W_total[row_offset+layer_size:row_offset+layer_size+next_layer_size,
                   row_offset:row_offset+layer_size] = -coupling_epsilon * C.T
        
        row_offset += layer_size
    
    # Compute singular values
    singular_values = np.linalg.svd(W_total, compute_uv=False)
    
    # Compute spectral radius (largest absolute eigenvalue)
    eigenvalues = np.linalg.eigvals(W_total)
    spectral_radius = np.max(np.abs(eigenvalues))
    
    return W_total, singular_values, spectral_radius


def analyze_weight_matrix(model, model_name, coupling_epsilon=None):
    """
    Analyze the total recurrent weight matrix of a model.
    
    Args:
        model: Either RandomizedOscillatorsNetwork or DeepRandomizedOscillatorsNetwork
        model_name: Name for display
        coupling_epsilon: Coupling strength (for antisymmetric models)
    
    Returns:
        Dictionary with analysis results including eigenvalues
    """
    print(f"\n{'='*60}")
    print(f"Weight Matrix Analysis: {model_name}")
    print(f"{'='*60}")
    
    if isinstance(model, RandomizedOscillatorsNetwork):
        W_total, singular_values, spectral_radius = compute_total_weight_matrix_1layer(model)
    elif isinstance(model, DeepRandomizedOscillatorsNetwork) and model.antisymmetric_coupling:
        if coupling_epsilon is None:
            coupling_epsilon = model.coupling_epsilon
        W_total, singular_values, spectral_radius = compute_total_weight_matrix_antisymmetric(
            model, coupling_epsilon
        )
    else:
        print("  Model does not support weight matrix analysis")
        return None
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(W_total)
    
    print(f"  Matrix size: {W_total.shape}")
    print(f"  Spectral radius (ρ): {spectral_radius:.6f}")
    print(f"  Number of singular values: {len(singular_values)}")
    print(f"  Largest singular value: {singular_values[0]:.6f}")
    print(f"  Smallest singular value: {singular_values[-1]:.6f}")
    print(f"  Condition number: {singular_values[0]/singular_values[-1]:.2e}")
    
    # Display top singular values
    print(f"\n  Top 10 singular values:")
    for i, sv in enumerate(singular_values[:10]):
        print(f"    σ_{i+1}: {sv:.6f}")
    
    # Check if matrix is (approximately) antisymmetric
    if W_total.shape[0] == W_total.shape[1]:
        antisymmetry_error = np.max(np.abs(W_total + W_total.T))
        print(f"\n  Antisymmetry check ||W + W^T||_∞: {antisymmetry_error:.6f}")
        if antisymmetry_error < 1e-6:
            print("    → Matrix is antisymmetric")
        else:
            print("    → Matrix is NOT antisymmetric (expected for coupled system)")
    
    # Generate filename for the plot
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')
    plot_filename = f"eigenspectrum_{safe_name}.png"
    
    # Create eigenvalue spectrum plot
    plot_eigenvalue_spectrum(
        eigenvalues, 
        title=f"Eigenvalue Spectrum: {model_name}",
        filename=plot_filename,
        spectral_radius=spectral_radius
    )
    
    return {
        'W_total': W_total,
        'singular_values': singular_values,
        'spectral_radius': spectral_radius,
        'condition_number': singular_values[0]/singular_values[-1],
        'eigenvalues': eigenvalues,
        'plot_filename': plot_filename
    }

print("=" * 80)
print("sMNIST: Standard 1-layer RON vs Antisymmetric 5-layer RON")
print("Coupling Strength Search")
print("=" * 80)

# Configuration
DATAROOT = "data"
BATCH_SIZE = 1000
N_HID = 700
N_TRIALS = 3  # Number of trials to reduce uncertainty

# Hyperparameters for 1-layer RON baseline
DT = 0.042
RHO_BASELINE = 0.9  # For 1-layer baseline
INP_SCALING = 1.0
EPSILON_CENTER = 0.51
EPSILON_RANGE = 0.5
GAMMA_CENTER = 2.7
GAMMA_RANGE = 1.0

# Different RHO values to test for 5-layer antisymmetric
#RHO_VALUES = [0.7, 0.8, 0.999]
RHO_VALUES = [0.9]
# Derived parameters
EPSILON_MIN = EPSILON_CENTER - EPSILON_RANGE
EPSILON_MAX = EPSILON_CENTER + EPSILON_RANGE
GAMMA_MIN = GAMMA_CENTER - GAMMA_RANGE  
GAMMA_MAX = GAMMA_CENTER + GAMMA_RANGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

print("Hyperparameters:")
print(f"  dt: {DT}")
print(f"  rho (baseline): {RHO_BASELINE}")
print(f"  rho (5-layer search): {RHO_VALUES}")
print(f"  input_scaling: {INP_SCALING}")
print(f"  epsilon: ({EPSILON_MIN}, {EPSILON_MAX})")
print(f"  gamma: ({GAMMA_MIN}, {GAMMA_MAX})")
print(f"  n_hid: {N_HID}")
print(f"  batch_size: {BATCH_SIZE}")
print(f"  trials: {N_TRIALS}\n")

n_inp = 1
n_out = 10


@torch.no_grad()
def test(data_loader, model, classifier, scaler):
    """Evaluate model on a dataset."""
    activations, ys = [], []
    for images, labels in tqdm(data_loader, desc="Testing", leave=False):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        
        # Handle both RON and DeepRON outputs
        output = model(images)
        if isinstance(output, tuple):
            output = output[0]  # Get hidden states
        
        # Get last timestep
        if len(output.shape) == 3:
            output = output[:, -1, :]
        
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    
    # Check for NaN or Inf
    if np.isnan(activations).any() or np.isinf(activations).any():
        print("  WARNING: Activations contain NaN or Inf!")
        return 0.0
    
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)


def train_and_evaluate(model_name, model, train_loader, valid_loader, test_loader, visualize_trajectory=False):
    """Train and evaluate a model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Collect training activations
    activations, ys = [], []
    print("Collecting training activations...")
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        
        # Handle both RON and DeepRON outputs
        output = model(images)
        if isinstance(output, tuple):
            output = output[0]  # Get hidden states
        
        # Get last timestep
        if len(output.shape) == 3:
            output = output[:, -1, :]
        
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    print(f"Activation statistics:")
    print(f"  Shape: {activations.shape}")
    print(f"  Mean: {activations.mean():.6f}, Std: {activations.std():.6f}")
    print(f"  Min: {activations.min():.6f}, Max: {activations.max():.6f}")
    
    # Check for NaN or Inf
    if np.isnan(activations).any() or np.isinf(activations).any():
        print("  ERROR: Activations contain NaN or Inf! Skipping this configuration.")
        return {
            'name': model_name,
            'train_acc': 0.0,
            'valid_acc': 0.0,
            'test_acc': 0.0,
            'saturated_pct': 0.0,
            'failed': True
        }
    
    # Check saturation
    saturated = np.sum(np.abs(activations) > 0.99) / activations.size
    print(f"  Saturated (|x| > 0.99): {saturated*100:.2f}%")
    
    # Scale and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    
    print("Training logistic regression classifier...")
    classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    classifier.fit(activations, ys)
    
    # Evaluate
    print("Evaluating...")
    train_acc = classifier.score(activations, ys)
    valid_acc = test(valid_loader, model, classifier, scaler)
    test_acc = test(test_loader, model, classifier, scaler)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Valid Accuracy: {valid_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    
    # Visualize trajectories if requested
    if visualize_trajectory:
        print("\n  Generating phase space trajectory visualizations...")
        device = next(model.parameters()).device
        
        # Collect trajectories from test set
        trajectory_data = collect_trajectories(model, test_loader, n_samples=5, device=device)
        
        # Generate safe filename
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')
        
        # Plot 3D phase space trajectories
        plot_phase_space_trajectories(
            trajectory_data,
            model_name=model_name,
            filename=f"trajectory_3d_{safe_name}.png",
            n_dims=3
        )
        
        # Plot 2D phase space trajectories
        plot_phase_space_trajectories(
            trajectory_data,
            model_name=model_name,
            filename=f"trajectory_2d_{safe_name}.png",
            n_dims=2
        )
        
        # Plot temporal evolution
        plot_temporal_evolution(
            trajectory_data,
            model_name=model_name,
            filename=f"temporal_{safe_name}.png"
        )
    
    return {
        'name': model_name,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'test_acc': test_acc,
        'saturated_pct': saturated * 100,
        'failed': False
    }


if __name__ == "__main__":
    # Load data
    print("Loading sMNIST dataset...")
    train_loader, valid_loader, test_loader = get_mnist_data(
        DATAROOT, 
        bs_train=BATCH_SIZE,
        bs_test=BATCH_SIZE
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} train, "
          f"{len(valid_loader.dataset)} valid, {len(test_loader.dataset)} test\n")
    
    results = []
    
    # ========================================
    # Baseline: Standard 1-layer RON (with trials)
    # ========================================
    print("\n" + "="*80)
    print(f"BASELINE: Standard 1-layer RON ({N_TRIALS} trials)")
    print("="*80)
    
    baseline_results = []
    for trial in range(N_TRIALS):
        print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
        
        model_standard = RandomizedOscillatorsNetwork(
            n_inp=n_inp,
            n_hid=N_HID,
            dt=DT,
            gamma=(GAMMA_MIN, GAMMA_MAX),
            epsilon=(EPSILON_MIN, EPSILON_MAX),
            rho=RHO_BASELINE,
            input_scaling=INP_SCALING,
            topology="full",
            device=device,
        ).to(device)
        
        if trial == 0:
            print(f"Model: 1-layer RON with {N_HID} units")
            print(f"Parameters: {sum(p.numel() for p in model_standard.parameters())}")
            
            # Analyze weight matrix (only for first trial)
            baseline_analysis = analyze_weight_matrix(model_standard, "1-layer RON Baseline")
        
        result_standard = train_and_evaluate(
            f"Standard 1-layer RON (trial {trial+1})",
            model_standard,
            train_loader,
            valid_loader,
            test_loader,
            visualize_trajectory=(trial == 0)  # Visualize only first trial
        )
        result_standard['trial'] = trial + 1
        result_standard['rho'] = RHO_BASELINE
        baseline_results.append(result_standard)
        
        # Clean up
        del model_standard
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Average baseline results
    baseline_avg = {
        'name': 'Standard 1-layer RON (avg)',
        'train_acc': np.mean([r['train_acc'] for r in baseline_results]),
        'valid_acc': np.mean([r['valid_acc'] for r in baseline_results]),
        'test_acc': np.mean([r['test_acc'] for r in baseline_results]),
        'train_std': np.std([r['train_acc'] for r in baseline_results]),
        'valid_std': np.std([r['valid_acc'] for r in baseline_results]),
        'test_std': np.std([r['test_acc'] for r in baseline_results]),
        'saturated_pct': np.mean([r['saturated_pct'] for r in baseline_results]),
        'failed': False,
        'rho': RHO_BASELINE,
    }
    # Add spectral properties from the first trial's analysis
    if baseline_analysis:
        baseline_avg['spectral_radius'] = baseline_analysis['spectral_radius']
        baseline_avg['condition_number'] = baseline_analysis['condition_number']
    results.append(baseline_avg)
    print(f"\nBaseline Average Results:")
    print(f"  Test Accuracy: {baseline_avg['test_acc']*100:.2f}% ± {baseline_avg['test_std']*100:.2f}%")
    
    # ========================================
    # Test: 5-layer Antisymmetric RON with different coupling strengths and RHO values
    # ========================================
    print("\n" + "="*80)
    print(f"COUPLING STRENGTH & RHO SEARCH: 5-layer Antisymmetric RON ({N_TRIALS} trials each)")
    print("="*80)
    
    # Test different coupling epsilon values
    coupling_values = [0.1, 5.0, 7, 10]
    
    for rho_val in RHO_VALUES:
        print(f"\n{'='*70}")
        print(f"RHO = {rho_val}")
        print(f"{'='*70}")
        
        for coup_eps in coupling_values:
            print(f"\n{'='*60}")
            print(f"Testing coupling_epsilon = {coup_eps}, rho = {rho_val}")
            print(f"{'='*60}")
            
            antisym_results = []
            for trial in range(N_TRIALS):
                print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
                
                model_antisym = DeepRandomizedOscillatorsNetwork(
                    n_inp=n_inp,
                    total_units=N_HID,
                    n_layers=5,
                    dt=DT,
                    gamma=(GAMMA_MIN, GAMMA_MAX),
                    epsilon=(EPSILON_MIN, EPSILON_MAX),
                    rho=rho_val,
                    input_scaling=INP_SCALING,
                    inter_scaling=INP_SCALING,
                    topology="full",
                    concat=True,
                    antisymmetric_coupling=True,
                    coupling_epsilon=coup_eps,
                    device=device,
                ).to(device)
                
                if trial == 0:
                    print(f"Model: 5-layer DeepRON with antisymmetric coupling")
                    print(f"Parameters: {sum(p.numel() for p in model_antisym.parameters())}")
                    
                    # Analyze weight matrix (only for first trial)
                    antisym_analysis = analyze_weight_matrix(
                        model_antisym, 
                        f"5-layer Antisymmetric RON (ε_c={coup_eps}, ρ={rho_val})",
                        coupling_epsilon=coup_eps
                    )
                
                result_antisym = train_and_evaluate(
                    f"Antisymmetric 5-layer RON (ε_c={coup_eps}, ρ={rho_val}, trial {trial+1})",
                    model_antisym,
                    train_loader,
                    valid_loader,
                    test_loader,
                    visualize_trajectory=(trial == 0)  # Visualize only first trial
                )
                result_antisym['coupling_epsilon'] = coup_eps
                result_antisym['rho'] = rho_val
                result_antisym['trial'] = trial + 1
                if trial == 0 and antisym_analysis:
                    result_antisym['spectral_radius'] = antisym_analysis['spectral_radius']
                    result_antisym['condition_number'] = antisym_analysis['condition_number']
                antisym_results.append(result_antisym)
                
                # Clean up
                del model_antisym
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Average results for this configuration
            antisym_avg = {
                'name': f'Antisymmetric 5-layer (ε_c={coup_eps}, ρ={rho_val}, avg)',
                'train_acc': np.mean([r['train_acc'] for r in antisym_results]),
                'valid_acc': np.mean([r['valid_acc'] for r in antisym_results]),
                'test_acc': np.mean([r['test_acc'] for r in antisym_results]),
                'train_std': np.std([r['train_acc'] for r in antisym_results]),
                'valid_std': np.std([r['valid_acc'] for r in antisym_results]),
                'test_std': np.std([r['test_acc'] for r in antisym_results]),
                'saturated_pct': np.mean([r['saturated_pct'] for r in antisym_results]),
                'coupling_epsilon': coup_eps,
                'rho': rho_val,
                'failed': False,
            }
            if 'spectral_radius' in antisym_results[0]:
                antisym_avg['spectral_radius'] = antisym_results[0]['spectral_radius']
                antisym_avg['condition_number'] = antisym_results[0]['condition_number']
            results.append(antisym_avg)
            print(f"\nAverage Results (ε_c={coup_eps}, ρ={rho_val}):")
            print(f"  Test Accuracy: {antisym_avg['test_acc']*100:.2f}% ± {antisym_avg['test_std']*100:.2f}%")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Filter out failed runs
    valid_results = [r for r in results if not r.get('failed', False)]
    
    if not valid_results:
        print("ERROR: All experiments failed!")
    else:
        # Create comparison plots
        baseline = [r for r in valid_results if 'Standard 1-layer RON' in r['name'] and 'avg' in r['name']]
        baseline_result = baseline[0] if baseline else None
        
        print("\n" + "="*80)
        print("GENERATING SPECTRAL ANALYSIS PLOTS")
        print("="*80)
        plot_spectral_radius_comparison(valid_results, baseline_result)
        
        # Sort by test accuracy
        results_sorted = sorted(valid_results, key=lambda x: x['test_acc'], reverse=True)
        
        print(f"\n{'Rank':<5} {'Model':<55} {'ε_c':<8} {'ρ_rec':<8} {'ρ_tot':<10} {'Test Acc':<18} {'Valid Acc':<18}")
        print("-"*130)
        
        for rank, r in enumerate(results_sorted, 1):
            coup_str = f"{r.get('coupling_epsilon', 0):.2f}" if 'coupling_epsilon' in r else "N/A"
            rho_rec_str = f"{r.get('rho', 0):.3f}" if 'rho' in r else "N/A"
            rho_tot_str = f"{r.get('spectral_radius', 0.0):.4f}" if 'spectral_radius' in r else "N/A"
            antisym_marker = "✓" if 'Antisymmetric' in r['name'] else " "
            
            # Format with std if available
            test_str = f"{r['test_acc']*100:.2f}%"
            if 'test_std' in r:
                test_str += f" ± {r['test_std']*100:.2f}%"
            
            valid_str = f"{r['valid_acc']*100:.2f}%"
            if 'valid_std' in r:
                valid_str += f" ± {r['valid_std']*100:.2f}%"
            
            print(f"{rank:<5} {antisym_marker} {r['name']:<53} {coup_str:<8} {rho_rec_str:<8} {rho_tot_str:<10} "
                  f"{test_str:<18} {valid_str:<18}")
        
        # Key findings
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        best = results_sorted[0]
        baseline = [r for r in valid_results if 'Standard 1-layer RON' in r['name'] and 'avg' in r['name']]
        baseline = baseline[0] if baseline else None
        
        antisym_results = [r for r in results_sorted if 'Antisymmetric' in r['name'] and 'avg' in r['name']]
        best_antisym = antisym_results[0] if antisym_results else None
        
        print(f"\nBest overall: {best['name']}")
        print(f"  Test Accuracy: {best['test_acc']*100:.2f}%")
        if 'coupling_epsilon' in best:
            print(f"  Coupling Epsilon: {best['coupling_epsilon']}")
        
        if baseline:
            print(f"\nBaseline (Standard 1-layer RON, avg over {N_TRIALS} trials):")
            print(f"  Test Accuracy: {baseline['test_acc']*100:.2f}% ± {baseline['test_std']*100:.2f}%")
            print(f"  Spectral Radius: {baseline.get('spectral_radius', 'N/A')}")
        
        if best_antisym:
            print(f"\nBest Antisymmetric (5-layer, avg over {N_TRIALS} trials):")
            print(f"  Test Accuracy: {best_antisym['test_acc']*100:.2f}% ± {best_antisym['test_std']*100:.2f}%")
            print(f"  Coupling Epsilon: {best_antisym['coupling_epsilon']}")
            print(f"  RHO (recurrent): {best_antisym['rho']}")
            print(f"  Spectral Radius (W_total): {best_antisym.get('spectral_radius', 'N/A')}")
            
            if baseline:
                improvement = (best_antisym['test_acc'] - baseline['test_acc']) * 100
                print(f"\nAntisymmetric vs Baseline: {improvement:+.2f} percentage points")
                
                if best_antisym['test_acc'] > baseline['test_acc']:
                    print("✓ Antisymmetric coupling provides improvement!")
                else:
                    print("✗ Antisymmetric coupling does not improve over baseline")
        
        if best_antisym:
            print(f"\nBest Antisymmetric (5-layer RON):")
            print(f"  Test Accuracy: {best_antisym['test_acc']*100:.2f}%")
            print(f"  Coupling Epsilon: {best_antisym['coupling_epsilon']}")
            
            if baseline:
                improvement = (best_antisym['test_acc'] - baseline['test_acc']) * 100
                print(f"\nAntisymmetric vs Baseline: {improvement:+.2f} percentage points")
                
                if best_antisym['test_acc'] > baseline['test_acc']:
                    print("✓ Antisymmetric coupling provides improvement!")
                else:
                    print("✗ Antisymmetric coupling does not improve over baseline")
        
        # Plot coupling strength vs accuracy
        print("\n" + "="*80)
        print("COUPLING STRENGTH vs SPECTRAL RADIUS ANALYSIS")
        print("="*80)
        
        print(f"\n{'Coupling ε':<12} {'Spectral ρ':<12} {'Test Acc':<12} {'Valid Acc':<12}")
        print("-"*48)
        antisym_only = [r for r in results_sorted if 'coupling_epsilon' in r]
        # Sort by coupling epsilon for this table
        antisym_only_sorted = sorted(antisym_only, key=lambda x: x['coupling_epsilon'])
        for r in antisym_only_sorted:
            rho_str = f"{r.get('spectral_radius', 0.0):.4f}" if 'spectral_radius' in r else "N/A"
            print(f"{r['coupling_epsilon']:<12.2f} {rho_str:<12} {r['test_acc']*100:<11.2f}% {r['valid_acc']*100:<11.2f}%")
    
    print("\n" + "="*80)
    
    # Save results to file
    result_file = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(result_file, 'w') as f:
        f.write("sMNIST RON Antisymmetric Coupling Results\n")
        f.write("="*80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  N_HID: {N_HID}\n")
        f.write(f"  N_TRIALS: {N_TRIALS}\n")
        f.write(f"  DT: {DT}\n")
        f.write(f"  RHO_BASELINE: {RHO_BASELINE}\n")
        f.write(f"  RHO_VALUES (5-layer): {RHO_VALUES}\n")
        f.write(f"  INPUT_SCALING: {INP_SCALING}\n")
        f.write(f"  EPSILON: ({EPSILON_MIN}, {EPSILON_MAX})\n")
        f.write(f"  GAMMA: ({GAMMA_MIN}, {GAMMA_MAX})\n")
        f.write(f"  COUPLING_VALUES: {coupling_values}\n\n")
        
        f.write("Results:\n")
        f.write("-"*80 + "\n")
        for r in results_sorted:
            f.write(f"{r['name']}\n")
            if 'coupling_epsilon' in r:
                f.write(f"  Coupling Epsilon: {r['coupling_epsilon']}\n")
            if 'rho' in r:
                f.write(f"  RHO (recurrent): {r['rho']}\n")
            if 'spectral_radius' in r:
                f.write(f"  Spectral Radius (W_total): {r['spectral_radius']:.6f}\n")
            if 'condition_number' in r:
                f.write(f"  Condition Number: {r['condition_number']:.2e}\n")
            f.write(f"  Train Acc: {r['train_acc']*100:.2f}%")
            if 'train_std' in r:
                f.write(f" ± {r['train_std']*100:.2f}%")
            f.write("\n")
            f.write(f"  Valid Acc: {r['valid_acc']*100:.2f}%")
            if 'valid_std' in r:
                f.write(f" ± {r['valid_std']*100:.2f}%")
            f.write("\n")
            f.write(f"  Test Acc: {r['test_acc']*100:.2f}%")
            if 'test_std' in r:
                f.write(f" ± {r['test_std']*100:.2f}%")
            f.write("\n")
            f.write(f"  Saturation: {r['saturated_pct']:.2f}%\n\n")
    
    print(f"\n{'='*80}")
    print("RESULTS SAVED")
    print(f"{'='*80}")
    print(f"Text results: {result_file}")
    print(f"Eigenvalue spectrum plots: {RESULTS_DIR}/eigenspectrum_*.png")
    print(f"Phase space trajectories (3D): {RESULTS_DIR}/trajectory_3d_*.png")
    print(f"Phase space trajectories (2D): {RESULTS_DIR}/trajectory_2d_*.png")
    print(f"Temporal dynamics: {RESULTS_DIR}/temporal_*.png")
    print(f"Spectral comparison: {RESULTS_DIR}/spectral_radius_comparison.png")
    print(f"{'='*80}\n")
