#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import argparse

# Add the parent directory to the path to import acds
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acds.archetypes import DeepReservoir, DeepRandomizedOscillatorsNetwork

def compute_spectral_properties(matrix):
    """Compute spectral radius and spectral norm of a matrix."""
    eigenvals = np.linalg.eigvals(matrix)
    spectral_radius = np.max(np.abs(eigenvals))
    spectral_norm = np.linalg.norm(matrix, ord=2)  # Largest singular value
    return spectral_radius, spectral_norm, eigenvals


def create_reservoir_architectures():
    """Create different reservoir architectures for spectral analysis with adjusted spectral radius."""
    architectures = [
        {"name": "Fully Connected 100", "n_layers": 1, "units_per_layer": 100, "tot_units": 100, "rho": 0.99, "cycle": False},
        {"name": "Simple Cycle 100 units", "n_layers": 1, "units_per_layer": 100, "tot_units": 100, "rho": 0.9, "cycle": True},
        {"name": "2 Layers 50 units", "n_layers": 2, "units_per_layer": 50, "tot_units": 100, "rho": 0.55, "cycle": True},
        {"name": "4 Layers 25 units", "n_layers": 4, "units_per_layer": 25, "tot_units": 100, "rho": 0.67, "cycle": True},
        {"name": "5 Layers 20 units NO CYCLE", "n_layers": 5, "units_per_layer": 20, "tot_units": 100, "rho": 0.90, "cycle": False},
        {"name": "10 Layers 10 units", "n_layers": 10, "units_per_layer": 10, "tot_units": 100, "rho": 0.685, "cycle": True},
        {"name": "20 Layers 5 units", "n_layers": 20, "units_per_layer": 5, "tot_units": 100, "rho": 0.67, "cycle": True},
        {"name": "50 Layers 2 units", "n_layers": 50, "units_per_layer": 2, "tot_units": 100, "rho": 0.695, "cycle": True},
        {"name": "100 Layers 1 unit", "n_layers": 100, "units_per_layer": 1, "tot_units": 100, "rho": 0.9, "cycle": True},
        #{"name": "Fully Connected 50", "n_layers": 1, "units_per_layer": 50, "tot_units": 50, "rho": 0.99},
        #{"name": "2 Layers 25 units", "n_layers": 2, "units_per_layer": 25, "tot_units": 50, "rho": 0.55},
        #{"name": "4 Layers 12 units", "n_layers": 4, "units_per_layer": 12, "tot_units": 50, "rho": 0.57},
        #{"name": "10 Layers 5 units", "n_layers": 10, "units_per_layer": 5, "tot_units": 50, "rho": 0.585},
        #{"name": "20 Layers 2 units", "n_layers": 20, "units_per_layer": 2, "tot_units": 50, "rho": 0.57},
        #{"name": "50 Layers 1 unit", "n_layers": 50, "units_per_layer": 1, "tot_units": 50, "rho": 0.695},
    ]

    return architectures


def construct_jacobian(model, device):
    """Construct block matrix form of the Jacobian of 
    The cycle network like follows:
    [ W_rec_0  W_proj_0  0         ... ]
    [ 0        W_rec_1  W_proj_1  ... ]
    [ 0        0        W_rec_2   ... ]
    ...
    [ W_proj_n-1  0       0       W_rec_n-1 ]
    
    where W_rec_i are the recurrent weight matrices of each layer and
    W_proj_i are the projection weights connecting layers in a cyclic manner.
    
    This is a simplified version that assumes all layers have recurrent connections
    and possibly projection connections in a cyclic topology.
    
    Returns a block diagonal matrix representing the Jacobian of the reservoir dynamics
    with respect to the reservoir states, which is useful for analyzing stability and dynamics.
    """
    jacobian_blocks = []
    
    # Determine which reservoir attribute to use (ESN uses 'reservoir', RON uses 'ron_reservoir')
    reservoir_layers = None
    if hasattr(model, 'reservoir'):
        reservoir_layers = model.reservoir
    elif hasattr(model, 'ron_reservoir'):
        reservoir_layers = model.ron_reservoir
    
    if reservoir_layers is None:
        return None
    
    # Extract weight matrices from each layer
    for layer_idx in range(model.n_layers):
        layer = reservoir_layers[layer_idx]
        
        # For ESN (DeepReservoir)
        if hasattr(layer, 'net') and hasattr(layer.net, 'recurrent_kernel'):
            W_rec = layer.net.recurrent_kernel.detach().cpu().numpy()
            jacobian_blocks.append(W_rec)
        # For RON (DeepRandomizedOscillatorsNetwork)
        elif hasattr(layer, 'h2h'):
            W_rec = layer.h2h.detach().cpu().numpy()
            jacobian_blocks.append(W_rec)
    
    if not jacobian_blocks:
        return None
    
    # For single layer, return the recurrent matrix
    if model.n_layers == 1:
        return jacobian_blocks[0]
    
    # For multi-layer cyclic networks, create the full Jacobian
    total_size = sum(W.shape[0] for W in jacobian_blocks)
    total_jacobian = np.zeros((total_size, total_size))
    
    # Fill block diagonal with recurrent weights
    row_start = 0
    for i, W in enumerate(jacobian_blocks):
        row_end = row_start + W.shape[0]
        col_start = row_start
        col_end = col_start + W.shape[1]
        total_jacobian[row_start:row_end, col_start:col_end] = W
        row_start = row_end
    
    # Add inter-layer connections for cyclic topology
    if model.cycle and model.n_layers > 1:
        # Connect last layer back to first layer
        for layer_idx in range(model.n_layers):
            next_layer_idx = (layer_idx + 1) % model.n_layers
            
            layer = reservoir_layers[layer_idx]
            
            # For ESN - Get projection kernel weights for cycle connections
            if hasattr(layer, 'net') and hasattr(layer.net, 'projection_kernel') and layer.net.projection_kernel is not None:
                W_proj = layer.net.projection_kernel.detach().cpu().numpy()
                
                # Calculate positions in the full matrix
                curr_start = sum(jacobian_blocks[j].shape[0] for j in range(layer_idx))
                curr_end = curr_start + jacobian_blocks[layer_idx].shape[0]
                next_start = sum(jacobian_blocks[j].shape[0] for j in range(next_layer_idx))
                next_end = next_start + jacobian_blocks[next_layer_idx].shape[0]
                
                # Add cycle connection
                if W_proj.shape[0] == (curr_end - curr_start) and W_proj.shape[1] == (next_end - next_start):
                    total_jacobian[next_start:next_end, curr_start:curr_end] = W_proj.T
            # For RON - Get cycle kernel weights
            elif hasattr(layer, 'cycle_kernel') and layer.cycle_kernel is not None:
                W_cycle = layer.cycle_kernel.detach().cpu().numpy()
                
                # Calculate positions in the full matrix
                curr_start = sum(jacobian_blocks[j].shape[0] for j in range(layer_idx))
                curr_end = curr_start + jacobian_blocks[layer_idx].shape[0]
                next_start = sum(jacobian_blocks[j].shape[0] for j in range(next_layer_idx))
                next_end = next_start + jacobian_blocks[next_layer_idx].shape[0]
                
                # Add cycle connection
                if W_cycle.shape[0] == (curr_end - curr_start) and W_cycle.shape[1] == (next_end - next_start):
                    total_jacobian[next_start:next_end, curr_start:curr_end] = W_cycle.T
            elif model.cycle and hasattr(model, 'no_projection') and model.no_projection:
                # For pure cycle without projection matrices, create identity connections
                curr_start = sum(jacobian_blocks[j].shape[0] for j in range(layer_idx))
                curr_end = curr_start + jacobian_blocks[layer_idx].shape[0]
                next_start = sum(jacobian_blocks[j].shape[0] for j in range(next_layer_idx))
                next_end = next_start + jacobian_blocks[next_layer_idx].shape[0]
                
                # Add identity connection (or truncated identity if dimensions differ)
                min_dim = min(curr_end - curr_start, next_end - next_start)
                total_jacobian[next_start:next_start+min_dim, curr_start:curr_start+min_dim] = np.eye(min_dim)
    
    return total_jacobian


def square_correlation(x, y):
    """Compute the squared correlation coefficient between two vectors."""
    correlation_matrix = np.corrcoef(x.flatten(), y.flatten())
    return correlation_matrix[0, 1] ** 2


def compute_memory_capacity(model, device, tot_units=100):
    """Compute memory capacity using the exact same approach as memorycapacity.py."""
    from sklearn import preprocessing
    from sklearn.linear_model import Ridge
    
    model.eval()
    
    # Set delay to 2 * total number of hidden units  
    delay = 2 * tot_units
    print(f"  Computing memory capacity up to delay {delay}...")
    
    try:
        # Set bias to zero (same as memorycapacity.py)
        for name, param in model.named_parameters():
            if "bias" in name:
                param.data.fill_(0)
        
        # Use same parameters as memorycapacity.py
        num_steps = 6000
        train_steps = 5000
        test_steps = 1000
        washout = 1000
        
        # Generate input signal (same as memorycapacity.py)
        u = np.random.uniform(-0.8, 0.8, size=(num_steps + delay, 1))
        u = u.astype(np.float32)
        
        # Get reservoir states (same as memorycapacity.py)
        states_u = model(torch.tensor(u[:-delay]).to(device).reshape(1, -1, 1))[0].cpu().numpy()
        
        # Handle concatenation (same as memorycapacity.py)
        if hasattr(model, 'concat') and model.concat:
            states_u = states_u.reshape(-1, tot_units)
        else:
            # If not concatenated, use the output size
            states_u = states_u.reshape(-1, states_u.shape[-1])
        
        # Check for NaN/Inf values
        if np.isnan(states_u).any() or np.isinf(states_u).any():
            print(f"    Warning: Model produces NaN/Inf values, skipping memory capacity")
            return 0.0, {}
        
        # Prepare data for single model that predicts all delays (same as memorycapacity.py)
        X_all = states_u[delay:num_steps, :]
        
        # Create target matrix with columns for each delay (same as memorycapacity.py)
        y_all = np.zeros((num_steps - delay, delay))
        for i in range(1, delay + 1):
            y_all[:, i-1] = u[delay-i:num_steps-i, 0]
        
        # Split into train and test (same as memorycapacity.py)
        split_idx_train = train_steps - delay
        split_idx_test = split_idx_train + test_steps
        
        X_train, X_test = X_all[:split_idx_train], X_all[split_idx_train:split_idx_test]
        y_train, y_test = y_all[:split_idx_train], y_all[split_idx_train:split_idx_test]
        
        # Add washout (same as memorycapacity.py)
        X_train = X_train[washout:]
        y_train = y_train[washout:]
        
        # Normalize the data (same as memorycapacity.py)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train classifier (same as memorycapacity.py)
        try:
            classifier = np.linalg.pinv(X_train) @ y_train
        except np.linalg.LinAlgError as e:
            ridge = Ridge(alpha=1e-6, max_iter=10000)
            ridge.fit(X_train, y_train)
            classifier = ridge.coef_.T
            if len(classifier.shape) == 1:
                classifier = classifier.reshape(-1, 1)
        
        y_hat_train = X_train @ classifier
        y_hat_test = X_test @ classifier
        
        # Calculate memory capacity for each delay (same as memorycapacity.py)
        total_memory_capacity = 0.0
        memory_per_delay = {}
        
        for i in range(1, delay + 1):
            col_idx = i - 1
            test_memory = square_correlation(y_hat_test[:, col_idx], y_test[:, col_idx])
            
            # Handle edge cases
            if test_memory > 1:
                print(f"    Warning: Test memory > 1 for delay {i}: {test_memory}")
                test_memory = 1.0
            if np.isnan(test_memory) or test_memory < 0:
                test_memory = 0.0
            
            total_memory_capacity += test_memory
            memory_per_delay[i] = test_memory
        
        return total_memory_capacity, memory_per_delay
        
    except Exception as e:
        print(f"    Warning: Failed to compute memory capacity: {e}")
        return 0.0, {}


def compute_jacobian_at_equilibrium(model, device, input_size=1, n_steps=50):
    """Compute a simplified Jacobian approximation."""
    model.eval()
    
    try:
        # Generate a small input sequence to reach equilibrium
        test_input = torch.zeros(1, n_steps, input_size).to(device)
        
        # Get the final state
        with torch.no_grad():
            states = model(test_input)[0]  # Shape: (batch, time, features)
            
            # Check for NaN/Inf
            if torch.isnan(states).any() or torch.isinf(states).any():
                print(f"    Warning: Model states contain NaN/Inf, using identity matrix")
                n_units = states.shape[-1]
                return np.eye(n_units)
            
            final_state = states[0, -1, :].clone()  # Last time step
        
        # For stability, return a simple approximation based on the weight structure
        return construct_jacobian(model, device)
        
    except Exception as e:
        print(f"    Warning: Failed to compute equilibrium Jacobian: {e}")
        # Return a conservative identity matrix
        try:
            test_input = torch.zeros(1, 10, input_size).to(device)
            states = model(test_input)[0]
            n_units = states.shape[-1]
            return np.eye(n_units)
        except:
            return np.eye(100)  # Default size


def analyze_reservoir_spectral_properties(device=torch.device("cpu"), use_ron=False):
    """Analyze spectral properties of different reservoir architectures."""
    architectures = create_reservoir_architectures()
    results = []
    
    print("="*60)
    if use_ron:
        print("Using RON (Randomized Oscillators Network)")
    else:
        print("Using ESN (Echo State Network / DeepReservoir)")
    print("="*60)
    
    for arch in tqdm(architectures, desc="Analyzing architectures"):
        
        # Reset seeds for each architecture to ensure reproducible results
        torch.manual_seed(42)
        np.random.seed(42)
        
        try:
            # Create the model with cycle connections and adjusted spectral radius
            units_per_layer = arch['tot_units'] // arch['n_layers']
            use_cycle = arch.get('cycle', True)  # Default to True if not specified
            
            if use_ron:
                # RON parameters as specified
                dt = 0.5923
                gamma = 1.632
                epsilon = 0.78
                input_scaling = 0.2
                inter_scaling = 0.2
                
                model = DeepRandomizedOscillatorsNetwork(
                    n_inp=1,
                    total_units=arch['tot_units'],
                    n_layers=arch['n_layers'],
                    dt=dt,
                    gamma=gamma,
                    epsilon=epsilon,
                    rho=arch['rho'],  # Use architecture-specific spectral radius
                    input_scaling=input_scaling,
                    inter_scaling=inter_scaling,
                    topology="full",
                    concat=True,
                    cycle=use_cycle,
                    device=device,
                ).to(device)
            else:
                # ESN parameters (original)
                # For cycle mode, we should not have self-recurrent connections, only cycle connections
                connectivity_recurrent_value = 0 if use_cycle else units_per_layer
                # For cycle mode, use fixed input scaling like SCR (v=0.5)
                # For non-cycle, use small value to avoid overwhelming reservoir dynamics
                input_scaling_value = 0.5 if use_cycle else 0.0051
                model = DeepReservoir(
                    input_size=1,
                    tot_units=arch['tot_units'],  # Use the specified total units
                    n_layers=arch['n_layers'],
                    concat=True,
                    spectral_radius=arch['rho'],  # Use architecture-specific spectral radius
                    inter_scaling=0.5,  # FIXED: Use same as input_scaling (like memorycapacity.py)
                    input_scaling=input_scaling_value,  # FIX: 0.5 for cycle (like SCR), 0.0051 otherwise
                    connectivity_recurrent=units_per_layer,  # FIX: 0 for cycle, units_per_layer otherwise
                    connectivity_input=units_per_layer,
                    connectivity_inter=1,  # FIXED: Use same as memorycapacity.y
                    leaky=1.0,
                    linear=True,
                    cycle=use_cycle,  # Use architecture-specific cycle setting
                ).to(device)
            

            
            # Set bias to zero for cleaner analysis
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(0)
            
            # Extract the effective Jacobian matrix from weights
            jacobian_matrix = construct_jacobian(model, device)
            
            # Compute Jacobian at equilibrium
            print(f"Computing Jacobian at equilibrium...")
            equilibrium_jacobian = compute_jacobian_at_equilibrium(model, device)
            
            # Compute memory capacity with delay = 2*tot_units
            print(f"Computing memory capacity (delay = 2*{arch['tot_units']} = {2*arch['tot_units']})...")
            total_mc, mc_per_delay = compute_memory_capacity(model, device, arch['tot_units'])
            
            if jacobian_matrix is not None:
                # Compute spectral properties of weight-based Jacobian
                spectral_radius, spectral_norm, eigenvals = compute_spectral_properties(jacobian_matrix)
                
                # Compute spectral properties of equilibrium Jacobian
                eq_spectral_radius, eq_spectral_norm, eq_eigenvals = compute_spectral_properties(equilibrium_jacobian)
                
                result = {
                    'architecture': arch['name'],
                    'n_layers': arch['n_layers'],
                    'units_per_layer': arch['units_per_layer'],
                    'target_rho': arch['rho'],  # Add target spectral radius
                    'cycle': use_cycle,  # Add cycle information
                    # Weight-based Jacobian properties
                    'spectral_radius': spectral_radius,
                    'spectral_norm': spectral_norm,
                    'eigenvalues': eigenvals,
                    'jacobian_shape': jacobian_matrix.shape,
                    'total_units': jacobian_matrix.shape[0],
                    # Equilibrium Jacobian properties
                    'eq_spectral_radius': eq_spectral_radius,
                    'eq_spectral_norm': eq_spectral_norm,
                    'eq_eigenvalues': eq_eigenvals,
                    'eq_jacobian_shape': equilibrium_jacobian.shape,
                    # Memory capacity
                    'total_memory_capacity': total_mc,
                    'memory_capacity_per_delay': mc_per_delay,
                    'avg_memory_per_delay': total_mc / len(mc_per_delay) if mc_per_delay else 0.0
                }
                
                results.append(result)
                
                print(f"Target ρ: {arch['rho']:.2f} | Actual Weight ρ: {spectral_radius:.6f}")
                print(f"Weight Jacobian - Spectral Norm: {spectral_norm:.6f}")
                print(f"Equilibrium Jacobian - Spectral Radius: {eq_spectral_radius:.6f}")
                print(f"Equilibrium Jacobian - Spectral Norm: {eq_spectral_norm:.6f}")
                print(f"Total Memory Capacity: {total_mc:.4f}")
                print(f"Avg Memory per Delay: {total_mc / len(mc_per_delay) if mc_per_delay else 0.0:.4f}")
                print(f"  {'🟢 STABLE' if spectral_radius < 1.0 else '🔴 UNSTABLE'} (ρ = {spectral_radius:.4f})")
            else:
                print(f"Failed to extract Jacobian matrix")
                
        except Exception as e:
            print(f"Error analyzing {arch['name']}: {e}")
            continue
    
    return results


def plot_eigenvalues_distribution(results, save_path=None):
    if not results:
        print("No results to plot")
        return None
    
    n_architectures = len(results)
    # Create a grid of subplots for eigenvalue distributions
    cols = 3
    rows = (n_architectures + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle('Eigenvalues Distribution for All Configurations', fontsize=16, fontweight='bold')
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_architectures > 1 else [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_architectures))
    
    for i, (result, color) in enumerate(zip(results, colors)):
        ax = axes_flat[i]
        eigenvals = result['eigenvalues']
        
        # Highlight non-cycle architectures with different marker
        cycle_label = "" if result.get('cycle', True) else " [NO CYCLE]"
        marker_style = 'o' if result.get('cycle', True) else 's'  # square for no cycle
        edge_color = 'black' if result.get('cycle', True) else 'red'
        edge_width = 0.5 if result.get('cycle', True) else 1.5
        
        # Plot eigenvalues
        ax.scatter(eigenvals.real, eigenvals.imag, alpha=0.7, s=40, c=[color], 
                  edgecolors=edge_color, linewidth=edge_width, marker=marker_style)
        
        # Add unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title(f"{result['n_layers']} Layers ({result['units_per_layer']} units/layer){cycle_label}\nρ={result['spectral_radius']:.4f}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set reasonable axis limits
        max_real = np.max(np.abs(eigenvals.real))
        max_imag = np.max(np.abs(eigenvals.imag))
        max_val = max(max_real, max_imag, 1.2)
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        
        # Add text with number of eigenvalues
        ax.text(0.02, 0.98, f'n={len(eigenvals)}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_architectures, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Eigenvalues distribution plot saved to: {save_path}")
    
    return fig


def plot_memory_capacity_vs_delay(results, save_path=None):
    if not results:
        print("No results to plot")
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Memory Capacity vs Delay for All Configurations', fontsize=16, fontweight='bold')
    
    # Get maximum delay across all configurations
    max_delays = []
    for result in results:
        if result['memory_capacity_per_delay']:
            max_delays.append(max(result['memory_capacity_per_delay'].keys()))
    
    if not max_delays:
        return None
    
    max_delay = max(max_delays)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (result, color) in enumerate(zip(results, colors)):
        if result['memory_capacity_per_delay']:
            delays = list(range(1, max_delay + 1))
            mc_values = [result['memory_capacity_per_delay'].get(d, 0) for d in delays]
            
            ax.plot(delays, mc_values, 'o-', alpha=0.7, linewidth=2, markersize=4,
                   label=f"{result['n_layers']} Layers ({result['units_per_layer']} units/layer)", 
                   color=color)
    
    ax.set_xlabel('Delay', fontsize=12)
    ax.set_ylabel('Memory Capacity', fontsize=12)
    ax.set_title('Memory Capacity vs Delay', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add vertical lines for key delays
    for units in [10, 20, 50, 100]:
        if units <= max_delay:
            ax.axvline(x=units, color='gray', linestyle=':', alpha=0.5)
            ax.text(units, ax.get_ylim()[1]*0.9, f'{units}', rotation=90, 
                   verticalalignment='top', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Memory capacity vs delay plot saved to: {save_path}")
    
    return fig


def plot_spectral_properties_vs_configuration(results, save_path=None):
    
    if not results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Spectral Properties vs Configuration (Adding More Layers)', fontsize=16, fontweight='bold')
    
    # Extract data
    n_layers = [r['n_layers'] for r in results]
    spectral_radii = [r['spectral_radius'] for r in results]
    spectral_norms = [r['spectral_norm'] for r in results]
    eq_spectral_radii = [r['eq_spectral_radius'] for r in results]
    eq_spectral_norms = [r['eq_spectral_norm'] for r in results]
    memory_capacities = [r['total_memory_capacity'] for r in results]
    
    # Plot 1: Spectral Radius vs Number of Layers
    axes[0, 0].plot(n_layers, spectral_radii, 'bo-', linewidth=2, markersize=8, alpha=0.7, label='Weight Jacobian')
    axes[0, 0].plot(n_layers, eq_spectral_radii, 'ro-', linewidth=2, markersize=8, alpha=0.7, label='Equilibrium Jacobian')
    axes[0, 0].set_xlabel('Number of Layers')
    axes[0, 0].set_ylabel('Spectral Radius (ρ)')
    axes[0, 0].set_title('Spectral Radius vs Number of Layers')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    axes[0, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Stability Threshold (ρ=1)')
    axes[0, 0].legend()
    
    # Add text annotations for each point
    for i, (x, y) in enumerate(zip(n_layers, spectral_radii)):
        axes[0, 0].annotate(f'{results[i]["units_per_layer"]}u', (x, y), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 2: Spectral Norm vs Number of Layers
    axes[0, 1].plot(n_layers, spectral_norms, 'go-', linewidth=2, markersize=8, alpha=0.7, label='Weight Jacobian')
    axes[0, 1].plot(n_layers, eq_spectral_norms, 'mo-', linewidth=2, markersize=8, alpha=0.7, label='Equilibrium Jacobian')
    axes[0, 1].set_xlabel('Number of Layers')
    axes[0, 1].set_ylabel('Spectral Norm (||·||₂)')
    axes[0, 1].set_title('Spectral Norm vs Number of Layers')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    
    # Add text annotations
    for i, (x, y) in enumerate(zip(n_layers, spectral_norms)):
        axes[0, 1].annotate(f'{results[i]["units_per_layer"]}u', (x, y), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 3: Memory Capacity vs Number of Layers
    axes[1, 0].plot(n_layers, memory_capacities, 'co-', linewidth=2, markersize=8, alpha=0.7)
    axes[1, 0].set_xlabel('Number of Layers')
    axes[1, 0].set_ylabel('Total Memory Capacity')
    axes[1, 0].set_title('Memory Capacity vs Number of Layers')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Add text annotations
    for i, (x, y) in enumerate(zip(n_layers, memory_capacities)):
        axes[1, 0].annotate(f'{results[i]["units_per_layer"]}u', (x, y), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 4: Combined comparison - multiple y-axes
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot spectral radius
    line1 = ax1.plot(n_layers, spectral_radii, 'b-o', linewidth=2, markersize=6, alpha=0.7, label='Spectral Radius')
    ax1.set_xlabel('Number of Layers')
    ax1.set_ylabel('Spectral Radius', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot spectral norm
    line2 = ax2.plot(n_layers, spectral_norms, 'g-s', linewidth=2, markersize=6, alpha=0.7, label='Spectral Norm')
    ax2.set_ylabel('Spectral Norm', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Plot memory capacity
    line3 = ax3.plot(n_layers, memory_capacities, 'r-^', linewidth=2, markersize=6, alpha=0.7, label='Memory Capacity')
    ax3.set_ylabel('Memory Capacity', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title('Combined: Spectral Properties & Memory Capacity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectral properties vs configuration plot saved to: {save_path}")
    
    return fig



def run_spectral_analysis(use_ron=False):
    """Main function to run the spectral analysis."""
    print("="*80)
    
    # Set device - force CPU usage
    device = torch.device("cpu")
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run analysis
    results = analyze_reservoir_spectral_properties(device, use_ron=use_ron)
    
    # Determine model type for file naming
    model_type = "ron" if use_ron else "esn"
    path = f'experiments/results_analysis_{model_type}'
    
    # Create the directory if it doesn't exist
    import os
    os.makedirs(path, exist_ok=True)
    
    # Save detailed log text file
    log_path = os.path.join(path, f'spectral_analysis_log_{model_type}.txt')
    with open(log_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write(f"Spectral Analysis Results - {'RON' if use_ron else 'ESN'}\n")
        f.write("="*120 + "\n\n")
        
        # Summary table
        f.write(f"{'Config':<25} {'L':<3} {'U/L':<4} {'Cycle':<6} {'Target ρ':<9} {'Actual ρ':<10} {'Eq ρ':<10} {'MC Total':<10} {'MC Avg':<8} {'Status':<8}\n")
        f.write("-" * 130 + "\n")
        
        for result in results:
            status = "STABLE" if result['spectral_radius'] < 1.0 else "UNSTAB"
            cycle_str = "YES" if result.get('cycle', True) else "NO"
            f.write(f"{result['architecture']:<25} "
                  f"{result['n_layers']:<3} "
                  f"{result['units_per_layer']:<4} "
                  f"{cycle_str:<6} "
                  f"{result['target_rho']:<9.2f} "
                  f"{result['spectral_radius']:<10.4f} "
                  f"{result['eq_spectral_radius']:<10.4f} "
                  f"{result['total_memory_capacity']:<10.4f} "
                  f"{result['avg_memory_per_delay']:<8.4f} "
                  f"{status:<8}\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write("Detailed Analysis:\n")
        f.write("="*120 + "\n\n")
        
        f.write("Stability (Spectral Radius < 1):\n")
        stable_weight = sum(1 for r in results if r['spectral_radius'] < 1.0)
        stable_eq = sum(1 for r in results if r['eq_spectral_radius'] < 1.0)
        f.write(f"  Weight Jacobian: {stable_weight}/{len(results)}\n")
        f.write(f"  Equilibrium Jacobian: {stable_eq}/{len(results)}\n\n")
        
        f.write("Memory Capacity:\n")
        best_mc = max(results, key=lambda r: r['total_memory_capacity'])
        worst_mc = min(results, key=lambda r: r['total_memory_capacity'])
        f.write(f"  Best: {best_mc['architecture']} (MC = {best_mc['total_memory_capacity']:.4f})\n")
        f.write(f"  Worst: {worst_mc['architecture']} (MC = {worst_mc['total_memory_capacity']:.4f})\n\n")
        
        # Detailed per-architecture results
        f.write("\n" + "="*120 + "\n")
        f.write("Per-Architecture Detailed Results:\n")
        f.write("="*120 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{i}. {result['architecture']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Number of Layers: {result['n_layers']}\n")
            f.write(f"  Units per Layer: {result['units_per_layer']}\n")
            f.write(f"  Total Units: {result['total_units']}\n")
            f.write(f"  Cycle Connections: {'YES' if result.get('cycle', True) else 'NO'}\n")
            f.write(f"  Target Spectral Radius: {result['target_rho']:.4f}\n")
            f.write(f"\n  Weight Jacobian:\n")
            f.write(f"    Spectral Radius: {result['spectral_radius']:.6f}\n")
            f.write(f"    Spectral Norm: {result['spectral_norm']:.6f}\n")
            f.write(f"    Matrix Shape: {result['jacobian_shape']}\n")
            f.write(f"\n  Equilibrium Jacobian:\n")
            f.write(f"    Spectral Radius: {result['eq_spectral_radius']:.6f}\n")
            f.write(f"    Spectral Norm: {result['eq_spectral_norm']:.6f}\n")
            f.write(f"    Matrix Shape: {result['eq_jacobian_shape']}\n")
            f.write(f"\n  Memory Capacity:\n")
            f.write(f"    Total MC: {result['total_memory_capacity']:.6f}\n")
            f.write(f"    Average MC per Delay: {result['avg_memory_per_delay']:.6f}\n")
            f.write(f"    Number of Delays: {len(result['memory_capacity_per_delay'])}\n")
            
            # Show first 10 and last 10 delays
            if result['memory_capacity_per_delay']:
                delays = sorted(result['memory_capacity_per_delay'].keys())
                f.write(f"\n    Memory Capacity by Delay (first 10):\n")
                for delay in delays[:10]:
                    mc_val = result['memory_capacity_per_delay'][delay]
                    f.write(f"      Delay {delay:3d}: {mc_val:.6f}\n")
                
                if len(delays) > 20:
                    f.write(f"      ... ({len(delays)-20} delays omitted) ...\n")
                
                if len(delays) > 10:
                    f.write(f"\n    Memory Capacity by Delay (last 10):\n")
                    for delay in delays[-10:]:
                        mc_val = result['memory_capacity_per_delay'][delay]
                        f.write(f"      Delay {delay:3d}: {mc_val:.6f}\n")
            
            f.write("\n")
    
    print(f"Detailed log saved to: {log_path}")
    
    # Print summary
    print("\n" + "="*120)
    print("="*120)
    
    print(f"{'Config':<25} {'L':<3} {'U/L':<4} {'Cycle':<6} {'Target ρ':<9} {'Actual ρ':<10} {'Eq ρ':<10} {'MC Total':<10} {'MC Avg':<8} {'Status':<8}")
    print("-" * 130)
    
    for result in results:
        status = "🟢STABLE" if result['spectral_radius'] < 1.0 else "🔴UNSTAB"
        cycle_str = "YES" if result.get('cycle', True) else "NO"
        print(f"{result['architecture']:<25} "
              f"{result['n_layers']:<3} "
              f"{result['units_per_layer']:<4} "
              f"{cycle_str:<6} "
              f"{result['target_rho']:<9.2f} "
              f"{result['spectral_radius']:<10.4f} "
              f"{result['eq_spectral_radius']:<10.4f} "
              f"{result['total_memory_capacity']:<10.4f} "
              f"{result['avg_memory_per_delay']:<8.4f} "
              f"{status:<8}")
    
    # Print detailed analysis
    print("\nDetailed Analysis:")
    print("="*120)
    
    print("\n Stability (Spectral Radius < 1):")
    stable_weight = sum(1 for r in results if r['spectral_radius'] < 1.0)
    stable_eq = sum(1 for r in results if r['eq_spectral_radius'] < 1.0)
    print(f"  Weight Jacobian: {stable_weight}/{len(results)}")
    print(f"  Equilibrium Jacobian: {stable_eq}/{len(results)}")
    
    print(f"\n Memory Capacity:")
    best_mc = max(results, key=lambda r: r['total_memory_capacity'])
    worst_mc = min(results, key=lambda r: r['total_memory_capacity'])
    print(f"  Migliore: {best_mc['architecture']} (MC = {best_mc['total_memory_capacity']:.4f})")
    print(f"  Peggiore: {worst_mc['architecture']} (MC = {worst_mc['total_memory_capacity']:.4f})")
    
    mc_values = [r['total_memory_capacity'] for r in results]
    weight_sr = [r['spectral_radius'] for r in results]
    eq_sr = [r['eq_spectral_radius'] for r in results]
    
    print("\nPlotting...")
    
    # Plot 1: Eigenvalues distribution for all configurations
    fig1 = plot_eigenvalues_distribution(results, os.path.join(path, f'eigenvalue_distributions_{model_type}.png'))
    
    # Plot 2: Memory capacity over delay until the last one
    fig2 = plot_memory_capacity_vs_delay(results, os.path.join(path, f'memory_capacity_vs_delay_{model_type}.png'))
    
    # Plot 3: Spectral properties vs configuration (adding more layers)
    fig3 = plot_spectral_properties_vs_configuration(results, os.path.join(path, f'spectral_properties_vs_configuration_{model_type}.png'))
    
    # Show the plots
    if fig1 is not None:
        plt.show()
    if fig2 is not None:
        plt.show() 
    if fig3 is not None:
        plt.show()
    
    print(f"  1.Eigenvalues Distribution: {path}/eigenvalue_distributions_{model_type}.png")
    print(f"  2.Memory Capacity vs Delay: {path}/memory_capacity_vs_delay_{model_type}.png") 
    print(f"  3.Spectral Properties vs Configuration: {path}/spectral_properties_vs_configuration_{model_type}.png")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spectral Analysis of Reservoir Architectures')
    parser.add_argument('--ron', action='store_true', 
                        help='Use RON (Randomized Oscillators Network) instead of ESN')
    args = parser.parse_args()
    
    results = run_spectral_analysis(use_ron=args.ron)