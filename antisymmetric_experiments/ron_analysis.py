"""
Analysis utilities for RON models.

This module provides functions for:
- Weight matrix analysis
- Spectral properties computation
- Eigenvalue and singular value analysis
"""
import numpy as np
from acds.archetypes import RandomizedOscillatorsNetwork, DeepRandomizedOscillatorsNetwork
from ron_visualization import plot_eigenvalue_spectrum


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


def analyze_weight_matrix(model, model_name, results_dir, coupling_epsilon=None):
    """
    Analyze the total recurrent weight matrix of a model.
    
    Args:
        model: Either RandomizedOscillatorsNetwork or DeepRandomizedOscillatorsNetwork
        model_name: Name for display
        results_dir: Directory to save plots
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
        results_dir=results_dir,
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
