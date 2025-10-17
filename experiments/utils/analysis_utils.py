"""
Analysis utilities for ESN models
Includes eigenspectrum and Jacobian computation
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple


def get_reservoir_matrices(model) -> List[np.ndarray]:
    """
    Extract recurrent weight matrices from the reservoir
    
    Args:
        model: ESN model with res_layers attribute
        
    Returns:
        List of recurrent weight matrices as numpy arrays
    """
    matrices = []
    
    if hasattr(model, 'res_layers'):
        for layer in model.res_layers:
            if hasattr(layer, 'recurrent_weights'):
                W_rec = layer.recurrent_weights.detach().cpu().numpy()
                matrices.append(W_rec)
    
    return matrices


def compute_jacobian_eigenvalues(
    model,
    sample_input: torch.Tensor,
    device: torch.device,
    max_units: int = 500
) -> np.ndarray:
    """
    Compute eigenvalues of the Jacobian matrix at a given state
    Uses automatic differentiation to compute the Jacobian
    
    Args:
        model: ESN model
        sample_input: Input tensor (batch_size, seq_len, features)
        device: torch device
        max_units: Maximum number of units to use (for computational efficiency)
        
    Returns:
        Eigenvalues of the Jacobian matrix
    """
    model.eval()
    sample_input = sample_input.to(device).requires_grad_(True)
    
    with torch.enable_grad():
        output = model(sample_input)
        if isinstance(output, tuple):
            states = output[0]
        else:
            states = output
        
        # Use final state for Jacobian computation
        final_state = states[:, -1, :]  # (batch, features)
        
        # If too many features, sample a subset
        n_features = final_state.shape[1]
        if n_features > max_units:
            indices = torch.linspace(0, n_features-1, max_units, dtype=torch.long)
            final_state = final_state[:, indices]
            n_features = max_units
        
        # Compute Jacobian for first sample
        final_state = final_state[0]  # (features,)
        
        # Compute Jacobian matrix
        input_dim = sample_input.shape[1] * sample_input.shape[2]
        jacobian = torch.zeros(n_features, input_dim, device=device)
        
        for i in range(n_features):
            if final_state[i].requires_grad:
                grad_outputs = torch.zeros_like(final_state)
                grad_outputs[i] = 1.0
                grads = torch.autograd.grad(
                    final_state, sample_input,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]
                if grads is not None:
                    jacobian[i] = grads[0].reshape(-1)
    
    # Convert to numpy and compute eigenvalues
    jacobian_np = jacobian.detach().cpu().numpy()
    
    # For non-square Jacobian, compute eigenvalues of J @ J.T
    if jacobian_np.shape[0] != jacobian_np.shape[1]:
        gram_matrix = jacobian_np @ jacobian_np.T
        eigenvalues = np.linalg.eigvals(gram_matrix)
        eigenvalues = np.sqrt(np.abs(eigenvalues))  # Singular values
    else:
        eigenvalues = np.linalg.eigvals(jacobian_np)
    
    return eigenvalues


def compute_spectral_properties(model) -> Dict[str, float]:
    """
    Compute spectral properties of the reservoir
    
    Args:
        model: ESN model
        
    Returns:
        Dictionary containing:
            - spectral_radius: Maximum eigenvalue magnitude
            - mean_eigenvalue_mag: Mean of eigenvalue magnitudes
            - std_eigenvalue_mag: Std of eigenvalue magnitudes
            - n_eigenvalues: Total number of eigenvalues
    """
    matrices = get_reservoir_matrices(model)
    
    if not matrices:
        return {
            'spectral_radius': 0.0,
            'mean_eigenvalue_mag': 0.0,
            'std_eigenvalue_mag': 0.0,
            'n_eigenvalues': 0
        }
    
    all_eigenvalues = []
    for W_rec in matrices:
        eigs = np.linalg.eigvals(W_rec)
        all_eigenvalues.extend(eigs)
    
    all_eigenvalues = np.array(all_eigenvalues)
    eigenvalue_magnitudes = np.abs(all_eigenvalues)
    
    return {
        'spectral_radius': float(np.max(eigenvalue_magnitudes)),
        'mean_eigenvalue_mag': float(np.mean(eigenvalue_magnitudes)),
        'std_eigenvalue_mag': float(np.std(eigenvalue_magnitudes)),
        'n_eigenvalues': len(all_eigenvalues)
    }


def analyze_stability(
    model,
    sample_input: torch.Tensor,
    device: torch.device,
    compute_jacobian: bool = True
) -> Dict[str, any]:
    """
    Comprehensive stability analysis of ESN model
    
    Args:
        model: ESN model
        sample_input: Sample input for Jacobian computation
        device: torch device
        compute_jacobian: Whether to compute Jacobian eigenvalues
        
    Returns:
        Dictionary with all stability metrics
    """
    # Compute spectral properties
    spectral_props = compute_spectral_properties(model)
    
    results = {
        'spectral_properties': spectral_props,
        'is_stable': spectral_props['spectral_radius'] <= 1.0,
        'echo_state_property': spectral_props['spectral_radius'] < 1.0,
    }
    
    # Compute Jacobian if requested
    if compute_jacobian:
        try:
            jac_eigs = compute_jacobian_eigenvalues(model, sample_input, device)
            jac_max = float(np.max(np.abs(jac_eigs)))
            results['jacobian_max_eigenvalue'] = jac_max
            results['jacobian_stable'] = jac_max <= 1.0
            results['jacobian_eigenvalues'] = jac_eigs
        except Exception as e:
            results['jacobian_error'] = str(e)
            results['jacobian_max_eigenvalue'] = None
            results['jacobian_stable'] = None
    
    return results
