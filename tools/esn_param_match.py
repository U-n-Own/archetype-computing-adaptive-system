
import math
import torch
import numpy as np
from acds.archetypes.esn import DeepReservoir


def get_units_for_target_params(architecture, n_layers=10, target_params=100000, input_size=1):
    """
    Calculates units_per_layer solving the quadratic equation:
    a * N^2 + b * N + c = 0
    where N is units_per_layer.
    """
    
    # Coefficients for aN^2 + bN + c = 0
    # a: scales with N^2 (Recurrent / Inter-layer weights)
    # b: scales with N (Input weights)
    # c: -target_params
    
    if architecture == "cycle":
        # Cycle uses: 
        # 1. Recurrent Matrix (N^2)
        # 2. Cycle Projection Matrix (N^2)
        # 3. Input Matrix (Input_dim * N) <-- Applied to ALL layers in your implementation
        
        a = 2 * n_layers
        b = input_size * n_layers 
        c = -target_params

    elif architecture == "antisymmetric":
        # Antisymmetric uses:
        # 1. Recurrent (N^2)
        # 2. Coupling C (N^2)
        # 3. Coupling C_T (N^2)
        # 4. Input/Inter-layer: 
        #    - Layer 0: Input_dim * N
        #    - Layers 1..L: N * N (from prev layer)
        
        # Total approx: L*3N^2 (Rec+C+CT) + (L-1)*N^2 (Inter) + 1*Input_dim*N
        # = (4L - 1) * N^2 + Input_dim * N
        
        a = 4 * n_layers - 1
        b = input_size
        c = -target_params
        
    elif architecture in ["baseline", "baseline_deep"]:
        # DeepESN uses:
        # 1. Recurrent (N^2)
        # 2. Input/Inter-layer:
        #    - Layer 0: Input_dim * N
        #    - Layers 1..L: N * N
        
        # Total: L*N^2 (Rec) + (L-1)*N^2 (Inter) + Input_dim*N
        # = (2L - 1) * N^2 + Input_dim * N
        
        a = 2 * n_layers - 1
        b = input_size
        c = -target_params
        
    else:
        # Fallback default (Naive)
        a = 2 * n_layers
        b = 0
        c = -target_params

    # Quadratic Formula: N = (-b + sqrt(b^2 - 4ac)) / 2a
    delta = b**2 - 4 * a * c
    if delta < 0:
        raise ValueError("Configuration results in negative delta, impossible to satisfy.")
        
    units_per_layer = (-b + np.sqrt(delta)) / (2 * a)
    
    # Calculate total units
    total_units = int(round(units_per_layer) * n_layers)

    return total_units

def compute_hidden_size(arch, n_layers, target_params=100_000, input_size=96):
    return get_units_for_target_params(arch, n_layers=n_layers, target_params=target_params, input_size=input_size)

def main():
    # Example usage
    arch = "antisymmetric"
    n_layers = 5
    target_params = 100000
    n_hid = compute_hidden_size(arch, n_layers, target_params)
    print(f"For architecture {arch} with {n_layers} layers to achieve {target_params} parameters, n_hid should be: {n_hid}")
    
    


if __name__ == "__main__":
    main()
