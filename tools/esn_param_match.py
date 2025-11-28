
import math
import torch
import numpy as np
from acds.archetypes.esn import DeepReservoir


def get_units_for_target_params(architecture, n_layers=10, target_params=100000):
    """
    Calculates the required total_units to achieve a target number of parameters.
    Based on Dense matrix implementation in ron.py.
    """
    if architecture == "antisymmetric":
        # Uses h2h, x2h, C_coupling, C_coupling_T_neg (4 dense matrices)
        matrices_per_layer = 4
    elif architecture == "cycle":
        # Uses h2h, cycle_kernel (2 dense matrices) + negligible x2h
        matrices_per_layer = 2
    elif architecture in ["baseline", "baseline_deep"]:
        # Baseline and Baseline_deep: both use h2h, x2h (2 dense matrices)
        # For baseline_deep, scaling is the same as baseline, just with n_layers > 1
        matrices_per_layer = 2
    else: # Standard fallback
        matrices_per_layer = 2

    # Calculate units per layer needed
    # Params approx = n_layers * matrices_per_layer * (units_per_layer^2)
    units_per_layer = np.sqrt(target_params / (n_layers * matrices_per_layer))

    # Round to nearest integer and calculate total units
    total_units = int(round(units_per_layer) * n_layers)

    return total_units

def compute_hidden_size(arch, n_layers, target_params=100_000):
    return get_units_for_target_params(arch, n_layers=n_layers, target_params=target_params)

def main():
    # Example usage
    arch = "antisymmetric"
    n_layers = 5
    target_params = 100000
    n_hid = compute_hidden_size(arch, n_layers, target_params)
    print(f"For architecture {arch} with {n_layers} layers to achieve {target_params} parameters, n_hid should be: {n_hid}")
    
    


if __name__ == "__main__":
    main()
