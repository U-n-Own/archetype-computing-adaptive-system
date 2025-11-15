#!/usr/bin/env python3
"""
Permuted Sequential MNIST (psMNIST) Classification Experiment

This script tests various reservoir computing models on the psMNIST dataset,
a challenging sequential task where MNIST pixel order is randomly permuted.
"""

import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks import get_psmnist_data


def count_parameters(model):
    """Count total parameters and reservoir parameters in the model.
    
    Returns:
        total_params: Total number of parameters in the model
        reservoir_params: Number of parameters in the reservoir (non-trainable)
        trainable_params: Number of trainable parameters (should be 0 for RC models)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reservoir_params = total_params - trainable_params
    return total_params, reservoir_params, trainable_params


def compute_spectral_properties(matrix):
    """Compute spectral radius and spectral norm of a matrix."""
    eigenvals = np.linalg.eigvals(matrix)
    spectral_radius = np.max(np.abs(eigenvals))
    spectral_norm = np.linalg.norm(matrix, ord=2)  # Largest singular value
    return spectral_radius, spectral_norm, eigenvals


def construct_jacobian(model, device):
    """Construct block matrix form of the Jacobian of the reservoir network.
    
    For cycle networks with n layers, the Jacobian structure is:
    [ W_rec_0     0         0       ...  W_proj_0   ]  ← layer 0 depends on layer n-1
    [ W_proj_1    W_rec_1   0       ...  0          ]  ← layer 1 depends on layer 0
    [ 0           W_proj_2  W_rec_2 ...  0          ]  ← layer 2 depends on layer 1
    ...
    [ 0           0         0       ...  W_rec_{n-1}]  ← layer n-1 depends on layer n-2
    
    where:
    - W_rec_i: recurrent weight matrices of each layer (diagonal blocks)
    - W_proj_i: projection weights from layer i-1 to layer i (sub-diagonal)
    - W_proj_0: projection from layer n-1 to layer 0 (top-right, closing the ring)
    """
    jacobian_blocks = []
    
    # Determine which reservoir attribute to use
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
    if hasattr(model, 'cycle') and model.cycle and model.n_layers > 1:
        for layer_idx in range(model.n_layers):
            prev_layer_idx = (layer_idx - 1) % model.n_layers
            layer = reservoir_layers[layer_idx]
            
            # For ESN
            if hasattr(layer, 'net') and hasattr(layer.net, 'projection_kernel') and layer.net.projection_kernel is not None:
                W_proj = layer.net.projection_kernel.detach().cpu().numpy()
                
                curr_start = sum(jacobian_blocks[j].shape[0] for j in range(layer_idx))
                curr_end = curr_start + jacobian_blocks[layer_idx].shape[0]
                prev_start = sum(jacobian_blocks[j].shape[0] for j in range(prev_layer_idx))
                prev_end = prev_start + jacobian_blocks[prev_layer_idx].shape[0]
                
                if W_proj.shape[0] == (curr_end - curr_start) and W_proj.shape[1] == (prev_end - prev_start):
                    total_jacobian[curr_start:curr_end, prev_start:prev_end] = W_proj
            
            # For RON
            elif hasattr(layer, 'cycle_kernel') and layer.cycle_kernel is not None:
                W_cycle = layer.cycle_kernel.detach().cpu().numpy()
                
                curr_start = sum(jacobian_blocks[j].shape[0] for j in range(layer_idx))
                curr_end = curr_start + jacobian_blocks[layer_idx].shape[0]
                prev_start = sum(jacobian_blocks[j].shape[0] for j in range(prev_layer_idx))
                prev_end = prev_start + jacobian_blocks[prev_layer_idx].shape[0]
                
                if W_cycle.shape[0] == (curr_end - curr_start) and W_cycle.shape[1] == (prev_end - prev_start):
                    total_jacobian[curr_start:curr_end, prev_start:prev_end] = W_cycle
    
    return total_jacobian


def compute_effective_jacobian(jacobian_matrix, leaky_rate):
    """Compute the effective Jacobian with leaky integration.
    
    For leaky integration: h(t) = (1-α)*h(t-1) + α*f(W*h(t-1) + input)
    The effective Jacobian is: J_eff = (1-α)*I + α*W
    """
    if jacobian_matrix is None:
        return None
    
    I = np.eye(jacobian_matrix.shape[0])
    J_eff = (1 - leaky_rate) * I + leaky_rate * jacobian_matrix
    return J_eff


parser = argparse.ArgumentParser(description="psMNIST Sequential Classification")
parser.add_argument("--dataroot", type=str, help="Path to data directory")
parser.add_argument("--resultroot", type=str, help="Path to results directory")
parser.add_argument("--resultsuffix", type=str, default="", help="Suffix for result file name")
parser.add_argument("--n_hid", type=int, default=256, help="Hidden size of reservoir")
parser.add_argument("--batch", type=int, default=1000, help="Batch size")
parser.add_argument("--dt", type=float, default=0.042, help="Step size for RON")
parser.add_argument("--gamma", type=float, default=2.7, help="Gamma parameter for RON")
parser.add_argument("--epsilon", type=float, default=4.7, help="Epsilon parameter for RON")
parser.add_argument("--gamma_range", type=float, default=0.0, help="Gamma range for RON")
parser.add_argument("--epsilon_range", type=float, default=0.0, help="Epsilon range for RON")
parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
parser.add_argument("--esn", action="store_true", help="Use ESN model")
parser.add_argument("--ron", action="store_true", help="Use RON model")
parser.add_argument("--pron", action="store_true", help="Use PRON model")
parser.add_argument("--mspron", action="store_true", help="Use MS-PRON model")
parser.add_argument("--deepron", action="store_true", help="Use DeepRON model")
parser.add_argument("--antisymmetric", action="store_true", help="Use antisymmetric coupling")
parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="Diffusive term")
parser.add_argument("--inp_scaling", type=float, default=1.0, help="Input scaling")
parser.add_argument("--rho", type=float, default=0.99, help="Spectral radius")
parser.add_argument("--leaky", type=float, default=1.0, help="Leaky parameter")
parser.add_argument("--use_test", action="store_true", help="Use test set instead of validation")
parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")
parser.add_argument("--n_layers", type=int, default=1, help="Number of layers")
parser.add_argument("--concat", action="store_true", help="Concatenate layer outputs for readout")
parser.add_argument("--cycle", action="store_true", help="Use cycle topology for deep reservoirs")
parser.add_argument(
    "--topology",
    type=str,
    default="full",
    choices=["full", "ring", "band", "lower", "toeplitz", "orthogonal", "antisymmetric"],
    help="Reservoir topology",
)
parser.add_argument("--sparsity", type=float, default=0.0, help="Reservoir sparsity [0, 1)")
parser.add_argument("--reservoir_scaler", type=float, default=1.0, help="Reservoir scaler")
parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation")
parser.add_argument("--coupling_epsilon", type=float, default=0.4, help="Coupling strength for antisymmetric inter-layer connections")

args = parser.parse_args()

if args.dataroot is None:
    warnings.warn("No dataroot provided. Using current location as default.")
    args.dataroot = os.getcwd()
if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()
assert os.path.exists(args.resultroot), \
    f"{args.resultroot} folder does not exist, please create it and run the script again."
assert 1.0 > args.sparsity >= 0.0, "Sparsity must be in [0, 1)"

@torch.no_grad()
def test(data_loader, classifier, scaler):
    """Evaluate model performance on a dataset."""
    activations, ys = [], []
    for images, labels in tqdm(data_loader, desc="Testing"):
        images = images.to(device)
        # Data is already flattened and permuted from dataset
        images = images.unsqueeze(-1)  # (batch, 784) -> (batch, 784, 1)
        output = model(images)[-1][0]
        # Handle case where output might be a list
        if isinstance(output, list):
            output = output[0]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)


device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)
print("Using device:", device)
print(f"Running {args.trials} trial(s)")
print(f"Permutation seed: {args.seed}")

n_inp = 1  # Sequential input (one pixel at a time)
n_out = 10  # 10 classes

gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

train_accs, valid_accs, test_accs = [], [], []

for trial in range(args.trials):
    print(f"\n{'='*60}")
    print(f"Trial {trial + 1}/{args.trials}")
    print(f"{'='*60}")
    
    # Create model based on args
    if args.esn:
        units_per_layer = args.n_hid // args.n_layers
        model = DeepReservoir(
            input_size=n_inp,
            tot_units=args.n_hid,
            spectral_radius=args.rho,
            input_scaling=args.inp_scaling,
            inter_scaling=args.inp_scaling,
            connectivity_recurrent=units_per_layer,
            connectivity_input=units_per_layer,
            connectivity_inter=units_per_layer,
            leaky=args.leaky,
            cycle=args.cycle,
            concat=args.concat,
            linear=False,
            antisymmetric=args.antisymmetric,
            epsilon=args.coupling_epsilon,
        ).to(device)
    elif args.ron:
        model = RandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.diffusive_gamma,
            args.rho,
            args.inp_scaling,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
            antisymmetric_coupling=args.antisymmetric,
            coupling_epsilon=args.coupling_epsilon,
        ).to(device)
    elif args.pron:
        model = PhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device,
        ).to(device)
    elif args.mspron:
        model = MultistablePhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device,
        ).to(device)
    elif args.deepron:
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=n_inp,
            total_units=args.n_hid,
            dt=args.dt,
            gamma=gamma,
            epsilon=epsilon,
            n_layers=args.n_layers,
            diffusive_gamma=args.diffusive_gamma,
            rho=args.rho,
            input_scaling=args.inp_scaling,
            inter_scaling=args.inp_scaling,
            device=device,
            antisymmetric_coupling=args.antisymmetric,
            coupling_epsilon=args.coupling_epsilon,
            cycle=args.cycle,
            concat=args.concat,
            connectivity_input=args.n_hid // args.n_layers,
            connectivity_inter=args.n_hid // args.n_layers,
        ).to(device)
    else:
        raise ValueError("Please specify a model: --esn, --ron, --pron, --mspron, or --deepron")

    # Count and display parameters
    total_params, reservoir_params, trainable_params = count_parameters(model)
    print(f"\n{'='*60}")
    print("MODEL PARAMETERS")
    print(f"{'='*60}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Reservoir parameters:  {reservoir_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"{'='*60}")

    # Compute spectral properties
    print("\n" + "="*60)
    print("SPECTRAL ANALYSIS")
    print("="*60)
    
    if hasattr(model, 'n_layers'):
        W_tot = construct_jacobian(model, device)
        
        if W_tot is not None:
            # Compute spectral properties of W_tot (weight matrix)
            rho_W, norm_W, _ = compute_spectral_properties(W_tot)
            
            # Compute effective Jacobian with leaky integration
            J_eff = compute_effective_jacobian(W_tot, args.leaky)
            rho_J, norm_J, _ = compute_spectral_properties(J_eff)
            
            print(f"\nConfiguration:")
            print(f"  Model: {'ESN' if args.esn else 'DeepRON' if args.deepron else 'RON'}")
            print(f"  Layers: {model.n_layers}")
            print(f"  Total units: {args.n_hid}")
            print(f"  Units per layer: {args.n_hid // model.n_layers}")
            print(f"  Target spectral radius: {args.rho}")
            print(f"  Leaky rate: {args.leaky}")
            print(f"  Cycle topology: {args.cycle if hasattr(args, 'cycle') else 'N/A'}")
            
            print(f"\nW_tot (Recurrent Weight Matrix):")
            print(f"  Matrix shape: {W_tot.shape}")
            print(f"  Spectral radius ρ(W_tot): {rho_W:.6f}")
            print(f"  Spectral norm ||W_tot||₂: {norm_W:.6f}")
            print(f"  Status: {'✓ STABLE' if rho_W < 1.0 else '✗ UNSTABLE'} (ρ < 1)")
            
            print(f"\nJ_eff (Effective Jacobian with leaky={args.leaky}):")
            print(f"  J_eff = (1-α)*I + α*W_tot, where α = {args.leaky}")
            print(f"  Spectral radius ρ(J_eff): {rho_J:.6f}")
            print(f"  Spectral norm ||J_eff||₂: {norm_J:.6f}")
            print(f"  Status: {'✓ STABLE' if rho_J < 1.0 else '⚠ POTENTIALLY UNSTABLE'} (ρ < 1)")
            
            print(f"\nSpectral Radius Comparison:")
            print(f"  Target ρ (initialization): {args.rho:.6f}")
            print(f"  Actual ρ(W_tot):            {rho_W:.6f} (diff: {rho_W - args.rho:+.6f})")
            print(f"  Effective ρ(J_eff):         {rho_J:.6f} (diff: {rho_J - args.rho:+.6f})")
            
            if args.leaky < 1.0:
                print(f"\n💡 Insight:")
                print(f"  With leaky integration (α={args.leaky}), the effective dynamics")
                print(f"  are governed by J_eff = (1-α)*I + α*W_tot")
                print(f"  This can increase the spectral radius compared to W_tot alone.")
                print(f"  The eigenvalue transformation is: λ_J = (1-α) + α*λ_W")
            
            # Save spectral properties to result file
            spectral_info = (
                f"\nSpectral Properties (Trial {trial + 1}):\n"
                f"  ρ(W_tot): {rho_W:.6f}, ||W_tot||₂: {norm_W:.6f}\n"
                f"  ρ(J_eff): {rho_J:.6f}, ||J_eff||₂: {norm_J:.6f}\n"
            )
        else:
            print("  Could not construct Jacobian matrix for this model.")
            spectral_info = "  Spectral analysis: N/A\n"
    else:
        print("  Model does not support multi-layer spectral analysis.")
        spectral_info = "  Spectral analysis: N/A\n"
    
    print("="*60)

    # Load psMNIST data
    print("\nLoading permuted sequential MNIST dataset...")
    train_loader, valid_loader, test_loader = get_psmnist_data(
        args.dataroot, args.batch, args.batch, seed=args.seed
    )

    # Extract reservoir activations and train readout
    print("Extracting training activations...")
    activations, ys = [], []
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        # Data is already flattened and permuted from dataset
        images = images.unsqueeze(-1)  # (batch, 784) -> (batch, 784, 1)
        output = model(images)[-1][0]
        # Handle case where output might be a list
        if isinstance(output, list):
            output = output[0]
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    
    print("Training readout classifier...")
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000, verbose=0).fit(activations, ys)
    
    # Evaluate
    print("Evaluating...")
    train_acc = test(train_loader, classifier, scaler)
    valid_acc = test(valid_loader, classifier, scaler) if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler) if args.use_test else 0.0
    
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)
    
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Valid Acc: {valid_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

# Save results
if args.ron:
    filename = f"psMNIST_log_RON_{args.topology}{args.resultsuffix}.txt"
elif args.pron:
    filename = f"psMNIST_log_PRON{args.resultsuffix}.txt"
elif args.mspron:
    filename = f"psMNIST_log_MSPRON{args.resultsuffix}.txt"
elif args.esn:
    filename = f"psMNIST_log_ESN{args.resultsuffix}.txt"
elif args.deepron:
    filename = f"psMNIST_log_DEEPRON{args.resultsuffix}.txt"
else:
    filename = f"psMNIST_log{args.resultsuffix}.txt"

filepath = os.path.join(args.resultroot, filename)
with open(filepath, "a") as f:
    ar = ""
    for k, v in vars(args).items():
        ar += f"{str(k)}: {str(v)}, "
    ar += (
        f"train: {[str(round(train_acc, 4)) for train_acc in train_accs]} "
        f"valid: {[str(round(valid_acc, 4)) for valid_acc in valid_accs]} "
        f"test: {[str(round(test_acc, 4)) for test_acc in test_accs]} "
        f"mean/std train: {np.mean(train_accs):.4f}±{np.std(train_accs):.4f} "
        f"mean/std valid: {np.mean(valid_accs):.4f}±{np.std(valid_accs):.4f} "
        f"mean/std test: {np.mean(test_accs):.4f}±{np.std(test_accs):.4f}"
    )
    # Add spectral info if available
    if 'spectral_info' in locals():
        ar += spectral_info
    f.write(ar + "\n")

print(f"\nResults saved to: {filepath}")
print(f"\n{'='*60}")
print("Final Results Summary:")
print(f"{'='*60}")
print(f"Train: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
print(f"Valid: {np.mean(valid_accs):.4f} ± {np.std(valid_accs):.4f}")
print(f"Test:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
print(f"{'='*60}")
