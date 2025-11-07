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

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks import get_psmnist_data

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
            connectivity_input=args.n_hid // args.n_layers,
            connectivity_inter=args.n_hid // args.n_layers,
        ).to(device)
    else:
        raise ValueError("Please specify a model: --esn, --ron, --pron, --mspron, or --deepron")

    # Load psMNIST data
    print("Loading permuted sequential MNIST dataset...")
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
    f.write(ar + "\n")

print(f"\nResults saved to: {filepath}")
print(f"\n{'='*60}")
print("Final Results Summary:")
print(f"{'='*60}")
print(f"Train: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
print(f"Valid: {np.mean(valid_accs):.4f} ± {np.std(valid_accs):.4f}")
print(f"Test:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
print(f"{'='*60}")
