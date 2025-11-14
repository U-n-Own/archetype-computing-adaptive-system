"""
Unified npCIFAR-10 Testing: ESN, DeepESN, RON, DeepRON
Tests both 1-layer and multi-layer architectures with various configurations:
- Standard 1-layer (ESN/RON baseline)
- N-layer with antisymmetric coupling (search coupling strength)
- N-layer with cycle topology

This version combines ESN and RON testing in a single script with consistent
seeding and evaluation methodology for the npCIFAR-10 task (CIFAR-10 with 
row-wise sequential presentation and random suffix to 1000 timesteps).
"""
import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork
)
from acds.benchmarks import get_cifar10_data

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.utils import set_seed

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test ESN and RON architectures on npCIFAR-10")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--model_type", type=str, default="both", 
                    choices=["esn", "ron", "both"],
                    help="Model type to test: esn, ron, or both")
parser.add_argument("--n_hid", type=int, default=500, help="Total number of hidden units")
parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
parser.add_argument("--trials", type=int, default=3, help="Number of trials per configuration")
parser.add_argument("--n_layers", type=int, default=5, help="Number of layers for deep architectures")
parser.add_argument("--dataroot", type=str, default="data", help="Data directory")
parser.add_argument("--resultroot", type=str, default="results_npcifar10_all_models", 
                    help="Results directory")
parser.add_argument("--seq_length", type=int, default=1000, 
                    help="Total sequence length (default: 1000)")
parser.add_argument("--use_test", action="store_true", help="Use test set instead of validation set")
parser.add_argument("--subset_size", type=int, default=None, 
                    help="Use stratified subset of training data (e.g., 5000 instead of 50000 for faster testing)")
parser.add_argument("--skip_baseline_validation", action="store_true", 
                    help="Skip validation/test evaluation for baseline 1-layer models")
args = parser.parse_args()

# Set the seed for reproducibility
set_seed(args.seed)
print(f"Random seed set to: {args.seed}")

# Create results directory
os.makedirs(args.resultroot, exist_ok=True)

print("=" * 80)
print("npCIFAR-10: Unified Model Comparison")
print(f"ESN vs RON | 1-layer vs {args.n_layers}-layer | Antisymmetric vs Cycle")
print("=" * 80)

# Configuration
DATAROOT = args.dataroot
BATCH_SIZE = args.batch_size
N_HID = args.n_hid
N_TRIALS = args.trials
SEQ_LENGTH = args.seq_length

# Hyperparameters - LEAVE BLANK FOR USER TO FILL
# ESN Hyperparameters
ESN_RHO_BASELINE = 900  # Spectral radius for 1-layer ESN
ESN_RHO_VALUES = [900, 90]  # List of rho values to test for 5-layer ESN, e.g., [0.7, 0.8, 0.9]
ESN_INPUT_SCALING = 1  # Input scaling for ESN
ESN_LEAKY = 0.001  # Leaky rate for ESN

# RON Hyperparameters
RON_DT_VALUES = [0.034]  # List of time step values to test (e.g., [0.034, 0.05, 0.1])
RON_RHO_BASELINE = 9  # Spectral radius for 1-layer RON
RON_RHO_VALUES = [9, 0.9]  # List of rho values to test for 5-layer RON
RON_INPUT_SCALING = 0.1  # Input scaling for RON
RON_EPSILON_CENTER_VALUES = [12.7]  # List of center values for epsilon range (e.g., [10, 12.7, 15])
RON_EPSILON_RANGE_VALUES = [0.5]  # List of range values for epsilon (e.g., [0.3, 0.5, 0.7])
RON_GAMMA_CENTER_VALUES = [1.3]  # List of center values for gamma range (e.g., [1.0, 1.3, 1.5])
RON_GAMMA_RANGE_VALUES = [0.5]  # List of range values for gamma (e.g., [0.3, 0.5, 0.7])

# Coupling strength values to test for antisymmetric architectures
COUPLING_VALUES = [0.5, 2, 5, 10, 20]  # e.g., [5.0, 10.0, 20.0, 50.0]

# Validate hyperparameters
if args.model_type in ["esn", "both"]:
    if ESN_RHO_BASELINE is None or ESN_INPUT_SCALING is None or ESN_LEAKY is None:
        raise ValueError("Please set ESN hyperparameters: ESN_RHO_BASELINE, ESN_INPUT_SCALING, ESN_LEAKY")
    if not ESN_RHO_VALUES:
        raise ValueError("Please set ESN_RHO_VALUES as a list of spectral radius values to test")
    if not COUPLING_VALUES:
        raise ValueError("Please set COUPLING_VALUES as a list of coupling strengths to test")

if args.model_type in ["ron", "both"]:
    if None in [RON_RHO_BASELINE, RON_INPUT_SCALING]:
        raise ValueError("Please set all RON hyperparameters")
    if not RON_DT_VALUES:
        raise ValueError("Please set RON_DT_VALUES as a list of time step values to test")
    if not RON_RHO_VALUES:
        raise ValueError("Please set RON_RHO_VALUES as a list of spectral radius values to test")
    if not RON_EPSILON_CENTER_VALUES:
        raise ValueError("Please set RON_EPSILON_CENTER_VALUES as a list")
    if not RON_EPSILON_RANGE_VALUES:
        raise ValueError("Please set RON_EPSILON_RANGE_VALUES as a list")
    if not RON_GAMMA_CENTER_VALUES:
        raise ValueError("Please set RON_GAMMA_CENTER_VALUES as a list")
    if not RON_GAMMA_RANGE_VALUES:
        raise ValueError("Please set RON_GAMMA_RANGE_VALUES as a list")
    if not COUPLING_VALUES:
        raise ValueError("Please set COUPLING_VALUES as a list of coupling strengths to test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# npCIFAR-10 configuration
n_inp = 96  # 32 pixels × 3 RGB channels per row
n_out = 10
image_rows = 32  # Number of rows in image (each becomes a time step)
row_size = 96  # 32 pixels × 3 RGB channels
random_suffix_length = SEQ_LENGTH - image_rows

print("npCIFAR-10 Configuration:")
print(f"  Input dimension per time step: {n_inp} (32 pixels × 3 RGB)")
print(f"  Image data: {image_rows} time steps")
print(f"  Random suffix: {random_suffix_length} time steps")
print(f"  Total sequence length: {SEQ_LENGTH} time steps\n")


def transform_to_npcifar10(images, seq_length=1000, device='cpu'):
    """
    Transform CIFAR-10 images to npCIFAR-10 sequential format.
    
    Args:
        images: torch.Tensor of shape (batch, 3, 32, 32) - RGB images
        seq_length: int - total sequence length (default 1000)
        device: torch device
    
    Returns:
        torch.Tensor of shape (batch, seq_length, 96) where:
        - First 32 time steps contain the image data (row-wise, RGB flattened)
        - Remaining time steps contain random noise
    """
    batch_size = images.shape[0]
    image_rows = 32
    row_size = 96  # 32 pixels × 3 RGB channels
    
    # Initialize output sequence
    sequences = torch.zeros((batch_size, seq_length, row_size), device=device)
    
    # Reshape images from (batch, 3, 32, 32) to (batch, 32, 32, 3)
    images_hwc = images.permute(0, 2, 3, 1)  # (batch, height=32, width=32, channels=3)
    
    # Flatten each row: (batch, 32, 32*3) = (batch, 32, 96)
    image_data = images_hwc.reshape(batch_size, image_rows, row_size)
    
    # Place image data in first 32 time steps
    sequences[:, :image_rows, :] = image_data
    
    # Fill remaining time steps with random noise
    random_suffix_length = seq_length - image_rows
    sequences[:, image_rows:, :] = torch.rand(
        (batch_size, random_suffix_length, row_size), 
        device=device
    ) * 2 - 1  # Scale to [-1, 1]
    
    return sequences


@torch.no_grad()
def evaluate_model(model, data_loader, classifier, scaler, device, seq_length):
    """Evaluate model on a dataset."""
    activations, ys = [], []
    for images, labels in tqdm(data_loader, leave=False):
        images = images.to(device)
        # Transform to npCIFAR-10 format: (batch, seq_length, 96)
        sequences = transform_to_npcifar10(images, seq_length, device)
        output = model(sequences)[-1][0]
        if isinstance(output, list):
            output = output[0]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)


def train_and_evaluate(model_name, model, train_loader, valid_loader, test_loader, device, seq_length, use_test=False, skip_validation=False):
    """Train readout and evaluate model."""
    print(f"\n  Training {model_name}...")
    
    # Extract training activations
    activations, ys = [], []
    for images, labels in tqdm(train_loader, desc="  Extracting activations", leave=False):
        images = images.to(device)
        # Transform to npCIFAR-10 format: (batch, seq_length, 96)
        sequences = transform_to_npcifar10(images, seq_length, device)
        output = model(sequences)[-1][0]
        if isinstance(output, list):
            output = output[0]
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    
    # Train readout
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=2000, verbose=0).fit(activations, ys)
    
    # Evaluate
    train_acc = evaluate_model(model, train_loader, classifier, scaler, device, seq_length)
    if skip_validation:
        valid_acc = 0.0
        test_acc = 0.0
    else:
        valid_acc = evaluate_model(model, valid_loader, classifier, scaler, device, seq_length) if not use_test else 0.0
        test_acc = evaluate_model(model, test_loader, classifier, scaler, device, seq_length) if use_test else 0.0
    
    return {
        'name': model_name,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'test_acc': test_acc,
    }


if __name__ == "__main__":
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, valid_loader, test_loader = get_cifar10_data(
        DATAROOT, 
        bs_train=BATCH_SIZE,
        bs_test=BATCH_SIZE,
        subset_size=args.subset_size,
        stratify=True,
        seed=args.seed
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} train, "
          f"{len(valid_loader.dataset)} valid, {len(test_loader.dataset)} test\n")
    
    results = []
    
    # ========================================
    # ESN MODELS
    # ========================================
    if args.model_type in ["esn", "both"]:
        print("\n" + "="*80)
        print("TESTING ESN MODELS")
        print("="*80)
        
        # 1-layer ESN Baseline
        print(f"\n{'='*70}")
        print(f"1-Layer ESN Baseline ({N_TRIALS} trials)")
        print(f"{'='*70}")
        
        baseline_esn_results = []
        for trial in range(N_TRIALS):
            print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
            
            model = DeepReservoir(
                input_size=n_inp,
                tot_units=N_HID,
                spectral_radius=ESN_RHO_BASELINE,
                input_scaling=ESN_INPUT_SCALING,
                inter_scaling=ESN_INPUT_SCALING,
                connectivity_recurrent=N_HID,
                connectivity_input=N_HID,
                connectivity_inter=N_HID,
                leaky=ESN_LEAKY,
                cycle=False,
                concat=False,
                linear=False,
            ).to(device)
            
            result = train_and_evaluate(
                f"ESN 1-layer (trial {trial+1})",
                model, train_loader, valid_loader, test_loader, device, SEQ_LENGTH, args.use_test, args.skip_baseline_validation
            )
            result['trial'] = trial + 1
            result['rho'] = ESN_RHO_BASELINE
            result['model_type'] = 'ESN'
            result['architecture'] = '1-layer'
            baseline_esn_results.append(result)
            
            if args.use_test:
                print(f"  Train: {result['train_acc']:.4f}, Test: {result['test_acc']:.4f}")
            else:
                print(f"  Train: {result['train_acc']:.4f}, Valid: {result['valid_acc']:.4f}")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average baseline results
        baseline_esn_avg = {
            'name': 'ESN 1-layer (avg)',
            'model_type': 'ESN',
            'architecture': '1-layer',
            'train_acc': np.mean([r['train_acc'] for r in baseline_esn_results]),
            'valid_acc': np.mean([r['valid_acc'] for r in baseline_esn_results]),
            'test_acc': np.mean([r['test_acc'] for r in baseline_esn_results]),
            'train_std': np.std([r['train_acc'] for r in baseline_esn_results]),
            'valid_std': np.std([r['valid_acc'] for r in baseline_esn_results]),
            'test_std': np.std([r['test_acc'] for r in baseline_esn_results]),
            'rho': ESN_RHO_BASELINE,
        }
        results.append(baseline_esn_avg)
        if args.use_test:
            print(f"\nBaseline ESN Average: Test {baseline_esn_avg['test_acc']*100:.2f}% ± {baseline_esn_avg['test_std']*100:.2f}%")
        else:
            print(f"\nBaseline ESN Average: Valid {baseline_esn_avg['valid_acc']*100:.2f}% ± {baseline_esn_avg['valid_std']*100:.2f}%")
        
        # N-layer DeepESN with Antisymmetric coupling
        print(f"\n{'='*70}")
        print(f"{args.n_layers}-Layer DeepESN with Antisymmetric Coupling")
        print(f"{'='*70}")
        
        for rho_val in ESN_RHO_VALUES:
            for coup_eps in COUPLING_VALUES:
                print(f"\n  Testing rho={rho_val}, coupling_epsilon={coup_eps}")
                
                antisym_esn_results = []
                for trial in range(N_TRIALS):
                    model = DeepReservoir(
                        input_size=n_inp,
                        tot_units=N_HID,
                        spectral_radius=rho_val,
                        input_scaling=ESN_INPUT_SCALING,
                        inter_scaling=ESN_INPUT_SCALING,
                        connectivity_recurrent=N_HID,
                        connectivity_input=N_HID,
                        connectivity_inter=N_HID,
                        leaky=ESN_LEAKY,
                        cycle=False,
                        concat=True,
                        linear=False,
                        antisymmetric=True,
                        epsilon=coup_eps,
                        n_layers=args.n_layers,
                    ).to(device)
                    
                    result = train_and_evaluate(
                        f"DeepESN {args.n_layers}-layer Antisym (ρ={rho_val}, ε_c={coup_eps}, trial {trial+1})",
                        model, train_loader, valid_loader, test_loader, device, SEQ_LENGTH, args.use_test
                    )
                    result['trial'] = trial + 1
                    result['rho'] = rho_val
                    result['coupling_epsilon'] = coup_eps
                    result['model_type'] = 'ESN'
                    result['architecture'] = '5-layer-antisymmetric'
                    antisym_esn_results.append(result)
                    
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Average results
                antisym_esn_avg = {
                    'name': f'DeepESN {args.n_layers}-layer Antisym (ρ={rho_val}, ε_c={coup_eps}, avg)',
                    'model_type': 'ESN',
                    'architecture': f'{args.n_layers}-layer-antisymmetric',
                    'train_acc': np.mean([r['train_acc'] for r in antisym_esn_results]),
                    'valid_acc': np.mean([r['valid_acc'] for r in antisym_esn_results]),
                    'test_acc': np.mean([r['test_acc'] for r in antisym_esn_results]),
                    'train_std': np.std([r['train_acc'] for r in antisym_esn_results]),
                    'valid_std': np.std([r['valid_acc'] for r in antisym_esn_results]),
                    'test_std': np.std([r['test_acc'] for r in antisym_esn_results]),
                    'rho': rho_val,
                    'coupling_epsilon': coup_eps,
                }
                results.append(antisym_esn_avg)
                if args.use_test:
                    print(f"    Average: Test {antisym_esn_avg['test_acc']*100:.2f}% ± {antisym_esn_avg['test_std']*100:.2f}%")
                else:
                    print(f"    Average: Valid {antisym_esn_avg['valid_acc']*100:.2f}% ± {antisym_esn_avg['valid_std']*100:.2f}%")
        
        # N-layer DeepESN with Cycle topology
        print(f"\n{'='*70}")
        print(f"{args.n_layers}-Layer DeepESN with Cycle Topology")
        print(f"{'='*70}")
        
        for rho_val in ESN_RHO_VALUES:
            print(f"\n  Testing rho={rho_val}")
            
            cycle_esn_results = []
            for trial in range(N_TRIALS):
                model = DeepReservoir(
                    input_size=n_inp,
                    tot_units=N_HID,
                    spectral_radius=rho_val,
                    input_scaling=ESN_INPUT_SCALING,
                    inter_scaling=ESN_INPUT_SCALING,
                    connectivity_recurrent=N_HID,
                    connectivity_input=N_HID,
                    connectivity_inter=N_HID,
                    leaky=ESN_LEAKY,
                    cycle=True,
                    concat=True,
                    linear=False,
                    n_layers=args.n_layers,
                ).to(device)
                
                result = train_and_evaluate(
                    f"DeepESN {args.n_layers}-layer Cycle (ρ={rho_val}, trial {trial+1})",
                    model, train_loader, valid_loader, test_loader, device, SEQ_LENGTH, args.use_test
                )
                result['trial'] = trial + 1
                result['rho'] = rho_val
                result['model_type'] = 'ESN'
                result['architecture'] = '5-layer-cycle'
                cycle_esn_results.append(result)
                
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Average results
            cycle_esn_avg = {
                'name': f'DeepESN {args.n_layers}-layer Cycle (ρ={rho_val}, avg)',
                'model_type': 'ESN',
                'architecture': f'{args.n_layers}-layer-cycle',
                'train_acc': np.mean([r['train_acc'] for r in cycle_esn_results]),
                'valid_acc': np.mean([r['valid_acc'] for r in cycle_esn_results]),
                'test_acc': np.mean([r['test_acc'] for r in cycle_esn_results]),
                'train_std': np.std([r['train_acc'] for r in cycle_esn_results]),
                'valid_std': np.std([r['valid_acc'] for r in cycle_esn_results]),
                'test_std': np.std([r['test_acc'] for r in cycle_esn_results]),
                'rho': rho_val,
            }
            results.append(cycle_esn_avg)
            if args.use_test:
                print(f"    Average: Test {cycle_esn_avg['test_acc']*100:.2f}% ± {cycle_esn_avg['test_std']*100:.2f}%")
            else:
                print(f"    Average: Valid {cycle_esn_avg['valid_acc']*100:.2f}% ± {cycle_esn_avg['valid_std']*100:.2f}%")
    
    # ========================================
    # RON MODELS
    # ========================================
    if args.model_type in ["ron", "both"]:
        print("\n" + "="*80)
        print("TESTING RON MODELS")
        print("="*80)
        
        # 1-layer RON Baseline - Test different dt, epsilon_center, epsilon_range, gamma_center, gamma_range
        print(f"\n{'='*70}")
        print(f"1-Layer RON Baseline ({N_TRIALS} trials)")
        print(f"Testing dt, epsilon_center, epsilon_range, gamma_center, gamma_range combinations")
        print(f"{'='*70}")
        
        for dt_val in RON_DT_VALUES:
            for eps_center in RON_EPSILON_CENTER_VALUES:
                for eps_range in RON_EPSILON_RANGE_VALUES:
                    for gamma_center in RON_GAMMA_CENTER_VALUES:
                        for gamma_range in RON_GAMMA_RANGE_VALUES:
                            eps_min = eps_center - eps_range / 2.0
                            eps_max = eps_center + eps_range / 2.0
                            gamma_min = gamma_center - gamma_range / 2.0
                            gamma_max = gamma_center + gamma_range / 2.0
                            
                            print(f"\n  Testing dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}")
                            
                            baseline_ron_results = []
                            for trial in range(N_TRIALS):
                                model = RandomizedOscillatorsNetwork(
                                    n_inp=n_inp,
                                    n_hid=N_HID,
                                    dt=dt_val,
                                    gamma=(gamma_min, gamma_max),
                                    epsilon=(eps_min, eps_max),
                                    rho=RON_RHO_BASELINE,
                                    input_scaling=RON_INPUT_SCALING,
                                    topology="full",
                                    device=device,
                                ).to(device)
                                
                                result = train_and_evaluate(
                                    f"RON 1-layer (dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}, trial {trial+1})",
                                    model, train_loader, valid_loader, test_loader, device, SEQ_LENGTH, args.use_test, args.skip_baseline_validation
                                )
                                result['trial'] = trial + 1
                                result['rho'] = RON_RHO_BASELINE
                                result['dt'] = dt_val
                                result['epsilon_center'] = eps_center
                                result['epsilon_range'] = eps_range
                                result['gamma_center'] = gamma_center
                                result['gamma_range'] = gamma_range
                                result['model_type'] = 'RON'
                                result['architecture'] = '1-layer'
                                baseline_ron_results.append(result)
                                
                                del model
                                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                            # Average baseline results
                            baseline_ron_avg = {
                                'name': f'RON 1-layer (dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}, avg)',
                                'model_type': 'RON',
                                'architecture': '1-layer',
                                'train_acc': np.mean([r['train_acc'] for r in baseline_ron_results]),
                                'valid_acc': np.mean([r['valid_acc'] for r in baseline_ron_results]),
                                'test_acc': np.mean([r['test_acc'] for r in baseline_ron_results]),
                                'train_std': np.std([r['train_acc'] for r in baseline_ron_results]),
                                'valid_std': np.std([r['valid_acc'] for r in baseline_ron_results]),
                                'test_std': np.std([r['test_acc'] for r in baseline_ron_results]),
                                'rho': RON_RHO_BASELINE,
                                'dt': dt_val,
                                'epsilon_center': eps_center,
                                'epsilon_range': eps_range,
                                'gamma_center': gamma_center,
                                'gamma_range': gamma_range,
                            }
                            results.append(baseline_ron_avg)
                            if args.use_test:
                                print(f"    Average: Test {baseline_ron_avg['test_acc']*100:.2f}% ± {baseline_ron_avg['test_std']*100:.2f}%")
                            else:
                                print(f"    Average: Valid {baseline_ron_avg['valid_acc']*100:.2f}% ± {baseline_ron_avg['valid_std']*100:.2f}%")
        
        # N-layer DeepRON with Antisymmetric coupling
        print(f"\n{'='*70}")
        print(f"{args.n_layers}-Layer DeepRON with Antisymmetric Coupling")
        print(f"Testing dt, epsilon_center, epsilon_range, gamma_center, gamma_range, rho, and coupling combinations")
        print(f"{'='*70}")
        
        for dt_val in RON_DT_VALUES:
            for eps_center in RON_EPSILON_CENTER_VALUES:
                for eps_range in RON_EPSILON_RANGE_VALUES:
                    for gamma_center in RON_GAMMA_CENTER_VALUES:
                        for gamma_range in RON_GAMMA_RANGE_VALUES:
                            eps_min = eps_center - eps_range / 2.0
                            eps_max = eps_center + eps_range / 2.0
                            gamma_min = gamma_center - gamma_range / 2.0
                            gamma_max = gamma_center + gamma_range / 2.0
                            
                            for rho_val in RON_RHO_VALUES:
                                for coup_eps in COUPLING_VALUES:
                                    print(f"\n  Testing dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}, rho={rho_val}, coupling_ε={coup_eps}")
                                    
                                    antisym_ron_results = []
                                    for trial in range(N_TRIALS):
                                        model = DeepRandomizedOscillatorsNetwork(
                                            n_inp=n_inp,
                                            total_units=N_HID,
                                            n_layers=args.n_layers,
                                            dt=dt_val,
                                            gamma=(gamma_min, gamma_max),
                                            epsilon=(eps_min, eps_max),
                                            rho=rho_val,
                                            input_scaling=RON_INPUT_SCALING,
                                            inter_scaling=RON_INPUT_SCALING,
                                            topology="full",
                                            concat=True,
                                            antisymmetric_coupling=True,
                                            coupling_epsilon=coup_eps,
                                            device=device,
                                        ).to(device)
                                        
                                        result = train_and_evaluate(
                                            f"DeepRON {args.n_layers}-layer Antisym (dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}, ρ={rho_val}, coup_ε={coup_eps}, trial {trial+1})",
                                            model, train_loader, valid_loader, test_loader, device, SEQ_LENGTH, args.use_test
                                        )
                                        result['trial'] = trial + 1
                                        result['rho'] = rho_val
                                        result['dt'] = dt_val
                                        result['epsilon_center'] = eps_center
                                        result['epsilon_range'] = eps_range
                                        result['gamma_center'] = gamma_center
                                        result['gamma_range'] = gamma_range
                                        result['coupling_epsilon'] = coup_eps
                                        result['model_type'] = 'RON'
                                        result['architecture'] = '5-layer-antisymmetric'
                                        antisym_ron_results.append(result)
                                        
                                        del model
                                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                                    
                                    # Average results
                                    antisym_ron_avg = {
                                        'name': f'DeepRON {args.n_layers}-layer Antisym (dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}, ρ={rho_val}, coup_ε={coup_eps}, avg)',
                                        'model_type': 'RON',
                                        'architecture': f'{args.n_layers}-layer-antisymmetric',
                                        'train_acc': np.mean([r['train_acc'] for r in antisym_ron_results]),
                                        'valid_acc': np.mean([r['valid_acc'] for r in antisym_ron_results]),
                                        'test_acc': np.mean([r['test_acc'] for r in antisym_ron_results]),
                                        'train_std': np.std([r['train_acc'] for r in antisym_ron_results]),
                                        'valid_std': np.std([r['valid_acc'] for r in antisym_ron_results]),
                                        'test_std': np.std([r['test_acc'] for r in antisym_ron_results]),
                                        'rho': rho_val,
                                        'dt': dt_val,
                                        'epsilon_center': eps_center,
                                        'epsilon_range': eps_range,
                                        'gamma_center': gamma_center,
                                        'gamma_range': gamma_range,
                                        'coupling_epsilon': coup_eps,
                                    }
                                    results.append(antisym_ron_avg)
                                    if args.use_test:
                                        print(f"    Average: Test {antisym_ron_avg['test_acc']*100:.2f}% ± {antisym_ron_avg['test_std']*100:.2f}%")
                                    else:
                                        print(f"    Average: Valid {antisym_ron_avg['valid_acc']*100:.2f}% ± {antisym_ron_avg['valid_std']*100:.2f}%")
        
        # N-layer DeepRON with Cycle topology
        print(f"\n{'='*70}")
        print(f"{args.n_layers}-Layer DeepRON with Cycle Topology")
        print(f"Testing dt, epsilon_center, epsilon_range, gamma_center, gamma_range, and rho combinations")
        print(f"{'='*70}")
        
        for dt_val in RON_DT_VALUES:
            for eps_center in RON_EPSILON_CENTER_VALUES:
                for eps_range in RON_EPSILON_RANGE_VALUES:
                    for gamma_center in RON_GAMMA_CENTER_VALUES:
                        for gamma_range in RON_GAMMA_RANGE_VALUES:
                            eps_min = eps_center - eps_range / 2.0
                            eps_max = eps_center + eps_range / 2.0
                            gamma_min = gamma_center - gamma_range / 2.0
                            gamma_max = gamma_center + gamma_range / 2.0
                            
                            for rho_val in RON_RHO_VALUES:
                                print(f"\n  Testing dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}, rho={rho_val}")
                                
                                cycle_ron_results = []
                                for trial in range(N_TRIALS):
                                    model = DeepRandomizedOscillatorsNetwork(
                                        n_inp=n_inp,
                                        total_units=N_HID,
                                        n_layers=args.n_layers,
                                        dt=dt_val,
                                        gamma=(gamma_min, gamma_max),
                                        epsilon=(eps_min, eps_max),
                                        rho=rho_val,
                                        input_scaling=RON_INPUT_SCALING,
                                        inter_scaling=RON_INPUT_SCALING,
                                        topology="full",
                                        concat=True,
                                        cycle=True,
                                        antisymmetric_coupling=False,
                                        device=device,
                                    ).to(device)
                                    
                                    result = train_and_evaluate(
                                        f"DeepRON {args.n_layers}-layer Cycle (dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}, ρ={rho_val}, trial {trial+1})",
                                        model, train_loader, valid_loader, test_loader, device, SEQ_LENGTH, args.use_test
                                    )
                                    result['trial'] = trial + 1
                                    result['rho'] = rho_val
                                    result['dt'] = dt_val
                                    result['epsilon_center'] = eps_center
                                    result['epsilon_range'] = eps_range
                                    result['gamma_center'] = gamma_center
                                    result['gamma_range'] = gamma_range
                                    result['model_type'] = 'RON'
                                    result['architecture'] = '5-layer-cycle'
                                    cycle_ron_results.append(result)
                                    
                                    del model
                                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                                
                                # Average results
                                cycle_ron_avg = {
                                    'name': f'DeepRON {args.n_layers}-layer Cycle (dt={dt_val}, ε_c={eps_center}, ε_r={eps_range}, γ_c={gamma_center}, γ_r={gamma_range}, ρ={rho_val}, avg)',
                                    'model_type': 'RON',
                                    'architecture': f'{args.n_layers}-layer-cycle',
                                    'train_acc': np.mean([r['train_acc'] for r in cycle_ron_results]),
                                    'valid_acc': np.mean([r['valid_acc'] for r in cycle_ron_results]),
                                    'test_acc': np.mean([r['test_acc'] for r in cycle_ron_results]),
                                    'train_std': np.std([r['train_acc'] for r in cycle_ron_results]),
                                    'valid_std': np.std([r['valid_acc'] for r in cycle_ron_results]),
                                    'test_std': np.std([r['test_acc'] for r in cycle_ron_results]),
                                    'rho': rho_val,
                                    'dt': dt_val,
                                    'epsilon_center': eps_center,
                                    'epsilon_range': eps_range,
                                    'gamma_center': gamma_center,
                                    'gamma_range': gamma_range,
                                }
                                results.append(cycle_ron_avg)
                                if args.use_test:
                                    print(f"    Average: Test {cycle_ron_avg['test_acc']*100:.2f}% ± {cycle_ron_avg['test_std']*100:.2f}%")
                                else:
                                    print(f"    Average: Valid {cycle_ron_avg['valid_acc']*100:.2f}% ± {cycle_ron_avg['valid_std']*100:.2f}%")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Sort by test or valid accuracy depending on use_test flag
    sort_key = 'test_acc' if args.use_test else 'valid_acc'
    results_sorted = sorted(results, key=lambda x: x[sort_key], reverse=True)
    
    acc_label = 'Test Acc' if args.use_test else 'Valid Acc'
    print(f"\n{'Rank':<5} {'Model':<10} {'Architecture':<25} {'Config':<50} {acc_label:<20}")
    print("-"*110)
    
    for rank, r in enumerate(results_sorted, 1):
        config_str = f"ρ={r.get('rho', 'N/A'):.3f}"
        if 'dt' in r:
            config_str += f", dt={r['dt']}"
        if 'epsilon_center' in r:
            config_str += f", ε_c={r['epsilon_center']}"
        if 'epsilon_range' in r:
            config_str += f", ε_r={r['epsilon_range']}"
        if 'gamma_center' in r:
            config_str += f", γ_c={r['gamma_center']}"
        if 'gamma_range' in r:
            config_str += f", γ_r={r['gamma_range']}"
        if 'coupling_epsilon' in r:
            config_str += f", coup_ε={r['coupling_epsilon']}"
        
        if args.use_test:
            acc_str = f"{r['test_acc']*100:.2f}%"
            if 'test_std' in r:
                acc_str += f" ± {r['test_std']*100:.2f}%"
        else:
            acc_str = f"{r['valid_acc']*100:.2f}%"
            if 'valid_std' in r:
                acc_str += f" ± {r['valid_std']*100:.2f}%"
        
        print(f"{rank:<5} {r['model_type']:<10} {r['architecture']:<25} {config_str:<30} {acc_str:<20}")
    
    # Save results to file
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.resultroot, f"results_summary_{args.n_layers}layer_{date_str}.txt")
    with open(result_file, 'w') as f:
        f.write("npCIFAR-10 Unified Model Comparison Results\n")
        f.write("="*80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Seed: {args.seed}\n")
        f.write(f"  N_HID: {N_HID}\n")
        f.write(f"  N_TRIALS: {N_TRIALS}\n")
        f.write(f"  BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"  SEQ_LENGTH: {SEQ_LENGTH}\n\n")
        
        if args.model_type in ["esn", "both"]:
            f.write("ESN Configuration:\n")
            f.write(f"  RHO_BASELINE: {ESN_RHO_BASELINE}\n")
            f.write(f"  RHO_VALUES: {ESN_RHO_VALUES}\n")
            f.write(f"  INPUT_SCALING: {ESN_INPUT_SCALING}\n")
            f.write(f"  LEAKY: {ESN_LEAKY}\n\n")
        
        if args.model_type in ["ron", "both"]:
            f.write("RON Configuration:\n")
            f.write(f"  DT_VALUES: {RON_DT_VALUES}\n")
            f.write(f"  RHO_BASELINE: {RON_RHO_BASELINE}\n")
            f.write(f"  RHO_VALUES: {RON_RHO_VALUES}\n")
            f.write(f"  INPUT_SCALING: {RON_INPUT_SCALING}\n")
            f.write(f"  EPSILON_CENTER_VALUES: {RON_EPSILON_CENTER_VALUES}\n")
            f.write(f"  EPSILON_RANGE_VALUES: {RON_EPSILON_RANGE_VALUES}\n")
            f.write(f"  GAMMA_CENTER_VALUES: {RON_GAMMA_CENTER_VALUES}\n")
            f.write(f"  GAMMA_RANGE_VALUES: {RON_GAMMA_RANGE_VALUES}\n\n")
        
        f.write(f"COUPLING_VALUES: {COUPLING_VALUES}\n\n")
        f.write("Results:\n")
        f.write("-"*80 + "\n")
        
        for r in results_sorted:
            f.write(f"{r['name']}\n")
            f.write(f"  Model Type: {r['model_type']}\n")
            f.write(f"  Architecture: {r['architecture']}\n")
            if 'dt' in r:
                f.write(f"  dt: {r['dt']}\n")
            if 'epsilon_center' in r:
                f.write(f"  Epsilon Center: {r['epsilon_center']}\n")
            if 'epsilon_range' in r:
                f.write(f"  Epsilon Range: {r['epsilon_range']}\n")
            if 'gamma_center' in r:
                f.write(f"  Gamma Center: {r['gamma_center']}\n")
            if 'gamma_range' in r:
                f.write(f"  Gamma Range: {r['gamma_range']}\n")
            if 'coupling_epsilon' in r:
                f.write(f"  Coupling Epsilon: {r['coupling_epsilon']}\n")
            if 'rho' in r:
                f.write(f"  Spectral Radius: {r['rho']}\n")
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
            f.write("\n\n")
    
    print(f"\nResults saved to: {result_file}")
    print("="*80)
