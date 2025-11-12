"""
Unified psMNIST Testing: ESN, DeepESN, RON, DeepRON
Tests both 1-layer and 5-layer architectures with various configurations:
- Standard 1-layer (ESN/RON baseline)
- 5-layer with antisymmetric coupling (search coupling strength)
- 5-layer with cycle topology

This version combines ESN and RON testing in a single script with consistent
seeding and evaluation methodology for the permuted sequential MNIST task.
"""
import os
import sys
import argparse
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork
)
from acds.benchmarks import get_psmnist_data

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.utils import set_seed

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test ESN and RON architectures on psMNIST")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--model_type", type=str, default="both", 
                    choices=["esn", "ron", "both"],
                    help="Model type to test: esn, ron, or both")
parser.add_argument("--n_hid", type=int, default=500, help="Total number of hidden units")
parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
parser.add_argument("--trials", type=int, default=3, help="Number of trials per configuration")
parser.add_argument("--dataroot", type=str, default="data", help="Data directory")
parser.add_argument("--resultroot", type=str, default="results_psmnist_all_models", 
                    help="Results directory")
parser.add_argument("--use_test", action="store_true", help="Use test set instead of validation set")
args = parser.parse_args()

# Set the seed for reproducibility
set_seed(args.seed)
print(f"Random seed set to: {args.seed}")

# Create results directory
os.makedirs(args.resultroot, exist_ok=True)

print("=" * 80)
print("psMNIST: Unified Model Comparison")
print("ESN vs RON | 1-layer vs 5-layer | Antisymmetric vs Cycle")
print("=" * 80)

# Configuration
DATAROOT = args.dataroot
BATCH_SIZE = args.batch_size
N_HID = args.n_hid
N_TRIALS = args.trials

# Hyperparameters - LEAVE BLANK FOR USER TO FILL
# ESN Hyperparameters
ESN_RHO_BASELINE = 0.999  # Spectral radius for 1-layer ESN
ESN_RHO_VALUES = [0.999]  # List of rho values to test for 5-layer ESN, e.g., [0.7, 0.8, 0.9]
ESN_INPUT_SCALING = 0.1  # Input scaling for ESN
ESN_LEAKY = 0.01  # Leaky rate for ESN

# RON Hyperparameters
RON_DT = 0.076  # Time step for RON
RON_RHO_BASELINE = 0.9  # Spectral radius for 1-layer RON
RON_RHO_VALUES = [0.9]  # List of rho values to test for 5-layer RON
RON_INPUT_SCALING = 1  # Input scaling for RON
RON_EPSILON_CENTER = 0.8  # Center value for epsilon range
RON_EPSILON_RANGE = 1  # Range for epsilon
RON_GAMMA_CENTER = 4  # Center value for gamma range
RON_GAMMA_RANGE = 1  # Range for gamma

# Coupling strength values to test for antisymmetric architectures
COUPLING_VALUES = [5, 10, 20]  # e.g., [5.0, 10.0, 20.0, 50.0]

# Validate hyperparameters
if args.model_type in ["esn", "both"]:
    if ESN_RHO_BASELINE is None or ESN_INPUT_SCALING is None or ESN_LEAKY is None:
        raise ValueError("Please set ESN hyperparameters: ESN_RHO_BASELINE, ESN_INPUT_SCALING, ESN_LEAKY")
    if not ESN_RHO_VALUES:
        raise ValueError("Please set ESN_RHO_VALUES as a list of spectral radius values to test")
    if not COUPLING_VALUES:
        raise ValueError("Please set COUPLING_VALUES as a list of coupling strengths to test")

if args.model_type in ["ron", "both"]:
    if None in [RON_DT, RON_RHO_BASELINE, RON_INPUT_SCALING, 
                RON_EPSILON_CENTER, RON_EPSILON_RANGE, 
                RON_GAMMA_CENTER, RON_GAMMA_RANGE]:
        raise ValueError("Please set all RON hyperparameters")
    if not RON_RHO_VALUES:
        raise ValueError("Please set RON_RHO_VALUES as a list of spectral radius values to test")
    if not COUPLING_VALUES:
        raise ValueError("Please set COUPLING_VALUES as a list of coupling strengths to test")

# Derived RON parameters
if args.model_type in ["ron", "both"]:
    EPSILON_MIN = RON_EPSILON_CENTER - RON_EPSILON_RANGE / 2.0
    EPSILON_MAX = RON_EPSILON_CENTER + RON_EPSILON_RANGE / 2.0
    GAMMA_MIN = RON_GAMMA_CENTER - RON_GAMMA_RANGE / 2.0
    GAMMA_MAX = RON_GAMMA_CENTER + RON_GAMMA_RANGE / 2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

n_inp = 1  # Sequential input (one pixel at a time)
n_out = 10


@torch.no_grad()
def evaluate_model(model, data_loader, classifier, scaler, device):
    """Evaluate model on a dataset."""
    activations, ys = [], []
    for images, labels in tqdm(data_loader, leave=False):
        images = images.to(device)
        # Data is already flattened and permuted from dataset
        images = images.unsqueeze(-1)  # (batch, 784) -> (batch, 784, 1)
        output = model(images)[-1][0]
        if isinstance(output, list):
            output = output[0]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)


def train_and_evaluate(model_name, model, train_loader, valid_loader, test_loader, device, use_test=False):
    """Train readout and evaluate model."""
    print(f"\n  Training {model_name}...")
    
    # Extract training activations
    activations, ys = [], []
    for images, labels in tqdm(train_loader, desc="  Extracting activations", leave=False):
        images = images.to(device)
        # Data is already flattened and permuted from dataset
        images = images.unsqueeze(-1)  # (batch, 784) -> (batch, 784, 1)
        output = model(images)[-1][0]
        if isinstance(output, list):
            output = output[0]
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    
    # Train readout
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000, verbose=0).fit(activations, ys)
    
    # Evaluate
    train_acc = evaluate_model(model, train_loader, classifier, scaler, device)
    valid_acc = evaluate_model(model, valid_loader, classifier, scaler, device) if not use_test else 0.0
    test_acc = evaluate_model(model, test_loader, classifier, scaler, device) if use_test else 0.0
    
    return {
        'name': model_name,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'test_acc': test_acc,
    }


if __name__ == "__main__":
    # Load data
    print("Loading psMNIST dataset...")
    train_loader, valid_loader, test_loader = get_psmnist_data(
        DATAROOT, 
        bs_train=BATCH_SIZE,
        bs_test=BATCH_SIZE,
        seed=args.seed
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} train, "
          f"{len(valid_loader.dataset)} valid, {len(test_loader.dataset)} test")
    print(f"Permutation seed: {args.seed}\n")
    
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
                model, train_loader, valid_loader, test_loader, device, args.use_test
            )
            result['trial'] = trial + 1
            result['rho'] = ESN_RHO_BASELINE
            result['model_type'] = 'ESN'
            result['architecture'] = '1-layer'
            baseline_esn_results.append(result)
            
            print(f"  Train: {result['train_acc']:.4f}, Valid: {result['valid_acc']:.4f}, Test: {result['test_acc']:.4f}")
            
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
        print(f"\nBaseline ESN Average: Test {baseline_esn_avg['test_acc']*100:.2f}% ± {baseline_esn_avg['test_std']*100:.2f}%")
        
        # 5-layer DeepESN with Antisymmetric coupling
        print(f"\n{'='*70}")
        print(f"5-Layer DeepESN with Antisymmetric Coupling")
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
                        connectivity_recurrent=N_HID // 5,
                        connectivity_input=N_HID // 5,
                        connectivity_inter=N_HID // 5,
                        leaky=ESN_LEAKY,
                        cycle=False,
                        concat=True,
                        linear=False,
                        antisymmetric=True,
                        epsilon=coup_eps,
                    ).to(device)
                    
                    result = train_and_evaluate(
                        f"DeepESN 5-layer Antisym (ρ={rho_val}, ε_c={coup_eps}, trial {trial+1})",
                        model, train_loader, valid_loader, test_loader, device, args.use_test
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
                    'name': f'DeepESN 5-layer Antisym (ρ={rho_val}, ε_c={coup_eps}, avg)',
                    'model_type': 'ESN',
                    'architecture': '5-layer-antisymmetric',
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
                print(f"    Average: Test {antisym_esn_avg['test_acc']*100:.2f}% ± {antisym_esn_avg['test_std']*100:.2f}%")
        
        # 5-layer DeepESN with Cycle topology
        print(f"\n{'='*70}")
        print(f"5-Layer DeepESN with Cycle Topology")
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
                    connectivity_recurrent=N_HID // 5,
                    connectivity_input=N_HID // 5,
                    connectivity_inter=N_HID // 5,
                    leaky=ESN_LEAKY,
                    cycle=True,
                    concat=True,
                    linear=False,
                ).to(device)
                
                result = train_and_evaluate(
                    f"DeepESN 5-layer Cycle (ρ={rho_val}, trial {trial+1})",
                    model, train_loader, valid_loader, test_loader, device, args.use_test
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
                'name': f'DeepESN 5-layer Cycle (ρ={rho_val}, avg)',
                'model_type': 'ESN',
                'architecture': '5-layer-cycle',
                'train_acc': np.mean([r['train_acc'] for r in cycle_esn_results]),
                'valid_acc': np.mean([r['valid_acc'] for r in cycle_esn_results]),
                'test_acc': np.mean([r['test_acc'] for r in cycle_esn_results]),
                'train_std': np.std([r['train_acc'] for r in cycle_esn_results]),
                'valid_std': np.std([r['valid_acc'] for r in cycle_esn_results]),
                'test_std': np.std([r['test_acc'] for r in cycle_esn_results]),
                'rho': rho_val,
            }
            results.append(cycle_esn_avg)
            print(f"    Average: Test {cycle_esn_avg['test_acc']*100:.2f}% ± {cycle_esn_avg['test_std']*100:.2f}%")
    
    # ========================================
    # RON MODELS
    # ========================================
    if args.model_type in ["ron", "both"]:
        print("\n" + "="*80)
        print("TESTING RON MODELS")
        print("="*80)
        
        # 1-layer RON Baseline
        print(f"\n{'='*70}")
        print(f"1-Layer RON Baseline ({N_TRIALS} trials)")
        print(f"{'='*70}")
        
        baseline_ron_results = []
        for trial in range(N_TRIALS):
            print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
            
            model = RandomizedOscillatorsNetwork(
                n_inp=n_inp,
                n_hid=N_HID,
                dt=RON_DT,
                gamma=(GAMMA_MIN, GAMMA_MAX),
                epsilon=(EPSILON_MIN, EPSILON_MAX),
                rho=RON_RHO_BASELINE,
                input_scaling=RON_INPUT_SCALING,
                topology="full",
                device=device,
            ).to(device)
            
            result = train_and_evaluate(
                f"RON 1-layer (trial {trial+1})",
                model, train_loader, valid_loader, test_loader, device, args.use_test
            )
            result['trial'] = trial + 1
            result['rho'] = RON_RHO_BASELINE
            result['model_type'] = 'RON'
            result['architecture'] = '1-layer'
            baseline_ron_results.append(result)
            
            print(f"  Train: {result['train_acc']:.4f}, Valid: {result['valid_acc']:.4f}, Test: {result['test_acc']:.4f}")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average baseline results
        baseline_ron_avg = {
            'name': 'RON 1-layer (avg)',
            'model_type': 'RON',
            'architecture': '1-layer',
            'train_acc': np.mean([r['train_acc'] for r in baseline_ron_results]),
            'valid_acc': np.mean([r['valid_acc'] for r in baseline_ron_results]),
            'test_acc': np.mean([r['test_acc'] for r in baseline_ron_results]),
            'train_std': np.std([r['train_acc'] for r in baseline_ron_results]),
            'valid_std': np.std([r['valid_acc'] for r in baseline_ron_results]),
            'test_std': np.std([r['test_acc'] for r in baseline_ron_results]),
            'rho': RON_RHO_BASELINE,
        }
        results.append(baseline_ron_avg)
        print(f"\nBaseline RON Average: Test {baseline_ron_avg['test_acc']*100:.2f}% ± {baseline_ron_avg['test_std']*100:.2f}%")
        
        # 5-layer DeepRON with Antisymmetric coupling
        print(f"\n{'='*70}")
        print(f"5-Layer DeepRON with Antisymmetric Coupling")
        print(f"{'='*70}")
        
        for rho_val in RON_RHO_VALUES:
            for coup_eps in COUPLING_VALUES:
                print(f"\n  Testing rho={rho_val}, coupling_epsilon={coup_eps}")
                
                antisym_ron_results = []
                for trial in range(N_TRIALS):
                    model = DeepRandomizedOscillatorsNetwork(
                        n_inp=n_inp,
                        total_units=N_HID,
                        n_layers=5,
                        dt=RON_DT,
                        gamma=(GAMMA_MIN, GAMMA_MAX),
                        epsilon=(EPSILON_MIN, EPSILON_MAX),
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
                        f"DeepRON 5-layer Antisym (ρ={rho_val}, ε_c={coup_eps}, trial {trial+1})",
                        model, train_loader, valid_loader, test_loader, device, args.use_test
                    )
                    result['trial'] = trial + 1
                    result['rho'] = rho_val
                    result['coupling_epsilon'] = coup_eps
                    result['model_type'] = 'RON'
                    result['architecture'] = '5-layer-antisymmetric'
                    antisym_ron_results.append(result)
                    
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Average results
                antisym_ron_avg = {
                    'name': f'DeepRON 5-layer Antisym (ρ={rho_val}, ε_c={coup_eps}, avg)',
                    'model_type': 'RON',
                    'architecture': '5-layer-antisymmetric',
                    'train_acc': np.mean([r['train_acc'] for r in antisym_ron_results]),
                    'valid_acc': np.mean([r['valid_acc'] for r in antisym_ron_results]),
                    'test_acc': np.mean([r['test_acc'] for r in antisym_ron_results]),
                    'train_std': np.std([r['train_acc'] for r in antisym_ron_results]),
                    'valid_std': np.std([r['valid_acc'] for r in antisym_ron_results]),
                    'test_std': np.std([r['test_acc'] for r in antisym_ron_results]),
                    'rho': rho_val,
                    'coupling_epsilon': coup_eps,
                }
                results.append(antisym_ron_avg)
                print(f"    Average: Test {antisym_ron_avg['test_acc']*100:.2f}% ± {antisym_ron_avg['test_std']*100:.2f}%")
        
        # 5-layer DeepRON with Cycle topology
        print(f"\n{'='*70}")
        print(f"5-Layer DeepRON with Cycle Topology")
        print(f"{'='*70}")
        
        for rho_val in RON_RHO_VALUES:
            print(f"\n  Testing rho={rho_val}")
            
            cycle_ron_results = []
            for trial in range(N_TRIALS):
                model = DeepRandomizedOscillatorsNetwork(
                    n_inp=n_inp,
                    total_units=N_HID,
                    n_layers=5,
                    dt=RON_DT,
                    gamma=(GAMMA_MIN, GAMMA_MAX),
                    epsilon=(EPSILON_MIN, EPSILON_MAX),
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
                    f"DeepRON 5-layer Cycle (ρ={rho_val}, trial {trial+1})",
                    model, train_loader, valid_loader, test_loader, device, args.use_test
                )
                result['trial'] = trial + 1
                result['rho'] = rho_val
                result['model_type'] = 'RON'
                result['architecture'] = '5-layer-cycle'
                cycle_ron_results.append(result)
                
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Average results
            cycle_ron_avg = {
                'name': f'DeepRON 5-layer Cycle (ρ={rho_val}, avg)',
                'model_type': 'RON',
                'architecture': '5-layer-cycle',
                'train_acc': np.mean([r['train_acc'] for r in cycle_ron_results]),
                'valid_acc': np.mean([r['valid_acc'] for r in cycle_ron_results]),
                'test_acc': np.mean([r['test_acc'] for r in cycle_ron_results]),
                'train_std': np.std([r['train_acc'] for r in cycle_ron_results]),
                'valid_std': np.std([r['valid_acc'] for r in cycle_ron_results]),
                'test_std': np.std([r['test_acc'] for r in cycle_ron_results]),
                'rho': rho_val,
            }
            results.append(cycle_ron_avg)
            print(f"    Average: Test {cycle_ron_avg['test_acc']*100:.2f}% ± {cycle_ron_avg['test_std']*100:.2f}%")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Sort by test accuracy
    results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<10} {'Architecture':<25} {'Config':<30} {'Test Acc':<20}")
    print("-"*90)
    
    for rank, r in enumerate(results_sorted, 1):
        config_str = f"ρ={r.get('rho', 'N/A'):.3f}"
        if 'coupling_epsilon' in r:
            config_str += f", ε_c={r['coupling_epsilon']}"
        
        test_str = f"{r['test_acc']*100:.2f}%"
        if 'test_std' in r:
            test_str += f" ± {r['test_std']*100:.2f}%"
        
        print(f"{rank:<5} {r['model_type']:<10} {r['architecture']:<25} {config_str:<30} {test_str:<20}")
    
    # Save results to file
    result_file = os.path.join(args.resultroot, "results_summary.txt")
    with open(result_file, 'w') as f:
        f.write("psMNIST Unified Model Comparison Results\n")
        f.write("="*80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Seed: {args.seed}\n")
        f.write(f"  N_HID: {N_HID}\n")
        f.write(f"  N_TRIALS: {N_TRIALS}\n")
        f.write(f"  BATCH_SIZE: {BATCH_SIZE}\n\n")
        
        if args.model_type in ["esn", "both"]:
            f.write("ESN Configuration:\n")
            f.write(f"  RHO_BASELINE: {ESN_RHO_BASELINE}\n")
            f.write(f"  RHO_VALUES: {ESN_RHO_VALUES}\n")
            f.write(f"  INPUT_SCALING: {ESN_INPUT_SCALING}\n")
            f.write(f"  LEAKY: {ESN_LEAKY}\n\n")
        
        if args.model_type in ["ron", "both"]:
            f.write("RON Configuration:\n")
            f.write(f"  DT: {RON_DT}\n")
            f.write(f"  RHO_BASELINE: {RON_RHO_BASELINE}\n")
            f.write(f"  RHO_VALUES: {RON_RHO_VALUES}\n")
            f.write(f"  INPUT_SCALING: {RON_INPUT_SCALING}\n")
            f.write(f"  EPSILON: ({EPSILON_MIN}, {EPSILON_MAX})\n")
            f.write(f"  GAMMA: ({GAMMA_MIN}, {GAMMA_MAX})\n\n")
        
        f.write(f"COUPLING_VALUES: {COUPLING_VALUES}\n\n")
        f.write("Results:\n")
        f.write("-"*80 + "\n")
        
        for r in results_sorted:
            f.write(f"{r['name']}\n")
            f.write(f"  Model Type: {r['model_type']}\n")
            f.write(f"  Architecture: {r['architecture']}\n")
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
