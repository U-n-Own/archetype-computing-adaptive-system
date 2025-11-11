"""
Test comparing standard 1-layer RON vs 5-layer deep architectures on sMNIST:
- Standard 1-layer RON (baseline)
- 5-layer Antisymmetric RON (with coupling strength search)
- 5-layer Cycle RON

This version uses modular imports:
- ron_visualization.py: Visualization functions
- ron_analysis.py: Weight matrix and spectral analysis
- ron_training.py: Training and evaluation functions
"""
import os
import sys
import argparse
import numpy as np
import torch

from acds.archetypes import RandomizedOscillatorsNetwork, DeepRandomizedOscillatorsNetwork
from acds.benchmarks import get_mnist_data

# Import utility modules
from ron_analysis import analyze_weight_matrix
from ron_training import train_and_evaluate
from ron_visualization import plot_spectral_radius_comparison

# Add parent directory to path to import from experiments
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.utils import set_seed

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test RON architectures on sMNIST")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

# Set the seed for reproducibility
set_seed(args.seed)
print(f"Random seed set to: {args.seed}")

# Create results directory
RESULTS_DIR = "results_smnist_ron_antisymmetric"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("sMNIST: Deep RON Architecture Comparison")
print("1-layer RON | 5-layer Antisymmetric RON | 5-layer Cycle RON")
print("=" * 80)

# Configuration
DATAROOT = "data"
BATCH_SIZE = 1000
N_HID = 500
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
RHO_VALUES = [0.7, 0.8, 0.999]

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
            baseline_analysis = analyze_weight_matrix(
                model_standard, 
                "1-layer RON Baseline",
                RESULTS_DIR
            )
        
        result_standard = train_and_evaluate(
            f"Standard 1-layer RON (trial {trial+1})",
            model_standard,
            train_loader,
            valid_loader,
            test_loader,
            RESULTS_DIR,
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
    coupling_values = [5.0, 10, 20, 50]
    
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
                        RESULTS_DIR,
                        coupling_epsilon=coup_eps
                    )
                
                result_antisym = train_and_evaluate(
                    f"Antisymmetric 5-layer RON (ε_c={coup_eps}, ρ={rho_val}, trial {trial+1})",
                    model_antisym,
                    train_loader,
                    valid_loader,
                    test_loader,
                    RESULTS_DIR,
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
    # Test: 5-layer Cycle RON with different RHO values
    # ========================================
    print("\n" + "="*80)
    print(f"CYCLE ARCHITECTURE: 5-layer Cycle RON ({N_TRIALS} trials each)")
    print("="*80)
    
    for rho_val in RHO_VALUES:
        print(f"\n{'='*70}")
        print(f"Testing 5-layer Cycle RON with RHO = {rho_val}")
        print(f"{'='*70}")
        
        cycle_results = []
        for trial in range(N_TRIALS):
            print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
            
            model_cycle = DeepRandomizedOscillatorsNetwork(
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
                cycle=True,
                antisymmetric_coupling=False,
                device=device,
            ).to(device)
            
            if trial == 0:
                print(f"Model: 5-layer DeepRON with cycle")
                print(f"Parameters: {sum(p.numel() for p in model_cycle.parameters())}")
                
                # Analyze weight matrix (only for first trial)
                cycle_analysis = analyze_weight_matrix(
                    model_cycle, 
                    f"5-layer Cycle RON (ρ={rho_val})",
                    RESULTS_DIR,
                )
            
            result_cycle = train_and_evaluate(
                f"Cycle 5-layer RON (ρ={rho_val}, trial {trial+1})",
                model_cycle,
                train_loader,
                valid_loader,
                test_loader,
                RESULTS_DIR,
                visualize_trajectory=(trial == 0)  # Visualize only first trial
            )
            result_cycle['rho'] = rho_val
            result_cycle['trial'] = trial + 1
            result_cycle['cycle'] = True
            if trial == 0 and cycle_analysis:
                result_cycle['spectral_radius'] = cycle_analysis['spectral_radius']
                result_cycle['condition_number'] = cycle_analysis['condition_number']
            cycle_results.append(result_cycle)
            
            # Clean up
            del model_cycle
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average results for this configuration
        cycle_avg = {
            'name': f'Cycle 5-layer (ρ={rho_val}, avg)',
            'train_acc': np.mean([r['train_acc'] for r in cycle_results]),
            'valid_acc': np.mean([r['valid_acc'] for r in cycle_results]),
            'test_acc': np.mean([r['test_acc'] for r in cycle_results]),
            'train_std': np.std([r['train_acc'] for r in cycle_results]),
            'valid_std': np.std([r['valid_acc'] for r in cycle_results]),
            'test_std': np.std([r['test_acc'] for r in cycle_results]),
            'saturated_pct': np.mean([r['saturated_pct'] for r in cycle_results]),
            'rho': rho_val,
            'cycle': True,
            'failed': False,
        }
        if 'spectral_radius' in cycle_results[0]:
            cycle_avg['spectral_radius'] = cycle_results[0]['spectral_radius']
            cycle_avg['condition_number'] = cycle_results[0]['condition_number']
        results.append(cycle_avg)
        print(f"\nAverage Results (Cycle, ρ={rho_val}):")
        print(f"  Test Accuracy: {cycle_avg['test_acc']*100:.2f}% ± {cycle_avg['test_std']*100:.2f}%")
    
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
        plot_spectral_radius_comparison(valid_results, RESULTS_DIR, baseline_result)
        
        # Sort by test accuracy
        results_sorted = sorted(valid_results, key=lambda x: x['test_acc'], reverse=True)
        
        print(f"\n{'Rank':<5} {'Type':<6} {'Model':<50} {'ε_c':<8} {'ρ_rec':<8} {'ρ_tot':<10} {'Test Acc':<18} {'Valid Acc':<18}")
        print("-"*135)
        print("Type: B=Baseline, A=Antisymmetric, C=Cycle")
        print("-"*135)
        
        for rank, r in enumerate(results_sorted, 1):
            coup_str = f"{r.get('coupling_epsilon', 0):.2f}" if 'coupling_epsilon' in r else "N/A"
            rho_rec_str = f"{r.get('rho', 0):.3f}" if 'rho' in r else "N/A"
            rho_tot_str = f"{r.get('spectral_radius', 0.0):.4f}" if 'spectral_radius' in r else "N/A"
            
            # Determine type marker
            if 'Antisymmetric' in r['name']:
                type_marker = "A"
            elif 'Cycle' in r['name']:
                type_marker = "C"
            else:
                type_marker = "B"  # Baseline
            
            # Format with std if available
            test_str = f"{r['test_acc']*100:.2f}%"
            if 'test_std' in r:
                test_str += f" ± {r['test_std']*100:.2f}%"
            
            valid_str = f"{r['valid_acc']*100:.2f}%"
            if 'valid_std' in r:
                valid_str += f" ± {r['valid_std']*100:.2f}%"
            
            print(f"{rank:<5} {type_marker:<6} {r['name']:<50} {coup_str:<8} {rho_rec_str:<8} {rho_tot_str:<10} "
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
        
        cycle_results_filtered = [r for r in results_sorted if 'Cycle' in r['name'] and 'avg' in r['name']]
        best_cycle = cycle_results_filtered[0] if cycle_results_filtered else None
        
        print(f"\nBest overall: {best['name']}")
        print(f"  Test Accuracy: {best['test_acc']*100:.2f}%")
        if 'coupling_epsilon' in best:
            print(f"  Coupling Epsilon: {best['coupling_epsilon']}")
        if 'cycle' in best:
            print(f"  Cycle: {best['cycle']}")
        
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
        
        if best_cycle:
            print(f"\nBest Cycle (5-layer, avg over {N_TRIALS} trials):")
            print(f"  Test Accuracy: {best_cycle['test_acc']*100:.2f}% ± {best_cycle['test_std']*100:.2f}%")
            print(f"  RHO (recurrent): {best_cycle['rho']}")
            print(f"  Spectral Radius (W_total): {best_cycle.get('spectral_radius', 'N/A')}")
            
            if baseline:
                improvement = (best_cycle['test_acc'] - baseline['test_acc']) * 100
                print(f"\nCycle vs Baseline: {improvement:+.2f} percentage points")
                
                if best_cycle['test_acc'] > baseline['test_acc']:
                    print("✓ Cycle architecture provides improvement!")
                else:
                    print("✗ Cycle architecture does not improve over baseline")
        
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
        f.write("sMNIST RON Architecture Comparison Results\n")
        f.write("Baseline | Antisymmetric | Cycle\n")
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
            if 'cycle' in r and r['cycle']:
                f.write(f"  Cycle: {r['cycle']}\n")
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
