"""
Parallelized version: Test comparing standard 1-layer RON vs 5-layer antisymmetric RON on sMNIST
with coupling strength search across multiple GPUs.

Uses multiprocessing to distribute experiments across 2 GPUs.
"""
import os
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import json
from datetime import datetime

from acds.archetypes import RandomizedOscillatorsNetwork, DeepRandomizedOscillatorsNetwork
from acds.benchmarks import get_mnist_data


def compute_total_weight_matrix_1layer(model):
    """
    Compute the total recurrent weight matrix for a 1-layer RON.
    """
    W_rec = model.h2h.detach().cpu().numpy()
    singular_values = np.linalg.svd(W_rec, compute_uv=False)
    eigenvalues = np.linalg.eigvals(W_rec)
    spectral_radius = np.max(np.abs(eigenvalues))
    return W_rec, singular_values, spectral_radius


def compute_total_weight_matrix_antisymmetric(model, coupling_epsilon):
    """
    Compute the total recurrent weight matrix for antisymmetric coupled DeepRON.
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
            C = layer.C_coupling.detach().cpu().numpy()
            
            # Forward coupling
            W_total[row_offset:row_offset+layer_size, 
                   row_offset+layer_size:row_offset+layer_size+next_layer_size] = coupling_epsilon * C
            
            # Backward coupling
            W_total[row_offset+layer_size:row_offset+layer_size+next_layer_size,
                   row_offset:row_offset+layer_size] = -coupling_epsilon * C.T
        
        row_offset += layer_size
    
    singular_values = np.linalg.svd(W_total, compute_uv=False)
    eigenvalues = np.linalg.eigvals(W_total)
    spectral_radius = np.max(np.abs(eigenvalues))
    
    return W_total, singular_values, spectral_radius


def run_single_experiment(config, gpu_id):
    """
    Run a single experiment on a specific GPU.
    
    Args:
        config: Dictionary with experiment configuration
        gpu_id: GPU device ID to use
    
    Returns:
        Dictionary with results
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Extract config
    model_type = config['model_type']
    trial = config['trial']
    coupling_epsilon = config.get('coupling_epsilon', None)
    rho = config['rho']
    
    # Configuration
    DATAROOT = config['DATAROOT']
    BATCH_SIZE = config['BATCH_SIZE']
    N_HID = config['N_HID']
    DT = config['DT']
    INP_SCALING = config['INP_SCALING']
    EPSILON_MIN = config['EPSILON_MIN']
    EPSILON_MAX = config['EPSILON_MAX']
    GAMMA_MIN = config['GAMMA_MIN']
    GAMMA_MAX = config['GAMMA_MAX']
    
    n_inp = 1
    n_out = 10
    
    # Load data
    train_loader, valid_loader, test_loader = get_mnist_data(
        DATAROOT, 
        bs_train=BATCH_SIZE,
        bs_test=BATCH_SIZE
    )
    
    # Build model
    if model_type == 'baseline':
        model = RandomizedOscillatorsNetwork(
            n_inp=n_inp,
            n_hid=N_HID,
            dt=DT,
            gamma=(GAMMA_MIN, GAMMA_MAX),
            epsilon=(EPSILON_MIN, EPSILON_MAX),
            rho=rho,
            input_scaling=INP_SCALING,
            topology="full",
            device=device,
        ).to(device)
        model_name = f"1-layer RON (trial {trial})"
    else:  # antisymmetric
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=n_inp,
            total_units=N_HID,
            n_layers=5,
            dt=DT,
            gamma=(GAMMA_MIN, GAMMA_MAX),
            epsilon=(EPSILON_MIN, EPSILON_MAX),
            rho=rho,
            input_scaling=INP_SCALING,
            inter_scaling=INP_SCALING,
            topology="full",
            concat=True,
            antisymmetric_coupling=True,
            coupling_epsilon=coupling_epsilon,
            device=device,
        ).to(device)
        model_name = f"5-layer Antisym (ε_c={coupling_epsilon}, ρ={rho}, trial {trial})"
    
    # Weight matrix analysis (only for first trial)
    spectral_radius = None
    condition_number = None
    if trial == 1:
        try:
            if model_type == 'baseline':
                _, singular_values, spectral_radius = compute_total_weight_matrix_1layer(model)
            else:
                _, singular_values, spectral_radius = compute_total_weight_matrix_antisymmetric(
                    model, coupling_epsilon
                )
            condition_number = singular_values[0] / singular_values[-1]
        except Exception as e:
            print(f"Warning: Weight matrix analysis failed for {model_name}: {e}")
    
    # Training
    try:
        # Collect training activations
        activations, ys = [], []
        for images, labels in train_loader:
            images = images.to(device)
            images = images.view(images.shape[0], -1).unsqueeze(-1)
            
            with torch.no_grad():
                output = model(images)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Get last timestep
                if len(output.shape) == 3:
                    output = output[:, -1, :]
                elif len(output.shape) == 2:
                    pass
                else:
                    output = output.reshape(output.shape[0], -1)
                
                activations.append(output.cpu())
                ys.append(labels)
        
        activations = torch.cat(activations, dim=0).numpy()
        ys = torch.cat(ys, dim=0).numpy()
        
        # Check for NaN/Inf
        if np.isnan(activations).any() or np.isinf(activations).any():
            return {
                'name': model_name,
                'model_type': model_type,
                'trial': trial,
                'coupling_epsilon': coupling_epsilon,
                'rho': rho,
                'failed': True,
                'error': 'NaN or Inf in activations'
            }
        
        # Scale and train classifier
        scaler = preprocessing.StandardScaler().fit(activations)
        activations = scaler.transform(activations)
        
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
        classifier.fit(activations, ys)
        
        # Evaluate
        train_acc = classifier.score(activations, ys)
        
        # Validation
        activations_val, ys_val = [], []
        for images, labels in valid_loader:
            images = images.to(device)
            images = images.view(images.shape[0], -1).unsqueeze(-1)
            
            with torch.no_grad():
                output = model(images)
                if isinstance(output, tuple):
                    output = output[0]
                
                if len(output.shape) == 3:
                    output = output[:, -1, :]
                elif len(output.shape) == 2:
                    pass
                else:
                    output = output.reshape(output.shape[0], -1)
                
                activations_val.append(output.cpu())
                ys_val.append(labels)
        
        activations_val = torch.cat(activations_val, dim=0).numpy()
        activations_val = scaler.transform(activations_val)
        ys_val = torch.cat(ys_val, dim=0).numpy()
        valid_acc = classifier.score(activations_val, ys_val)
        
        # Test
        activations_test, ys_test = [], []
        for images, labels in test_loader:
            images = images.to(device)
            images = images.view(images.shape[0], -1).unsqueeze(-1)
            
            with torch.no_grad():
                output = model(images)
                if isinstance(output, tuple):
                    output = output[0]
                
                if len(output.shape) == 3:
                    output = output[:, -1, :]
                elif len(output.shape) == 2:
                    pass
                else:
                    output = output.reshape(output.shape[0], -1)
                
                activations_test.append(output.cpu())
                ys_test.append(labels)
        
        activations_test = torch.cat(activations_test, dim=0).numpy()
        activations_test = scaler.transform(activations_test)
        ys_test = torch.cat(ys_test, dim=0).numpy()
        test_acc = classifier.score(activations_test, ys_test)
        
        # Saturation check
        saturated = np.sum(np.abs(activations) > 0.99) / activations.size
        
        result = {
            'name': model_name,
            'model_type': model_type,
            'trial': trial,
            'coupling_epsilon': coupling_epsilon,
            'rho': rho,
            'train_acc': float(train_acc),
            'valid_acc': float(valid_acc),
            'test_acc': float(test_acc),
            'saturated_pct': float(saturated * 100),
            'failed': False,
            'gpu_id': gpu_id
        }
        
        if spectral_radius is not None:
            result['spectral_radius'] = float(spectral_radius)
        if condition_number is not None:
            result['condition_number'] = float(condition_number)
        
        print(f"[GPU {gpu_id}] {model_name}: Test Acc = {test_acc*100:.2f}%")
        
        return result
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR in {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': model_name,
            'model_type': model_type,
            'trial': trial,
            'coupling_epsilon': coupling_epsilon,
            'rho': rho,
            'failed': True,
            'error': str(e),
            'gpu_id': gpu_id
        }


def main():
    print("=" * 80)
    print("PARALLELIZED sMNIST: 1-layer RON vs Antisymmetric 5-layer RON")
    print("Running on 2 GPUs in parallel")
    print("=" * 80)
    
    # Configuration
    DATAROOT = "data"
    BATCH_SIZE = 1000
    N_HID = 500
    N_TRIALS = 3
    
    DT = 0.042
    RHO_BASELINE = 0.9
    INP_SCALING = 1.0
    EPSILON_CENTER = 0.51
    EPSILON_RANGE = 0.5
    GAMMA_CENTER = 2.7
    GAMMA_RANGE = 1.0
    
    EPSILON_MIN = EPSILON_CENTER - EPSILON_RANGE
    EPSILON_MAX = EPSILON_CENTER + EPSILON_RANGE
    GAMMA_MIN = GAMMA_CENTER - GAMMA_RANGE
    GAMMA_MAX = GAMMA_CENTER + GAMMA_RANGE
    
    RHO_VALUES = [0.7, 0.8, 0.999]
    coupling_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"\nConfiguration:")
    print(f"  N_HID: {N_HID}")
    print(f"  N_TRIALS: {N_TRIALS}")
    print(f"  RHO_BASELINE: {RHO_BASELINE}")
    print(f"  RHO_VALUES (5-layer): {RHO_VALUES}")
    print(f"  COUPLING_VALUES: {coupling_values}")
    print(f"  GPUs available: 2")
    
    # Build experiment queue
    experiments = []
    
    # Baseline experiments
    for trial in range(1, N_TRIALS + 1):
        exp = {
            'model_type': 'baseline',
            'trial': trial,
            'rho': RHO_BASELINE,
            'DATAROOT': DATAROOT,
            'BATCH_SIZE': BATCH_SIZE,
            'N_HID': N_HID,
            'DT': DT,
            'INP_SCALING': INP_SCALING,
            'EPSILON_MIN': EPSILON_MIN,
            'EPSILON_MAX': EPSILON_MAX,
            'GAMMA_MIN': GAMMA_MIN,
            'GAMMA_MAX': GAMMA_MAX,
        }
        experiments.append(exp)
    
    # Antisymmetric experiments
    for rho in RHO_VALUES:
        for coup_eps in coupling_values:
            for trial in range(1, N_TRIALS + 1):
                exp = {
                    'model_type': 'antisymmetric',
                    'trial': trial,
                    'rho': rho,
                    'coupling_epsilon': coup_eps,
                    'DATAROOT': DATAROOT,
                    'BATCH_SIZE': BATCH_SIZE,
                    'N_HID': N_HID,
                    'DT': DT,
                    'INP_SCALING': INP_SCALING,
                    'EPSILON_MIN': EPSILON_MIN,
                    'EPSILON_MAX': EPSILON_MAX,
                    'GAMMA_MIN': GAMMA_MIN,
                    'GAMMA_MAX': GAMMA_MAX,
                }
                experiments.append(exp)
    
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Estimated time: ~{len(experiments)//2 * 3} minutes (if 3 min per experiment)")
    print(f"\nStarting parallel execution...\n")
    
    # Run experiments in parallel on 2 GPUs
    # Use a pool of 2 workers (one per GPU)
    results = []
    
    # Create a pool with 2 processes
    with mp.Pool(processes=2) as pool:
        # Distribute experiments across GPUs in round-robin fashion
        gpu_assignments = [(exp, i % 2) for i, exp in enumerate(experiments)]
        
        # Map experiments to workers
        results = pool.starmap(run_single_experiment, gpu_assignments)
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)
    
    # Aggregate results by configuration
    aggregated_results = []
    
    # Baseline aggregation
    baseline_trials = [r for r in results if r['model_type'] == 'baseline' and not r.get('failed', False)]
    if baseline_trials:
        baseline_avg = {
            'name': f'Standard 1-layer RON (avg over {len(baseline_trials)} trials)',
            'model_type': 'baseline',
            'rho': RHO_BASELINE,
            'train_acc': np.mean([r['train_acc'] for r in baseline_trials]),
            'valid_acc': np.mean([r['valid_acc'] for r in baseline_trials]),
            'test_acc': np.mean([r['test_acc'] for r in baseline_trials]),
            'train_std': np.std([r['train_acc'] for r in baseline_trials]),
            'valid_std': np.std([r['valid_acc'] for r in baseline_trials]),
            'test_std': np.std([r['test_acc'] for r in baseline_trials]),
            'saturated_pct': np.mean([r['saturated_pct'] for r in baseline_trials]),
            'failed': False,
        }
        if any('spectral_radius' in r for r in baseline_trials):
            baseline_avg['spectral_radius'] = next(r['spectral_radius'] for r in baseline_trials if 'spectral_radius' in r)
        aggregated_results.append(baseline_avg)
    
    # Antisymmetric aggregation
    for rho in RHO_VALUES:
        for coup_eps in coupling_values:
            trials = [r for r in results 
                     if r['model_type'] == 'antisymmetric' 
                     and r['rho'] == rho 
                     and r['coupling_epsilon'] == coup_eps
                     and not r.get('failed', False)]
            
            if trials:
                avg = {
                    'name': f'Antisymmetric 5-layer (ε_c={coup_eps}, ρ={rho}, avg)',
                    'model_type': 'antisymmetric',
                    'coupling_epsilon': coup_eps,
                    'rho': rho,
                    'train_acc': np.mean([r['train_acc'] for r in trials]),
                    'valid_acc': np.mean([r['valid_acc'] for r in trials]),
                    'test_acc': np.mean([r['test_acc'] for r in trials]),
                    'train_std': np.std([r['train_acc'] for r in trials]),
                    'valid_std': np.std([r['valid_acc'] for r in trials]),
                    'test_std': np.std([r['test_acc'] for r in trials]),
                    'saturated_pct': np.mean([r['saturated_pct'] for r in trials]),
                    'failed': False,
                }
                if any('spectral_radius' in r for r in trials):
                    avg['spectral_radius'] = next(r['spectral_radius'] for r in trials if 'spectral_radius' in r)
                    avg['condition_number'] = next(r['condition_number'] for r in trials if 'condition_number' in r)
                aggregated_results.append(avg)
    
    # Sort by test accuracy
    aggregated_results_sorted = sorted(aggregated_results, key=lambda x: x['test_acc'], reverse=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Averaged over trials)")
    print("=" * 80)
    
    print(f"\n{'Rank':<5} {'Model':<55} {'ε_c':<8} {'ρ_rec':<8} {'ρ_tot':<10} {'Test Acc':<18} {'Valid Acc':<18}")
    print("-"*130)
    
    for rank, r in enumerate(aggregated_results_sorted, 1):
        coup_str = f"{r.get('coupling_epsilon', 0):.2f}" if 'coupling_epsilon' in r else "N/A"
        rho_rec_str = f"{r.get('rho', 0):.3f}" if 'rho' in r else "N/A"
        rho_tot_str = f"{r.get('spectral_radius', 0.0):.4f}" if 'spectral_radius' in r else "N/A"
        marker = "✓" if r['model_type'] == 'antisymmetric' else " "
        
        test_str = f"{r['test_acc']*100:.2f}% ± {r['test_std']*100:.2f}%"
        valid_str = f"{r['valid_acc']*100:.2f}% ± {r['valid_std']*100:.2f}%"
        
        print(f"{rank:<5} {marker} {r['name']:<53} {coup_str:<8} {rho_rec_str:<8} {rho_tot_str:<10} "
              f"{test_str:<18} {valid_str:<18}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    best = aggregated_results_sorted[0]
    baseline = next((r for r in aggregated_results if r['model_type'] == 'baseline'), None)
    best_antisym = next((r for r in aggregated_results_sorted if r['model_type'] == 'antisymmetric'), None)
    
    print(f"\nBest overall: {best['name']}")
    print(f"  Test Accuracy: {best['test_acc']*100:.2f}% ± {best['test_std']*100:.2f}%")
    
    if baseline:
        print(f"\nBaseline (1-layer RON):")
        print(f"  Test Accuracy: {baseline['test_acc']*100:.2f}% ± {baseline['test_std']*100:.2f}%")
    
    if best_antisym and baseline:
        improvement = (best_antisym['test_acc'] - baseline['test_acc']) * 100
        print(f"\nBest Antisymmetric vs Baseline: {improvement:+.2f} percentage points")
        
        if best_antisym['test_acc'] > baseline['test_acc']:
            print("✓ Antisymmetric coupling provides improvement!")
        else:
            print("✗ Antisymmetric coupling does not improve over baseline")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results_smnist_ron_parallel_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'N_HID': N_HID,
                'N_TRIALS': N_TRIALS,
                'RHO_BASELINE': RHO_BASELINE,
                'RHO_VALUES': RHO_VALUES,
                'coupling_values': coupling_values,
            },
            'all_results': results,
            'aggregated_results': aggregated_results_sorted,
        }, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    # Set start method for multiprocessing (important for CUDA)
    mp.set_start_method('spawn', force=True)
    main()
