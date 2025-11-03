"""
Memory Capacity test comparing standard RON vs antisymmetric coupled RON
Testing different layer configurations while keeping total units at 100.

Test configurations:
- 1 layer × 100 units
- 2 layers × 50 units each
- 4 layers × 25 units each
- 50 layers × 2 units each
- 100 layers × 1 unit each

RON Hyperparameters:
- dt = 0.95
- gamma = 1.14
- epsilon = 0.88
- rho = 0.99
- input_scaling = 0.2

Testing coupling strengths: 0.1 and 10.0
Averaging over 3 trials per configuration.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import defaultdict
from acds.archetypes import RandomizedOscillatorsNetwork, DeepRandomizedOscillatorsNetwork

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Create results directory
RESULTS_DIR = "results_mc_antisymmetric"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("Memory Capacity: Standard RON vs Antisymmetric Coupled RON")
print("=" * 80)

# Configuration
TOTAL_UNITS = 100
N_TRIALS = 3
DELAY = 100
WASHOUT = 1000

# RON Hyperparameters (as specified)
DT = 0.95
GAMMA = 1.14
EPSILON = 0.88
RHO = 0.99
INPUT_SCALING = 0.2

# Coupling strengths to test
COUPLING_STRENGTHS = [0.1, 10.0]

# Layer configurations: (n_layers, description)
LAYER_CONFIGS = [
    (1, "1 layer × 100 units"),
    (2, "2 layers × 50 units"),
    (4, "4 layers × 25 units"),
    (50, "50 layers × 2 units"),
    (100, "100 layers × 1 unit"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

print("Configuration:")
print(f"  Total units: {TOTAL_UNITS}")
print(f"  Trials per config: {N_TRIALS}")
print(f"  Delay: {DELAY}")
print(f"  Washout: {WASHOUT}")
print(f"  dt: {DT}")
print(f"  gamma: {GAMMA}")
print(f"  epsilon: {EPSILON}")
print(f"  rho: {RHO}")
print(f"  input_scaling: {INPUT_SCALING}")
print(f"  coupling_strengths: {COUPLING_STRENGTHS}\n")

# Data generation parameters
NUM_STEPS = 6000
TRAIN_STEPS = 5000
TEST_STEPS = 1000


def square_correlation(output, target):
    """Calculate squared correlation coefficient."""
    return (np.corrcoef(output.flatten(), target.flatten())[0, 1]) ** 2


def compute_memory_capacity(model, delay, device, concat=True, units_per_layer=None):
    """
    Compute memory capacity for a given model.
    
    Returns:
        train_memory_dict: dict mapping delay -> list of memory values
        test_memory_dict: dict mapping delay -> list of memory values
        total_train_memory: total memory capacity on train set
        total_test_memory: total memory capacity on test set
    """
    # Generate random input sequence
    u = np.random.uniform(-0.8, 0.8, size=(NUM_STEPS + delay, 1))
    u = u.astype(np.float32)
    
    # Get reservoir states
    with torch.no_grad():
        states_u = model(torch.tensor(u[:-delay]).to(device).reshape(1, -1, 1))[0].cpu().numpy()
    
    # Reshape based on concatenation
    if concat:
        states_u = states_u.reshape(-1, TOTAL_UNITS)
    else:
        states_u = states_u.reshape(-1, units_per_layer)
    
    # Prepare data for all delays
    X_all = states_u[delay:NUM_STEPS, :]
    
    # Create target matrix with columns for each delay
    y_all = np.zeros((NUM_STEPS - delay, delay))
    for i in range(1, delay + 1):
        y_all[:, i-1] = u[delay-i:NUM_STEPS-i, 0]
    
    # Split into train and test
    split_idx_train = TRAIN_STEPS - delay
    split_idx_test = split_idx_train + TEST_STEPS
    
    X_train, X_test = X_all[:split_idx_train], X_all[split_idx_train:split_idx_test]
    y_train, y_test = y_all[:split_idx_train], y_all[split_idx_train:split_idx_test]
    
    # Add washout
    X_train = X_train[WASHOUT:]
    y_train = y_train[WASHOUT:]
    
    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train classifier using pseudo-inverse
    try:
        classifier = np.linalg.pinv(X_train) @ y_train
    except np.linalg.LinAlgError as e:
        print(f"  Warning: Pseudo-inverse failed, using Ridge regression")
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1e-6, max_iter=10000)
        ridge.fit(X_train, y_train)
        classifier = ridge.coef_.T
        if len(classifier.shape) == 1:
            classifier = classifier.reshape(-1, 1)
    
    y_hat_train = X_train @ classifier
    y_hat_test = X_test @ classifier
    
    # Calculate memory capacity for each delay
    train_memory_dict = {}
    test_memory_dict = {}
    total_train_memory = 0
    total_test_memory = 0
    
    for i in range(1, delay + 1):
        col_idx = i - 1
        train_memory = square_correlation(y_hat_train[:, col_idx], y_train[:, col_idx])
        test_memory = square_correlation(y_hat_test[:, col_idx], y_test[:, col_idx])
        
        # Clamp values to [0, 1]
        train_memory = np.clip(train_memory, 0, 1)
        test_memory = np.clip(test_memory, 0, 1)
        
        train_memory_dict[i] = train_memory
        test_memory_dict[i] = test_memory
        total_train_memory += train_memory
        total_test_memory += test_memory
    
    return train_memory_dict, test_memory_dict, total_train_memory, total_test_memory


def run_experiment(n_layers, description, antisymmetric=False, coupling_epsilon=None):
    """
    Run memory capacity experiment for a given configuration.
    
    Returns:
        results: dict with averaged results
    """
    print(f"\n{'='*70}")
    if antisymmetric:
        print(f"{description} - ANTISYMMETRIC (ε_c={coupling_epsilon})")
    else:
        print(f"{description} - STANDARD")
    print(f"{'='*70}")
    
    trial_results = []
    
    for trial in range(N_TRIALS):
        print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
        
        # Create model
        if n_layers == 1:
            # Single layer RON
            model = RandomizedOscillatorsNetwork(
                n_inp=1,
                n_hid=TOTAL_UNITS,
                dt=DT,
                gamma=GAMMA,
                epsilon=EPSILON,
                rho=RHO,
                input_scaling=INPUT_SCALING,
                topology="full",
                device=device,
                antisymmetric_coupling=antisymmetric,
                coupling_epsilon=coupling_epsilon if antisymmetric else 0.1,
            ).to(device)
            concat = True
            units_per_layer = TOTAL_UNITS
        else:
            # Multi-layer DeepRON
            model = DeepRandomizedOscillatorsNetwork(
                n_inp=1,
                total_units=TOTAL_UNITS,
                n_layers=n_layers,
                dt=DT,
                gamma=GAMMA,
                epsilon=EPSILON,
                rho=RHO,
                input_scaling=INPUT_SCALING,
                inter_scaling=INPUT_SCALING,
                topology="full",
                concat=True,
                device=device,
                antisymmetric_coupling=antisymmetric,
                coupling_epsilon=coupling_epsilon if antisymmetric else 0.1,
            ).to(device)
            concat = True
            units_per_layer = TOTAL_UNITS // n_layers
        
        if trial == 0:
            print(f"  Model: {model.__class__.__name__}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
            print(f"  Units per layer: {units_per_layer}")
        
        # Set bias to zero
        for name, param in model.named_parameters():
            if "bias" in name:
                param.data.fill_(0)
        
        # Compute memory capacity
        train_mc_dict, test_mc_dict, total_train, total_test = compute_memory_capacity(
            model, DELAY, device, concat=concat, units_per_layer=units_per_layer
        )
        
        trial_results.append({
            'train_mc_dict': train_mc_dict,
            'test_mc_dict': test_mc_dict,
            'total_train': total_train,
            'total_test': total_test,
        })
        
        print(f"  Total Train MC: {total_train:.4f}")
        print(f"  Total Test MC: {total_test:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Average results across trials
    avg_train_mc = defaultdict(list)
    avg_test_mc = defaultdict(list)
    
    for result in trial_results:
        for delay, value in result['train_mc_dict'].items():
            avg_train_mc[delay].append(value)
        for delay, value in result['test_mc_dict'].items():
            avg_test_mc[delay].append(value)
    
    # Compute mean and std
    train_mc_mean = {k: np.mean(v) for k, v in avg_train_mc.items()}
    train_mc_std = {k: np.std(v) for k, v in avg_train_mc.items()}
    test_mc_mean = {k: np.mean(v) for k, v in avg_test_mc.items()}
    test_mc_std = {k: np.std(v) for k, v in avg_test_mc.items()}
    
    total_train_mean = np.mean([r['total_train'] for r in trial_results])
    total_train_std = np.std([r['total_train'] for r in trial_results])
    total_test_mean = np.mean([r['total_test'] for r in trial_results])
    total_test_std = np.std([r['total_test'] for r in trial_results])
    
    print(f"\n{'='*60}")
    print(f"AVERAGED RESULTS (over {N_TRIALS} trials)")
    print(f"{'='*60}")
    print(f"Total Train MC: {total_train_mean:.4f} ± {total_train_std:.4f}")
    print(f"Total Test MC: {total_test_mean:.4f} ± {total_test_std:.4f}")
    
    return {
        'n_layers': n_layers,
        'description': description,
        'antisymmetric': antisymmetric,
        'coupling_epsilon': coupling_epsilon,
        'train_mc_mean': train_mc_mean,
        'train_mc_std': train_mc_std,
        'test_mc_mean': test_mc_mean,
        'test_mc_std': test_mc_std,
        'total_train_mean': total_train_mean,
        'total_train_std': total_train_std,
        'total_test_mean': total_test_mean,
        'total_test_std': total_test_std,
    }


# ========================================
# Main Experiments
# ========================================
all_results = []

for n_layers, description in LAYER_CONFIGS:
    # Test standard (no antisymmetric coupling)
    result_standard = run_experiment(n_layers, description, antisymmetric=False)
    all_results.append(result_standard)
    
    # Test with antisymmetric coupling at different strengths
    if n_layers > 1:  # Antisymmetric coupling only makes sense for multi-layer
        for coupling_eps in COUPLING_STRENGTHS:
            result_antisym = run_experiment(
                n_layers, description, 
                antisymmetric=True, 
                coupling_epsilon=coupling_eps
            )
            all_results.append(result_antisym)


# ========================================
# Summary and Visualization
# ========================================
print("\n" + "="*80)
print("SUMMARY - Total Memory Capacity (Test Set)")
print("="*80)

print(f"\n{'Config':<25} {'Type':<20} {'ε_c':<8} {'Test MC':<20}")
print("-"*80)

for result in all_results:
    config_str = result['description']
    if result['antisymmetric']:
        type_str = "Antisymmetric"
        coupling_str = f"{result['coupling_epsilon']:.1f}"
    else:
        type_str = "Standard"
        coupling_str = "N/A"
    
    mc_str = f"{result['total_test_mean']:.4f} ± {result['total_test_std']:.4f}"
    
    print(f"{config_str:<25} {type_str:<20} {coupling_str:<8} {mc_str:<20}")


# ========================================
# Plot Results
# ========================================
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

# Plot 1: Total MC vs Number of Layers
fig, ax = plt.subplots(figsize=(12, 6))

# Group results by configuration
configs = {}
for result in all_results:
    n_layers = result['n_layers']
    if n_layers not in configs:
        configs[n_layers] = {'standard': None, 'antisym': {}}
    
    if result['antisymmetric']:
        configs[n_layers]['antisym'][result['coupling_epsilon']] = result
    else:
        configs[n_layers]['standard'] = result

# Plot standard configuration
n_layers_list = sorted(configs.keys())
standard_mc = []
standard_std = []

for n_layers in n_layers_list:
    if configs[n_layers]['standard']:
        standard_mc.append(configs[n_layers]['standard']['total_test_mean'])
        standard_std.append(configs[n_layers]['standard']['total_test_std'])
    else:
        standard_mc.append(0)
        standard_std.append(0)

ax.errorbar(n_layers_list, standard_mc, yerr=standard_std, 
            marker='o', linestyle='-', linewidth=2, markersize=8,
            label='Standard RON', capsize=5)

# Plot antisymmetric configurations
colors = ['red', 'green']
for idx, coupling_eps in enumerate(COUPLING_STRENGTHS):
    antisym_mc = []
    antisym_std = []
    
    for n_layers in n_layers_list:
        if coupling_eps in configs[n_layers]['antisym']:
            antisym_mc.append(configs[n_layers]['antisym'][coupling_eps]['total_test_mean'])
            antisym_std.append(configs[n_layers]['antisym'][coupling_eps]['total_test_std'])
        else:
            antisym_mc.append(None)
            antisym_std.append(0)
    
    # Remove None values for plotting
    valid_indices = [i for i, v in enumerate(antisym_mc) if v is not None]
    valid_n_layers = [n_layers_list[i] for i in valid_indices]
    valid_mc = [antisym_mc[i] for i in valid_indices]
    valid_std = [antisym_std[i] for i in valid_indices]
    
    if valid_mc:
        ax.errorbar(valid_n_layers, valid_mc, yerr=valid_std,
                    marker='s', linestyle='--', linewidth=2, markersize=8,
                    label=f'Antisymmetric (ε_c={coupling_eps})', 
                    capsize=5, color=colors[idx])

ax.set_xlabel('Number of Layers', fontsize=12)
ax.set_ylabel('Total Memory Capacity (Test)', fontsize=12)
ax.set_title('Memory Capacity vs Number of Layers (100 total units)', fontsize=14)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'mc_vs_layers.png'), dpi=300)
print(f"Saved: {os.path.join(RESULTS_DIR, 'mc_vs_layers.png')}")

# Plot 2: MC vs Delay for each configuration (selected configs)
selected_configs = [(1, "1×100"), (2, "2×50"), (4, "4×25")]

for n_layers, short_desc in selected_configs:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot standard
    if configs[n_layers]['standard']:
        result = configs[n_layers]['standard']
        delays = sorted(result['test_mc_mean'].keys())
        mc_values = [result['test_mc_mean'][d] for d in delays]
        mc_stds = [result['test_mc_std'][d] for d in delays]
        
        ax.plot(delays, mc_values, linewidth=2, label='Standard', alpha=0.8)
        ax.fill_between(delays, 
                        [mc_values[i] - mc_stds[i] for i in range(len(delays))],
                        [mc_values[i] + mc_stds[i] for i in range(len(delays))],
                        alpha=0.2)
    
    # Plot antisymmetric
    for idx, coupling_eps in enumerate(COUPLING_STRENGTHS):
        if coupling_eps in configs[n_layers]['antisym']:
            result = configs[n_layers]['antisym'][coupling_eps]
            delays = sorted(result['test_mc_mean'].keys())
            mc_values = [result['test_mc_mean'][d] for d in delays]
            mc_stds = [result['test_mc_std'][d] for d in delays]
            
            ax.plot(delays, mc_values, linewidth=2, linestyle='--',
                   label=f'Antisymmetric (ε_c={coupling_eps})', alpha=0.8)
            ax.fill_between(delays,
                           [mc_values[i] - mc_stds[i] for i in range(len(delays))],
                           [mc_values[i] + mc_stds[i] for i in range(len(delays))],
                           alpha=0.2)
    
    ax.set_xlabel('Delay Step', fontsize=12)
    ax.set_ylabel('Memory Capacity', fontsize=12)
    ax.set_title(f'Memory Capacity vs Delay - {short_desc}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    filename = f'mc_vs_delay_{n_layers}layers.png'
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    print(f"Saved: {os.path.join(RESULTS_DIR, filename)}")

plt.close('all')

# ========================================
# Save Results to File
# ========================================
result_file = os.path.join(RESULTS_DIR, "results_summary.txt")
with open(result_file, 'w') as f:
    f.write("Memory Capacity: Standard RON vs Antisymmetric Coupled RON\n")
    f.write("="*80 + "\n\n")
    
    f.write("Configuration:\n")
    f.write(f"  Total Units: {TOTAL_UNITS}\n")
    f.write(f"  Trials: {N_TRIALS}\n")
    f.write(f"  Delay: {DELAY}\n")
    f.write(f"  Washout: {WASHOUT}\n")
    f.write(f"  dt: {DT}\n")
    f.write(f"  gamma: {GAMMA}\n")
    f.write(f"  epsilon: {EPSILON}\n")
    f.write(f"  rho: {RHO}\n")
    f.write(f"  input_scaling: {INPUT_SCALING}\n")
    f.write(f"  coupling_strengths: {COUPLING_STRENGTHS}\n\n")
    
    f.write("Results:\n")
    f.write("-"*80 + "\n\n")
    
    for result in all_results:
        f.write(f"{result['description']}\n")
        if result['antisymmetric']:
            f.write(f"  Type: Antisymmetric (ε_c={result['coupling_epsilon']})\n")
        else:
            f.write(f"  Type: Standard\n")
        f.write(f"  Total Train MC: {result['total_train_mean']:.4f} ± {result['total_train_std']:.4f}\n")
        f.write(f"  Total Test MC: {result['total_test_mean']:.4f} ± {result['total_test_std']:.4f}\n\n")

print(f"\nResults saved to: {result_file}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETED")
print("="*80)
print(f"Results directory: {RESULTS_DIR}")
print("="*80)
