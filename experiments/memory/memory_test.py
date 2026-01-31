import argparse
import numpy as np
import torch

# Force GPU usage if available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("[INFO] Using GPU (cuda)")
else:
    DEVICE = torch.device('cpu')
    print("[WARNING] CUDA not available, using CPU")
import itertools
import json
import csv
import os
from datetime import datetime
from sklearn.linear_model import Ridge
from acds.archetypes.esn import DeepReservoir
from acds.archetypes.ron import DeepRandomizedOscillatorsNetwork

# --- 1. Benchmark Data Generators ---

def generate_narma(n=10, length=7000):
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(n, length):
        sum_y = np.sum(y[t-n:t])
        y[t] = 0.3 * y[t-1] + 0.01 * y[t-1] * sum_y + 1.5 * u[t-n] * u[t-1] + 0.1
    return u, y

def generate_ctxor(delay=10, length=7000):
    u = np.random.uniform(-0.8, 0.8, length)
    y = np.zeros(length)
    for t in range(delay + 1, length):
        r = u[t-delay-1] * u[t-delay]
        y[t] = (r**2) * np.sign(r)
    return u, y

def generate_sinmem(delay=10, length=7000):
    u = np.random.uniform(-0.8, 0.8, length)
    y = np.zeros(length)
    for t in range(delay, length):
        y[t] = np.sin(np.pi * u[t-delay])
    return u, y

def get_bench_data(name):
    if "narma10" in name: return generate_narma(n=10)
    elif "narma30" in name: return generate_narma(n=30)
    elif "ctxor5" in name: return generate_ctxor(delay=5)
    elif "ctxor10" in name: return generate_ctxor(delay=10)
    elif "sinmem5" in name: return generate_sinmem(delay=5)
    elif "sinmem10" in name: return generate_sinmem(delay=10)
    raise ValueError(f"Unknown task: {name}")

# --- 2. Model Factory ---

def create_model(model_type, variant, params):
    n_layers = params['n_layers']
    tot_units = 100
    # Calculate dynamic connectivity to keep density consistent across depths
    # e.g., if n_layers=10, units_per_layer=10, so connectivity=10 (full)
    # if n_layers=2, units_per_layer=50, so connectivity=50 (full)
    units_per_layer = tot_units // n_layers
    dyn_connectivity = max(1, units_per_layer)

    if model_type == 'deepesn':
        cfg = {
            'input_size': 1,
            'tot_units': tot_units,
            'n_layers': n_layers,
            'concat': True, #
            'spectral_radius': params.get('rho', 0.99),
            'leaky': params.get('leaky', 1.0),
            'input_scaling': params.get('inp_scaling', 1.0),
            'connectivity_recurrent': dyn_connectivity,
            'connectivity_input': dyn_connectivity,
            'connectivity_inter': dyn_connectivity,
            'cycle': False,
            'antisymmetric': False
        }
        if variant == 'cycle':
            cfg['cycle'] = True
        elif variant == 'cycle_zero':
            cfg['cycle'] = True
            cfg['connectivity_recurrent'] = 0 #
        elif variant == 'antisymmetric':
            cfg['antisymmetric'] = True
            cfg['epsilon'] = params.get('epsilon_coupling', 0.1)
        return DeepReservoir(**cfg)

    elif model_type == 'deepron':
        cfg = {
            'n_inp': 1,
            'total_units': tot_units,
            'n_layers': n_layers,
            'concat': True, #
            'dt': params.get('dt', 0.05),
            'rho': params.get('rho', 0.99),
            'gamma': params.get('gamma', 1.0),
            'epsilon': params.get('epsilon', 5.0),
            'input_scaling': params.get('inp_scaling', 1.0),
            'connectivity_recurrent': dyn_connectivity,
            'cycle': False,
            'antisymmetric_coupling': False
        }
        if variant == 'cycle':
            cfg['cycle'] = True
        elif variant == 'cycle_zero':
            cfg['cycle'] = True
            cfg['connectivity_recurrent'] = 0 #
        elif variant == 'antisymmetric':
            cfg['antisymmetric_coupling'] = True
            cfg['coupling_epsilon'] = params.get('epsilon_coupling', 0.1)
        return DeepRandomizedOscillatorsNetwork(**cfg)

# --- 3. Training and Evaluation (NRMSE) ---

def run_evaluation(model, u, y, reg_alpha=1e-5):
    model = model.to(DEVICE)
    u_torch = torch.from_numpy(u).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)

    with torch.no_grad():
        # States are concatenated because concat=True
        states, _ = model(u_torch)

    states = states.squeeze(0).cpu().numpy()
    X_train, y_train = states[100:5000], y[100:5000]
    X_val, y_val = states[5000:6000], y[5000:6000]
    X_test, y_test = states[6000:7000], y[6000:7000]

    readout = Ridge(alpha=reg_alpha)
    readout.fit(X_train, y_train)

    val_pred = readout.predict(X_val)
    test_pred = readout.predict(X_test)

    # NRMSE: sqrt(MSE) / (y_max - y_min)
    val_nrmse = np.sqrt(np.mean((y_val - val_pred)**2)) / (np.max(y_val) - np.min(y_val))
    test_nrmse = np.sqrt(np.mean((y_test - test_pred)**2)) / (np.max(y_test) - np.min(y_test))

    return val_nrmse, test_nrmse

# --- 4. Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Deep RC NRMSE Benchmark Suite")
    parser.add_argument('--task', choices=['narma10', 'narma30', 'ctxor5', 'ctxor10', 'sinmem5', 'sinmem10'], required=True)
    parser.add_argument('--model', choices=['deepesn', 'deepron'], required=True)
    parser.add_argument('--variant', choices=['standard', 'cycle', 'cycle_zero', 'antisymmetric'], default='standard')
    parser.add_argument('--output_dir', default='results_benchmarks', help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    u, y = get_bench_data(args.task)
    
    # Grid parameters: n_layers is now part of the search
    if args.model == 'deepesn':
        grid = {
            'n_layers': [2, 5],
            'leaky': [0.1, 0.5, 0.9],
            'rho': [0.9, 0.99],
            'inp_scaling': [1.0, 10.0],
            'reg': [1e-8, 1e-2]
        }
    else: # deepron
        grid = {
            'n_layers': [2, 5, 10],
            'dt': [0.01, 0.05],
            'rho': [0.9, 0.99],
            'inp_scaling': [1, 10],
            'epsilon': [5, 10],
            'reg': [1e-8, 1e-5, 1e-2]
        }

    if args.variant == 'antisymmetric':
        grid['epsilon_coupling'] = [0.01, 0.1, 0.5]

    best_val_nrmse = float('inf')
    best_params = None
    
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"--- Starting Grid Search for {args.model} {args.variant} on {args.task} ---")
    total_combinations = len(combinations)
    for params in combinations:
        # write progress
        print(f"Evaluating {params} ... ({combinations.index(params)+1}/{total_combinations})", end='\r')
        model = create_model(args.model, args.variant, params)
        val_err, _ = run_evaluation(model, u, y, reg_alpha=params['reg'])
        if val_err < best_val_nrmse:
            print(f"\n New Best! Val NRMSE: {val_err:.6f} with params: {params}")
            best_val_nrmse = val_err
            best_params = params

    print(f"\nBest Config: {best_params} | Val NRMSE: {best_val_nrmse:.6f}")

    # Final 3 Trials
    test_results = []
    for t in range(3):
        model = create_model(args.model, args.variant, best_params)
        _, test_err = run_evaluation(model, u, y, reg_alpha=best_params['reg'])
        test_results.append(test_err)
        print(f" Trial {t+1}: Test NRMSE = {test_err:.6f}")

    mean_nrmse = np.mean(test_results)
    # standard deviation
    std_nrmse = np.std(test_results)
    

    # File Saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{args.model}_{args.variant}_{args.task}_{timestamp}.json"
    with open(os.path.join(args.output_dir, json_filename), 'w') as f:
        json.dump({
            'best_params': best_params, 
            'val_nrmse': float(best_val_nrmse),
            'test_mean': float(mean_nrmse),
            'test_std': float(std_nrmse)
        }, f, indent=4)

    csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['Timestamp', 'Model', 'Variant', 'Task', 'Layers', 'Mean_NRMSE', 'Std_NRMSE'])
        writer.writerow([timestamp, args.model, args.variant, args.task, best_params['n_layers'], mean_nrmse, std_nrmse])

    print(f"\nFinal Mean NRMSE: {mean_nrmse:.6f} | Std: {std_nrmse:.2e}")

if __name__ == "__main__":
    main()