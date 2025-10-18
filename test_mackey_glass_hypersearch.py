#!/usr/bin/env python3
"""
COMPLETE Mackey-Glass test with:
1. Hyperparameter search for ESN configs (using validation)
2. Ridge alpha tuning for each config (using validation)
3. Final model selection (using validation)
4. Test performance on held-out data

This is the PROPER way to evaluate!
"""

import torch
import numpy as np
from acds.archetypes.esn import DeepReservoir
from acds.benchmarks import get_mackey_glass
from sklearn import preprocessing
from sklearn.linear_model import Ridge
import time
import os
from itertools import product
import random

def extract_states(model, data, device, washout=200):
    """Extract reservoir states from time series data"""
    data_tensor = data.unsqueeze(-1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(data_tensor)[0]
        if len(output.shape) == 3:
            output = output[0]
    
    return output[washout:].cpu().numpy()

def evaluate_config(config, train_data, train_target, val_data, val_target, device, washout=200):
    """
    Evaluate a single ESN configuration
    Returns validation NRMSE (for model selection) and best Ridge alpha
    """
    units_per_layer = config['tot_units'] // config['n_layers']
    
    try:
        # Create model
        model = DeepReservoir(
            input_size=1,
            tot_units=config['tot_units'],
            n_layers=config['n_layers'],
            concat=True,
            spectral_radius=config['rho'],
            input_scaling=config['input_scaling'],
            leaky=config['leaky'],
            connectivity_recurrent=units_per_layer,
            connectivity_input=units_per_layer,
            connectivity_inter=units_per_layer,
            antisymmetric=config['antisymmetric'],
            epsilon=config.get('epsilon', 0.0),
            cycle=False,
            linear=False,
        ).to(device)
        
        # Extract states
        train_states = extract_states(model, train_data, device, washout)
        val_states = extract_states(model, val_data, device, washout)
        
        # Check stability
        if np.isnan(train_states).any() or np.isinf(train_states).any():
            return None, None, 'unstable'
        
        # Scale states
        scaler = preprocessing.StandardScaler()
        train_states_scaled = scaler.fit_transform(train_states)
        val_states_scaled = scaler.transform(val_states)
        
        # Search for best Ridge alpha using validation
        alphas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        best_alpha = None
        best_val_nrmse = float('inf')
        
        for alpha in alphas:
            readout = Ridge(alpha=alpha)
            readout.fit(train_states_scaled, train_target.cpu().numpy())
            val_pred = readout.predict(val_states_scaled)
            
            mse = np.mean((val_target.cpu().numpy() - val_pred) ** 2)
            rmse = np.sqrt(mse)
            val_nrmse = rmse / np.std(val_target.cpu().numpy())
            
            if val_nrmse < best_val_nrmse:
                best_val_nrmse = val_nrmse
                best_alpha = alpha
        
        return float(best_val_nrmse), float(best_alpha), 'ok'
        
    except Exception as e:
        return None, None, 'error'

def main():
    print("=" * 80)
    print("MACKEY-GLASS: COMPLETE HYPERPARAMETER SEARCH + PROPER VALIDATION")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading Mackey-Glass data...")
    dataroot = "./acds/benchmarks/raw/"
    
    (train_data, train_target), (val_data, val_target), (test_data, test_target) = \
        get_mackey_glass(dataroot, lag=84, washout=200)
    
    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_data)} timesteps (for fitting)")
    print(f"  Val:   {len(val_data)} timesteps (for hyperparameter selection)")
    print(f"  Test:  {len(test_data)} timesteps (held-out, final evaluation only)")
    
    # Define hyperparameter search space
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH SPACE")
    print("=" * 80)
    
    # Search space for STANDARD (1-layer) models
    standard_space = {
        'n_layers': [1],
        'tot_units': [300, 500, 700],
        'leaky': [0.001, 0.01, 0.1, 0.3],
        'rho': [0.9, 0.95, 0.99],
        'input_scaling': [1.0],
        'antisymmetric': [False],
        'epsilon': [0.0],
    }
    
    # Search space for ANTISYMMETRIC models
    antisym_space = {
        'n_layers': [3, 5],
        'tot_units': [300, 500],
        'leaky': [0.05, 0.1, 0.2],
        'rho': [0.85, 0.9, 0.95],
        'input_scaling': [1.0],
        'antisymmetric': [True],
        'epsilon': [0.1, 0.2, 0.3, 0.5],
    }
    
    # Generate all combinations
    def generate_configs(space, name_prefix):
        keys = list(space.keys())
        configs = []
        for values in product(*[space[k] for k in keys]):
            config = dict(zip(keys, values))
            config['name'] = f"{name_prefix}_L{config['n_layers']}_U{config['tot_units']}_" \
                           f"leak{config['leaky']}_rho{config['rho']}"
            if config['antisymmetric']:
                config['name'] += f"_eps{config['epsilon']}"
            configs.append(config)
        return configs
    
    standard_configs = generate_configs(standard_space, "Standard")
    antisym_configs = generate_configs(antisym_space, "Antisym")
    
    all_configs = standard_configs + antisym_configs
    
    print(f"\nTotal configurations:")
    print(f"  Standard (1-layer): {len(standard_configs)}")
    print(f"  Antisymmetric (multi-layer): {len(antisym_configs)}")
    print(f"  Total: {len(all_configs)}")
    
    # Option to sample if too many
    max_configs = 50
    if len(all_configs) > max_configs:
        print(f"\n⚠️  Too many configs ({len(all_configs)}), randomly sampling {max_configs}")
        random.seed(42)
        # Keep all standard 1-layer with leaky=0.001 or 0.1
        important_standard = [c for c in standard_configs if c['leaky'] in [0.001, 0.1]]
        # Sample from the rest
        other_configs = [c for c in all_configs if c not in important_standard]
        sampled_configs = important_standard + random.sample(other_configs, 
                                                             min(max_configs - len(important_standard), 
                                                                 len(other_configs)))
        all_configs = sampled_configs
        print(f"  Sampling {len(all_configs)} configurations (keeping important baselines)")
    
    # Hyperparameter search
    print("\n" + "=" * 80)
    print("PHASE 1: HYPERPARAMETER SEARCH (on validation set)")
    print("=" * 80)
    
    results = []
    
    for i, config in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] Testing: {config['name']}")
        print(f"  layers={config['n_layers']}, units={config['tot_units']}, "
              f"leaky={config['leaky']}, rho={config['rho']}", end="")
        if config['antisymmetric']:
            print(f", epsilon={config['epsilon']}")
        else:
            print()
        
        val_nrmse, best_alpha, status = evaluate_config(
            config, train_data, train_target, val_data, val_target, device
        )
        
        if status == 'ok':
            print(f"  ✓ Val NRMSE: {val_nrmse:.4f}, Best Ridge alpha: {best_alpha:.0e}")
            results.append({
                **config,
                'val_nrmse': val_nrmse,
                'best_alpha': best_alpha,
                'status': status
            })
        else:
            print(f"  ❌ {status.upper()}")
            results.append({
                **config,
                'val_nrmse': 999.0,
                'best_alpha': None,
                'status': status
            })
    
    # Model selection based on validation
    print("\n" + "=" * 80)
    print("PHASE 2: MODEL SELECTION (based on validation NRMSE)")
    print("=" * 80)
    
    valid_results = [r for r in results if r['status'] == 'ok']
    
    if not valid_results:
        print("\n❌ No valid configurations found!")
        return
    
    valid_results_sorted = sorted(valid_results, key=lambda x: x['val_nrmse'])
    
    print(f"\nTop 10 configurations (by validation NRMSE):")
    print(f"{'Config':<50} {'Val NRMSE':<12} {'Alpha'}")
    print("-" * 80)
    for r in valid_results_sorted[:10]:
        antisym_marker = "🔗" if r['antisymmetric'] else "  "
        print(f"{antisym_marker} {r['name']:<48} {r['val_nrmse']:<12.4f} {r['best_alpha']:.0e}")
    
    best_config = valid_results_sorted[0]
    
    print(f"\n✓ SELECTED MODEL (best validation NRMSE): {best_config['name']}")
    print(f"  Validation NRMSE: {best_config['val_nrmse']:.4f}")
    print(f"  Configuration:")
    print(f"    n_layers: {best_config['n_layers']}")
    print(f"    tot_units: {best_config['tot_units']}")
    print(f"    leaky: {best_config['leaky']}")
    print(f"    rho: {best_config['rho']}")
    print(f"    antisymmetric: {best_config['antisymmetric']}")
    if best_config['antisymmetric']:
        print(f"    epsilon: {best_config['epsilon']}")
    print(f"    Ridge alpha: {best_config['best_alpha']:.0e}")
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("PHASE 3: FINAL TEST EVALUATION (on held-out test set)")
    print("=" * 80)
    
    print(f"\nRe-training best model and evaluating on test set...")
    
    # Recreate best model
    units_per_layer = best_config['tot_units'] // best_config['n_layers']
    
    final_model = DeepReservoir(
        input_size=1,
        tot_units=best_config['tot_units'],
        n_layers=best_config['n_layers'],
        concat=True,
        spectral_radius=best_config['rho'],
        input_scaling=best_config['input_scaling'],
        leaky=best_config['leaky'],
        connectivity_recurrent=units_per_layer,
        connectivity_input=units_per_layer,
        connectivity_inter=units_per_layer,
        antisymmetric=best_config['antisymmetric'],
        epsilon=best_config.get('epsilon', 0.0),
        cycle=False,
        linear=False,
    ).to(device)
    
    # Extract states
    train_states = extract_states(final_model, train_data, device, washout=200)
    test_states = extract_states(final_model, test_data, device, washout=200)
    
    # Scale and train final readout
    scaler = preprocessing.StandardScaler()
    train_states_scaled = scaler.fit_transform(train_states)
    test_states_scaled = scaler.transform(test_states)
    
    final_readout = Ridge(alpha=best_config['best_alpha'])
    final_readout.fit(train_states_scaled, train_target.cpu().numpy())
    
    # Final test prediction
    test_pred = final_readout.predict(test_states_scaled)
    
    mse = np.mean((test_target.cpu().numpy() - test_pred) ** 2)
    rmse = np.sqrt(mse)
    test_nrmse = rmse / np.std(test_target.cpu().numpy())
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nSelected Model: {best_config['name']}")
    print(f"  Validation NRMSE: {best_config['val_nrmse']:.4f}")
    print(f"  Test NRMSE:       {test_nrmse:.4f}")
    
    # Compare with best standard baseline
    best_standard = next((r for r in valid_results_sorted if not r['antisymmetric']), None)
    
    if best_standard:
        print(f"\nBest Standard (1-layer) Baseline: {best_standard['name']}")
        print(f"  Validation NRMSE: {best_standard['val_nrmse']:.4f}")
        
        # Evaluate baseline on test set
        units_per_layer_std = best_standard['tot_units'] // best_standard['n_layers']
        baseline_model = DeepReservoir(
            input_size=1,
            tot_units=best_standard['tot_units'],
            n_layers=best_standard['n_layers'],
            concat=True,
            spectral_radius=best_standard['rho'],
            input_scaling=1.0,
            leaky=best_standard['leaky'],
            connectivity_recurrent=units_per_layer_std,
            connectivity_input=units_per_layer_std,
            connectivity_inter=units_per_layer_std,
            antisymmetric=False,
            epsilon=0.0,
            cycle=False,
            linear=False,
        ).to(device)
        
        baseline_train = extract_states(baseline_model, train_data, device, washout=200)
        baseline_test = extract_states(baseline_model, test_data, device, washout=200)
        
        baseline_scaler = preprocessing.StandardScaler()
        baseline_train_scaled = baseline_scaler.fit_transform(baseline_train)
        baseline_test_scaled = baseline_scaler.transform(baseline_test)
        
        baseline_readout = Ridge(alpha=best_standard['best_alpha'])
        baseline_readout.fit(baseline_train_scaled, train_target.cpu().numpy())
        baseline_test_pred = baseline_readout.predict(baseline_test_scaled)
        
        baseline_test_nrmse = np.sqrt(np.mean((test_target.cpu().numpy() - 
                                               baseline_test_pred) ** 2)) / np.std(test_target.cpu().numpy())
        
        print(f"  Test NRMSE:       {baseline_test_nrmse:.4f}")
        
        improvement = ((baseline_test_nrmse - test_nrmse) / baseline_test_nrmse * 100)
        print(f"\nImprovement over baseline: {improvement:+.2f}%")
    
    # Save results
    import json
    with open('mackey_glass_hypersearch_results.json', 'w') as f:
        json.dump({
            'selected_model': best_config,
            'selected_test_nrmse': float(test_nrmse),
            'best_standard_baseline': best_standard,
            'best_standard_test_nrmse': float(baseline_test_nrmse) if best_standard else None,
            'top_10_configs': valid_results_sorted[:10],
            'all_results': results
        }, f, indent=2)
    print(f"\n✓ Full results saved to: mackey_glass_hypersearch_results.json")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if best_config['antisymmetric']:
        print("\n🎉 ANTISYMMETRIC COUPLING WINS!")
        print(f"   Selected: {best_config['name']}")
        print(f"   Test NRMSE: {test_nrmse:.4f}")
        if best_standard:
            print(f"   vs Best 1-layer: {baseline_test_nrmse:.4f} ({improvement:+.1f}%)")
        print("\n   Chaotic dynamics benefit from hierarchical temporal integration")
        print("   via antisymmetric coupling between reservoir layers.")
    else:
        print(f"\n✓ Standard (1-layer) architecture wins")
        print(f"   Selected: {best_config['name']}")
        print(f"   Test NRMSE: {test_nrmse:.4f}")
        print("\n   For this task, simple echo state is sufficient.")

if __name__ == '__main__':
    main()
