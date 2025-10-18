#!/usr/bin/env python3
"""
Test antisymmetric coupling on sMNIST with PROPER leaky rates
The key insight: leaky=0.001 disables inter-layer communication!
"""

import torch
import numpy as np
from acds.archetypes.esn import DeepReservoir
from acds.benchmarks.mnist import get_mnist_data
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import time

def extract_states(data_loader, model, device, desc="Processing"):
    """Extract reservoir states from data"""
    activations, ys = [], []
    for images, labels in tqdm(data_loader, desc=desc, leave=False):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)  # (batch, 784, 1)
        with torch.no_grad():
            output = model(images)[0]
            if len(output.shape) == 3:  # (batch, seq, features)
                output = output[:, -1, :]  # Take final timestep
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    return activations, ys

def test_config(config, train_loader, valid_loader, test_loader, device):
    """Test a single configuration"""
    print(f"\nTesting: {config['name']}")
    print(f"  n_layers={config['n_layers']}, tot_units={config['tot_units']}")
    print(f"  leaky={config['leaky']}, rho={config['rho']}, epsilon={config['epsilon']}")
    print(f"  antisymmetric={config['antisymmetric']}")
    
    # Create model
    units_per_layer = config['tot_units'] // config['n_layers']
    
    start_time = time.time()
    model = DeepReservoir(
        input_size=1,
        tot_units=config['tot_units'],
        n_layers=config['n_layers'],
        concat=True,
        spectral_radius=config['rho'],
        input_scaling=1.0,
        leaky=config['leaky'],
        connectivity_recurrent=units_per_layer,
        connectivity_input=units_per_layer,
        connectivity_inter=units_per_layer,
        antisymmetric=config['antisymmetric'],
        epsilon=config['epsilon'],
        cycle=False,
        linear=False,
    ).to(device)
    
    # Extract states
    print("  Extracting training states...")
    train_acts, train_ys = extract_states(train_loader, model, device, "Train")
    
    print("  Extracting validation states...")
    valid_acts, valid_ys = extract_states(valid_loader, model, device, "Valid")
    
    print("  Extracting test states...")
    test_acts, test_ys = extract_states(test_loader, model, device, "Test")
    
    # Check for NaN/Inf
    if np.isnan(train_acts).any() or np.isinf(train_acts).any():
        print("  ❌ UNSTABLE: NaN/Inf in training states!")
        return {**config, 'train_acc': 0.0, 'valid_acc': 0.0, 'test_acc': 0.0, 'status': 'unstable'}
    
    # Check state statistics
    print(f"  State stats: mean={train_acts.mean():.4f}, std={train_acts.std():.4f}, "
          f"max={np.abs(train_acts).max():.4f}")
    
    # Scale and train classifier
    scaler = preprocessing.StandardScaler()
    train_acts_scaled = scaler.fit_transform(train_acts)
    valid_acts_scaled = scaler.transform(valid_acts)
    test_acts_scaled = scaler.transform(test_acts)
    
    print("  Training classifier...")
    classifier = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    classifier.fit(train_acts_scaled, train_ys)
    
    # Evaluate
    train_acc = classifier.score(train_acts_scaled, train_ys)
    valid_acc = classifier.score(valid_acts_scaled, valid_ys)
    test_acc = classifier.score(test_acts_scaled, test_ys)
    
    elapsed = time.time() - start_time
    
    print(f"  ✓ Train: {100*train_acc:.2f}%, Valid: {100*valid_acc:.2f}%, Test: {100*test_acc:.2f}%")
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        **config,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'test_acc': test_acc,
        'time': elapsed,
        'status': 'ok'
    }

def main():
    print("=" * 70)
    print("sMNIST: Testing Antisymmetric with Proper Leaky Rates")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading sMNIST data...")
    train_loader, valid_loader, test_loader = get_mnist_data(
        dataroot='./data',
        batch_size=1000,
        download=True
    )
    print(f"✓ Data loaded")
    
    # Test configurations
    configs = [
        # Baseline: 1-layer with low leaky (your current best)
        {
            'name': '1-Layer Low-Leaky (Baseline)',
            'n_layers': 1,
            'tot_units': 256,
            'leaky': 0.001,
            'rho': 0.999,
            'epsilon': 0.0,
            'antisymmetric': False,
        },
        
        # 5-layer standard with low leaky (fails on your tests)
        {
            'name': '5-Layer Standard Low-Leaky',
            'n_layers': 5,
            'tot_units': 250,  # 50 per layer
            'leaky': 0.001,
            'rho': 0.999,
            'epsilon': 0.0,
            'antisymmetric': False,
        },
        
        # 5-layer antisymmetric with PROPER leaky for long sequences
        {
            'name': '5-Layer Antisym Mid-Leaky',
            'n_layers': 5,
            'tot_units': 250,
            'leaky': 0.05,  # 50x higher!
            'rho': 0.95,
            'epsilon': 0.3,
            'antisymmetric': True,
        },
        
        {
            'name': '5-Layer Antisym High-Leaky',
            'n_layers': 5,
            'tot_units': 250,
            'leaky': 0.1,
            'rho': 0.9,
            'epsilon': 0.2,
            'antisymmetric': True,
        },
        
        # Alternative: fewer layers, more units per layer
        {
            'name': '3-Layer Antisym Balanced',
            'n_layers': 3,
            'tot_units': 300,  # 100 per layer
            'leaky': 0.05,
            'rho': 0.95,
            'epsilon': 0.3,
            'antisymmetric': True,
        },
        
        # Conservative antisymmetric (lower epsilon)
        {
            'name': '5-Layer Antisym Conservative',
            'n_layers': 5,
            'tot_units': 250,
            'leaky': 0.05,
            'rho': 0.95,
            'epsilon': 0.1,  # Lower epsilon for stability
            'antisymmetric': True,
        },
    ]
    
    results = []
    
    for config in configs:
        try:
            result = test_config(config, train_loader, valid_loader, test_loader, device)
            results.append(result)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({**config, 'train_acc': 0.0, 'valid_acc': 0.0, 
                          'test_acc': 0.0, 'status': 'error'})
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)
    
    print(f"\n{'Config':<35} {'Layers':<7} {'Leaky':<8} {'Test Acc':<10} {'Valid Acc':<10}")
    print("-" * 75)
    for r in results_sorted:
        print(f"{r['name']:<35} {r['n_layers']:<7} {r['leaky']:<8.3f} "
              f"{100*r['test_acc']:>8.2f}%  {100*r['valid_acc']:>8.2f}%")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    best = results_sorted[0]
    baseline = next((r for r in results if 'Baseline' in r['name']), None)
    
    if best['status'] == 'ok':
        print(f"\n✓ Best: {best['name']}")
        print(f"  Test accuracy: {100*best['test_acc']:.2f}%")
        
        if baseline:
            improvement = best['test_acc'] - baseline['test_acc']
            print(f"\n  vs Baseline (1-layer low-leaky):")
            print(f"    Baseline: {100*baseline['test_acc']:.2f}%")
            print(f"    Improvement: {100*improvement:+.2f}%")
            
            if improvement > 0.02:  # >2% improvement
                print(f"\n  🎉 Antisymmetric coupling WORKS for sMNIST!")
                print(f"     Key: Higher leaky rate enables inter-layer communication")
            elif improvement > -0.02:  # within 2%
                print(f"\n  ⚖️  Antisymmetric performs similarly to baseline")
            else:
                print(f"\n  ⚠️  Antisymmetric underperforms baseline")
    
    # Save results
    import json
    with open('smnist_antisym_leaky_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: smnist_antisym_leaky_results.json")

if __name__ == '__main__':
    main()
