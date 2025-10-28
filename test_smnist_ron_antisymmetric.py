"""
Test comparing standard 1-layer RON vs 5-layer antisymmetric RON on sMNIST
with coupling strength search.

Based on smnist.py experiment.
"""
import os
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import RandomizedOscillatorsNetwork, DeepRandomizedOscillatorsNetwork
from acds.benchmarks import get_mnist_data

print("=" * 80)
print("sMNIST: Standard 1-layer RON vs Antisymmetric 5-layer RON")
print("Coupling Strength Search")
print("=" * 80)

# Configuration
DATAROOT = "data"
BATCH_SIZE = 1000
N_HID = 500

# Hyperparameters for 1-layer RON baseline
DT = 0.042
RHO = 0.9  # Note: paper says rho=9 but that's likely a typo, using 0.9
INP_SCALING = 1.0
EPSILON_CENTER = 0.51
EPSILON_RANGE = 0.5
GAMMA_CENTER = 2.7
GAMMA_RANGE = 1.0

# Derived parameters
EPSILON_MIN = EPSILON_CENTER - EPSILON_RANGE
EPSILON_MAX = EPSILON_CENTER + EPSILON_RANGE
GAMMA_MIN = GAMMA_CENTER - GAMMA_RANGE  
GAMMA_MAX = GAMMA_CENTER + GAMMA_RANGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

print("Hyperparameters:")
print(f"  dt: {DT}")
print(f"  rho: {RHO}")
print(f"  input_scaling: {INP_SCALING}")
print(f"  epsilon: ({EPSILON_MIN}, {EPSILON_MAX})")
print(f"  gamma: ({GAMMA_MIN}, {GAMMA_MAX})")
print(f"  n_hid: {N_HID}")
print(f"  batch_size: {BATCH_SIZE}\n")

n_inp = 1
n_out = 10


@torch.no_grad()
def test(data_loader, model, classifier, scaler):
    """Evaluate model on a dataset."""
    activations, ys = [], []
    for images, labels in tqdm(data_loader, desc="Testing", leave=False):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        
        # Handle both RON and DeepRON outputs
        output = model(images)
        if isinstance(output, tuple):
            output = output[0]  # Get hidden states
        
        # Get last timestep
        if len(output.shape) == 3:
            output = output[:, -1, :]
        
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    
    # Check for NaN or Inf
    if np.isnan(activations).any() or np.isinf(activations).any():
        print("  WARNING: Activations contain NaN or Inf!")
        return 0.0
    
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)


def train_and_evaluate(model_name, model, train_loader, valid_loader, test_loader):
    """Train and evaluate a model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Collect training activations
    activations, ys = [], []
    print("Collecting training activations...")
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        
        # Handle both RON and DeepRON outputs
        output = model(images)
        if isinstance(output, tuple):
            output = output[0]  # Get hidden states
        
        # Get last timestep
        if len(output.shape) == 3:
            output = output[:, -1, :]
        
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    print(f"Activation statistics:")
    print(f"  Shape: {activations.shape}")
    print(f"  Mean: {activations.mean():.6f}, Std: {activations.std():.6f}")
    print(f"  Min: {activations.min():.6f}, Max: {activations.max():.6f}")
    
    # Check for NaN or Inf
    if np.isnan(activations).any() or np.isinf(activations).any():
        print("  ERROR: Activations contain NaN or Inf! Skipping this configuration.")
        return {
            'name': model_name,
            'train_acc': 0.0,
            'valid_acc': 0.0,
            'test_acc': 0.0,
            'saturated_pct': 0.0,
            'failed': True
        }
    
    # Check saturation
    saturated = np.sum(np.abs(activations) > 0.99) / activations.size
    print(f"  Saturated (|x| > 0.99): {saturated*100:.2f}%")
    
    # Scale and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    
    print("Training logistic regression classifier...")
    classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    classifier.fit(activations, ys)
    
    # Evaluate
    print("Evaluating...")
    train_acc = classifier.score(activations, ys)
    valid_acc = test(valid_loader, model, classifier, scaler)
    test_acc = test(test_loader, model, classifier, scaler)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Valid Accuracy: {valid_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    
    return {
        'name': model_name,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'test_acc': test_acc,
        'saturated_pct': saturated * 100,
        'failed': False
    }


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
    # Baseline: Standard 1-layer RON
    # ========================================
    print("\n" + "="*80)
    print("BASELINE: Standard 1-layer RON")
    print("="*80)
    
    model_standard = RandomizedOscillatorsNetwork(
        n_inp=n_inp,
        n_hid=N_HID,
        dt=DT,
        gamma=(GAMMA_MIN, GAMMA_MAX),
        epsilon=(EPSILON_MIN, EPSILON_MAX),
        rho=RHO,
        input_scaling=INP_SCALING,
        topology="full",
        device=device,
    ).to(device)
    
    print(f"Model: 1-layer RON with {N_HID} units")
    print(f"Parameters: {sum(p.numel() for p in model_standard.parameters())}")
    
    result_standard = train_and_evaluate(
        "Standard 1-layer RON",
        model_standard,
        train_loader,
        valid_loader,
        test_loader
    )
    results.append(result_standard)
    
    # Clean up
    del model_standard
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # Test: 5-layer Antisymmetric RON with different coupling strengths
    # ========================================
    print("\n" + "="*80)
    print("COUPLING STRENGTH SEARCH: 5-layer Antisymmetric RON")
    print("="*80)
    
    # Test different coupling epsilon values
    coupling_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    for coup_eps in coupling_values:
        print(f"\n{'='*60}")
        print(f"Testing coupling_epsilon = {coup_eps}")
        print(f"{'='*60}")
        
        model_antisym = DeepRandomizedOscillatorsNetwork(
            n_inp=n_inp,
            total_units=N_HID,
            n_layers=5,
            dt=DT,
            gamma=(GAMMA_MIN, GAMMA_MAX),
            epsilon=(EPSILON_MIN, EPSILON_MAX),
            rho=RHO,
            input_scaling=INP_SCALING,
            inter_scaling=INP_SCALING,
            topology="full",
            concat=True,
            antisymmetric_coupling=True,
            coupling_epsilon=coup_eps,
            device=device,
        ).to(device)
        
        print(f"Model: 5-layer DeepRON with antisymmetric coupling")
        print(f"Parameters: {sum(p.numel() for p in model_antisym.parameters())}")
        
        result_antisym = train_and_evaluate(
            f"Antisymmetric 5-layer RON (ε_c={coup_eps})",
            model_antisym,
            train_loader,
            valid_loader,
            test_loader
        )
        result_antisym['coupling_epsilon'] = coup_eps
        results.append(result_antisym)
        
        # Clean up
        del model_antisym
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
        # Sort by test accuracy
        results_sorted = sorted(valid_results, key=lambda x: x['test_acc'], reverse=True)
        
        print(f"\n{'Rank':<5} {'Model':<50} {'Coupling ε':<12} {'Test Acc':<12} {'Valid Acc':<12} {'Sat %'}")
        print("-"*105)
        
        for rank, r in enumerate(results_sorted, 1):
            coup_str = f"{r.get('coupling_epsilon', 'N/A'):.2f}" if 'coupling_epsilon' in r else "N/A"
            antisym_marker = "✓" if 'Antisymmetric' in r['name'] else " "
            print(f"{rank:<5} {antisym_marker} {r['name']:<48} {coup_str:<12} "
                  f"{r['test_acc']*100:<11.2f}% {r['valid_acc']*100:<11.2f}% {r['saturated_pct']:.2f}%")
        
        # Key findings
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        best = results_sorted[0]
        baseline = [r for r in valid_results if r['name'] == 'Standard 1-layer RON']
        baseline = baseline[0] if baseline else None
        
        antisym_results = [r for r in results_sorted if 'Antisymmetric' in r['name']]
        best_antisym = antisym_results[0] if antisym_results else None
        
        print(f"\nBest overall: {best['name']}")
        print(f"  Test Accuracy: {best['test_acc']*100:.2f}%")
        if 'coupling_epsilon' in best:
            print(f"  Coupling Epsilon: {best['coupling_epsilon']}")
        
        if baseline:
            print(f"\nBaseline (Standard 1-layer RON):")
            print(f"  Test Accuracy: {baseline['test_acc']*100:.2f}%")
        
        if best_antisym:
            print(f"\nBest Antisymmetric (5-layer RON):")
            print(f"  Test Accuracy: {best_antisym['test_acc']*100:.2f}%")
            print(f"  Coupling Epsilon: {best_antisym['coupling_epsilon']}")
            
            if baseline:
                improvement = (best_antisym['test_acc'] - baseline['test_acc']) * 100
                print(f"\nAntisymmetric vs Baseline: {improvement:+.2f} percentage points")
                
                if best_antisym['test_acc'] > baseline['test_acc']:
                    print("✓ Antisymmetric coupling provides improvement!")
                else:
                    print("✗ Antisymmetric coupling does not improve over baseline")
        
        # Plot coupling strength vs accuracy
        print("\n" + "="*80)
        print("COUPLING STRENGTH ANALYSIS")
        print("="*80)
        
        print(f"\n{'Coupling ε':<12} {'Test Acc':<12} {'Valid Acc':<12}")
        print("-"*36)
        for r in results_sorted:
            if 'coupling_epsilon' in r:
                print(f"{r['coupling_epsilon']:<12.2f} {r['test_acc']*100:<11.2f}% {r['valid_acc']*100:<11.2f}%")
    
    print("\n" + "="*80)
    
    # Save results to file
    result_file = "results_smnist_ron_antisymmetric.txt"
    with open(result_file, 'w') as f:
        f.write("sMNIST RON Antisymmetric Coupling Results\n")
        f.write("="*80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  N_HID: {N_HID}\n")
        f.write(f"  DT: {DT}\n")
        f.write(f"  RHO: {RHO}\n")
        f.write(f"  INPUT_SCALING: {INP_SCALING}\n")
        f.write(f"  EPSILON: ({EPSILON_MIN}, {EPSILON_MAX})\n")
        f.write(f"  GAMMA: ({GAMMA_MIN}, {GAMMA_MAX})\n\n")
        
        f.write("Results:\n")
        f.write("-"*80 + "\n")
        for r in results_sorted:
            f.write(f"{r['name']}\n")
            if 'coupling_epsilon' in r:
                f.write(f"  Coupling Epsilon: {r['coupling_epsilon']}\n")
            f.write(f"  Train Acc: {r['train_acc']*100:.2f}%\n")
            f.write(f"  Valid Acc: {r['valid_acc']*100:.2f}%\n")
            f.write(f"  Test Acc: {r['test_acc']*100:.2f}%\n")
            f.write(f"  Saturation: {r['saturated_pct']:.2f}%\n\n")
    
    print(f"Results saved to: {result_file}")
