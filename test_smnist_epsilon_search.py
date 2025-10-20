"""
Test comparing standard 1-layer ESN vs 5-layer antisymmetric ESN on sMNIST
with epsilon coupling strength search.
"""
import os
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import DeepReservoir
from acds.benchmarks import get_mnist_data

print("=" * 80)
print("sMNIST: Standard 1-layer vs Antisymmetric 5-layer with Epsilon Search")
print("=" * 80)

# Configuration
DATAROOT = "data"
BATCH_SIZE = 1000
N_HID = 500
RHO = 0.9
INP_SCALING = 1.0
LEAKY = 0.001  # Low leaky rate as used in previous sMNIST experiments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

n_inp = 1
n_out = 10


@torch.no_grad()
def test(data_loader, model, classifier, scaler):
    """Evaluate model on a dataset."""
    activations, ys = [], []
    for images, labels in tqdm(data_loader, desc="Testing", leave=False):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output = model(images)[-1][0]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
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
        output = model(images)[-1][0]
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    print(f"Activation statistics:")
    print(f"  Shape: {activations.shape}")
    print(f"  Mean: {activations.mean():.6f}, Std: {activations.std():.6f}")
    print(f"  Min: {activations.min():.6f}, Max: {activations.max():.6f}")
    
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
        'saturated_pct': saturated * 100
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
    # Baseline: Standard 1-layer ESN
    # ========================================
    print("\n" + "="*80)
    print("BASELINE: Standard 1-layer ESN")
    print("="*80)
    
    model_standard = DeepReservoir(
        input_size=n_inp,
        tot_units=N_HID,
        spectral_radius=RHO,
        n_layers=1,
        input_scaling=INP_SCALING,
        connectivity_recurrent=N_HID,
        connectivity_input=N_HID,
        connectivity_inter=N_HID,
        leaky=LEAKY,
        concat=True,
        antisymmetric=False,
    ).to(device)
    
    result_standard = train_and_evaluate(
        "Standard 1-layer",
        model_standard,
        train_loader,
        valid_loader,
        test_loader
    )
    results.append(result_standard)
    
    # ========================================
    # Test: 5-layer Antisymmetric with different epsilon values
    # ========================================
    print("\n" + "="*80)
    print("EPSILON SEARCH: 5-layer Antisymmetric ESN")
    print("="*80)
    
    # Test different epsilon values
    epsilon_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    for eps in epsilon_values:
        model_antisym = DeepReservoir(
            input_size=n_inp,
            tot_units=N_HID,
            spectral_radius=RHO,
            n_layers=5,
            input_scaling=INP_SCALING,
            connectivity_recurrent=N_HID // 5,
            connectivity_input=N_HID // 5,
            connectivity_inter=N_HID // 5,
            leaky=LEAKY,
            epsilon=eps,
            concat=True,
            antisymmetric=True,
        ).to(device)
        
        result_antisym = train_and_evaluate(
            f"Antisymmetric 5-layer (ε={eps})",
            model_antisym,
            train_loader,
            valid_loader,
            test_loader
        )
        result_antisym['epsilon'] = eps
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
    
    # Sort by test accuracy
    results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<40} {'Epsilon':<10} {'Test Acc':<12} {'Valid Acc':<12} {'Saturated %'}")
    print("-"*95)
    
    for rank, r in enumerate(results_sorted, 1):
        eps_str = f"{r.get('epsilon', 'N/A'):.3f}" if 'epsilon' in r else "N/A"
        antisym_marker = "✓" if 'Antisymmetric' in r['name'] else " "
        print(f"{rank:<5} {antisym_marker} {r['name']:<38} {eps_str:<10} "
              f"{r['test_acc']*100:<11.2f}% {r['valid_acc']*100:<11.2f}% {r['saturated_pct']:.2f}%")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    best = results_sorted[0]
    baseline = [r for r in results if r['name'] == 'Standard 1-layer'][0]
    best_antisym = [r for r in results_sorted if 'Antisymmetric' in r['name']][0]
    
    print(f"\nBest overall: {best['name']}")
    print(f"  Test Accuracy: {best['test_acc']*100:.2f}%")
    if 'epsilon' in best:
        print(f"  Epsilon: {best['epsilon']}")
    
    print(f"\nBaseline (Standard 1-layer):")
    print(f"  Test Accuracy: {baseline['test_acc']*100:.2f}%")
    
    print(f"\nBest Antisymmetric (5-layer):")
    print(f"  Test Accuracy: {best_antisym['test_acc']*100:.2f}%")
    print(f"  Epsilon: {best_antisym['epsilon']}")
    
    improvement = (best_antisym['test_acc'] - baseline['test_acc']) * 100
    print(f"\nAntisymmetric vs Baseline: {improvement:+.2f} percentage points")
    
    if best_antisym['test_acc'] > baseline['test_acc']:
        print("✓ Antisymmetric coupling provides improvement!")
    else:
        print("✗ Antisymmetric coupling does not improve over baseline")
    
    print("\n" + "="*80)
