"""
Path-X Benchmark: Compare 1-layer ESN vs 5-layer Antisymmetric ESN
Tests long-range dependency capabilities on the challenging Path-X task.
"""
import argparse
import os
import time
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd

from acds.archetypes import DeepReservoir
from acds.benchmarks.pathx import get_pathx_data, visualize_pathx_samples, visualize_pathx_predictions

parser = argparse.ArgumentParser(description="Path-X benchmark for ESN architectures")
parser.add_argument("--resultroot", type=str, default="./experiments")
parser.add_argument("--resolution", type=int, default=16, help="Grid resolution (16x16 = 256 seq len)")
parser.add_argument("--n_hid", type=int, default=512, help="Total hidden units")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--train_samples", type=int, default=8000, help="Training samples")
parser.add_argument("--val_samples", type=int, default=1000, help="Validation samples")
parser.add_argument("--test_samples", type=int, default=1000, help="Test samples")
parser.add_argument("--inp_scaling", type=float, default=1.0, help="ESN input scaling")
parser.add_argument("--rho", type=float, default=0.99, help="ESN spectral radius")
parser.add_argument("--leaky", type=float, default=1.0, help="ESN leaky rate")
parser.add_argument("--epsilon", type=float, default=2.0, help="Antisymmetric coupling strength")
parser.add_argument("--trials", type=int, default=3, help="Number of trials per architecture")
parser.add_argument("--difficulty", type=float, default=0.3, help="Path-X difficulty (wall probability)")
parser.add_argument("--visualize", action="store_true", help="Generate visualizations of Path-X grids")
parser.add_argument("--cpu", action="store_true")

args = parser.parse_args()

os.makedirs(args.resultroot, exist_ok=True)

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)

print("="*80)
print("PATH-X BENCHMARK: ESN Architecture Comparison")
print("="*80)
print(f"Device: {device}")
print(f"Grid Resolution: {args.resolution}x{args.resolution}")
print(f"Sequence Length: {args.resolution * args.resolution}")
print(f"Hidden Units: {args.n_hid}")
print(f"Difficulty: {args.difficulty}")
print(f"Trials: {args.trials}")
print("="*80)


@torch.no_grad()
def test(data_loader, classifier, scaler, model):
    """Test the model on a given dataset."""
    activations, ys = [], []
    for sequences, labels in tqdm(data_loader, desc="Testing", leave=False):
        sequences = sequences.to(device)
        sequences = sequences.unsqueeze(-1)  # Add feature dimension
        
        # Get reservoir output
        output = model(sequences)[-1][0]  # states_last[0]
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    accuracy = classifier.score(activations, ys)
    predictions = classifier.predict(activations)
    
    return accuracy, predictions


def train_and_evaluate(architecture_name, n_layers, antisymmetric, epsilon, trial_idx):
    """Train and evaluate ESN with specified architecture."""
    trial_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"{architecture_name} - Trial {trial_idx + 1}/{args.trials}")
    print(f"{'='*70}")
    print(f"Layers: {n_layers}, Antisymmetric: {antisymmetric}, Epsilon: {epsilon}")
    
    # Create model
    model_start = time.time()
    units_per_layer = args.n_hid // n_layers
    model = DeepReservoir(
        input_size=1,
        tot_units=args.n_hid,
        n_layers=n_layers,
        spectral_radius=args.rho,
        input_scaling=args.inp_scaling,
        inter_scaling=args.inp_scaling,
        connectivity_recurrent=units_per_layer,
        connectivity_input=units_per_layer,
        connectivity_inter=units_per_layer,
        leaky=args.leaky,
        cycle=False,
        linear=False,
        antisymmetric=antisymmetric,
        epsilon=epsilon,
    ).to(device)
    model_time = time.time() - model_start
    print(f"Model creation: {model_time:.2f}s")
    
    # Load data
    data_start = time.time()
    train_loader, val_loader, test_loader = get_pathx_data(
        resolution=args.resolution,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        batch_size=args.batch,
        difficulty=args.difficulty,
    )
    data_time = time.time() - data_start
    print(f"Data generation: {data_time:.2f}s")
    
    # Extract features
    print("Extracting training features...")
    feature_start = time.time()
    activations, ys = [], []
    for sequences, labels in tqdm(train_loader, desc="Training features"):
        sequences = sequences.to(device)
        sequences = sequences.unsqueeze(-1)  # Add feature dimension
        
        # Get reservoir output
        output = model(sequences)[-1][0]  # states_last[0]
        activations.append(output.cpu())
        ys.append(labels)
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    feature_time = time.time() - feature_start
    print(f"Feature extraction: {feature_time:.2f}s")
    print(f"Features shape: {activations.shape}")
    
    # Train classifier
    readout_start = time.time()
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000, solver='lbfgs').fit(activations, ys)
    readout_time = time.time() - readout_start
    print(f"Readout training: {readout_time:.2f}s")
    
    # Evaluate
    print("Evaluating...")
    eval_start = time.time()
    train_acc, train_preds = test(train_loader, classifier, scaler, model)
    val_acc, val_preds = test(val_loader, classifier, scaler, model)
    test_acc, test_preds = test(test_loader, classifier, scaler, model)
    eval_time = time.time() - eval_start
    total_time = time.time() - trial_start
    
    print(f"\n--- Results ---")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc:   {val_acc:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Total Time: {total_time:.2f}s")
    
    return {
        "architecture": architecture_name,
        "n_layers": n_layers,
        "antisymmetric": antisymmetric,
        "epsilon": epsilon,
        "trial": trial_idx + 1,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "model_time": model_time,
        "data_time": data_time,
        "feature_time": feature_time,
        "readout_time": readout_time,
        "eval_time": eval_time,
        "total_time": total_time,
        "test_loader": test_loader,
        "test_preds": test_preds,
    }


# Run experiments
results = []

# Test 1: Standard 1-layer ESN
print("\n" + "="*80)
print("ARCHITECTURE 1: Standard 1-Layer ESN")
print("="*80)
for trial in range(args.trials):
    result = train_and_evaluate(
        architecture_name="ESN-1Layer",
        n_layers=1,
        antisymmetric=False,
        epsilon=0.0,
        trial_idx=trial
    )
    results.append(result)

# Test 2: Deep 5-layer ESN with Antisymmetric Coupling
print("\n" + "="*80)
print("ARCHITECTURE 2: Deep 5-Layer Antisymmetric ESN")
print("="*80)
for trial in range(args.trials):
    result = train_and_evaluate(
        architecture_name="ESN-5Layer-Antisymmetric",
        n_layers=5,
        antisymmetric=True,
        epsilon=args.epsilon,
        trial_idx=trial
    )
    results.append(result)

# Create results DataFrame
df = pd.DataFrame(results)

# Print summary
print("\n" + "="*80)
print("FINAL COMPARISON SUMMARY")
print("="*80)

for arch_name in df["architecture"].unique():
    arch_results = df[df["architecture"] == arch_name]
    
    train_mean = arch_results["train_acc"].mean()
    train_std = arch_results["train_acc"].std()
    val_mean = arch_results["val_acc"].mean()
    val_std = arch_results["val_acc"].std()
    test_mean = arch_results["test_acc"].mean()
    test_std = arch_results["test_acc"].std()
    time_mean = arch_results["total_time"].mean()
    time_std = arch_results["total_time"].std()
    
    print(f"\n{arch_name}:")
    print(f"  Train Accuracy: {train_mean:.4f} ± {train_std:.4f}")
    print(f"  Val Accuracy:   {val_mean:.4f} ± {val_std:.4f}")
    print(f"  Test Accuracy:  {test_mean:.4f} ± {test_std:.4f}")
    print(f"  Total Time:     {time_mean:.1f}s ± {time_std:.1f}s")
    print(f"  Layers: {arch_results.iloc[0]['n_layers']}")
    print(f"  Antisymmetric: {arch_results.iloc[0]['antisymmetric']}")

# Save results
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(args.resultroot, "results_pathx")
os.makedirs(results_dir, exist_ok=True)

detailed_file = os.path.join(results_dir, f"pathx_detailed_{timestamp}.csv")
df.to_csv(detailed_file, index=False)
print(f"\nDetailed results saved to: {detailed_file}")

# Summary statistics
summary_stats = []
for arch_name in df["architecture"].unique():
    arch_results = df[df["architecture"] == arch_name]
    summary_stats.append({
        "Architecture": arch_name,
        "Layers": arch_results.iloc[0]['n_layers'],
        "Antisymmetric": arch_results.iloc[0]['antisymmetric'],
        "Train Acc": f"{arch_results['train_acc'].mean():.4f}±{arch_results['train_acc'].std():.4f}",
        "Val Acc": f"{arch_results['val_acc'].mean():.4f}±{arch_results['val_acc'].std():.4f}",
        "Test Acc": f"{arch_results['test_acc'].mean():.4f}±{arch_results['test_acc'].std():.4f}",
        "Time (s)": f"{arch_results['total_time'].mean():.1f}±{arch_results['total_time'].std():.1f}",
    })

summary_df = pd.DataFrame(summary_stats)
summary_file = os.path.join(results_dir, f"pathx_summary_{timestamp}.csv")
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved to: {summary_file}")

# Generate visualizations if requested
if args.visualize:
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Visualize sample grids
    print("\nGenerating sample Path-X grids...")
    sample_dataset = results[0]["test_loader"].dataset
    viz_path = os.path.join(results_dir, f"pathx_samples_{timestamp}.png")
    visualize_pathx_samples(sample_dataset, num_samples=8, save_path=viz_path)
    
    # Visualize predictions for each architecture
    for arch_name in df["architecture"].unique():
        print(f"\nGenerating prediction visualization for {arch_name}...")
        arch_results = df[df["architecture"] == arch_name]
        
        # Get the last trial's test results
        last_result = results[[i for i, r in enumerate(results) 
                              if r["architecture"] == arch_name][-1]]
        
        test_dataset = last_result["test_loader"].dataset
        predictions = last_result["test_preds"]
        
        # Show 8 random test samples
        indices = np.random.choice(len(test_dataset), min(8, len(test_dataset)), replace=False)
        viz_pred_path = os.path.join(results_dir, 
                                     f"pathx_predictions_{arch_name}_{timestamp}.png")
        visualize_pathx_predictions(test_dataset, predictions, indices, 
                                   save_path=viz_pred_path)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
