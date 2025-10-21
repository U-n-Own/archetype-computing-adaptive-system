"""
Test comparing standard 1-layer ESN vs 5-layer antisymmetric ESN on Speech Commands
with epsilon coupling strength search.
"""
import os
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import DeepReservoir

print("=" * 80)
print("Speech Commands: Standard 1-layer vs Antisymmetric 5-layer with Epsilon Search")
print("=" * 80)

# Configuration
DATAROOT = "speech"
BATCH_SIZE = 64
N_HID = 500
RHO = 0.9
INP_SCALING = 1.0
LEAKY = 0.001  # Low leaky rate
SEED = 42

# Set random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


def load_speech_data(dataroot: str):
    """Load speech data from .npy files in the specified directory."""
    # Constants for the speech dataset
    TIME_STEPS = 101
    FEATURE_DIM = 40
    
    speech_files = [f for f in os.listdir(dataroot) if f.endswith('.npy')]
    if not speech_files:
        raise ValueError(f"No .npy files found in {dataroot}")
    
    all_data = []
    all_labels = []
    class_names = []
    
    for class_idx, file_name in enumerate(tqdm(speech_files, desc="Loading data files")):
        filepath = os.path.join(dataroot, file_name)
        class_name = file_name.split('.')[0]  # Use filename without extension as class name
        class_names.append(class_name)
        
        try:
            data = np.load(filepath, allow_pickle=True)
            # Basic validation
            if data.ndim != 3 or data.shape[1] != TIME_STEPS or data.shape[2] != FEATURE_DIM:
                print(f"Skipping {file_name}: unexpected shape {data.shape}")
                continue
            if data.shape[0] == 0:
                print(f"Skipping {file_name}: no samples found")
                continue

            all_data.append(data)
            all_labels.append(np.full(data.shape[0], class_idx))  # Assign class index as label
            print(f"Loaded {file_name}: {data.shape[0]} samples")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data loaded. Check files and expected dimensions.")
    
    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    # Split into train and test sets
    np.random.seed(SEED)
    indices = np.random.permutation(combined_data.shape[0])
    train_size = int(combined_data.shape[0] * 0.8)  # 80% train
    
    X_train = combined_data[indices[:train_size]]
    y_train = combined_labels[indices[:train_size]]
    X_test = combined_data[indices[train_size:]]
    y_test = combined_labels[indices[train_size:]]
    
    # Normalize input features
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    input_feature_scaler = preprocessing.StandardScaler()
    X_train_scaled_reshaped = input_feature_scaler.fit_transform(X_train_reshaped)
    X_test_scaled_reshaped = input_feature_scaler.transform(X_test_reshaped)
    
    # Reshape back to original: (samples, time_steps, features)
    X_train = X_train_scaled_reshaped.reshape(X_train.shape)
    X_test = X_test_scaled_reshaped.reshape(X_test.shape)
    
    print("Input features normalized.")
    
    return X_train, y_train, X_test, y_test, class_names


@torch.no_grad()
def test(model, X_data, y_data, scaler, classifier, batch_size=256):
    """Evaluate model on a dataset using mean pooling."""
    activations = []
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    
    for i in tqdm(range(0, X_data.shape[0], batch_size), desc="Evaluating", leave=False):
        batch_X = X_tensor[i:i+batch_size].to(device)
        output = model(batch_X)
        
        if isinstance(output, tuple):
            states = output[0]  # (batch, time, features)
        else:
            states = output
        
        # Mean pooling over time dimension
        pooled_states = states.mean(dim=1).cpu().numpy()
        activations.append(pooled_states)
    
    activations = np.concatenate(activations, axis=0)
    activations = scaler.transform(activations)
    
    return classifier.score(activations, y_data)


def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test, batch_size=64):
    """Train and evaluate a model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Collect training activations
    activations = []
    print("Collecting training activations...")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    for j in tqdm(range(0, X_train.shape[0], batch_size), desc="Processing train batches"):
        batch_X = X_train_tensor[j:j+batch_size].to(device)
        output = model(batch_X)
        
        if isinstance(output, tuple):
            states = output[0]  # (batch, time, features)
        else:
            states = output
        
        # Mean pooling over time dimension
        pooled_states = states.mean(dim=1).cpu().numpy()
        activations.append(pooled_states)
    
    activations = np.concatenate(activations, axis=0)
    
    print(f"Activation statistics:")
    print(f"  Shape: {activations.shape}")
    print(f"  Mean: {activations.mean():.6f}, Std: {activations.std():.6f}")
    print(f"  Min: {activations.min():.6f}, Max: {activations.max():.6f}")
    
    # Check saturation
    saturated = np.sum(np.abs(activations) > 0.99) / activations.size
    print(f"  Saturated (|x| > 0.99): {saturated*100:.2f}%")
    
    # Scale and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations_scaled = scaler.transform(activations)
    
    print("Training logistic regression classifier...")
    classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    classifier.fit(activations_scaled, y_train)
    
    # Evaluate
    print("Evaluating...")
    train_acc = classifier.score(activations_scaled, y_train)
    test_acc = test(model, X_test, y_test, scaler, classifier, batch_size=256)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    
    return {
        'name': model_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'saturated_pct': saturated * 100
    }


if __name__ == "__main__":
    # Load data
    print("Loading Speech Commands dataset...")
    X_train, y_train, X_test, y_test, class_names = load_speech_data(DATAROOT)
    n_inp = X_train.shape[2]  # Feature dimension (should be 40)
    n_out = len(class_names)  # Number of classes
    print(f"Loaded data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Input dimension: {n_inp}, Number of classes: {n_out}\n")
    
    results = []
    
    # ========================================
    # Baseline: Standard 1-layer ESN
    # ========================================
    print("\n" + "="*80)
    print("BASELINE: Standard 1-layer ESN")
    print("="*80)
    
    units_per_layer = N_HID // 1
    model_standard = DeepReservoir(
        input_size=n_inp,
        tot_units=N_HID,
        spectral_radius=RHO,
        n_layers=1,
        input_scaling=INP_SCALING,
        inter_scaling=INP_SCALING,
        connectivity_recurrent=units_per_layer,
        connectivity_input=units_per_layer,
        connectivity_inter=units_per_layer,
        leaky=LEAKY,
        cycle=False,
        linear=False,
        antisymmetric=False,
    ).to(device)
    
    result_standard = train_and_evaluate(
        "Standard 1-layer",
        model_standard,
        X_train, y_train,
        X_test, y_test,
        batch_size=BATCH_SIZE
    )
    results.append(result_standard)
    
    # Clean up
    del model_standard
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # Test: 5-layer Antisymmetric with different epsilon values
    # ========================================
    print("\n" + "="*80)
    print("EPSILON SEARCH: 5-layer Antisymmetric ESN")
    print("="*80)
    
    # Test different epsilon values
    epsilon_values = [0.001, 0.01, 0.5]
    
    for eps in epsilon_values:
        units_per_layer_antisym = N_HID // 5
        model_antisym = DeepReservoir(
            input_size=n_inp,
            tot_units=N_HID,
            spectral_radius=RHO,
            n_layers=5,
            input_scaling=INP_SCALING,
            inter_scaling=INP_SCALING,
            connectivity_recurrent=units_per_layer_antisym,
            connectivity_input=units_per_layer_antisym,
            connectivity_inter=units_per_layer_antisym,
            leaky=LEAKY,
            cycle=False,
            linear=False,
            epsilon=eps,
            antisymmetric=True,
        ).to(device)
        
        result_antisym = train_and_evaluate(
            f"Antisymmetric 5-layer (ε={eps})",
            model_antisym,
            X_train, y_train,
            X_test, y_test,
            batch_size=BATCH_SIZE
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
    
    print(f"\n{'Rank':<5} {'Model':<45} {'Epsilon':<10} {'Test Acc':<12} {'Train Acc':<12} {'Saturated %'}")
    print("-"*100)
    
    for rank, r in enumerate(results_sorted, 1):
        eps_str = f"{r.get('epsilon', 'N/A'):.3f}" if 'epsilon' in r else "N/A"
        antisym_marker = "✓" if 'Antisymmetric' in r['name'] else " "
        print(f"{rank:<5} {antisym_marker} {r['name']:<43} {eps_str:<10} "
              f"{r['test_acc']*100:<11.2f}% {r['train_acc']*100:<11.2f}% {r['saturated_pct']:.2f}%")
    
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
