import argparse
import os
import warnings
import numpy as np
import torch
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import json
from datetime import datetime
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# import loguniform sampling

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.archetypes.utils import count_parameters

# Constants for the speech dataset
TIME_STEPS = 101
FEATURE_DIM = 40

parser = argparse.ArgumentParser(description="Speech Random Hyperparameter Search")
# Data Args
parser.add_argument("--dataroot", type=str, required=True, help="Directory containing speech .npy files (one per class)")
parser.add_argument("--resultroot", type=str, help="Directory to save results")
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")

# Search Args
parser.add_argument("--n_trials", type=int, default=50, help="Number of random trials")
parser.add_argument("--parallel", action="store_true", help="Run trials in parallel")
parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")

# Model args (which are fixed)
parser.add_argument("--esn", action="store_true", help="Use Echo State Network")
parser.add_argument("--ron", action="store_true", help="Use Randomized Oscillator Network")
parser.add_argument("--deepron", action="store_true", help="Use Deep Randomized Oscillator Network")
parser.add_argument("--pron", action="store_true", help="Use Physically Implementable RON")
parser.add_argument("--topology", type=str, default="full", 
                    choices=["full", "ring", "band", "lower", "toeplitz", "orthogonal", "antisymmetric"],
                    help="Topology of the reservoir")

parser.add_argument("--use_last_state", action="store_true", help="Use last hidden state instead of all states")
parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
parser.add_argument("--batch", type=int, default=64, help="Batch size")

args = parser.parse_args()

# Validate arguments
if args.dataroot is None:
    raise ValueError("--dataroot is required")

if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()
    
if not os.path.exists(args.resultroot):
    os.makedirs(args.resultroot, exist_ok=True)

if not any([args.esn, args.ron, args.pron, args.deepron]):
    parser.error("Must specify at least one model type (--esn, --ron, --pron, or --deepron)")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
print(f"Using device: {device}")

# Set deterministic behavior for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_speech_data(dataroot):
    """Load speech data from .npy files in the specified directory."""
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
    np.random.seed(42)
    indices = np.random.permutation(combined_data.shape[0])
    train_size = int(combined_data.shape[0] * 0.7)  # 70% train
    
    X_train = combined_data[indices[:train_size]]
    y_train = combined_labels[indices[:train_size]]
    X_test = combined_data[indices[train_size:]]
    y_test = combined_labels[indices[train_size:]]
    
    return X_train, y_train, X_test, y_test, class_names


def get_random_config():
    """Generate random hyperparameter configuration based on model type."""
    config = {}
    
    # Common parameters
    if args.esn or args.deepron:
        config["rho"] = np.random.uniform(0.1, 90)  # Spectral radius
        config["leaky"] = np.random.uniform(0.1, 1.0)  # Leaky integration
    
    if args.ron or args.deepron or args.pron:
        # sample with loguniform distribution to ensure good coverage
        config["dt"] = np.random.uniform(np.log10(0.001), np.log10(1))
        config["gamma"] = np.random.uniform(0.5, 5.0)  # Gamma parameter
        config["gamma_range"] = np.random.uniform(0.0, 2.0)  # Gamma range
        config["epsilon"] = np.random.uniform(0.5, 5.0)  # Epsilon parameter
        config["epsilon_range"] = np.random.uniform(0.0, 2.0)  # Epsilon range
        
    # All models
    config["n_hid"] = int(np.random.choice([100]))  # Hidden size
    config["input_scaling"] = np.random.uniform(0.1, 1.0)  # Input scaling
    
    # For deep models
    if args.deepron:
        n_layers = int(np.random.choice([1]))
        config["n_layers"] = n_layers
        config["inter_scaling"] = np.random.uniform(0.1, 1.0)  # Inter-layer scaling
    else:
        config["n_layers"] = 1
        config["inter_scaling"] = np.random.uniform(0.1, 1)  # Default to input scaling
    # For non-physically implementable models
    if args.esn or args.ron or args.deepron:
        config["sparsity"] = np.random.uniform(0.0, 0.0)  # Sparsity
    
    return config


@torch.no_grad()
def evaluate_model(model, X_data, y_data, scaler, classifier, use_last_state=False, batch_size=64):
    """Evaluate a model on the given data."""
    activations = []
    
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    
    for i in range(0, X_data.shape[0], batch_size):
        batch_X = X_tensor[i:i+batch_size].to(device)
        output = model(batch_X)
        
        if isinstance(output, tuple):
            states = output[0]
            last_hidden = output[1]
        else:
            states = output
            last_hidden = states[:, -1, :]
        
        if use_last_state:
            if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                final_state = last_hidden[-1].cpu().numpy()
            else:
                final_state = last_hidden.cpu().numpy()
            activations.append(final_state)
        else:
            pooled_states = states.mean(dim=1).cpu().numpy()
            activations.append(pooled_states)
    
    activations = np.concatenate(activations, axis=0)
    activations = scaler.transform(activations)
    
    return classifier.score(activations, y_data)


def train_model(config, X_train, y_train, X_test, y_test):
    """Train and evaluate a model with the given configuration."""
    try:
        # Set up model parameters from config
        n_inp = FEATURE_DIM
        
        if args.esn:
            units_per_layer = config["n_hid"] // config["n_layers"]
            model = DeepReservoir(
                input_size=n_inp,
                tot_units=config["n_hid"],
                spectral_radius=config["rho"],
                input_scaling=config["input_scaling"],
                inter_scaling=config.get("inter_scaling", config["input_scaling"]),
                connectivity_recurrent=int((1 - config["sparsity"]) * units_per_layer),
                connectivity_input=int((1 - config["sparsity"]) * units_per_layer),
                connectivity_inter=int((1 - config["sparsity"]) * units_per_layer),
                leaky=config["leaky"],
                cycle=True,
                n_layers=config["n_layers"],
            ).to(device)
        
        elif args.ron:
            gamma = (config["gamma"] - config["gamma_range"] / 2.0, 
                    config["gamma"] + config["gamma_range"] / 2.0)
            epsilon = (config["epsilon"] - config["epsilon_range"] / 2.0, 
                      config["epsilon"] + config["epsilon_range"] / 2.0)
            
            model = RandomizedOscillatorsNetwork(
                n_inp,
                config["n_hid"],
                config["dt"],
                gamma,
                epsilon,
                0.0,  # Diffusive gamma
                config.get("rho", 0.99),
                config["input_scaling"],
                topology=args.topology,
                sparsity=config["sparsity"],
                device=device,
            ).to(device)
        
        elif args.deepron:
            gamma = (config["gamma"] - config["gamma_range"] / 2.0, 
                    config["gamma"] + config["gamma_range"] / 2.0)
            epsilon = (config["epsilon"] - config["epsilon_range"] / 2.0, 
                      config["epsilon"] + config["epsilon_range"] / 2.0)
            
            # Create n_hid_layers list where each layer has n_hid / n_layers units
            n_hid_per_layer = config["n_hid"] // config["n_layers"]
            n_hid_layers = [n_hid_per_layer] * config["n_layers"]
            
            model = DeepRandomizedOscillatorsNetwork(
                n_inp,
                config["n_hid"],
                n_hid_layers,
                config["dt"],
                gamma,
                epsilon,
                0.0,  # Diffusive gamma
                config["rho"],
                config["input_scaling"],
                device=device,
            ).to(device)
        
        elif args.pron:
            gamma = (config["gamma"] - config["gamma_range"] / 2.0, 
                    config["gamma"] + config["gamma_range"] / 2.0)
            epsilon = (config["epsilon"] - config["epsilon_range"] / 2.0, 
                      config["epsilon"] + config["epsilon_range"] / 2.0)
            
            model = PhysicallyImplementableRandomizedOscillatorsNetwork(
                n_inp,
                config["n_hid"],
                config["dt"],
                gamma,
                epsilon,
                config["input_scaling"],
                device=device,
            ).to(device)
        
        else:
            raise ValueError("No model type specified")
        
        # Process training data
        model.eval()
        activations = []
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        batch_size = args.batch
        
        for j in range(0, X_train.shape[0], batch_size):
            batch_X = X_train_tensor[j:j+batch_size].to(device)
            output = model(batch_X)
            
            if isinstance(output, tuple):
                states = output[0]
                last_hidden = output[1]
            else:
                states = output
                last_hidden = states[:, -1, :]
            
            if args.use_last_state:
                if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                    final_state = last_hidden[-1].cpu().numpy()
                else:
                    final_state = last_hidden.cpu().numpy()
                activations.append(final_state)
            else:
                pooled_states = states.mean(dim=1).cpu().numpy()
                activations.append(pooled_states)
        
        activations = np.concatenate(activations, axis=0)
        
        # Scale data and train classifier
        scaler = preprocessing.StandardScaler().fit(activations)
        activations = scaler.transform(activations)
        classifier = LogisticRegression(max_iter=1000).fit(activations, y_train)
        
        # Evaluate
        train_acc = evaluate_model(model, X_train, y_train, scaler, classifier, args.use_last_state, args.batch)
        test_acc = evaluate_model(model, X_test, y_test, scaler, classifier, args.use_last_state, args.batch)
        
        # Calculate model complexity
        n_params = count_parameters(model)
        
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "n_parameters": n_params,
            "config": config,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error training model: {e}")
        return {
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "n_parameters": 0,
            "config": config,
            "status": f"error: {str(e)}"
        }


def run_trial(trial_id, X_train, y_train, X_test, y_test):
    """Run a single trial with random configuration."""
    np.random.seed(42 + trial_id)  # Ensure different configs for different trials
    config = get_random_config()
    
    print(f"Trial {trial_id+1}/{args.n_trials}: Starting with config: {config}")
    start_time = time.time()
    
    result = train_model(config, X_train, y_train, X_test, y_test)
    result["trial_id"] = trial_id
    result["execution_time"] = time.time() - start_time
    
    print(f"Trial {trial_id+1}/{args.n_trials}: "
          f"train_accuracy={result['train_accuracy']:.4f}, "
          f"test_accuracy={result['test_accuracy']:.4f}, "
          f"execution_time={result['execution_time']:.2f}s")
    
    return result


def main():
    # Load speech data
    print("Loading speech data...")
    X_train, y_train, X_test, y_test, class_names = load_speech_data(args.dataroot)
    print(f"Loaded data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Number of classes: {len(class_names)}")
    
    # Define which model type we're searching
    model_type = "ESN" if args.esn else "RON" if args.ron else "DeepRON" if args.deepron else "PRON" if args.pron else "Unknown"
    print(f"Running random search for model type: {model_type}")
    
    # Create a timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        args.resultroot, 
        f"speech_random_search_{model_type}{args.resultsuffix}_{timestamp}.json"
    )
    
    # Run trials
    results = []
    
    if args.parallel:
        # Parallel execution
        print(f"Running {args.n_trials} trials in parallel")
        num_workers = args.num_workers if args.num_workers is not None else mp.cpu_count()
        print(f"Using {num_workers} workers")
        
        with mp.Pool(num_workers) as pool:
            partial_run_trial = partial(run_trial, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            results = list(tqdm(pool.imap(partial_run_trial, range(args.n_trials)), total=args.n_trials))
    else:
        # Sequential execution
        print(f"Running {args.n_trials} trials sequentially")
        for i in tqdm(range(args.n_trials)):
            result = run_trial(i, X_train, y_train, X_test, y_test)
            results.append(result)
            
            # Save intermediate results
            if (i + 1) % 5 == 0:
                with open(results_file, 'w') as f:
                    json.dump({
                        'model_type': model_type,
                        'timestamp': timestamp,
                        'args': vars(args),
                        'results': results
                    }, f, indent=2)
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['test_accuracy'])
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump({
            'model_type': model_type,
            'timestamp': timestamp,
            'args': vars(args),
            'best_result': best_result,
            'results': results
        }, f, indent=2)
    
    print("\nRandom Search Complete!")
    print(f"Best test accuracy: {best_result['test_accuracy']:.4f}")
    print(f"Best configuration: {best_result['config']}")
    print(f"Results saved to: {results_file}")


    

if __name__ == "__main__":
    main()