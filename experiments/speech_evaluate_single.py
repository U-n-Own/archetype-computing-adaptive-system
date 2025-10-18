#!/usr/bin/env python3
"""
Evaluate a single ESN model on Speech Dataset and generate visualizations
Use this to test specific configurations or analyze saved results
"""

import torch
import numpy as np
import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acds.archetypes.esn import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# Import modular utilities
from utils import (
    plot_predictions,
    plot_state_dynamics,
    plot_eigenspectrum_analysis,
    test_model
)


def load_speech_data(dataroot: str, seed: int = 42):
    """Load and preprocess speech data"""
    from tqdm import tqdm
    
    TIME_STEPS = 101
    FEATURE_DIM = 40
    
    speech_files = [f for f in os.listdir(dataroot) if f.endswith('.npy')]
    if not speech_files:
        raise ValueError(f"No .npy files found in {dataroot}")
    
    all_data = []
    all_labels = []
    class_names = []
    
    print("Loading speech data files...")
    for class_idx, file_name in enumerate(tqdm(speech_files, desc="Loading")):
        filepath = os.path.join(dataroot, file_name)
        class_name = file_name.split('.')[0]
        class_names.append(class_name)
        
        try:
            data = np.load(filepath, allow_pickle=True)
            if data.ndim != 3 or data.shape[1] != TIME_STEPS or data.shape[2] != FEATURE_DIM:
                print(f"Skipping {file_name}: unexpected shape {data.shape}")
                continue
            if data.shape[0] == 0:
                print(f"Skipping {file_name}: no samples found")
                continue

            all_data.append(data)
            all_labels.append(np.full(data.shape[0], class_idx))
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data loaded")
    
    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    # Split into train, validation, and test sets
    np.random.seed(seed)
    indices = np.random.permutation(combined_data.shape[0])
    train_size = int(combined_data.shape[0] * 0.7)
    valid_size = int(combined_data.shape[0] * 0.15)
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]
    
    X_train = combined_data[train_indices]
    y_train = combined_labels[train_indices]
    X_valid = combined_data[valid_indices]
    y_valid = combined_labels[valid_indices]
    X_test = combined_data[test_indices]
    y_test = combined_labels[test_indices]
    
    # Normalize
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_valid_reshaped = X_valid.reshape(-1, X_valid.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    input_scaler = preprocessing.StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train_reshaped)
    X_valid_scaled = input_scaler.transform(X_valid_reshaped)
    X_test_scaled = input_scaler.transform(X_test_reshaped)
    
    X_train = X_train_scaled.reshape(X_train.shape)
    X_valid = X_valid_scaled.reshape(X_valid.shape)
    X_test = X_test_scaled.reshape(X_test.shape)
    
    print(f"\nData: Train={X_train.shape[0]}, Valid={X_valid.shape[0]}, Test={X_test.shape[0]}")
    print(f"Classes: {len(class_names)}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, class_names


def create_model(config: dict, n_inp: int, device: torch.device):
    """
    Create ESN model from configuration
    
    Args:
        config: Dict with model hyperparameters
        n_inp: Input dimension
        device: torch device
        
    Returns:
        model, mode_name
    """
    units_per_layer = config['tot_units'] // config['n_layers']
    
    # Determine mode
    if config.get('antisymmetric', False):
        mode_name = 'Antisymmetric'
    elif config.get('cycle', False):
        mode_name = 'Cycle'
    else:
        mode_name = 'Standard'
    
    model = DeepReservoir(
        input_size=n_inp,
        tot_units=config['tot_units'],
        n_layers=config['n_layers'],
        concat=True,
        spectral_radius=config['spectral_radius'],
        input_scaling=config['input_scaling'],
        leaky=config['leaky'],
        connectivity_recurrent=units_per_layer,
        connectivity_input=units_per_layer,
        connectivity_inter=units_per_layer,
        antisymmetric=config.get('antisymmetric', False),
        epsilon=config.get('epsilon', 0.1),
        cycle=config.get('cycle', False),
        linear=False,
    ).to(device)
    
    return model, mode_name


def train_readout(model, X_train, y_train, device: torch.device):
    """
    Train readout layer
    
    Returns:
        classifier, scaler
    """
    print("Training readout layer...")
    activations = []
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    batch_size = 64
    
    for i in range(0, X_train.shape[0], batch_size):
        batch_X = X_tensor[i:i+batch_size].to(device)
        
        with torch.no_grad():
            output = model(batch_X)
            if isinstance(output, tuple):
                states = output[0]
            else:
                states = output
            
            if len(states.shape) == 3:
                final_state = states[:, -1, :].cpu().numpy()
            else:
                final_state = states.cpu().numpy()
            
            activations.append(final_state)
    
    activations = np.concatenate(activations, axis=0)
    
    scaler = preprocessing.StandardScaler().fit(activations)
    activations_scaled = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations_scaled, y_train)
    
    print("✓ Readout trained")
    return classifier, scaler


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate single ESN configuration on Speech Dataset'
    )
    parser.add_argument('--dataroot', type=str, default='./speech',
                       help='Path to speech data directory')
    parser.add_argument('--mode', type=str, default='antisymmetric',
                       choices=['antisymmetric', 'cycle', 'standard'],
                       help='ESN mode to evaluate')
    parser.add_argument('--n_layers', type=int, default=5,
                       help='Number of layers')
    parser.add_argument('--tot_units', type=int, default=500,
                       help='Total number of reservoir units')
    parser.add_argument('--spectral_radius', type=float, default=0.99,
                       help='Spectral radius')
    parser.add_argument('--leaky', type=float, default=0.1,
                       help='Leaky integration rate')
    parser.add_argument('--input_scaling', type=float, default=1.0,
                       help='Input scaling')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Epsilon for antisymmetric mode')
    parser.add_argument('--save_dir', type=str, default='./experiments/plots/speech_single_eval',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--load_config', type=str, default=None,
                       help='Path to JSON config file (overrides other params)')
    parser.add_argument('--n_examples', type=int, default=6,
                       help='Number of prediction examples to visualize')
    
    args = parser.parse_args()
    
    # Setup
    print(f"\n{'='*70}")
    print("SINGLE MODEL EVALUATION - SPEECH DATASET")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load configuration
    if args.load_config:
        print(f"Loading configuration from: {args.load_config}")
        with open(args.load_config, 'r') as f:
            config_data = json.load(f)
        
        # Extract config from saved results
        if 'antisymmetric' in config_data and 'best_config' in config_data['antisymmetric']:
            config = config_data['antisymmetric']['best_config'].copy()
            config['antisymmetric'] = True
            config['cycle'] = False
            config['n_layers'] = config_data.get('n_layers', args.n_layers)
            mode_name = 'Antisymmetric'
        elif 'cycle' in config_data and 'best_config' in config_data['cycle']:
            config = config_data['cycle']['best_config'].copy()
            config['antisymmetric'] = False
            config['cycle'] = True
            config['n_layers'] = config_data.get('n_layers', args.n_layers)
            mode_name = 'Cycle'
        else:
            raise ValueError("Could not find valid config in JSON file")
        
        print(f"Loaded {mode_name} configuration")
    else:
        # Build config from command line args
        config = {
            'n_layers': args.n_layers,
            'tot_units': args.tot_units,
            'spectral_radius': args.spectral_radius,
            'leaky': args.leaky,
            'input_scaling': args.input_scaling,
            'epsilon': args.epsilon,
            'antisymmetric': args.mode == 'antisymmetric',
            'cycle': args.mode == 'cycle',
        }
        mode_name = args.mode.capitalize()
    
    print(f"\nModel Configuration ({mode_name}):")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print()
    
    # Load data
    print("Loading speech data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names = load_speech_data(
        args.dataroot, seed=args.seed
    )
    n_inp = X_train.shape[2]
    print(f"✓ Data loaded (input_dim={n_inp}, n_classes={len(class_names)})\n")
    
    # Create model
    print(f"Creating {mode_name} model...")
    model, detected_mode = create_model(config, n_inp, device)
    print(f"✓ Model created\n")
    
    # Train readout
    classifier, scaler = train_readout(model, X_train, y_train, device)
    
    # Evaluate
    print("\nEvaluating model...")
    train_acc = test_model(model, (X_train, y_train), classifier, scaler, device, is_loader=False)
    valid_acc = test_model(model, (X_valid, y_valid), classifier, scaler, device, is_loader=False)
    test_acc = test_model(model, (X_test, y_test), classifier, scaler, device, is_loader=False)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Train Accuracy:      {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Validation Accuracy: {valid_acc:.4f} ({valid_acc*100:.2f}%)")
    print(f"Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = {
        'mode': mode_name,
        'config': config,
        'train_accuracy': float(train_acc),
        'valid_accuracy': float(valid_acc),
        'test_accuracy': float(test_acc),
        'n_classes': len(class_names),
        'class_names': class_names,
    }
    
    with open(f"{args.save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {args.save_dir}/results.json")
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # Create models_dict for visualization functions
    models_dict = {
        mode_name.lower(): {
            'model': model,
            'classifier': classifier,
            'scaler': scaler,
            'test_acc': test_acc
        }
    }
    
    # Add a dummy second model for comparison plots (same model)
    models_dict['reference'] = models_dict[mode_name.lower()]
    
    print("Generating prediction examples...")
    plot_predictions(
        models_dict, X_test, y_test,
        class_names=class_names,
        device=device, save_dir=args.save_dir,
        n_examples=args.n_examples,
        mode1_name=mode_name,
        mode2_name=f"{mode_name} (ref)"
    )
    
    print("Generating state dynamics analysis...")
    plot_state_dynamics(
        models_dict, X_test, y_test,
        device=device, save_dir=args.save_dir,
        n_samples=100,
        mode1_name=mode_name,
        mode2_name=f"{mode_name} (ref)"
    )
    
    print("Generating eigenspectrum analysis...")
    sample_input = torch.tensor(X_test[:1], dtype=torch.float32).to(device)
    plot_eigenspectrum_analysis(
        models_dict, sample_input,
        device=device, save_dir=args.save_dir,
        mode1_name=mode_name,
        mode2_name=f"{mode_name} (ref)"
    )
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Results and plots saved to: {args.save_dir}")
    print(f"\nGenerated files:")
    print(f"  - results.json")
    print(f"  - prediction_examples.png")
    print(f"  - state_dynamics.png")
    print(f"  - eigenspectrum_analysis.png")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
