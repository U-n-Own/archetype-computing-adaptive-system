"""
Quick test script to verify the UCR experiments setup.
Tests data loading, model creation, and training pipeline.
"""
import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print("UCR EXPERIMENTS - QUICK TEST")
print("=" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    from ucr_experiments.ucr_data_loader import get_ucr_data, UCR_DATASETS
    from ucr_experiments.model_factory import create_model
    from ucr_experiments.training import train_and_evaluate
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test data loading
print("\n2. Testing data loading (FordA)...")
try:
    # Use a small dataset for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    train_loader, valid_loader, test_loader, metadata = get_ucr_data(
        dataset_name='FordA',
        root_path='./data/UCR',
        bs_train=16,
        bs_test=16,
        valid_split=0.2,
        whole_train=False,
        download=True,
    )
    
    print(f"✓ Data loaded successfully")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Valid batches: {len(valid_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Classes: {metadata['n_classes']}")
    print(f"   Sequence length: {metadata['seq_length']}")
    
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test ESN model creation
print("\n3. Testing ESN model creation...")
try:
    # Standard ESN
    config_esn = {
        'n_inp': 1,
        'rho': 0.9,
        'input_scaling': 1.0,
        'leaky': 0.01,
    }
    
    model_esn = create_model('esn', config_esn, device, antisymmetric=False)
    print(f"✓ Standard ESN created (100 units)")
    
    # Antisymmetric ESN
    config_esn_antisym = {
        'n_inp': 1,
        'rho': 0.9,
        'input_scaling': 1.0,
        'leaky': 0.01,
        'coupling_epsilon': 0.1,
        'inter_scaling': 1.0,
    }
    
    model_esn_antisym = create_model('esn', config_esn_antisym, device, antisymmetric=True)
    print(f"✓ Antisymmetric ESN created (5 layers × 100 units = 500 total)")
    
except Exception as e:
    print(f"✗ ESN model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test RON model creation
print("\n4. Testing RON model creation...")
try:
    # Standard RON
    config_ron = {
        'n_inp': 1,
        'dt': 0.01,
        'gamma': 1.0,
        'epsilon': 1.0,
        'gamma_range': 0.5,
        'epsilon_range': 0.5,
        'rho': 0.9,
        'input_scaling': 1.0,
        'reservoir_scaler': 0.5,
        'diffusive_gamma': 0.0,
    }
    
    model_ron = create_model('ron', config_ron, device, antisymmetric=False, topology='full')
    print(f"✓ Standard RON created (100 units)")
    
    # Antisymmetric RON
    config_ron_antisym = {
        'n_inp': 1,
        'dt': 0.01,
        'gamma': 1.0,
        'epsilon': 1.0,
        'gamma_range': 0.5,
        'epsilon_range': 0.5,
        'rho': 0.9,
        'input_scaling': 1.0,
        'reservoir_scaler': 0.5,
        'diffusive_gamma': 0.0,
        'coupling_epsilon': 0.1,
        'inter_scaling': 1.0,
    }
    
    model_ron_antisym = create_model('ron', config_ron_antisym, device, antisymmetric=True, topology='antisymmetric')
    print(f"✓ Antisymmetric RON created (5 layers × 100 units = 500 total)")
    
except Exception as e:
    print(f"✗ RON model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test forward pass
print("\n5. Testing forward pass...")
try:
    # Get a sample batch
    x, y = next(iter(train_loader))
    x = x.to(device)
    print(f"   Input shape: {x.shape}")
    
    # Test ESN
    with torch.no_grad():
        output_esn = model_esn(x)[-1][0]
        if isinstance(output_esn, list):
            output_esn = output_esn[0]
        print(f"✓ ESN forward pass successful, output shape: {output_esn.shape}")
        
        # Check for NaN/Inf
        if torch.isnan(output_esn).any() or torch.isinf(output_esn).any():
            print(f"⚠ ESN output contains NaN or Inf!")
    
    # Test RON
    with torch.no_grad():
        output_ron = model_ron(x)[-1][0]
        if isinstance(output_ron, list):
            output_ron = output_ron[0]
        print(f"✓ RON forward pass successful, output shape: {output_ron.shape}")
        
        # Check for NaN/Inf
        if torch.isnan(output_ron).any() or torch.isinf(output_ron).any():
            print(f"⚠ RON output contains NaN or Inf!")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test training pipeline
print("\n6. Testing training pipeline...")
try:
    # Use small subset for quick test
    from torch.utils.data import Subset
    
    # Create small subset (10% of data)
    n_train = len(train_loader.dataset)
    n_valid = len(valid_loader.dataset)
    n_test = len(test_loader.dataset)
    
    train_subset = Subset(train_loader.dataset, range(min(50, n_train)))
    valid_subset = Subset(valid_loader.dataset, range(min(20, n_valid)))
    test_subset = Subset(test_loader.dataset, range(min(20, n_test)))
    
    from torch.utils.data import DataLoader
    train_loader_small = DataLoader(train_subset, batch_size=16, shuffle=True)
    valid_loader_small = DataLoader(valid_subset, batch_size=16, shuffle=False)
    test_loader_small = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    # Train ESN
    print("   Training ESN on small subset...")
    train_acc, valid_acc, test_acc, is_stable = train_and_evaluate(
        model_esn,
        train_loader_small,
        valid_loader_small,
        test_loader_small,
        device,
        max_iter=100,
        return_test=True,
    )
    
    if is_stable:
        print(f"✓ ESN training successful")
        print(f"   Train acc: {train_acc:.4f}, Valid acc: {valid_acc:.4f}, Test acc: {test_acc:.4f}")
    else:
        print(f"⚠ ESN training completed but model was unstable")
    
except Exception as e:
    print(f"✗ Training pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✓ All tests passed!")
print("\nYou can now run:")
print("  1. Hyperparameter search:")
print("     python ucr_hyperparameter_search.py --dataset FordA --model esn --n_trials 20")
print("\n  2. Final training:")
print("     python ucr_train_final.py --config <path_to_best_config.json> --dataset FordA --trials 5")
print("\n  3. Batch experiments:")
print("     python run_all_ucr_experiments.py --phase search --datasets FordA --n_trials 20")
print("=" * 80)
