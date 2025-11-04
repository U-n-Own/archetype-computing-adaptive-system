#!/usr/bin/env python3
"""
Verification script to check if all files and imports are working correctly
for UCR experiments on remote machine.
"""

import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ MISSING {description}: {filepath}")
        return False

def check_import(module_name, item_name=None):
    """Check if a module/item can be imported."""
    try:
        if item_name:
            exec(f"from {module_name} import {item_name}")
            print(f"✓ Import successful: from {module_name} import {item_name}")
        else:
            exec(f"import {module_name}")
            print(f"✓ Import successful: import {module_name}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {module_name}.{item_name if item_name else ''}")
        print(f"  Error: {e}")
        return False

def main():
    print("="*80)
    print("UCR EXPERIMENTS VERIFICATION")
    print("="*80)
    print()
    
    all_good = True
    
    # Check experiment files
    print("Checking experiment files...")
    all_good &= check_file("ucr_hyperparameter_search.py", "Hyperparameter search script")
    all_good &= check_file("ucr_train_final.py", "Training script")
    all_good &= check_file("run_all_ucr_experiments.py", "Batch runner script")
    print()
    
    # Check ucr_experiments package
    print("Checking ucr_experiments package...")
    all_good &= check_file("ucr_experiments/__init__.py", "Package init file")
    all_good &= check_file("ucr_experiments/model_factory.py", "Model factory")
    all_good &= check_file("ucr_experiments/ucr_data_loader.py", "Data loader")
    all_good &= check_file("ucr_experiments/training.py", "Training utilities")
    print()
    
    # Check ACDS package
    print("Checking ACDS package...")
    all_good &= check_file("acds/__init__.py", "ACDS package init")
    all_good &= check_file("acds/archetypes/__init__.py", "Archetypes package init")
    print()
    
    # Check imports
    print("Checking Python imports...")
    all_good &= check_import("ucr_experiments.ucr_data_loader", "UCR_DATASETS")
    all_good &= check_import("ucr_experiments.ucr_data_loader", "get_ucr_data")
    all_good &= check_import("ucr_experiments.model_factory", "create_model")
    all_good &= check_import("ucr_experiments.training")
    all_good &= check_import("acds.archetypes", "DeepRandomizedOscillatorsNetwork")
    all_good &= check_import("acds.archetypes", "DeepReservoir")
    print()
    
    # Test model creation
    print("Testing model creation...")
    try:
        from ucr_experiments.model_factory import create_model
        
        # Test RON
        config_ron = {
            'n_inp': 1,
            'total_units': 100,
            'output_size': 2,
            'dt': 0.01,
            'gamma': 1.0,
            'epsilon': 1.0,
            'gamma_range': 0.5,
            'epsilon_range': 0.5,
        }
        model_ron = create_model('ron', config_ron, device='cpu', antisymmetric=False)
        print(f"✓ RON model creation successful: {type(model_ron).__name__}")
        
        # Test ESN
        config_esn = {
            'n_inp': 1,
            'total_units': 100,
            'output_size': 2,
            'rho': 0.9,
            'input_scaling': 1.0,
            'leaky': 0.1,
        }
        model_esn = create_model('esn', config_esn, device='cpu', antisymmetric=False)
        print(f"✓ ESN model creation successful: {type(model_esn).__name__}")
        
        # Test ESN antisymmetric
        config_esn_antisym = {
            'n_inp': 1,
            'total_units': 100,
            'output_size': 2,
            'rho': 0.9,
            'input_scaling': 1.0,
            'leaky': 0.1,
            'coupling_epsilon': 0.05,
            'inter_scaling': 0.5,
        }
        model_esn_antisym = create_model('esn', config_esn_antisym, device='cpu', antisymmetric=True)
        print(f"✓ ESN antisymmetric model creation successful: {type(model_esn_antisym).__name__}")
        print(f"  Layers: {model_esn_antisym.n_layers}, Antisymmetric: {model_esn_antisym.antisymmetric}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        all_good = False
    print()
    
    # Check CUDA availability
    print("Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
    print()
    
    # Summary
    print("="*80)
    if all_good:
        print("✓ ALL CHECKS PASSED - Ready to run experiments!")
        print()
        print("Next steps:")
        print("  1. Test with small trial: python run_all_ucr_experiments.py --phase search --datasets FordA --n_trials 2 --device cuda")
        print("  2. Run full experiments: python run_all_ucr_experiments.py --phase search --datasets FordA FordB --n_trials 50 --device cuda")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
