"""
Training utilities for RON models.

This module provides functions for:
- Model training and evaluation
- Classifier training
- Accuracy computation
"""
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import DeepRandomizedOscillatorsNetwork
from ron_visualization import collect_trajectories, plot_phase_space_trajectories, plot_temporal_evolution


@torch.no_grad()
def test(data_loader, model, classifier, scaler):
    """Evaluate model on a dataset."""
    device = next(model.parameters()).device
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


def train_and_evaluate(model_name, model, train_loader, valid_loader, test_loader, 
                      results_dir, visualize_trajectory=False):
    """Train and evaluate a model."""
    device = next(model.parameters()).device
    
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
    
    # Visualize trajectories if requested
    if visualize_trajectory:
        print("\n  Generating phase space trajectory visualizations...")
        
        # Collect trajectories from test set
        trajectory_data = collect_trajectories(model, test_loader, n_samples=5, device=device)
        
        # Generate safe filename
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')
        
        # Plot 3D phase space trajectories
        plot_phase_space_trajectories(
            trajectory_data,
            model_name=model_name,
            filename=f"trajectory_3d_{safe_name}.png",
            results_dir=results_dir,
            n_dims=3
        )
        
        # Plot 2D phase space trajectories
        plot_phase_space_trajectories(
            trajectory_data,
            model_name=model_name,
            filename=f"trajectory_2d_{safe_name}.png",
            results_dir=results_dir,
            n_dims=2
        )
        
        # Plot temporal evolution
        plot_temporal_evolution(
            trajectory_data,
            model_name=model_name,
            filename=f"temporal_{safe_name}.png",
            results_dir=results_dir
        )
    
    return {
        'name': model_name,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'test_acc': test_acc,
        'saturated_pct': saturated * 100,
        'failed': False
    }
