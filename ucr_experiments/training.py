"""
Training utilities for reservoir computing models.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_reservoir_states(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract reservoir states from a model.
    
    Args:
        model: Reservoir computing model
        data_loader: DataLoader with input data
        device: Device to run on
        
    Returns:
        Tuple of (activations, labels) as numpy arrays
    """
    model.eval()
    activations, ys = [], []
    
    for x, y in data_loader:
        x = x.to(device)
        output = model(x)[-1][0]
        
        # Handle different output formats
        if isinstance(output, list):
            output = output[0]
        
        activations.append(output.cpu())
        ys.append(y)
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    return activations, ys


def train_readout(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    max_iter: int = 1000,
    alpha: float = 1e-8,
    use_ridge: bool = False,
) -> Tuple[object, preprocessing.StandardScaler]:
    """Train a linear readout on reservoir states.
    
    Args:
        model: Reservoir computing model
        train_loader: Training data loader
        device: Device to run on
        max_iter: Maximum iterations for solver
        alpha: Regularization parameter for Ridge regression
        use_ridge: If True, use Ridge regression instead of Logistic Regression
        
    Returns:
        Tuple of (classifier, scaler)
    """
    # Extract activations
    activations, ys = extract_reservoir_states(model, train_loader, device)
    
    # Standardize
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    
    # Train classifier
    if use_ridge:
        classifier = Ridge(alpha=alpha, max_iter=max_iter).fit(activations, ys)
    else:
        classifier = LogisticRegression(max_iter=max_iter, random_state=42).fit(activations, ys)
    
    return classifier, scaler


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    classifier: object,
    scaler: preprocessing.StandardScaler,
    device: torch.device,
) -> float:
    """Evaluate model accuracy.
    
    Args:
        model: Reservoir computing model
        data_loader: Data loader
        classifier: Trained readout classifier
        scaler: Feature scaler
        device: Device to run on
        
    Returns:
        Accuracy score
    """
    model.eval()
    
    # Extract activations
    activations, ys = extract_reservoir_states(model, data_loader, device)
    
    # Standardize
    activations = scaler.transform(activations)
    
    # Evaluate
    accuracy = classifier.score(activations, ys)
    
    return accuracy


def check_for_instability(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[bool, Optional[str]]:
    """Check if model produces unstable outputs (NaN or Inf).
    
    Args:
        model: Model to check
        data_loader: Data loader
        device: Device to run on
        
    Returns:
        Tuple of (is_unstable, error_message)
    """
    model.eval()
    
    try:
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                output = model(x)[-1][0]
                
                if isinstance(output, list):
                    output = output[0]
                
                if torch.isnan(output).any():
                    return True, "NaN detected in output"
                if torch.isinf(output).any():
                    return True, "Inf detected in output"
                
                # Only check first batch for speed
                break
                
        return False, None
        
    except Exception as e:
        return True, f"Error during forward pass: {str(e)}"


def train_and_evaluate(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    max_iter: int = 1000,
    use_ridge: bool = False,
    return_test: bool = False,
) -> Tuple[float, float, float, bool]:
    """Complete training and evaluation pipeline.
    
    Args:
        model: Reservoir computing model
        train_loader: Training data loader
        valid_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run on
        max_iter: Maximum iterations for readout training
        use_ridge: Use Ridge regression instead of Logistic Regression
        return_test: If True, also return test accuracy
        
    Returns:
        Tuple of (train_acc, valid_acc, test_acc, is_stable)
    """
    # Check for instability
    is_unstable, error_msg = check_for_instability(model, train_loader, device)
    if is_unstable:
        print(f"⚠ Model unstable: {error_msg}")
        return 0.0, 0.0, 0.0, False
    
    # Train readout
    try:
        classifier, scaler = train_readout(
            model, train_loader, device, max_iter=max_iter, use_ridge=use_ridge
        )
    except Exception as e:
        print(f"⚠ Error training readout: {e}")
        return 0.0, 0.0, 0.0, False
    
    # Evaluate
    try:
        train_acc = evaluate_model(model, train_loader, classifier, scaler, device)
        valid_acc = evaluate_model(model, valid_loader, classifier, scaler, device) if len(valid_loader) > 0 else 0.0
        test_acc = evaluate_model(model, test_loader, classifier, scaler, device) if return_test else 0.0
        
        return train_acc, valid_acc, test_acc, True
        
    except Exception as e:
        print(f"⚠ Error during evaluation: {e}")
        return 0.0, 0.0, 0.0, False
