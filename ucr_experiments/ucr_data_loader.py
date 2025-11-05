"""
UCR Time Series Dataset Loader
Handles downloading and loading UCR time series datasets using aeon toolkit.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

try:
    from aeon.datasets import load_classification
    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False
    print("Warning: aeon library not available. Install with: pip install aeon")

from acds.benchmarks.rc_dataset import RCDataset


# UCR Dataset metadata
UCR_DATASETS = {
    'FordA': {
        'url': 'https://timeseriesclassification.com/aeon-toolkit/FordA.zip',
        'n_classes': 2,
        'length': 500,
    },
    'FordB': {
        'url': 'https://timeseriesclassification.com/aeon-toolkit/FordB.zip',
        'n_classes': 2,
        'length': 500,
    },
    'Adiac': {
        'url': 'https://timeseriesclassification.com/aeon-toolkit/Adiac.zip',
        'n_classes': 37,
        'length': 176,
    },
    'OliveOil': {
        'url': 'https://timeseriesclassification.com/aeon-toolkit/OliveOil.zip',
        'n_classes': 4,
        'length': 570,
    },
    'CinCECGTorso': {
        'url': 'https://timeseriesclassification.com/aeon-toolkit/CinCECGTorso.zip',
        'n_classes': 4,
        'length': 1639,
    },
}


def load_ucr_with_aeon(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load UCR dataset using aeon library.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'FordA', 'OliveOil')
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    if not AEON_AVAILABLE:
        raise ImportError(
            "aeon library is required but not installed. "
            "Install with: pip install aeon"
        )
    
    print(f"Loading {dataset_name} using aeon...")
    
    # Load data from aeon
    X_train, y_train = load_classification(dataset_name, split="train", return_metadata=False)
    X_test, y_test = load_classification(dataset_name, split="test", return_metadata=False)
    
    # Convert from 3D (n_samples, n_channels, n_timepoints) to 2D (n_samples, n_timepoints)
    # Most UCR datasets are univariate
    if len(X_train.shape) == 3 and X_train.shape[1] == 1:
        X_train = X_train.squeeze(1)  # Remove channel dimension
    if len(X_test.shape) == 3 and X_test.shape[1] == 1:
        X_test = X_test.squeeze(1)
    
    return X_train, y_train, X_test, y_test


def get_ucr_data(
    dataset_name: str,
    root_path: os.PathLike = None,
    bs_train: int = 32,
    bs_test: int = 32,
    valid_split: float = 0.2,
    whole_train: bool = False,
    for_rc: bool = True,
    download: bool = True,
    use_aeon: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Load a UCR time series dataset.
    
    Args:
        dataset_name: Name of the UCR dataset
        root_path: Root directory for datasets (not used when use_aeon=True)
        bs_train: Batch size for training
        bs_test: Batch size for validation/test
        valid_split: Fraction of training data to use for validation
        whole_train: If True, use all training data (no validation split)
        for_rc: If True, return RCDataset format
        download: If True, download dataset if not found (handled by aeon)
        use_aeon: If True, use aeon library to load data (recommended)
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, metadata)
    """
    # Load data using aeon (preferred method)
    if use_aeon:
        train_series, train_labels, test_series, test_labels = load_ucr_with_aeon(dataset_name)
    else:
        # Fallback: manual loading (deprecated)
        raise NotImplementedError(
            "Manual loading is deprecated. Please use aeon library. "
            "Install with: pip install aeon"
        )
    
    # Encode labels to be 0-indexed
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([train_labels, test_labels])
    label_encoder.fit(all_labels)
    
    train_labels_encoded = label_encoder.transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    # Split training into train/validation
    if whole_train:
        valid_series = train_series[0:0]  # Empty
        valid_labels_encoded = train_labels_encoded[0:0]
    else:
        n_valid = int(len(train_series) * valid_split)
        if n_valid > 0:
            # Shuffle before split
            indices = np.random.permutation(len(train_series))
            train_series = train_series[indices]
            train_labels_encoded = train_labels_encoded[indices]
            
            valid_series = train_series[-n_valid:]
            valid_labels_encoded = train_labels_encoded[-n_valid:]
            train_series = train_series[:-n_valid]
            train_labels_encoded = train_labels_encoded[:-n_valid]
        else:
            valid_series = train_series[0:0]
            valid_labels_encoded = train_labels_encoded[0:0]
    
    # Create input-output pairs
    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            # Add channel dimension for univariate time series
            sample = (data_x[i:i+1, :].T, data_y[i])  # Shape: (seq_len, 1)
            mydata.append(sample)
        return mydata
    
    train_data = inp_out_pairs(train_series, train_labels_encoded)
    valid_data = inp_out_pairs(valid_series, valid_labels_encoded)
    test_data = inp_out_pairs(test_series, test_labels_encoded)
    
    # Create datasets
    if for_rc:
        train_dataset = RCDataset(train_data)
        valid_dataset = RCDataset(valid_data)
        test_dataset = RCDataset(test_data)
    else:
        # Use a simple torch dataset
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(
            torch.FloatTensor(train_series).unsqueeze(-1),
            torch.LongTensor(train_labels_encoded)
        )
        valid_dataset = TensorDataset(
            torch.FloatTensor(valid_series).unsqueeze(-1),
            torch.LongTensor(valid_labels_encoded)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(test_series).unsqueeze(-1),
            torch.LongTensor(test_labels_encoded)
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=bs_train, shuffle=True, drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=bs_test, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs_test, shuffle=False, drop_last=False
    )
    
    # Metadata
    metadata = {
        'dataset_name': dataset_name,
        'n_classes': len(label_encoder.classes_),
        'seq_length': train_series.shape[1],
        'n_train': len(train_series),
        'n_valid': len(valid_series),
        'n_test': len(test_series),
        'label_encoder': label_encoder,
    }
    
    print(f"✓ Loaded {dataset_name}:")
    print(f"    Classes: {metadata['n_classes']}")
    print(f"    Sequence length: {metadata['seq_length']}")
    print(f"    Train samples: {metadata['n_train']}")
    print(f"    Valid samples: {metadata['n_valid']}")
    print(f"    Test samples: {metadata['n_test']}")
    
    return train_loader, valid_loader, test_loader, metadata
