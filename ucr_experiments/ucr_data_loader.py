"""
UCR Time Series Dataset Loader
Handles downloading and loading UCR time series datasets using aeon library.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from acds.benchmarks.rc_dataset import RCDataset

# Try to import aeon
try:
    from aeon.datasets import load_classification
    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False
    print("Warning: aeon library not available. Install with: pip install aeon")


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
        dataset_name: Name of the dataset (e.g., 'Adiac', 'FordA')
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
        X arrays have shape (n_samples, n_channels, seq_length)
        y arrays have shape (n_samples,)
    """
    if not AEON_AVAILABLE:
        raise ImportError(
            "aeon library is required. Install with: pip install aeon"
        )
    
    print(f"Loading {dataset_name} using aeon...")
    
    # Load the dataset
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    
    # Convert to numpy arrays if needed
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # aeon returns shape (n_samples, n_channels, seq_length)
    # For univariate datasets, squeeze the channel dimension
    if X_train.shape[1] == 1:
        X_train = X_train.squeeze(1)  # (n_samples, seq_length)
        X_test = X_test.squeeze(1)
    
    print(f"✓ Loaded {dataset_name} with aeon:")
    print(f"    Train shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"    Test shape: {X_test.shape}, labels: {y_test.shape}")
    print(f"    Unique classes: {np.unique(y_train)}")
    
    return X_train, y_train, X_test, y_test


def download_ucr_dataset(dataset_name: str, root_path: Path) -> Path:
    """Download a UCR dataset if it doesn't exist.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'FordA')
        root_path: Root directory to store datasets
        
    Returns:
        Path to the dataset directory
    """
    if dataset_name not in UCR_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(UCR_DATASETS.keys())}")
    
    dataset_path = root_path / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded (try different extensions)
    already_downloaded = False
    for ext in ['.tsv', '.txt', '.ts', '.arff']:
        train_file = dataset_path / f"{dataset_name}_TRAIN{ext}"
        test_file = dataset_path / f"{dataset_name}_TEST{ext}"
        if train_file.exists() and test_file.exists():
            already_downloaded = True
            break
    
    if already_downloaded:
        print(f"✓ Dataset {dataset_name} already exists at {dataset_path}")
        return dataset_path
    
    # Download the dataset
    url = UCR_DATASETS[dataset_name]['url']
    zip_path = dataset_path / f"{dataset_name}.zip"
    
    print(f"Downloading {dataset_name} from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"✓ Downloaded to {zip_path}")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print(f"✓ Extracted to {dataset_path}")
        
        # Clean up zip file
        zip_path.unlink()
        
    except Exception as e:
        raise RuntimeError(f"Failed to download {dataset_name}: {e}")
    
    return dataset_path


def load_ucr_file(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a UCR time series file.
    
    Args:
        file_path: Path to the data file (.tsv, .txt, .ts, etc.)
        
    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    data = None
    last_error = None
    
    # Try different delimiters
    # Try whitespace first since it's most common for UCR datasets
    for delimiter in [None, '\t', ',']:  # None means whitespace
        try:
            if delimiter is None:
                data = np.genfromtxt(file_path)
            else:
                data = np.genfromtxt(file_path, delimiter=delimiter)
            
            # Check if data was loaded successfully and has the right shape
            # UCR datasets should be 2D: (n_samples, n_features+1)
            if data is not None and data.size > 0 and len(data.shape) == 2:
                break
            else:
                # Wrong shape, try next delimiter
                data = None
                continue
        except Exception as e:
            last_error = e
            data = None
            continue
    
    if data is None or data.size == 0:
        raise RuntimeError(
            f"Failed to load {file_path}. Last error: {last_error}"
        )
    
    # Handle both 1D and 2D arrays
    if len(data.shape) == 1:
        # If it's 1D, might be a single sample or malformed file
        raise ValueError(
            f"Loaded data from {file_path} is 1-dimensional with shape {data.shape}. "
            f"Expected 2-dimensional data with shape (n_samples, n_features+1)."
        )
    
    # First column is label, rest is time series
    labels = data[:, 0]
    series = data[:, 1:]
    
    return series, labels


def get_ucr_data(
    dataset_name: str,
    root_path: os.PathLike,
    bs_train: int,
    bs_test: int,
    valid_split: float = 0.2,
    whole_train: bool = False,
    for_rc: bool = True,
    download: bool = True,
    use_aeon: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Load a UCR time series dataset.
    
    Args:
        dataset_name: Name of the UCR dataset
        root_path: Root directory for datasets
        bs_train: Batch size for training
        bs_test: Batch size for validation/test
        valid_split: Fraction of training data to use for validation
        whole_train: If True, use all training data (no validation split)
        for_rc: If True, return RCDataset format
        download: If True, download dataset if not found
        use_aeon: If True, use aeon library to load data (recommended)
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, metadata)
    """
    root_path = Path(root_path)
    
    # Use aeon if available and requested
    if use_aeon and AEON_AVAILABLE:
        try:
            train_series, train_labels, test_series, test_labels = load_ucr_with_aeon(dataset_name)
        except Exception as e:
            print(f"Warning: Failed to load with aeon ({e}), falling back to manual loading")
            use_aeon = False
    else:
        use_aeon = False
    
    # Fall back to manual loading if aeon not available or failed
    if not use_aeon:
        # Download if needed
        if download:
            dataset_path = download_ucr_dataset(dataset_name, root_path)
        else:
            dataset_path = root_path / dataset_name
            
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        # Load train and test files
        # Try different file extensions and naming conventions
        train_file = None
        test_file = None
        
        for ext in ['.tsv', '.txt', '.ts', '.arff']:
            potential_train = dataset_path / f"{dataset_name}_TRAIN{ext}"
            potential_test = dataset_path / f"{dataset_name}_TEST{ext}"
            
            if potential_train.exists() and potential_test.exists():
                train_file = potential_train
                test_file = potential_test
                break
        
        # Try alternative naming (lowercase)
        if train_file is None:
            for ext in ['.tsv', '.txt', '.ts', '.arff']:
                potential_train = dataset_path / f"train{ext}"
                potential_test = dataset_path / f"test{ext}"
                
                if potential_train.exists() and potential_test.exists():
                    train_file = potential_train
                    test_file = potential_test
                    break
        
        if train_file is None or test_file is None:
            raise FileNotFoundError(
                f"Could not find train/test files in {dataset_path}. "
                f"Available files: {list(dataset_path.iterdir())}"
            )
            
        print(f"Loading {dataset_name}...")
        print(f"  Train file: {train_file}")
        print(f"  Test file: {test_file}")
        
        train_series, train_labels = load_ucr_file(train_file)
        test_series, test_labels = load_ucr_file(test_file)
    
    # Encode labels to be 0-indexed
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([train_labels, test_labels])
    label_encoder.fit(all_labels)
    
    train_labels = label_encoder.transform(train_labels)
    test_labels = label_encoder.transform(test_labels)
    
    # Split training into train/validation with stratification
    if whole_train:
        valid_series = train_series[0:0]  # Empty
        valid_labels = train_labels[0:0]
    else:
        if valid_split > 0 and len(train_series) > 10:
            # Use stratified split with shuffling
            train_series, valid_series, train_labels, valid_labels = train_test_split(
                train_series,
                train_labels,
                test_size=valid_split,
                random_state=42,
                stratify=train_labels,
                shuffle=True
            )
        else:
            # Too few samples for stratification
            valid_series = train_series[0:0]
            valid_labels = train_labels[0:0]
    
    # Create input-output pairs
    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            # Add channel dimension for univariate time series
            sample = (data_x[i:i+1, :].T, data_y[i])  # Shape: (seq_len, 1)
            mydata.append(sample)
        return mydata
    
    train_data = inp_out_pairs(train_series, train_labels)
    valid_data = inp_out_pairs(valid_series, valid_labels)
    test_data = inp_out_pairs(test_series, test_labels)
    
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
            torch.LongTensor(train_labels)
        )
        valid_dataset = TensorDataset(
            torch.FloatTensor(valid_series).unsqueeze(-1),
            torch.LongTensor(valid_labels)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(test_series).unsqueeze(-1),
            torch.LongTensor(test_labels)
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
