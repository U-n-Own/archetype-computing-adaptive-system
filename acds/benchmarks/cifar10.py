import os

import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def get_cifar10_data(
    root: os.PathLike, bs_train: int, bs_test: int, valid_perc: int = 10,
    subset_size: int = None, stratify: bool = True, seed: int = 42
):
    """Get the CIFAR-10 dataset and return the train, validation and test dataloaders.

    Args:
        root (os.PathLike): Path to the folder containing the CIFAR-10 dataset.
        bs_train (int): Batch size for the train dataloader.
        bs_test (int): Batch size for the validation and test dataloaders.
        valid_perc (int): Percentage of the train dataset to use for
            validation. Defaults to 10.
        subset_size (int, optional): If provided, use only this many training samples
            (e.g., 5000 instead of 50000). Useful for faster testing.
        stratify (bool): If True and subset_size is set, maintain class distribution
            when sampling subset. Defaults to True.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    # Normalize with CIFAR-10 mean and std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, transform=transform
    )

    # If subset_size is specified, create a stratified subset
    if subset_size is not None and subset_size < len(train_dataset):
        # Get all labels
        labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        indices = np.arange(len(train_dataset))
        
        if stratify:
            # Use stratified sampling to maintain class distribution
            subset_indices, _ = train_test_split(
                indices,
                train_size=subset_size,
                stratify=labels,
                random_state=seed
            )
        else:
            # Random sampling without stratification
            rng = np.random.RandomState(seed)
            subset_indices = rng.choice(indices, size=subset_size, replace=False)
        
        train_dataset = Subset(train_dataset, subset_indices)
        print(f"Using stratified subset: {len(train_dataset)} / 50000 training samples")

    valid_size = int(len(train_dataset) * (valid_perc / 100.0))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs_test, shuffle=False)

    return train_loader, valid_loader, test_loader
