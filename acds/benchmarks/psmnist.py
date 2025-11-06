import os

import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PermutedMNIST(Dataset):
    """Wrapper for MNIST dataset that applies a fixed permutation to pixels."""
    
    def __init__(self, mnist_dataset, permutation=None, seed=42):
        """
        Args:
            mnist_dataset: Original MNIST dataset
            permutation: Fixed permutation to apply (if None, generates one)
            seed: Random seed for permutation generation
        """
        self.mnist_dataset = mnist_dataset
        
        if permutation is None:
            rng = np.random.RandomState(seed)
            self.permutation = rng.permutation(784)
        else:
            self.permutation = permutation
    
    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        # Flatten and permute
        image_flat = image.view(-1)
        image_permuted = image_flat[self.permutation]
        return image_permuted, label


def get_psmnist_data(
    root: os.PathLike, bs_train: int, bs_test: int, valid_perc: int = 10, seed: int = 42
):
    """Get the permuted sequential MNIST dataset.
    
    The pixels are permuted with a fixed random permutation, making it a 
    challenging sequential task where spatial structure is destroyed.

    Args:
        root (os.PathLike): Path to the folder containing the MNIST dataset.
        bs_train (int): Batch size for the train dataloader.
        bs_test (int): Batch size for the validation and test dataloaders.
        valid_perc (int): Percentage of the train dataset to use for
            validation. Defaults to 10.
        seed (int): Random seed for the permutation. Defaults to 42.
    """
    train_dataset_base = torchvision.datasets.MNIST(
        root=root, train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset_base = torchvision.datasets.MNIST(
        root=root, train=False, transform=transforms.ToTensor()
    )

    # Create the same permutation for all datasets
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(784)

    # Apply permutation to datasets
    train_dataset_full = PermutedMNIST(train_dataset_base, permutation=permutation)
    test_dataset = PermutedMNIST(test_dataset_base, permutation=permutation)

    # Split train into train and validation
    valid_size = int(len(train_dataset_full) * (valid_perc / 100.0))
    train_size = len(train_dataset_full) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset_full, [train_size, valid_size]
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs_test, shuffle=False)

    return train_loader, valid_loader, test_loader
