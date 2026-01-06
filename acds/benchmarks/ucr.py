from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .rc_dataset import RCDataset


class UCRDataset(torch.utils.data.Dataset):
    """Generic UCR dataset for non-RC pipelines (no one-hot by default).

    Data items are tuples (x, y) where x is a 1D numpy array (time series)
    and y is an integer class label (0-indexed). The series is reshaped to
    (T, 1) on retrieval to match (seq_len, input_dim) convention.
    """

    def __init__(self, data: List[Tuple[np.ndarray, int]]):
        self.data = data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[idx]
        x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        y_t = torch.tensor([y], dtype=torch.float32)
        return x_t, y_t

    def __len__(self) -> int:
        return len(self.data)


def _load_ucr_txt_as_numpy(root_path: os.PathLike, split: str) -> np.ndarray:
    """Load a UCR dataset split (train/test) from a txt file into a numpy array.

    The first column is the 1-based label; remaining columns are the time series.
    """
    def _find_split_file() -> Path:
        root = Path(root_path)
        dataset_name = root.name
        suffix = "TRAIN" if split.lower() == "train" else "TEST"

        def _extend_variants(name: str) -> List[str]:
            base_variants = {name, name.lower(), name.upper(), name.capitalize()}
            sanitized = name.replace(" ", "").replace("-", "").replace("_", "")
            base_variants.update({sanitized, sanitized.lower(), sanitized.upper(), sanitized.capitalize()})
            return [variant for variant in base_variants if variant]

        name_variants = _extend_variants(dataset_name)

        candidate_dirs: List[Path] = [root]
        parent = root.parent
        if parent and parent not in candidate_dirs:
            candidate_dirs.append(parent)

        if not root.exists():
            # Fall back to a directory that matches the dataset name if the suggested root is missing
            guessed_dir = parent / dataset_name if parent else Path(dataset_name)
            if guessed_dir not in candidate_dirs:
                candidate_dirs.append(guessed_dir)

        candidates: List[Path] = []
        seen = set()

        for directory in candidate_dirs:
            candidates_in_dir = [
                directory / f"{split}.txt",
                directory / f"{split.upper()}.txt",
                directory / f"{split.capitalize()}.txt",
                directory / f"{split}.TXT",
                directory / f"{split.upper()}.TXT",
                directory / f"{split.capitalize()}.TXT",
            ]
            for name in name_variants:
                candidates_in_dir.extend(
                    [
                        directory / f"{name}_{suffix}.txt",
                        directory / f"{name}_{suffix}.TXT",
                    ]
                )

            for candidate in candidates_in_dir:
                if candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(" | ".join(str(c) for c in candidates) + " not found.")

    path = _find_split_file()
    data = np.genfromtxt(path, dtype="float64")
    # Ensure at least 3 cols (parity with ADIAC loader); pad if needed
    rows: List[List[float]] = []
    for row in data:
        el = list(row)
        while len(el) < 3:
            el.append(np.nan)
        rows.append(el)
    return np.array(rows)


def _pairs_from_arrays(series: np.ndarray, targets: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    pairs: List[Tuple[np.ndarray, int]] = []
    for i in range(len(targets)):
        pairs.append((series[i, :], int(targets[i])))
    return pairs


def get_ucr_data(
    root_path: os.PathLike,
    bs_train: int,
    bs_test: int,
    whole_train: bool = False,
    for_rc: bool = True,
    valid_ratio: float = 0.2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generic loader for UCR-style datasets with train.txt and test.txt.

    - Labels in the first column (1-based) are converted to 0-based.
    - Validation split is taken from the tail of the training set unless
      whole_train=True.

    Args:
        root_path: Folder containing train.txt and test.txt
        bs_train: Batch size for train
        bs_test: Batch size for valid/test
        whole_train: If True, use entire training set (no validation)
        for_rc: If True, return RCDataset (preferred for RC/ESN pipelines);
                otherwise return a simple UCRDataset (no one-hot)
        valid_ratio: Fraction of training samples to reserve for validation

    Returns:
        (train_loader, valid_loader, test_loader)
    """

    # Load splits
    train_arr = _load_ucr_txt_as_numpy(root_path, "train")
    test_arr = _load_ucr_txt_as_numpy(root_path, "test")

    # Split into series and labels; convert 1-based labels to 0-based ints
    train_labels = train_arr[:, 0] - 1
    train_series = train_arr[:, 1:]
    test_labels = test_arr[:, 0] - 1
    test_series = test_arr[:, 1:]

    n_train = train_series.shape[0]
    if whole_train:
        val_len = 0
    else:
        val_len = int(round(valid_ratio * n_train))
        # Ensure at least 1 sample if ratio > 0 and dataset is tiny
        if valid_ratio > 0 and val_len == 0 and n_train > 1:
            val_len = 1

    if val_len == 0:
        tr_series, tr_labels = train_series, train_labels
        val_series = train_series[0:0, :]  # empty
        val_labels = train_labels[0:0]
    else:
        tr_series, tr_labels = train_series[:-val_len, :], train_labels[:-val_len]
        val_series, val_labels = train_series[-val_len:, :], train_labels[-val_len:]

    # Build (x, y) pairs
    train_data = _pairs_from_arrays(tr_series, tr_labels)
    valid_data = _pairs_from_arrays(val_series, val_labels)
    test_data = _pairs_from_arrays(test_series, test_labels)

    data_builder = RCDataset if for_rc else UCRDataset

    train_ds = data_builder(train_data)
    valid_ds = data_builder(valid_data)
    test_ds = data_builder(test_data)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=bs_test, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=bs_test, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader


# Convenience wrappers for common UCR datasets
def get_olive_oil_data(*args, **kwargs):
    return get_ucr_data(*args, **kwargs)


def get_forda_data(*args, **kwargs):
    return get_ucr_data(*args, **kwargs)


def get_fordb_data(*args, **kwargs):
    return get_ucr_data(*args, **kwargs)
