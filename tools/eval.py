#!/usr/bin/env python3
"""Run a single Optuna-style evaluation for DeepReservoir models.

This script mirrors the evaluation pipeline used inside ``ray/bayesian_search.py``
without running a hyper-parameter search. It allows you to test specific
configurations (dataset, architecture, hyper-parameters) and verify that the
standalone experiment scripts remain consistent with the Optuna workflow.
"""
from __future__ import annotations

import argparse
import os
from statistics import mean, pstdev
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from experiments.utils import set_seed
from acds.benchmarks import get_mnist_data, get_cifar10_data, get_psmnist_data
from acds.archetypes import DeepReservoir


def get_data_config(
    dataset_name: str,
    dataroot: str,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[Tuple[torch.utils.data.DataLoader, ...], int, Callable[[torch.Tensor], torch.Tensor]]:
    """Mimic the data loader + preprocessing setup from bayesian_search."""
    if dataset_name == "mnist":
        loaders = get_mnist_data(root=dataroot, bs_train=batch_size, bs_test=1000)

        def preprocess(images: torch.Tensor) -> torch.Tensor:
            images = images.to(device)
            return images.view(images.shape[0], -1).unsqueeze(-1)

        return loaders, 1, preprocess

    if dataset_name == "psmnist":
        loaders = get_psmnist_data(root=dataroot, bs_train=batch_size, bs_test=1000, seed=seed)

        def preprocess(images: torch.Tensor) -> torch.Tensor:
            images = images.to(device)
            return images.unsqueeze(-1)

        return loaders, 1, preprocess

    if dataset_name == "npcifar10":
        loaders = get_cifar10_data(root=dataroot, bs_train=batch_size, bs_test=1000)

        def preprocess(images: torch.Tensor) -> torch.Tensor:
            images = images.to(device)
            batch_size_local = images.shape[0]
            seq_length = 1000
            row_size = 96
            image_rows = 32
            sequences = torch.zeros((batch_size_local, seq_length, row_size), device=device)
            images_hwc = images.permute(0, 2, 3, 1)
            image_data = images_hwc.reshape(batch_size_local, image_rows, row_size)
            sequences[:, :image_rows, :] = image_data
            sequences[:, image_rows:, :] = torch.rand(
                (batch_size_local, seq_length - image_rows, row_size), device=device
            ) * 2 - 1
            return sequences

        return loaders, 96, preprocess

    raise ValueError(f"Unknown dataset: {dataset_name}")


@torch.no_grad()
def evaluate(
    model: DeepReservoir,
    data_loader: torch.utils.data.DataLoader,
    clf: LogisticRegression,
    scaler: preprocessing.StandardScaler,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    activations: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for images, ys in data_loader:
        states, _ = model(preprocess_fn(images))
        activations.append(states[:, -1, :].cpu().numpy())
        labels.append(ys.numpy())
    X = scaler.transform(np.concatenate(activations, axis=0))
    y = np.concatenate(labels, axis=0)
    return float(clf.score(X, y))


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reservoir = total - trainable
    return total, reservoir


def build_model(args: argparse.Namespace, input_dim: int, device: torch.device, zero_recurrence: bool) -> DeepReservoir:
    units_per_layer = args.n_hid // args.n_layers if args.n_layers else args.n_hid
    connectivity_recurrent = 0 if zero_recurrence else units_per_layer

    return DeepReservoir(
        input_size=input_dim,
        tot_units=args.n_hid,
        n_layers=args.n_layers,
        spectral_radius=args.rho,
        input_scaling=args.inp_scaling,
        inter_scaling=args.inp_scaling,
        connectivity_recurrent=connectivity_recurrent,
        connectivity_input=units_per_layer,
        connectivity_inter=units_per_layer,
        leaky=args.leaky,
        cycle=args.cycle,
        concat=args.concat,
        linear=False,
        epsilon=args.coupling_epsilon,
        antisymmetric=args.antisymmetric,
        gamma=args.diffusive_gamma,
    ).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna-style reservoir evaluation")
    parser.add_argument("--dataset", choices=["mnist", "psmnist", "npcifar10"], default="psmnist")
    parser.add_argument("--arch", choices=["baseline", "baseline_deep", "cycle", "cycle_zero", "antisymmetric"], default="baseline")
    parser.add_argument("--dataroot", type=str, default="./data")
    parser.add_argument("--batch", type=int, default=812)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--use_test", action="store_true", help="Report test accuracy instead of validation accuracy")
    parser.add_argument("--trial", type=int, default=0, help="Starting trial index (added to seed)")

    parser.add_argument("--n_layers", type=int, default=10)
    parser.add_argument("--n_hid", type=int, default=500)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--leaky", type=float, default=0.1)
    parser.add_argument("--inp_scaling", type=float, default=0.5)
    parser.add_argument("--coupling_epsilon", type=float, default=20.0)
    parser.add_argument("--diffusive_gamma", type=float, default=0.0)

    parser.add_argument("--concat", dest="concat", action="store_true", default=True)
    parser.add_argument("--no_concat", dest="concat", action="store_false")
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--antisymmetric", action="store_true")
    parser.add_argument("--zero_recurrence", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--logistic_max_iter", type=int, default=1000)
    return parser.parse_args()


def configure_architecture(args: argparse.Namespace) -> argparse.Namespace:
    if args.arch == "baseline":
        args.cycle = False
        args.antisymmetric = False
        args.zero_recurrence = False
    elif args.arch == "baseline_deep":
        args.cycle = False
        args.antisymmetric = False
        args.zero_recurrence = False
    elif args.arch == "cycle":
        args.cycle = True
        args.antisymmetric = False
    elif args.arch == "cycle_zero":
        args.cycle = True
        args.antisymmetric = False
        args.zero_recurrence = True
    elif args.arch == "antisymmetric":
        args.cycle = False
        args.antisymmetric = True
    return args


def main() -> int:
    args = parse_args()
    args = configure_architecture(args)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    results: Dict[str, List[float]] = {"train": [], "valid": [], "test": []}

    for trial_idx in range(args.trials):
        current_seed = args.seed + args.trial + trial_idx
        print("\n" + "=" * 60)
        print(f"Trial {trial_idx + 1}/{args.trials} | seed {current_seed}")
        print("=" * 60)

        set_seed(current_seed)

        (train_loader, valid_loader, test_loader), input_dim, preprocess_fn = get_data_config(
            args.dataset, args.dataroot, args.batch, current_seed, device
        )

        model = build_model(args, input_dim, device, args.zero_recurrence)
        total_params, reservoir_params = count_parameters(model)
        print(f"Total parameters: {total_params:,} | Reservoir parameters: {reservoir_params:,}")

        train_activations: List[torch.Tensor] = []
        train_labels: List[torch.Tensor] = []
        for images, labels in train_loader:
            with torch.no_grad():
                states, _ = model(preprocess_fn(images))
                train_activations.append(states[:, -1, :].cpu())
                train_labels.append(labels)

        X = torch.cat(train_activations, dim=0).numpy()
        y = torch.cat(train_labels, dim=0).numpy()

        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        clf = LogisticRegression(max_iter=args.logistic_max_iter).fit(X_scaled, y)

        train_acc = evaluate(model, train_loader, clf, scaler, preprocess_fn)
        valid_acc = evaluate(model, valid_loader, clf, scaler, preprocess_fn)
        test_acc = evaluate(model, test_loader, clf, scaler, preprocess_fn)

        results["train"].append(train_acc)
        results["valid"].append(valid_acc)
        results["test"].append(test_acc)

        print(f"Train Acc: {train_acc:.4f}")
        print(f"Valid Acc: {valid_acc:.4f}")
        print(f"Test Acc : {test_acc:.4f}")

    def summarize(values: List[float]) -> str:
        if not values:
            return "0.0000 ± 0.0000"
        return f"{mean(values):.4f} ± {pstdev(values):.4f}"

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Train: {summarize(results['train'])}")
    if args.use_test:
        print(f"Test: {summarize(results['test'])}")
    else:
        print(f"Valid: {summarize(results['valid'])}")

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        f"Train: {summarize(results['train'])}",
                        f"Valid: {summarize(results['valid'])}",
                        f"Test: {summarize(results['test'])}",
                    ]
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
