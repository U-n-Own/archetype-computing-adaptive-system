import os
import json
import time
import numpy as np
import torch
import optuna
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tools.esn_param_match import get_units_for_target_params 

from experiments.utils import set_seed
# Import all necessary getters
from acds.benchmarks import get_mnist_data, get_cifar10_data, get_psmnist_data
from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork,
)

# -------------------------------------------------------------
# Helper: Dataset Configuration Strategy
# -------------------------------------------------------------
def get_data_config(dataset_name, dataroot, batch_size, seed, device):
    """
    Returns:
        loaders: (train, valid, test)
        input_size: int (size of input dimension per step)
        preprocess_fn: function(images) -> sequences [batch, time, input_size]
    """
    
    if dataset_name == "mnist":
        loaders = get_mnist_data(root=dataroot, bs_train=batch_size, bs_test=1000)
        
        def preprocess(images):
            # MNIST: (B, 1, 28, 28) -> (B, 784, 1)
            images = images.to(device)
            return images.view(images.shape[0], -1).unsqueeze(-1)
            
        return loaders, 1, preprocess

    elif dataset_name == "psmnist":
        # psMNIST requires a fixed permutation seed usually
        loaders = get_psmnist_data(root=dataroot, bs_train=batch_size, bs_test=1000, seed=seed)
        
        def preprocess(images):
            # psMNIST: (B, 784) -> (B, 784, 1)
            images = images.to(device)
            return images.unsqueeze(-1)
            
        return loaders, 1, preprocess

    elif dataset_name == "npcifar10":
        # Use standard CIFAR10 loader
        loaders = get_cifar10_data(root=dataroot, bs_train=batch_size, bs_test=1000)
        
        def preprocess(images):
            # npCIFAR10: (B, 3, 32, 32) -> (B, 1000, 96) with noise padding
            # Logic adapted from your test_npcifar10_all_models.py
            images = images.to(device)
            batch_size = images.shape[0]
            seq_length = 1000
            row_size = 96 # 32 * 3
            image_rows = 32
            
            sequences = torch.zeros((batch_size, seq_length, row_size), device=device)
            
            # Reshape (B, 3, 32, 32) -> (B, 32, 32, 3) -> (B, 32, 96)
            images_hwc = images.permute(0, 2, 3, 1)
            image_data = images_hwc.reshape(batch_size, image_rows, row_size)
            
            sequences[:, :image_rows, :] = image_data
            
            # Noise padding
            sequences[:, image_rows:, :] = torch.rand(
                (batch_size, seq_length - image_rows, row_size), 
                device=device
            ) * 2 - 1
            
            return sequences

        return loaders, 96, preprocess

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# -------------------------------------------------------------
# Helper: Count Params
# -------------------------------------------------------------

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reservoir_params = total_params - trainable_params
    return total_params, reservoir_params

# Count trainable parameters in sklearn LogisticRegression
def count_logreg_params(clf):
    n_weights = clf.coef_.size
    n_bias = clf.intercept_.size
    return n_weights + n_bias

# -------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data_loader, clf, scaler, preprocess_fn):
    activations = []
    ys = []

    for images, labels in data_loader:
        # Use the strategy function for preprocessing
        states, _ = model(preprocess_fn(images))
        output = states[:, -1, :]
        activations.append(output.cpu())
        ys.append(labels)

    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()

    return clf.score(activations, ys)


# -------------------------------------------------------------
# Objective Function
# -------------------------------------------------------------
def objective(trial, arch, dataset_name, model_type="esn"):

    # 1. Configure constraints
    if arch == "baseline":
        n_layers_opts = [1]
    # n_hid will be set dynamically below
    elif arch == "baseline_deep":
            n_layers_opts = [5, 10]
    elif arch in ["cycle", "antisymmetric"]:
        n_layers_opts = [5, 10]

    n_layers = trial.suggest_categorical("n_layers", n_layers_opts)
    n_hid = get_units_for_target_params(architecture=arch, n_layers=n_layers, target_params=100_000)

    # 2. Hyperparameters
    config = {
        "dataset": dataset_name,
        "model": model_type,
        "arch": arch,
        "n_hid": n_hid,
        "n_layers": n_layers,
        "rho": trial.suggest_float("rho", 0.1, 9, log=True), # Adjusted range usually better for DeepESN
        "inp_scaling": trial.suggest_float("inp_scaling", 0.1, 1, log=True),
        "leaky": trial.suggest_float("leaky", 0.001, 1, log=True),
        "coupling_epsilon": 20.0, # Consider optimizing this too if antisym
        "concat": True,
        "batch": 812,
        "seed": 42,
        "dataroot": "./data",
        "logdir": f"./logs/bayesopt_{dataset_name}_{arch}",
    }
    
    # Logic flags
    cycle_flag = (arch == "cycle")
    antisymmetric_flag = (arch == "antisymmetric")

    # 3. Load Data & Config
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [ELEGANCE] Retrieve loaders AND the specific preprocessing function
    (train_loader, valid_loader, test_loader), input_dim, preprocess_fn = get_data_config(
        config["dataset"], config["dataroot"], config["batch"], config["seed"], device
    )

    # 4. Build Model
    if config["model"] == "esn":
        units_per_layer = config["n_hid"] // config["n_layers"]
        model = DeepReservoir(
            input_size=input_dim, # Dynamic based on dataset
            tot_units=config["n_hid"],
            n_layers=config["n_layers"],
            spectral_radius=config["rho"],
            input_scaling=config["inp_scaling"],
            inter_scaling=config["inp_scaling"],
            connectivity_recurrent=units_per_layer,
            connectivity_input=units_per_layer,
            connectivity_inter=units_per_layer,
            leaky=config["leaky"],
            cycle=cycle_flag,
            concat=config["concat"],
            linear=False,
            epsilon=config["coupling_epsilon"],
            antisymmetric=antisymmetric_flag,
        ).to(device)
    else:
        raise ValueError("Model type not supported here.")

    # 5. Forward Pass (Training)
    activations = []
    labels_all = []

    for images, labels in train_loader:
        # [ELEGANCE] Clean loop, complex logic hidden in preprocess_fn
        states, _ = model(preprocess_fn(images))
        out = states[:, -1, :]
        activations.append(out.cpu())
        labels_all.append(labels)

    activations = torch.cat(activations).numpy()
    y = torch.cat(labels_all).numpy()

    # 6. Readout Training
    scaler = preprocessing.StandardScaler().fit(activations)
    X = scaler.transform(activations)
    clf = LogisticRegression(max_iter=1000).fit(X, y)

    # 7. Validation
    valid_acc = evaluate(model, valid_loader, clf, scaler, preprocess_fn)

    # 8. Logging
    os.makedirs(config["logdir"], exist_ok=True)
    total_params, reservoir_params = count_parameters(model)
    readout_params = count_logreg_params(clf)
    log_record = dict(config)
    log_record["valid_accuracy"] = float(valid_acc)
    log_record["reservoir_params"] = reservoir_params
    log_record["readout_params"] = readout_params
    log_record["total_params"] = total_params + readout_params
    if multi == True:
        log_file = os.path.join(config["logdir"], f"trial_log_multi_{dataset_name}_{arch}.jsonl")
    else:
        log_file = os.path.join(config["logdir"], "trial_log.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_record) + "\n")

    return valid_acc

if __name__ == "__main__":
    multi = False 
    # List of datasets to run
    DATASETS = ["mnist", "psmnist", "npcifar10"]
    architectures = ["baseline", "cycle", "antisymmetric", "baseline_deep"]

    for dataset in DATASETS:
        for arch in architectures:
            print(f"\n=== Optimizing {arch} on {dataset} ===")

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42), 
                study_name=f"{dataset}_{arch}"
            )

            study.optimize(
                lambda trial: objective(trial, arch, dataset, model_type="esn"),
                n_trials=100,
                show_progress_bar=True,
            )

            print("\nBest:", study.best_params, "Acc:", study.best_value)