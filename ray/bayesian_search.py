import os
import json
import time
import numpy as np
import torch
import optuna
from codecarbon import EmissionsTracker
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
def objective(trial, arch, dataset_name, model_type="esn", multi=None):
    input_size = 1  # Default input size for npcifar10; adjust as needed per dataset

    # Handle cycle_zero architecture variant
    zero_recurrence = False
    
    if arch == "cycle_zero":
        real_arch = "cycle"
        zero_recurrence = True
    else:
        real_arch = arch

    # 1. Configure constraints
    #n_layers_opts = [1] # Default to avoid UnboundLocalError
    if real_arch == "baseline":
        n_layers_opts = [1]
    elif real_arch == "baseline_deep":
        n_layers_opts = [5, 10]
    elif real_arch in ["cycle", "antisymmetric"]:
        n_layers_opts = [5, 10]

    # Single-layer for shallow RON to avoid undefined cycle kernels
    if model_type == "ron":
        n_layers_opts = [1]

    n_layers = trial.suggest_categorical("n_layers", n_layers_opts)
    if model_type == "ron":
        n_layers = 1

    if dataset_name == "mnist" or dataset_name == "psmnist":
        input_size = 1
    elif dataset_name == "npcifar10":
        input_size = 96

    n_hid = get_units_for_target_params(
        architecture=real_arch,
        n_layers=n_layers,
        target_params=100_000,
        input_size=input_size,
        zero_recurrence=zero_recurrence,
    )

    cycle_flag = real_arch == "cycle"
    antisymmetric_flag = real_arch == "antisymmetric"
    topology_choices = ["full", "orthogonal", "antisymmetric"]
    if antisymmetric_flag:
        topology_choices = ["antisymmetric"]
    elif real_arch in ["baseline", "baseline_deep", "cycle", "cycle_zero"]:
        topology_choices = ["full", "orthogonal"]

    # 2. Hyperparameters (model-specific pieces appended below)
    config = {
        "dataset": dataset_name,
        "model": model_type,
        "arch": arch,  # Keep original name for logging
        "n_hid": n_hid,
        "n_layers": n_layers,
        "batch": 1024,
        "seed": 42,
        "dataroot": "./data",
        "logdir": f"./logs/bayesopt_{model_type}_{dataset_name}_{arch}",
    }

    if model_type == "esn":
        config.update(
            {
                "rho": trial.suggest_float("rho", 0.1, 9, log=True),
                "inp_scaling": trial.suggest_float("inp_scaling", 0.1, 1, log=True),
                "leaky": trial.suggest_float("leaky", 0.001, 1, log=True),
                "coupling_epsilon": 20.0,
                "concat": True,
                "diffusive_gamma": trial.suggest_float("diffusive_gamma", 0.0001, 0.5, log=True),
            }
        )
    elif model_type in {"ron", "deepron"}:
        config.update(
            {
                "dt": trial.suggest_float("dt", 1e-4, 1, log=True),
                "gamma": 1,
                "gamma_range": 0.1,
                "epsilon": 1,
                "epsilon_range": 0.1,
                "rho": trial.suggest_float("rho", 0.1, 9.0, log=True),
                "inp_scaling": trial.suggest_float("inp_scaling", 0.05, 2.0, log=True),
                "coupling_epsilon": trial.suggest_float(
                    "coupling_epsilon", 0.01, 5.0, log=True
                ),
                "diffusive_gamma": trial.suggest_float("diffusive_gamma", 0.0001, 0.5, log=True),
                "concat": True,
            }
        )
        if model_type == "deepron":
            config["inter_scaling"] = 0
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 3. Load Data & Config
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (train_loader, valid_loader, test_loader), input_dim, preprocess_fn = get_data_config(
        config["dataset"], config["dataroot"], config["batch"], config["seed"], device
    )

    # 4. Build Model
    if config["model"] == "esn":
        units_per_layer = config["n_hid"] // config["n_layers"]
        connectivity_recurrent = 0 if zero_recurrence else units_per_layer

        model = DeepReservoir(
            input_size=input_dim,
            tot_units=config["n_hid"],
            n_layers=config["n_layers"],
            spectral_radius=config["rho"],
            input_scaling=config["inp_scaling"],
            inter_scaling=config["inp_scaling"],
            connectivity_recurrent=connectivity_recurrent,
            connectivity_input=units_per_layer,
            connectivity_inter=units_per_layer,
            leaky=config["leaky"],
            cycle=cycle_flag,
            concat=config["concat"],
            linear=False,
            epsilon=config["coupling_epsilon"],
            antisymmetric=antisymmetric_flag,
            gamma=config["diffusive_gamma"],
        ).to(device)
    elif config["model"] == "ron":
        gamma = (
            config["gamma"] - config["gamma_range"] / 2.0,
            config["gamma"] + config["gamma_range"] / 2.0,
        )
        epsilon = (
            config["epsilon"] - config["epsilon_range"] / 2.0,
            config["epsilon"] + config["epsilon_range"] / 2.0,
        )
        model = RandomizedOscillatorsNetwork(
            n_inp=input_dim,
            n_hid=config["n_hid"],
            dt=config["dt"],
            gamma=gamma,
            epsilon=epsilon,
            diffusive_gamma=config["diffusive_gamma"],
            rho=config["rho"],
            input_scaling=config["inp_scaling"],
            device=device,
            linear=False,
            cycle=False,  # single-layer RON lacks cycle kernels
            antisymmetric_coupling=antisymmetric_flag,
            coupling_epsilon=config["coupling_epsilon"],
        ).to(device)
    elif config["model"] == "deepron":
        gamma = (
            config["gamma"] - config["gamma_range"] / 2.0,
            config["gamma"] + config["gamma_range"] / 2.0,
        )
        epsilon = (
            config["epsilon"] - config["epsilon_range"] / 2.0,
            config["epsilon"] + config["epsilon_range"] / 2.0,
        )
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=input_dim,
            total_units=config["n_hid"],
            dt=config["dt"],
            gamma=gamma,
            epsilon=epsilon,
            n_layers=config["n_layers"],
            diffusive_gamma=config["diffusive_gamma"],
            rho=config["rho"],
            input_scaling=config["inp_scaling"],
            inter_scaling=config.get("inter_scaling", config["inp_scaling"]),
            device=device,
            antisymmetric_coupling=antisymmetric_flag,
            coupling_epsilon=config["coupling_epsilon"],
            cycle=cycle_flag,
            concat=config["concat"],
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

    # Support both the legacy `multi` flag and the newer `MULTI_MODE` global.
    multi_mode = MULTI_MODE if multi is None else bool(multi)
    if multi_mode:
        log_file = os.path.join(
            config["logdir"], f"trial_log_multi_{dataset_name}_{arch}_{model_type}.jsonl"
        )
    else:
        log_file = os.path.join(config["logdir"], f"trial_log_{model_type}.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_record) + "\n")

    return valid_acc

MULTI_MODE = False

if __name__ == "__main__":
    tracker = EmissionsTracker(project_name="bayesian_search", output_dir="./logs/emissions")
    tracker.start()
    try:
        models_env = os.environ.get("BAYESIAN_MODELS")
        
        model_types = (
            [m.strip() for m in models_env.split(",") if m.strip()]
            if models_env
            else ["esn", "ron", "deepron"]
        )
        n_trials = int(os.environ.get("BAYESIAN_N_TRIALS", "100"))

        MULTI_MODE = len(model_types) > 1
        DATASETS = ["mnist", "psmnist", "npcifar10"]
        architectures = ["antisymmetric, cycle, cycle_zero"]#["baseline", "cycle", "cycle_zero", "antisymmetric", "baseline_deep"]

        for dataset in DATASETS:
            for arch in architectures:
                for model_type in model_types:
                    print(f"\n=== Optimizing {model_type} | {arch} on {dataset} ===")

                    study = optuna.create_study(
                        direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=42),
                        study_name=f"{dataset}_{arch}_{model_type}",
                    )

                    study.optimize(
                        lambda trial, m=model_type, a=arch, d=dataset: objective(
                            trial, a, d, model_type=m
                        ),
                        n_trials=n_trials,
                        show_progress_bar=True,
                    )

                    print("\nBest:", study.best_params, "Acc:", study.best_value)
    finally:
        tracker.stop()
