import os
import json
import time
import numpy as np
import torch
import optuna
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from experiments.utils import set_seed
from acds.benchmarks import get_mnist_data

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork,
)

# -------------------------------------------------------------
# Helper: Count Params
# -------------------------------------------------------------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reservoir_params = total_params - trainable_params
    return total_params, reservoir_params, trainable_params


# -------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data_loader, clf, scaler, device):
    activations = []
    ys = []

    for images, labels in data_loader:
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        states, _ = model(images)
        output = states[:, -1, :]
        activations.append(output.cpu())
        ys.append(labels)

    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()

    return clf.score(activations, ys)


# -------------------------------------------------------------
# Objective Function for Optuna
# -------------------------------------------------------------
def objective(trial, arch, model_type="esn"):

    # ------------------ Hyperparameter Sampling ------------------
    config = {
        "model": model_type,
        "arch": arch,
        "n_hid": 500,
        "n_layers": trial.suggest_categorical("n_layers", [1, 5, 10]),
        "rho": trial.suggest_float("rho", 0.999, 90, log=True),
        "inp_scaling": trial.suggest_float("inp_scaling", 0.1, 1, log=True),
        "leaky": trial.suggest_float("leaky", 0.001, 1, log=True),
        "coupling_epsilon": 20.0,
        "concat": True,
        "batch": 256,
        "seed": 42,
        "dataroot": "./data",
        "logdir": f"./logs/bayesopt_esn_{arch}",
    }

    # ------------------ Rule logic keeping same behaviour -------------
    if arch == "baseline":
        config["n_layers"] = 1
        cycle_flag = False
        antisymmetric_flag = False
    elif config["n_layers"] == 1:
        cycle_flag = False
        antisymmetric_flag = False
    else:
        cycle_flag = (arch == "cycle")
        antisymmetric_flag = (arch == "antisymmetric")

    # ------------------ Load Data ------------------
    set_seed(config["seed"])
    train_loader, valid_loader, test_loader = get_mnist_data(
        root=config["dataroot"], bs_train=config["batch"], bs_test=1000
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ Build Model ------------------
    if config["model"] == "esn":
        units_per_layer = config["n_hid"] // config["n_layers"]
        model = DeepReservoir(
            input_size=1,
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

    # ------------------ Reservoir Forward Pass ------------------
    t0 = time.time()
    activations = []
    labels_all = []

    for images, labels in train_loader:
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        states, _ = model(images)
        out = states[:, -1, :]
        activations.append(out.cpu())
        labels_all.append(labels)

    activations = torch.cat(activations).numpy()
    y = torch.cat(labels_all).numpy()
    print(f"[PROFILE] Forward: {time.time()-t0:.2f}s")

    # ---------------- LogReg ------------------
    scaler = preprocessing.StandardScaler().fit(activations)
    X = scaler.transform(activations)

    clf = LogisticRegression(max_iter=1000).fit(X, y)

    # ---------------- Validation ------------------
    valid_acc = evaluate(model, valid_loader, clf, scaler, device)

    # ---------------- Log to JSONL ------------------
    os.makedirs(config["logdir"], exist_ok=True)
    with open(os.path.join(config["logdir"], "trial_log.jsonl"), "a") as f:
        log_record = dict(config)
        log_record["valid_accuracy"] = float(valid_acc)
        _, reservoir_params, _ = count_parameters(model)
        log_record["reservoir_params"] = reservoir_params
        f.write(json.dumps(log_record) + "\n")

    return valid_acc


# -------------------------------------------------------------
# MAIN LOOP (Replaces Ray)
# -------------------------------------------------------------
if __name__ == "__main__":

    architectures = ["cycle", "antisymmetric", "baseline"]

    for arch in architectures:
        print(f"\n============================")
        print(f" Optimizing architecture: {arch}")
        print(f"============================\n")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42), 
        )

        study.optimize(
            lambda trial: objective(trial, arch, model_type="esn"),
            n_trials=50,
            show_progress_bar=True,
        )

        print("\nBest:", study.best_params, "Acc:", study.best_value)
