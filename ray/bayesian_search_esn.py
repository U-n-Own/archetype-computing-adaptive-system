import os
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import json

from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler

from experiments.utils import set_seed
from acds.benchmarks import get_mnist_data

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork,
)

# -----------------------------------------------------------
# Helper: identical to your smnist “count_parameters”
# -----------------------------------------------------------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reservoir_params = total_params - trainable_params
    return total_params, reservoir_params, trainable_params


# -----------------------------------------------------------
# Ray Tune trainable function (this runs one trial)
# -----------------------------------------------------------
def train_smnist_ray(config):

    # Reproducibility
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_inp = 1
    n_out = 10

    arch = config["arch"]
    n_layers = config["n_layers"]
    
    # --- Rule 1: baseline forces 1 layer ---
    if arch == "baseline":
        n_layers = 1  # override
        cycle_flag = False
        antisymmetric_flag = False

    # --- Rule 2: if user tries multilayer baseline, force it to 1 ---
    elif n_layers == 1:
        cycle_flag = False
        antisymmetric_flag = False

    # --- Rule 3: multilayer non-baseline architectures ---
    else:
        cycle_flag = (arch == "cycle")
        antisymmetric_flag = (arch == "antisymmetric")
        
    # -------------------------------------------------------
    # Build model exactly like smnist.py
    # -------------------------------------------------------
    if config["model"] == "esn":
        units_per_layer = config["n_hid"] // config["n_layers"]
        model = DeepReservoir(
            input_size=n_inp,
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

    elif config["model"] == "ron":
        model = RandomizedOscillatorsNetwork(
            n_inp=n_inp,
            n_hid=config["n_hid"],
            dt=config["dt"],
            gamma=config["gamma"],
            epsilon=config["epsilon"],
            diffusive_gamma=config["diffusive_gamma"],
            rho=config["rho"],
            input_scaling=config["inp_scaling"],
            topology=config["topology"],
            sparsity=config["sparsity"],
            reservoir_scaler=config["reservoir_scaler"],
            device=device,
        ).to(device)

    elif config["model"] == "deepron":
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=n_inp,
            total_units=config["n_hid"],
            dt=config["dt"],
            gamma=config["gamma"],
            epsilon=config["epsilon"],
            n_layers=config["n_layers"],
            rho=config["rho"],
            input_scaling=config["inp_scaling"],
            inter_scaling=config["inp_scaling"],
            device=device,
            antisymmetric_coupling=config["antisymmetric"],
            coupling_epsilon=config["coupling_epsilon"],
            concat=config["concat"],
            cycle=config["cycle"]
        ).to(device)

    else:
        raise ValueError("Unknown model type.")

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------
    train_loader, valid_loader, test_loader = get_mnist_data(
        config["dataroot"],
        config["batch"],
        config["batch"],
    )

    # -------------------------------------------------------
    # Train reservoir → collect activations → logistic regression
    # -------------------------------------------------------
    activations = []
    ys = []

    for images, labels in train_loader:
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        states, _ = model(images)
        output = states[:, -1, :]
        activations.append(output.cpu())
        ys.append(labels)

    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()

    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)

    clf = LogisticRegression(max_iter=1000).fit(activations, ys)

    # -------------------------------------------------------
    # Validation accuracy → what we optimize
    # -------------------------------------------------------
    valid_acc = evaluate(model, valid_loader, clf, scaler, device)
    
    # Build flat record
    log_record = {**config}
    log_record["valid_accuracy"] = float(valid_acc)

    # Clean any numpy / tensor types
    for k, v in list(log_record.items()):
        if hasattr(v, "item"):
            log_record[k] = v.item()
        if isinstance(v, np.generic):
            log_record[k] = np.asscalar(v)

    # Write JSON line for this trial
    os.makedirs(config["logdir"], exist_ok=True)
    log_path = os.path.join(config["logdir"], "trial_log.jsonl")

    with open(log_path, "a") as f:
        f.write(json.dumps(log_record) + "\n")

    # -------------------------------------------------------
    # Report to Ray Tune (must come after logging)
    # -------------------------------------------------------
    tune.report(valid_accuracy=valid_acc)

# -----------------------------------------------------------
# Evaluate function identical to smnist.py
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":

    # ----------------------------
    # YOU define the search space
    # ----------------------------
    search_space = {
        "arch": tune.choice(["cycle", "antisymmetric", "baseline"]),
        "model": "esn",
        "n_hid": tune.choice([500]),
        "n_layers": tune.choice([1, 5, 10]),
        "rho": tune.uniform(0.999,9, 90),
        "inp_scaling": tune.uniform(0.1, 1),
        "leaky": tune.loguniform(0.001, 1),
        "coupling_epsilon": tune.uniform(20),
        "concat": True,
        "batch": 1000,
        "seed": 42,
        "dataroot": "./data",
    }

    algo = BayesOptSearch(metric="valid_accuracy", mode="max")

    tuner = tune.Tuner(
        train_smnist_ray,
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=200,  # how many trials you want
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    print("Best result:", results.get_best_result(metric="valid_accuracy", mode="max"))


