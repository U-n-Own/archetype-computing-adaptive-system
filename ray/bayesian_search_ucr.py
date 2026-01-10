import json
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import optuna
import torch
from codecarbon import EmissionsTracker
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge

from tools.esn_param_match import get_units_for_target_params

from experiments.utils import set_seed
from acds.benchmarks import get_mackey_glass
from acds.benchmarks.ucr import get_forda_data, get_fordb_data, get_olive_oil_data
from acds.archetypes import (
    DeepRandomizedOscillatorsNetwork,
    DeepReservoir,
    RandomizedOscillatorsNetwork,
)


# -------------------------------------------------------------
# DeepRON helpers
# -------------------------------------------------------------
def _deepron_state_sequence(model: DeepRandomizedOscillatorsNetwork, x: torch.Tensor) -> torch.Tensor:
    """Stream timesteps to avoid materializing the full tensor stack.

    Returns features per timestep shaped [B, T, F], where F is either the
    concatenated hidden size (concat=True) or the last-layer hidden size.
    """
    batch_size, seq_len, _ = x.shape
    h_states = [torch.zeros(batch_size, layer.n_hid, device=x.device, dtype=x.dtype) for layer in model.ron_reservoir]
    hz_states = [torch.zeros(batch_size, layer.n_hid, device=x.device, dtype=x.dtype) for layer in model.ron_reservoir]

    features: List[torch.Tensor] = []

    if model.antisymmetric_coupling:
        for t in range(seq_len):
            new_h_states: List[torch.Tensor] = []
            new_hz_states: List[torch.Tensor] = []
            for i, ron_layer in enumerate(model.ron_reservoir):
                if i == 0:
                    layer_input = x[:, t, :]
                else:
                    layer_input = new_h_states[i - 1]

                h_prev_layer = h_states[i - 1] if i > 0 else None
                h_next_layer = h_states[i + 1] if i < len(model.ron_reservoir) - 1 else None

                new_h, new_hz = ron_layer.cell(
                    layer_input,
                    h_states[i],
                    hz_states[i],
                    first_layer=(i == 0),
                    h_last=None,
                    h_prev_layer=h_prev_layer,
                    h_next_layer=h_next_layer,
                )
                new_h_states.append(new_h)
                new_hz_states.append(new_hz)

            h_states = new_h_states
            hz_states = new_hz_states
            features.append(torch.cat(h_states, dim=1) if model.concat else h_states[-1])

    elif model.cycle:
        for t in range(seq_len):
            current_input = x[:, t, :]
            new_h_states = []
            new_hz_states = []
            for i, ron_layer in enumerate(model.ron_reservoir):
                if i == 0:
                    prev_layer_output = (
                        h_states[-1]
                        if len(model.ron_reservoir) > 1
                        else torch.zeros(batch_size, ron_layer.n_hid, device=x.device, dtype=x.dtype)
                    )
                else:
                    prev_layer_output = new_h_states[-1]

                new_h, new_hz = ron_layer.cell(
                    current_input,
                    h_states[i],
                    hz_states[i],
                    first_layer=True,
                    h_last=prev_layer_output,
                )
                new_h_states.append(new_h)
                new_hz_states.append(new_hz)

            h_states = new_h_states
            hz_states = new_hz_states
            features.append(torch.cat(h_states, dim=1) if model.concat else h_states[-1])

    else:
        for t in range(seq_len):
            layer_input = x[:, t, :]
            new_h_states = []
            new_hz_states = []
            for i, ron_layer in enumerate(model.ron_reservoir):
                new_h, new_hz = ron_layer.cell(
                    layer_input,
                    h_states[i],
                    hz_states[i],
                    first_layer=(i == 0),
                    h_last=None,
                )
                new_h_states.append(new_h)
                new_hz_states.append(new_hz)
                layer_input = new_h

            h_states = new_h_states
            hz_states = new_hz_states
            features.append(torch.cat(h_states, dim=1) if model.concat else h_states[-1])

    return torch.stack(features, dim=1)


def _deepron_last_state(model: DeepRandomizedOscillatorsNetwork, x: torch.Tensor) -> torch.Tensor:
    """Return only the final activation to save memory for long sequences."""
    return _deepron_state_sequence(model, x)[:, -1, :]


def _model_last_activation(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, DeepRandomizedOscillatorsNetwork):
        return _deepron_last_state(model, x)
    states, _ = model(x)
    return states[:, -1, :]


def _model_state_sequence(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, DeepRandomizedOscillatorsNetwork):
        return _deepron_state_sequence(model, x)
    states, _ = model(x)
    return states


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reservoir_params = total_params - trainable_params
    return total_params, reservoir_params


def count_linear_params(clf) -> int:
    n_weights = clf.coef_.size
    n_bias = clf.intercept_.size
    return n_weights + n_bias


@torch.no_grad()
def evaluate_classification(model: torch.nn.Module, data_loader, clf, scaler, preprocess_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
    activations = []
    ys = []
    for x, y in data_loader:
        output = _model_last_activation(model, preprocess_fn(x))
        activations.append(output.cpu())
        ys.append(y)

    activations_np = torch.cat(activations, dim=0).numpy()
    activations_np = scaler.transform(activations_np)
    ys_np = torch.cat(ys, dim=0).squeeze().long().numpy()
    return clf.score(activations_np, ys_np)


def compute_nrmse(predictions: np.ndarray, target: np.ndarray) -> float:
    """Compute Normalized Root Mean Squared Error.
    
    NRMSE = RMSE / std(target)
    """
    mse = np.mean((predictions - target) ** 2)
    rmse = np.sqrt(mse)
    target_std = np.std(target)
    return rmse / target_std if target_std > 0 else rmse


@torch.no_grad()
def evaluate_mackey_glass(model: torch.nn.Module, dataset: torch.Tensor, target: torch.Tensor, clf, scaler, washout: int) -> float:
    seq = dataset.reshape(1, -1, 1).to(next(model.parameters()).device)
    features = _model_state_sequence(model, seq)[:, washout:, :]
    features = features.reshape(-1, features.shape[-1]).cpu().numpy()
    features = scaler.transform(features)
    preds = clf.predict(features)
    target_np = target.reshape(-1, 1).numpy()
    nrmse = compute_nrmse(preds, target_np)
    return nrmse


def resolve_dataset_root(dataset_name: str, base_root: str) -> str:
    env_override = os.environ.get(f"BAYESIAN_DATAROOT_{dataset_name.upper()}")
    if env_override:
        return env_override

    candidates: Dict[str, List[str]] = {
        "forda": ["FordA", "forda"],
        "fordb": ["FordB", "fordb"],
        "oliveoil": ["OliveOil", "oliveoil"],
        "mackeyglass": ["mackey_glass", "mackeyglass", "MackeyGlass"],
    }
    for candidate in candidates.get(dataset_name, []):
        path = os.path.join(base_root, candidate)
        if os.path.exists(path):
            return path
    return os.path.join(base_root, dataset_name)


def get_data_config(dataset_name: str, dataroot: str, batch_size: int, seed: int, device: torch.device, mg_lag: int, mg_washout: int):
    if dataset_name in {"forda", "fordb", "oliveoil"}:
        loader_map = {
            "forda": get_forda_data,
            "fordb": get_fordb_data,
            "oliveoil": get_olive_oil_data,
        }
        loaders = loader_map[dataset_name](
            root_path=dataroot,
            bs_train=batch_size,
            bs_test=max(batch_size, 256),
            whole_train=False,
            for_rc=True,
            valid_ratio=0.2,
        )

        def preprocess(x: torch.Tensor) -> torch.Tensor:
            return x.to(device)

        return {
            "task": "classification",
            "loaders": loaders,
            "input_size": 1,
            "preprocess": preprocess,
        }

    if dataset_name == "mackeyglass":
        train_data, valid_data, test_data = get_mackey_glass(dataroot, lag=mg_lag)
        return {
            "task": "regression",
            "train": train_data,
            "valid": valid_data,
            "test": test_data,
            "input_size": 1,
            "washout": mg_washout,
        }

    raise ValueError(f"Unknown dataset: {dataset_name}")


# -------------------------------------------------------------
# Objective
# -------------------------------------------------------------
def objective(trial: optuna.Trial, arch: str, dataset_name: str, model_type: str, base_dataroot: str, mg_lag: int, mg_washout: int) -> float:
    zero_recurrence = arch == "cycle_zero"
    real_arch = "cycle" if arch == "cycle_zero" else arch

    if real_arch == "baseline":
        n_layers_opts = [1]
    elif real_arch == "baseline_deep":
        n_layers_opts = [5, 10]
    elif real_arch in {"cycle", "antisymmetric"}:
        n_layers_opts = [5, 10]
    else:
        n_layers_opts = [1]

    if model_type == "ron":
        n_layers_opts = [1]
    if model_type == "deepron":
        n_layers_opts = [5, 10]

    n_layers = trial.suggest_categorical("n_layers", n_layers_opts)
    if model_type == "ron":
        n_layers = 1

    input_size = 1

    n_hid = get_units_for_target_params(
        architecture=real_arch,
        n_layers=n_layers,
        target_params=100_000,
        input_size=input_size,
        zero_recurrence=zero_recurrence,
    )

    cycle_flag = real_arch == "cycle"
    antisymmetric_flag = real_arch == "antisymmetric"

    config = {
        "dataset": dataset_name,
        "model": model_type,
        "arch": arch,
        "n_hid": n_hid,
        "n_layers": n_layers,
        "batch": int(os.environ.get("BAYESIAN_BATCH", "64")),
        "seed": 42,
        "dataroot_base": base_dataroot,
        "logdir": f"./logs/bayesopt_{model_type}_{dataset_name}_{arch}",
        "mg_lag": mg_lag,
        "mg_washout": mg_washout,
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

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = resolve_dataset_root(dataset_name, base_root=base_dataroot)
    data_config = get_data_config(dataset_name, dataset_root, config["batch"], config["seed"], device, mg_lag, mg_washout)

    # Build model
    if config["model"] == "esn":
        units_per_layer = config["n_hid"] // config["n_layers"]
        connectivity_recurrent = 0 if zero_recurrence else units_per_layer

        model = DeepReservoir(
            input_size=input_size,
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
            n_inp=input_size,
            n_hid=config["n_hid"],
            dt=config["dt"],
            gamma=gamma,
            epsilon=epsilon,
            diffusive_gamma=config["diffusive_gamma"],
            rho=config["rho"],
            input_scaling=config["inp_scaling"],
            device=device,
            linear=False,
            cycle=False,
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
            n_inp=input_size,
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
            connectivity_recurrent=0 if zero_recurrence else None,
        ).to(device)
    else:
        raise ValueError("Model type not supported here.")

    model.eval()

    # Training
    if data_config["task"] == "classification":
        train_loader, valid_loader, test_loader = data_config["loaders"]
        activations = []
        labels_all = []
        with torch.no_grad():
            for images, labels in train_loader:
                out = _model_last_activation(model, data_config["preprocess"](images))
                activations.append(out.cpu())
                labels_all.append(labels)

        activations_np = torch.cat(activations).numpy()
        y_np = torch.cat(labels_all).squeeze().long().numpy()

        scaler = preprocessing.StandardScaler().fit(activations_np)
        X = scaler.transform(activations_np)
        clf = LogisticRegression(max_iter=1000).fit(X, y_np)

        valid_score = evaluate_classification(model, valid_loader, clf, scaler, data_config["preprocess"])
    else:
        train_dataset, train_target = data_config["train"]
        valid_dataset, valid_target = data_config["valid"]

        train_seq = train_dataset.reshape(1, -1, 1).to(device)
        features = _model_state_sequence(model, train_seq)[:, data_config["washout"] :, :]
        features = features.reshape(-1, features.shape[-1]).cpu().numpy()
        scaler = preprocessing.StandardScaler().fit(features)
        X = scaler.transform(features)
        y = train_target.reshape(-1, 1).numpy()
        clf = Ridge(max_iter=1000).fit(X, y)

        valid_score = evaluate_mackey_glass(
            model,
            valid_dataset,
            valid_target,
            clf,
            scaler,
            data_config["washout"],
        )

    # Logging
    os.makedirs(config["logdir"], exist_ok=True)
    total_params, reservoir_params = count_parameters(model)
    readout_params = count_linear_params(clf)
    log_record = dict(config)
    log_record["valid_score"] = float(valid_score)
    log_record["reservoir_params"] = reservoir_params
    log_record["readout_params"] = readout_params
    log_record["total_params"] = total_params + readout_params

    multi_mode = MULTI_MODE
    if multi_mode:
        log_file = os.path.join(
            config["logdir"], f"trial_log_multi_{dataset_name}_{arch}_{model_type}.jsonl"
        )
    else:
        log_file = os.path.join(config["logdir"], f"trial_log_{model_type}.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return valid_score


MULTI_MODE = False

if __name__ == "__main__":
    tracker = EmissionsTracker(project_name="bayesian_search_ucr", output_dir="./logs/emissions", measure_power_secs=180)
    tracker.start()
    try:
        models_env = os.environ.get("BAYESIAN_MODELS")
        datasets_env = os.environ.get("BAYESIAN_DATASETS")
        arch_env = os.environ.get("BAYESIAN_ARCHS")

        model_types = (
            [m.strip() for m in models_env.split(",") if m.strip()]
            if models_env
            else ["esn", "ron", "deepron"]
        )
        dataset_list = (
            [d.strip().lower() for d in datasets_env.split(",") if d.strip()]
            if datasets_env
            else ["forda", "fordb", "oliveoil", "mackeyglass"]
        )
        architectures = (
            [a.strip() for a in arch_env.split(",") if a.strip()]
            if arch_env
            else ["baseline", "cycle", "antisymmetric", "cycle_zero"]
        )
        n_trials = int(os.environ.get("BAYESIAN_N_TRIALS", "50"))
        base_dataroot = os.environ.get("BAYESIAN_DATAROOT", "./data")
        mg_lag = int(os.environ.get("MG_LAG", "1"))
        mg_washout = int(os.environ.get("MG_WASHOUT", "200"))

        MULTI_MODE = len(model_types) > 1

        for dataset in dataset_list:
            direction = "minimize" if dataset == "mackeyglass" else "maximize"
            for arch in architectures:
                for model_type in model_types:
                    print(f"\n=== Optimizing {model_type} | {arch} on {dataset} ===")

                    study = optuna.create_study(
                        direction=direction,
                        sampler=optuna.samplers.TPESampler(seed=42),
                        study_name=f"{dataset}_{arch}_{model_type}",
                    )

                    study.optimize(
                        lambda trial, m=model_type, a=arch, d=dataset: objective(
                            trial,
                            a,
                            d,
                            model_type=m,
                            base_dataroot=base_dataroot,
                            mg_lag=mg_lag,
                            mg_washout=mg_washout,
                        ),
                        n_trials=n_trials,
                        show_progress_bar=True,
                    )

                    print("\nBest:", study.best_params, "Score:", study.best_value)
    finally:
        tracker.stop()
