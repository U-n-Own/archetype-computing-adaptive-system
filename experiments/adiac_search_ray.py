import argparse
import torch
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import plotly.tools as tls
from acds.archetypes.utils import count_parameters
from collections import defaultdict
import os
import tensorboard
import random
from experiments.utils import set_seed

# import List
from typing import List

import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge

from acds.benchmarks import get_adiac_data
import warnings


from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork
)
parser = argparse.ArgumentParser(description="training parameters")

parser.add_argument("--resultroot", type=str)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--delay", type=int, default=200)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")
parser.add_argument("--deepron", action="store_true")
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--n_hid", type=int, default=100)
parser.add_argument("--dt", type=float, default=0.0075)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--gamma_range", type=float, default=0)
parser.add_argument("--epsilon_range", type=float, default=0)
parser.add_argument("--rho", type=float, default=0.99)
parser.add_argument("--inp_scaling", type=float, default=1)
parser.add_argument("--leaky", type=float, default=1.0, help="ESN spectral radius")
parser.add_argument("--n_layers", type=int, default=1, help="Number of layers of ESN")
parser.add_argument(
    "--sparsity", type=float, default=0.0, help="Sparsity of the reservoir"
)
parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
parser.add_argument("--topology", type=str, default="full", choices=["full", "antisymmetric", "orthogonal"], help="Topology of the hidden-to-hidden matrix")
parser.add_argument("--use_test", action="store_true")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--resultsuffix", type=str, default="")
parser.add_argument("--bayesian", action="store_true")
parser.add_argument("--cycle", action="store_true")
parser.add_argument("--dataroot", type=str)
parser.add_argument("--concat", action="store_true")
args = parser.parse_args()

# Set random seed for reproducibility (match adiac.py and utils.py)
SEED = 42
set_seed(SEED)


def train_adiac(config):
    
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    
    assert args.dataroot is not None, "No dataroot provided."
    if args.resultroot is None:
        warnings.warn("No resultroot provided. Using current location as default.")
        args.resultroot = os.getcwd()
    assert os.path.exists(args.resultroot), \
        f"{args.resultroot} folder does not exist, please create it and run the script again."


    assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not args.cpu
        else torch.device("cpu")
    )

    
    # Initialize model with config parameters
    if args.esn:
        model = DeepReservoir(
            input_size=1,
            tot_units=int(config["n_hid"]),
            spectral_radius=config["rho"],
            n_layers=int(config["n_layers"]),
            input_scaling=config["input_scaling"],
            inter_scaling=config["inter_scaling"],
            leaky=config["leaky"],
            concat=True,
            connectivity_input=int(config["n_hid"]),
            connectivity_recurrent=int(config["n_hid"] / config["n_layers"]),
            cycle=args.cycle,
        )
    elif args.deepron:
        gamma = (config["gamma"] - config["gamma_range"] / 2.0, config["gamma"] + config["gamma_range"] / 2.0)
        epsilon = (config["epsilon"] - config["epsilon_range"] / 2.0, config["epsilon"] + config["epsilon_range"] / 2.0)
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=1,
            n_layers=int(config["n_layers"]),
            total_units=int(config["n_hid"]),
            dt=config["dt"],
            gamma=gamma,
            epsilon=epsilon,
            input_scaling=config["input_scaling"],
            inter_scaling=config["inter_scaling"],
            reservoir_scaler=config["input_scaling"],
            connectivity_input=int(1-args.sparsity * 1),
            connectivity_inter=int(config["n_hid"] / config["n_layers"]),
            rho=config["rho"],
            cycle=args.cycle,
            concat=True,
            device=device,
            topology=args.topology,
        )
   
   # run adiac.py with the selected model

    @torch.no_grad()
    def test(data_loader, classifier, scaler):
        activations, ys = [], []
        for x, y in tqdm(data_loader):
            x = x.to(device)
            output = model(x)[-1][0]
            if isinstance(output, list):
                output = output[0]  # or torch.stack(output) if needed
            activations.append(output.cpu())
            ys.append(y)
        activations = torch.cat(activations, dim=0).numpy()
        activations = scaler.transform(activations)
        ys = torch.cat(ys, dim=0).numpy()
        return classifier.score(activations, ys)


    n_inp = 1
    n_out = 37  # classes
    gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
    epsilon = (
        args.epsilon - args.epsilon_range / 2.0,
        args.epsilon + args.epsilon_range / 2.0,
    )

    max_test_accs: List[float] = []
    if args.trials > 1:
        assert args.use_test, "Multiple runs are only for the final test phase with the test set."
        train_loader, valid_loader, test_loader = get_adiac_data(
            args.dataroot, args.batch, args.batch, whole_train=True
        )
    else:
        train_loader, valid_loader, test_loader = get_adiac_data(
            args.dataroot, args.batch, args.batch
        )
        

    train_accs, valid_accs, test_accs = [], [], []


    activations, ys = [], []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        if isinstance(output, list):
            output = output[0]  # or torch.stack(output), depending on your model
        activations.append(output.cpu())
        #output = model(x)[-1][0]
        #activations.append(output.cpu())
        ys.append(y)
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)
    train_acc = test(train_loader, classifier, scaler)
    valid_acc = test(valid_loader, classifier, scaler) if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler) if args.use_test else 0.0

    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)
    
    max_test_accs.append(max(test_accs))
    
    return {
        "accuracy": max_test_accs[-1],
        "train_accuracy": train_accs[-1],
        "valid_accuracy": valid_accs[-1],
    }

def run_hyperparameter_search():
    # Bayesian optimization search 
    
    oldsearch_space = {
        "gamma": tune.uniform(1, 4),
        "epsilon": tune.uniform(0.5, 4),
        "dt": tune.loguniform(0.05, 0.1),
        "rho": tune.uniform(0.99, 90),
        #"alpha": tune.uniform(1e-9, 1e-9),
        "gamma_range": tune.uniform(0.5, 2),
        "epsilon_range": tune.uniform(0, 1),
        "n_hid": 100,
        #"n_layers": tune.randint(1, 11),
        "n_layers": tune.quniform(1, 10, 1),  # returns float values 1.0, 2.0, ..., 10.0
        "input_scaling": tune.uniform(0, 20),
        "inter_scaling": 1,
        #"leaky": tune.loguniform(0.00001, 0.001)
        #"inter_scaling": tune.uniform(0.1, 0.2),
    }
    
    search_space = {
        "gamma": 3.32,
        "epsilon": 1.147,
        "dt": 0.077,
        "rho": 83.046,
        #"alpha": tune.uniform(1e-9, 1e-9),
        "gamma_range": 1.909,
        "epsilon_range": 0.969,
        "n_hid": 100,
        #"n_layers": tune.randint(1, 11),
        "n_layers": 6,  # returns float values 1.0, 2.0, ..., 10.0
        "input_scaling": 17.89,
        "inter_scaling": 1,
        #"leaky": tune.loguniform(0.00001, 0.001)
        #"inter_scaling": tune.uniform(0.1, 0.2),
    }

    if args.bayesian:
        # Configure Bayesian optimization
        bayesopt = BayesOptSearch(
            metric="accuracy",
            mode="max",
            utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}
        )
        
        bayesopt = ConcurrencyLimiter(bayesopt, max_concurrent=8)
    else:
        # Just use random search
        bayesopt = None


    tuner = tune.Tuner(
        train_adiac,
        tune_config=tune.TuneConfig(
            # if bayesopt is None do random search
            search_alg=bayesopt if bayesopt is not None else None,
            num_samples=1,  # Number of trials
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print(f"Best accuracy: {best_result.metrics['accuracy']}")
    print(f"Best config: {best_result.config}")

if __name__ == "__main__":
    

    # Add weights and biases logging
    if args.wandb:
        wandb.init(project="deep-ron-thesis", entity="vincent", config=args, sync_tensorboard=True, job_type="Adiac search")
        wandb.run.save()

        
    # set seed
    # use all cpus    
    ray.init(num_cpus=8)
    # check how many cpus are being used
    print(ray.available_resources())
    
    # log with wandb
    
    run_hyperparameter_search()

    #ray.shutdown()

