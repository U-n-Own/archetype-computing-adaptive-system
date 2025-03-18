import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import warnings
import wandb
import matplotlib.pyplot as plt
import plotly.tools as tls
import cProfile
import io
import pstats
import logging
from acds.archetypes.utils import count_parameters
from collections import defaultdict

import lightning as L
import optuna as op

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge

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
parser.add_argument("--gamma_range", type=float, default=0.5)
parser.add_argument("--epsilon_range", type=float, default=1)
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

args = parser.parse_args()


# setup gamma, epsilon and their range for optuna then calculate the final values and return as trial suggest
def gamma_r(trial):

    gamma = trial.suggest_float("gamma", 0.1, 0.5, 2.0, 3)
    gamma_range = trial.suggest_float("gamma_range", 0.1, 0.5)

    return (gamma - gamma_range / 2.0, gamma + gamma_range / 2.0) 

def epsilon_r(trial):
    
    epsilon = trial.suggest_float("epsilon", 0.1, 0.5, 3.0)
    epsilon_range = trial.suggest_float("epsilon_range", 0.1, 0.5)
    
    return (epsilon - epsilon_range / 2.0, epsilon + epsilon_range / 2.0)

# setup optuna for hyperparameter optimization over the reservoir, we want to maimize square correlation between output and target

def square_correlation(output, target):
    return (np.corrcoef(output.flatten(), target.flatten())[0, 1])**2

def define_model(trial):
    
    if args.esn:
        model = DeepReservoir(
            input_size=1,
            tot_units=args.n_hid,
            n_layers=args.n_layers,
            concat=True,
            spectral_radius=trial.suggest_float("rho", 0.95, 0.99),
            input_scaling=0.1,
            inter_scaling=0.1,
            leaky=1,
            connectivity_recurrent=int(args.n_hid / args.n_layers),
            connectivity_input=int(args.n_hid / args.n_layers),
            connectivity_inter=int(args.n_hid / args.n_layers),
        ).to(dtype=torch.float64)
    elif args.ron:
        model = RandomizedOscillatorsNetwork(
            n_inp=1,
            total_units= args.n_hid,
            dt=trial.suggest_float("dt", 0.005, 0.05, 0.1, 0.3, 0.9, 1),
            gamma = gamma_r(trial),
            epsilon= epsilon_r(trial),
            input_scaling=0.2,
            inter_scaling=0.2,
            connectivity_input=int(args.n_hid / args.n_layers),
            connectivity_inter=int(args.n_hid / args.n_layers),
            rho=trial.suggest_float("rho", 0.95, 0.99),
            n_layers=args.n_layers,
            sparsity=trial.suggest_float("sparsity", 0.0, 0.5, 0.7),
            diffusive_gamma=args.diffusive_gamma,
            topology=trial.suggest_categorical("topology", ["full", "antisymmetric", "orthogonal"]),
        ).to(dtype=torch.float64)
    elif args.deepron:
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=1,
            total_units=args.n_hid,
            dt=trial.suggest_float("dt", 0.005, 0.05, 0.1, 0.3, 0.9, 1),
            gamma = gamma_r(trial),
            epsilon= epsilon_r(trial),
            input_scaling=0.2,
            inter_scaling=0.2,
            connectivity_input=int(args.n_hid / args.n_layers),
            connectivity_inter=int(args.n_hid / args.n_layers),
            rho=trial.suggest_float("rho", 0.95, 0.99),
            n_layers=args.n_layers,
            sparsity=trial.suggest_float("sparsity", 0.0, 0.5, 0.7),
            diffusive_gamma=args.diffusive_gamma,
            topology=trial.suggest_categorical("topology", ["full", "antisymmetric", "orthogonal"]),
        )
        
    return model

def objective(trial):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # setup the model
    model = define_model(trial).to(device)
     
    delay = 2*args.n_hid
    washout = 100
    alpha = trial.suggest_float("alpha", 1e-6, 1e-3)
    # setup the optimizer linear regression
    optimizer = Ridge(alpha=1e-6, max_iter=1000)
    
    # setup the dataset
    num_steps = 6000
    train_steps = 4000
    valid_step = 1000
    test_steps = 1000
    
    u = np.random.uniform(-0.8, 0.8, size=(num_steps+delay, 1))
    u = u.astype(np.float32)

    # compute activation before delay
    states_u = model(torch.tensor(u[:-delay]).to(device).reshape(1, -1, 1))[0].cpu().numpy()
    states_u = states_u.reshape(-1, args.n_hid)
    
    delay_score = 0
    
    for i in range(delay):
        
        states_i = states_u[i:num_steps, :]
        target = u[: num_steps - i, 0]
            
        split_idx_train = train_steps - i
        split_idx_valid = split_idx_train + valid_step
        split_idx_test = split_idx_valid + test_steps
   
       # Splits
        y_train, y_valid, y_test = (target[:split_idx_train], 
                                   target[split_idx_train:split_idx_valid], 
                                   target[split_idx_valid:split_idx_test])
        
        X_train, X_valid, X_test = (states_i[:split_idx_train, :],
                                    states_i[split_idx_train:split_idx_valid, :],
                                    states_i[split_idx_valid:split_idx_test, :])
        
        # add washout
        y_train, y_test, y_valid = y_train[washout:], y_test[washout:], y_valid[washout:]
        X_train, X_test, X_valid = X_train[washout:], X_test[washout:], X_valid[washout:] 
        
        
        # Normalize the data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_valid = scaler.transform(X_valid)
            
        # Validation
        optimizer.fit(X_train, y_train)

        y_pred = optimizer.predict(X_valid)
        
        
        cur_delay_score = square_correlation(y_pred, y_valid)
        
        delay_score += cur_delay_score
        
    trial.report(delay_score, i)    
    
    if trial.should_prune():
        raise op.exceptions.TrialPruned()
    
    return delay_score
        
    
if __name__ == "__main__":

    study = op.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    
    print("done")