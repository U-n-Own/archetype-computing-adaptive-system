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
# import plotly
#import plotly.express as px

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork   
)


# set a seed for all randomness sources
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description="training parameters")

parser.add_argument("--resultroot", type=str)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--delay", type=int, default=100)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")
parser.add_argument("--deepron", action="store_true")
parser.add_argument("--concat", action="store_true")
parser.add_argument("--cycle", action="store_true")
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--n_hid", type=int, default=100)
parser.add_argument("--dt", type=float, default=0.0075)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--gamma_range", type=float, default=0)
parser.add_argument("--epsilon_range", type=float, default=0)
parser.add_argument("--rho", type=float, default=0.99)
parser.add_argument("--inp_scaling", type=float, default=1)
parser.add_argument("--inter_scaling", type=float, default=1)
parser.add_argument("--leaky", type=float, default=1.0, help="ESN spectral radius")
parser.add_argument("--n_layers", type=int, default=1, help="Number of layers of ESN")
parser.add_argument(
    "--sparsity", type=float, default=0.0, help="Sparsity of the reservoir"
)
parser.add_argument("--ron_leaky_esn", action="store_true", help="Use leaky ESN with RON")
parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
parser.add_argument("--topology", type=str, default="full", choices=["full", "antisymmetric", "orthogonal"], help="Topology of the hidden-to-hidden matrix")
parser.add_argument("--use_test", action="store_true")
parser.add_argument("--trials", type=int, default=1)
# save on the data/user folder
parser.add_argument("--remote", action="store_true")
parser.add_argument("--resultsuffix", type=str, default="")

args = parser.parse_args()

# --- Wandb logging ---
# disable local logging
if args.wandb == True:
    wandb.init(project="deep-ron-thesis",
            config={"architecture": "DeepRON" if args.deepron else "RON" if args.ron else "ESN",
                    "trials": args.trials, 
                    "n_hid": args.n_hid,
                    "delay": args.delay,
                    "n_layers": args.n_layers,
                    "gamma": args.gamma,
                    "diffusive_gamma": args.diffusive_gamma,
                    "epsilon": args.epsilon,
                    "rho": args.rho,
                    "inp_scaling": args.inp_scaling,
                    "leaky": args.leaky,
                    "sparsity": args.sparsity,
                    "topology": args.topology,
                    "use_test": args.use_test,
                    "epsilon_range": args.epsilon_range,
                    "gamma_range": args.gamma_range,
                    "resultsuffix": args.resultsuffix
                    }

            )
else:
    warnings.warn("Wandb is not enabled. No logging will be done.")
    # disable wandb logging
    wandb.disabled = True

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)
print("Using device ", device)

if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()

n_inp = 1
n_out = 1
washout = 1000
# TODO this should be set automatically to double of units of the model
delay = args.delay

def square_correlation(output, target):
    return (np.corrcoef(output.flatten(), target.flatten())[0, 1])**2

# set custom criterion eval to square correlation
def criterion_eval(output, target):
    return square_correlation(output, target)


def plot_statistics(results_dict, test_dict, model):
    """Use the dictionary with results for each trial and plot the mean, std and variance over trials

    Args:
        results_dict (int, list): Integer representing the trial along the list with the memory values
    """
    # Sum for each trial the memory values of each step
    results_dict_sum = {k: sum(v) for k, v in results_dict.items()}
    
    # Divide by the trials to get the mean of the memory values
    results_dict_mean = {k: v / args.trials for k, v in results_dict_sum.items()}
    results_dict_mean_test = {k: sum(v) / args.trials for k, v in test_dict.items()}
   
    # for each step we want the variance, staring with dividing the steps in result_dict for trials so first delay steps are for trial 1 and so on
    # then we want the variance between the step across trials

    variance_between_steps = {k: np.var(v) for k, v in results_dict.items()}
    
    variance_between_steps_test = {k: np.var(v) for k, v in test_dict.items()}
    
    # plot the mean, std and variance
    plt.figure(figsize=(12, 6))
    plt.plot(list(results_dict_mean.keys()), list(results_dict_mean.values()), label="Mean")
    plt.plot(list(results_dict_mean_test.keys()), list(results_dict_mean_test.values()), label="Mean Test")
    
    # plot the variance for each step across trials of variance_between_steps
    plt.fill_between(
        list(variance_between_steps.keys()),
        [results_dict_mean[k] - np.sqrt(variance_between_steps[k]) for k in variance_between_steps.keys()],
        [results_dict_mean[k] + np.sqrt(variance_between_steps[k]) for k in variance_between_steps.keys()],
        alpha=0.3,
        label="Variance",
    )
    
    plt.fill_between(
        list(variance_between_steps_test.keys()),
        [results_dict_mean_test[k] - np.sqrt(variance_between_steps_test[k]) for k in variance_between_steps_test.keys()],
        [results_dict_mean_test[k] + np.sqrt(variance_between_steps_test[k]) for k in variance_between_steps_test.keys()],
        alpha=0.3,
        label="Variance Test")
    
    
    plt.grid(True, which="both", linestyle="--")
    plt.xlabel("Delay")
    plt.ylabel("Memory Capacity")
    # add the model name to the title
    plt.title(f"Memory Capacity over delay steps for {model.__class__.__name__}{args.n_layers} layers")
    # add total of result dict and test dict to the plot as values
    plt.text(
    0.5, 1.05,
    f"Total memory: {sum(results_dict_mean.values()):.2f}, Total test memory: {sum(results_dict_mean_test.values()):.2f}",
    ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=12
    )
    plt.legend()
    
    return plt
 
 
if args.ron_leaky_esn:
    gamma = 1 
    epsilon = 1/args.dt
else:
    gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
    epsilon = (
        args.epsilon - args.epsilon_range / 2.0,
        args.epsilon + args.epsilon_range / 2.0,
    )

train_memory, valid_memory, test_memory = 0.0, 0.0, 0.0
train_nrmse, valid_nrmse, test_nrmse = 0.0, 0.0, 0.0

train_memory_dict, valid_memory_dict, test_memory_dict = defaultdict(list), defaultdict(list), defaultdict(list)
train_nrmse_list, valid_nrmse_list, test_nrmse_list = [], [], []
var_train, var_test = [], []

for t in range(args.trials):
    if args.esn:
        units_per_layer = args.n_hid // args.n_layers
        model = DeepReservoir(
            input_size=n_inp,
            tot_units=args.n_hid,
            n_layers=args.n_layers,
            concat=args.concat,
            spectral_radius=args.rho,
            inter_scaling=args.inp_scaling,
            input_scaling=args.inp_scaling,
            # Since we are using tot unit and dividing them by the number of layers we need to adjust the connectivity
            connectivity_recurrent=units_per_layer,
            connectivity_input=units_per_layer,
            connectivity_inter=1,
            leaky=args.leaky,
            cycle=args.cycle,
            linear=True,  
        ).to(device)
    elif args.ron:
        model = RandomizedOscillatorsNetwork(
            n_inp=1,
            n_hid=args.n_hid,
            dt=args.dt,
            gamma=gamma, 
            epsilon=epsilon,
            diffusive_gamma=args.diffusive_gamma,
            rho=args.rho,
            input_scaling=args.inp_scaling,
            topology=args.topology,
            device=device,
        ).to(device)
    elif args.deepron:
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=1,
            total_units=args.n_hid,
            dt=args.dt,
            gamma=gamma,
            epsilon=epsilon,
            n_layers=args.n_layers,
            diffusive_gamma=args.diffusive_gamma,
            rho=args.rho,
            input_scaling=args.inp_scaling,
            inter_scaling=args.inter_scaling,
            # This is not used in ron, to scale internal recurrent use reservoir scalre
            reservoir_scaler=args.inp_scaling,
            device=device,
            connectivity_input=int(args.n_hid / args.n_layers),
            connectivity_inter=int(args.n_hid / args.n_layers),
            concat=args.concat,
            cycle=args.cycle,
        ).to(device)
        
    else:
        raise ValueError("Select exactly one of --esn / --ron / --deepron")

    DEBUG_PARAMS = False
    if DEBUG_PARAMS:
        print(model)
        # print number of parameters and subtract bias
        def count_parameters(model):
            # count only bias and subtract it from the total number of parameters
            # count only bias
            #count each type of parameter
            num_params = 0
            for name, param in model.named_parameters():
                if "bias" in name:
                    num_params += param.numel()
            return num_params
        
        print(f"Number of parameters (including bias): {sum(p.numel() for p in model.parameters())}")
        # print number of parameters and subtract bias
        print(f"Number of parameters (excluding bias): {sum(p.numel() for p in model.parameters()) - count_parameters(model)}")
    else:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        #print(f"Number of parameters per layer: {count_parameters(model)}")

    if DEBUG_PARAMS:    
        # print number of units in the model
        for name, params in model.named_parameters():
            print(f"{name}: {params.shape}")
        
        for name, param in model.named_parameters():    
            if "bias" in name:
                print(name, param.shape)
    
    # set to 0 the bias
    for name, param in model.named_parameters():
        if "bias" in name:
            param.data.fill_(0)
    
    total_train_memory, total_test_memory, total_val_memory = 0, 0, 0 
        
    num_steps = 6000
    train_steps = 5000
    test_steps = 1000
    
    u = np.random.uniform(-0.8, 0.8, size=(num_steps+delay, 1))
    u = u.astype(np.float32)

    states_u = model(torch.tensor(u[:-delay]).to(device).reshape(1, -1, 1))[0].cpu().numpy()
    
    if args.concat:
        states_u = states_u.reshape(-1, args.n_hid)
    # this when we want just to take last state, because
    # if hidden states are not concatenated we have to reshape accordingly
    else:
        states_u = states_u.reshape(-1, args.n_hid // args.n_layers)
    
    # Instead of training separate models for each delay, prepare data for a single model
    # that predicts all delays simultaneously
    
    # Use the same states for all delays
    X_all = states_u[delay:num_steps, :]
    
    # Create a target matrix with columns for each delay
    y_all = np.zeros((num_steps - delay, delay))
    for i in range(1, delay + 1):
        y_all[:, i-1] = u[delay-i:num_steps-i, 0]
    
    # Split into train and test
    split_idx_train = train_steps - delay
    split_idx_test = split_idx_train + test_steps
    
    X_train, X_test = X_all[:split_idx_train], X_all[split_idx_train:split_idx_test]
    y_train, y_test = y_all[:split_idx_train], y_all[split_idx_train:split_idx_test]
    
    # Add washout
    X_train = X_train[washout:]
    y_train = y_train[washout:]
    
    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train a single classifier that outputs all delays
    # Try pseudo-inverse first, fall back to Ridge regression with regularization if it fails
    try:
        classifier = np.linalg.pinv(X_train) @ y_train
        print("Using pseudo-inverse for classification")
    except np.linalg.LinAlgError as e:
        print(f"Pseudo-inverse failed ({e}), using Ridge regression with regularization")
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1e-6, max_iter=10000)
        ridge.fit(X_train, y_train)
        classifier = ridge.coef_.T
        if len(classifier.shape) == 1:
            classifier = classifier.reshape(-1, 1)
    
    y_hat_train = X_train @ classifier
    y_hat_test = X_test @ classifier
    
    #y_hat_train = classifier.predict(X_train)
    #y_hat_test = classifier.predict(X_test)
    
    # Calculate memory capacity for each delay column
    for i in range(1, delay + 1):
        col_idx = i - 1
        train_memory = square_correlation(y_hat_train[:, col_idx], y_train[:, col_idx])
        test_memory = square_correlation(y_hat_test[:, col_idx], y_test[:, col_idx])
        
        # Handle if larger than 1 with an error
        if train_memory > 1:
            raise ValueError("Train memory is larger than 1")
        if test_memory > 1:
            raise ValueError("Test memory is larger than 1")
        
        print("Train memory: ", train_memory, "Test memory: ", test_memory)
        total_train_memory += train_memory
        total_test_memory += test_memory
        train_memory_dict[i].append(train_memory)
        test_memory_dict[i].append(test_memory)
        
        print(
            f"Trial {t}, delay {i}/{delay}, "  
            f"train memory: {round(train_memory, 2)}, "
            f"test memory: {round(test_memory, 2)}, "
            # print current total memory
            "\n",
            f"total train memory: {round(total_train_memory, 2)}, "
            f"total test memory: {round(total_test_memory, 2)}",
            f"\n"
        )
# these is the variance between the trials
var_test.append(total_test_memory)
var_train.append(total_train_memory)
#reset total_test_memory
total_test_memory = 0
total_train_memory = 0
    
if args.ron:
    f = open(os.path.join(args.resultroot, f"MemoryCapacity_log_RON_{args.topology}{args.resultsuffix}.txt"), "a")
elif args.deepron:
    f = open(os.path.join(args.resultroot, f"MemoryCapacity_log_DEEPRON{args.resultsuffix}.txt"), "a")
elif args.esn:
    f = open(os.path.join(args.resultroot, f"MemoryCapacity_log_ESN{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")

# sum train, valid and test memory dict lists and divide by the number of trials
train_memory = sum([sum(v) for k, v in train_memory_dict.items()]) / args.trials
test_memory = sum([sum(v) for k, v in test_memory_dict.items()]) / args.trials

if args.cycle:
    cycle = "simple cycle"
else:
    cycle = "no cycle"
    
plt = plot_statistics(train_memory_dict, test_memory_dict, model=model)
plt.savefig(os.path.join(args.resultroot, f"MemoryCapacity_plot{args.resultsuffix}{args.delay}{model.__class__.__name__}{args.n_layers}{cycle}.png"))
plotly_fig = tls.mpl_to_plotly(plt.gcf())

if args.wandb:
    # save the plot as a wandb artifact
    wandb.log({"Memory Capacity": plotly_fig})
    plt.savefig(os.path.join(args.resultroot, f"MemoryCapacity_plot{args.resultsuffix}{args.delay}{model.__class__.__name__}{args.n_layers}.png"))
    wandb.log({"train_memory": train_memory, "test_memory": test_memory})
    
    # TODO plotly broken
    #for i in range(args.trials):
        #wandb.Image(train_memory_list[::delay], caption="Debug MC plot")
    
ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, " 
ar += (
    f"Memory capacity for train: {train_memory} for test: {test_memory}"
    f"variance train: {np.var(var_train)}, variance test: {np.var(var_test)}"
)
f.write(ar + "\n")
f.close()
