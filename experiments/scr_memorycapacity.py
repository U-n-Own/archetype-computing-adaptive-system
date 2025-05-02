import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV
#from sklearn.linear_model import Ridge


from acds.archetypes.scr import SimpleCycleReservoir
from acds.archetypes.esn import DeepReservoir



# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description="SCR Memory Capacity Evaluation")
parser.add_argument("--resultroot", type=str, default="./results")
parser.add_argument("--delay", type=int, default=100)
parser.add_argument("--n_reservoir", type=int, default=100)
parser.add_argument("--r", type=float, default=0.5, help="Recurrent weight (cycle weight)")
parser.add_argument("--v", type=float, default=0.5, help="Input weight magnitude")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--washout", type=int, default=100)
parser.add_argument("--concat", action="store_true", help="Concatenate hidden states")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device:", device)

# set seed
torch.manual_seed(seed)
np.random.seed(seed)


os.makedirs(args.resultroot, exist_ok=True)

def square_correlation(output, target):
    """Calculate squared correlation coefficient"""
    return (np.corrcoef(output.flatten(), target.flatten())[0, 1])**2

def plot_statistics(results_dict, test_dict, model):
    """Plot memory capacity across different delay steps"""
    # Calculate mean for each delay value
    results_dict_mean = {k: sum(v) / args.trials for k, v in results_dict.items()}
    results_dict_mean_test = {k: sum(v) / args.trials for k, v in test_dict.items()}
   
    # Calculate variance for each delay value
    variance_between_steps = {k: np.var(v) for k, v in results_dict.items()}
    variance_between_steps_test = {k: np.var(v) for k, v in test_dict.items()}
    
    plt.figure(figsize=(12, 6))
    plt.plot(list(results_dict_mean.keys()), list(results_dict_mean.values()), label="Mean (Train)")
    plt.plot(list(results_dict_mean_test.keys()), list(results_dict_mean_test.values()), label="Mean (Test)")
    
    # Add confidence intervals
    plt.fill_between(
        list(variance_between_steps.keys()),
        [results_dict_mean[k] - np.sqrt(variance_between_steps[k]) for k in variance_between_steps.keys()],
        [results_dict_mean[k] + np.sqrt(variance_between_steps[k]) for k in variance_between_steps.keys()],
        alpha=0.3,
        label="Variance (Train)",
    )
    
    plt.fill_between(
        list(variance_between_steps_test.keys()),
        [results_dict_mean_test[k] - np.sqrt(variance_between_steps_test[k]) for k in variance_between_steps_test.keys()],
        [results_dict_mean_test[k] + np.sqrt(variance_between_steps_test[k]) for k in variance_between_steps_test.keys()],
        alpha=0.3,
        label="Variance (Test)",
    )
    
    plt.grid(True, which="both", linestyle="--")
    plt.xlabel("Delay")
    plt.ylabel("Memory Capacity")
    plt.title(f"Memory Capacity over delay steps for SCR (n_reservoir={args.n_reservoir})")
    # Add total memory capacity to the plot
    plt.text(
        0.5, 1.05,
        f"Total memory: {sum(results_dict_mean.values()):.2f}, Total test memory: {sum(results_dict_mean_test.values()):.2f}",
        ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=12
    ) 
    plt.legend()
    
    return plt

# Initialize dictionaries to store results
train_memory_dict, test_memory_dict = defaultdict(list), defaultdict(list)

for t in range(args.trials):
    print(f"Trial {t+1}/{args.trials}")
    
    n_layers = 20
# try ESN
    model = DeepReservoir(
        input_size=1,
        tot_units=args.n_reservoir,
        n_layers=n_layers,
        input_scaling=0.1,
        leaky=1,
        connectivity_input=int(args.n_reservoir / n_layers),
        spectral_radius=args.r,
        connectivity_recurrent=int(args.n_reservoir / n_layers),
        connectivity_inter=int(args.n_reservoir / n_layers),
        concat=args.concat,
    ).to(device)
    
        # Initialize SCR model
    model = SimpleCycleReservoir(
        input_size=1,
        n_reservoir=args.n_reservoir,
        r=args.r,
        v=args.v
    ).to(device)

    print(f"SCR model created with {args.n_reservoir} reservoir units")
    
    # Generate data
    num_steps = 6000
    train_steps = 5000
    test_steps = 1000
    
    u = np.random.uniform(-0.8, 0.8, size=(num_steps+args.delay, 1))
    u = u.astype(np.float32)

    # Get reservoir states
    states_u = model(torch.tensor(u[:-args.delay]).to(device).reshape(1, -1, 1))[0].cpu().numpy()
    states_u = states_u.reshape(-1, args.n_reservoir)
   
   
    if args.concat:
        states_u = states_u.reshape(-1, args.n_reservoir)
    # this when we want just to take last state, because
    # if hidden states are not concatenated we have to reshape accordingly
    else:
        states_u = states_u.reshape(-1, args.n_reservoir // n_layers)
    
    # Prepare data for all delays simultaneously
    X_all = states_u[args.delay:num_steps, :]
    
    # Create target matrix with columns for each delay
    y_all = np.zeros((num_steps - args.delay, args.delay))
    for i in range(1, args.delay + 1):
        y_all[:, i-1] = u[args.delay-i:num_steps-i, 0]
    
    # Split into train and test
    split_idx_train = train_steps - args.delay
    split_idx_test = split_idx_train + test_steps
    
    X_train, X_test = X_all[:split_idx_train], X_all[split_idx_train:split_idx_test]
    y_train, y_test = y_all[:split_idx_train], y_all[split_idx_train:split_idx_test]
    
    # Apply washout
    X_train, X_test = X_train[args.washout:], X_test[args.washout:]
    y_train, y_test = y_train[args.washout:], y_test[args.washout:]
    
    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    

    # train with a pseudo invers of outputsize 100
    classifier = np.linalg.pinv(X_train) @ y_train
    y_hat_train = X_train @ classifier
    y_hat_test = X_test @ classifier
    
    #y_hat_train = classifier.predict(X_train)
    #y_hat_test = classifier.predict(X_test)
    
    # Calculate memory capacity for each delay
    total_train_memory, total_test_memory = 0, 0
    for i in range(1, args.delay + 1):
        col_idx = i - 1
        train_memory = square_correlation(y_hat_train[:, col_idx], y_train[:, col_idx])
        test_memory = square_correlation(y_hat_test[:, col_idx], y_test[:, col_idx])
        
        total_train_memory += train_memory
        total_test_memory += test_memory
        
        train_memory_dict[i].append(train_memory)
        test_memory_dict[i].append(test_memory)
        
        if True:  # Print progress for early delays and every 10th delay
            print(
                f"Delay {i}/{args.delay}: "
                f"train memory = {train_memory:.4f}, "
                f"test memory = {test_memory:.4f}"
            )
    
    print(f"Trial {t+1} total memory: train = {total_train_memory:.4f}, test = {total_test_memory:.4f}")

# Calculate overall memory capacity
train_memory = sum([sum(v) for k, v in train_memory_dict.items()]) / args.trials
test_memory = sum([sum(v) for k, v in test_memory_dict.items()]) / args.trials

print(f"Average total memory capacity: train = {train_memory:.4f}, test = {test_memory:.4f}")

# Plot results
plt = plot_statistics(train_memory_dict, test_memory_dict, model)
plt.savefig(os.path.join(args.resultroot, f"SCR_MemoryCapacity_n{args.n_reservoir}_r{args.r}_v{args.v}.png"))
print(f"Plot saved to {args.resultroot}/SCR_MemoryCapacity_n{args.n_reservoir}_r{args.r}_v{args.v}.png")

# Save numerical results
with open(os.path.join(args.resultroot, f"SCR_MemoryCapacity_results.txt"), "a") as f:
    f.write(f"SCR(n_reservoir={args.n_reservoir}, r={args.r}, v={args.v}): "
            f"train_memory={train_memory:.4f}, test_memory={test_memory:.4f}\n")
