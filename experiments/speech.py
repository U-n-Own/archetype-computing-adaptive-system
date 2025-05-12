import argparse
import os
import warnings
import numpy as np
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork,
)

from typing import List, Tuple

parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--dataroot", type=str, help="Path to the directory containing speech data .npy files")
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=500, help="hidden size of recurrent net"
)
parser.add_argument("--n_hid_layers", type=str, default="500, 500", help="hidden size of recurrent net")
parser.add_argument("--batch", type=int, default=64, help="batch size")
parser.add_argument(
    "--dt", type=float, default=0.042, help="step size <dt> of the coRNN"
)
parser.add_argument(
    "--gamma", type=float, default=2.7, help="y controle parameter <gamma> of the coRNN"
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=4.7,
    help="z controle parameter <epsilon> of the coRNN",
)
parser.add_argument(
    "--gamma_range",
    type=float,
    default=2.7,
    help="y controle parameter <gamma> of the coRNN",
)
parser.add_argument(
    "--epsilon_range",
    type=float,
    default=4.7,
    help="z controle parameter <epsilon> of the coRNN",
)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")
parser.add_argument("--pron", action="store_true")
parser.add_argument("--mspron", action="store_true")
parser.add_argument("--deepron", action="store_true")
parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
parser.add_argument("--inp_scaling", type=float, default=1.0, help="ESN input scaling")
parser.add_argument("--concat", action="store_true", help="Use concatenation in the network")
parser.add_argument("--rho", type=float, default=0.99, help="ESN spectral radius")
parser.add_argument("--leaky", type=float, default=1.0, help="ESN leaky integration")
parser.add_argument("--use_test", action="store_true")
parser.add_argument(
    "--trials", type=int, default=1, help="How many times to run the experiment"
)
parser.add_argument("--ron_leaky", action="store_true", help="Use leaky integration in the RON")
parser.add_argument("--n_layers", type=int, default=1, help="Number of layers in the network")
parser.add_argument("--cycle", action="store_true", help="Use cycle in the network")
parser.add_argument(
    "--topology",
    type=str,
    default="full",
    choices=["full", "ring", "band", "lower", "toeplitz", "orthogonal", "antisymmetric"],
    help="Topology of the reservoir",
)
parser.add_argument(
    "--sparsity", type=float, default=0.0, help="Sparsity of the reservoir"
)
parser.add_argument(
    "--reservoir_scaler",
    type=float,
    default=1.0,
    help="Scaler in case of ring/band/toeplitz reservoir",
)
parser.add_argument(
    "--use_last_state", 
    action="store_true", 
    help="Use last hidden state instead of all states for readout"
)
# if no --use_last_state is provided, use all states

args = parser.parse_args()
# make sure that n_hid_layers is a list of integers
args.n_hid_layers = [int(x) for x in args.n_hid_layers.split(",")]

if args.dataroot is None:
    warnings.warn("No dataroot provided. Using current location as default.")
    args.dataroot = os.getcwd()
if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()
assert os.path.exists(args.resultroot), \
    f"{args.resultroot} folder does not exist, please create it and run the script again."

assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"


def load_speech_data(dataroot: str):
    """Load speech data from .npy files in the specified directory."""
    # Constants for the speech dataset
    TIME_STEPS = 101
    FEATURE_DIM = 40
    
    speech_files = [f for f in os.listdir(dataroot) if f.endswith('.npy')]
    if not speech_files:
        raise ValueError(f"No .npy files found in {dataroot}")
    
    all_data = []
    all_labels = []
    class_names = []
    
    for class_idx, file_name in enumerate(tqdm(speech_files, desc="Loading data files")):
        filepath = os.path.join(dataroot, file_name)
        class_name = file_name.split('.')[0]  # Use filename without extension as class name
        class_names.append(class_name)
        
        try:
            data = np.load(filepath, allow_pickle=True)
            # Basic validation
            if data.ndim != 3 or data.shape[1] != TIME_STEPS or data.shape[2] != FEATURE_DIM:
                print(f"Skipping {file_name}: unexpected shape {data.shape}")
                continue
            if data.shape[0] == 0:
                print(f"Skipping {file_name}: no samples found")
                continue

            all_data.append(data)
            all_labels.append(np.full(data.shape[0], class_idx))  # Assign class index as label
            print(f"Loaded {file_name}: {data.shape[0]} samples")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data loaded. Check files and expected dimensions.")
    
    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    # Split into train and test sets
    np.random.seed(42)
    indices = np.random.permutation(combined_data.shape[0])
    train_size = int(combined_data.shape[0] * 0.8)  # 80% train
    test_size = combined_data.shape[0] - train_size
    
    X_train = combined_data[indices[:train_size]]
    y_train = combined_labels[indices[:train_size]]
    X_test = combined_data[indices[train_size:]]
    y_test = combined_labels[indices[train_size:]]
    
    return X_train, y_train, X_test, y_test, class_names


@torch.no_grad()
def test(model, X_data, y_data, scaler, classifier, use_last_state=False, batch_size=256):
    activations = []
    
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    
    for i in tqdm(range(0, X_data.shape[0], batch_size), desc="Evaluating"):
        batch_X = X_tensor[i:i+batch_size].to(device)
        output = model(batch_X)
        
        # Debug information about output
        if i == 0:
            print(f"DEBUG - Output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"DEBUG - Output[0] shape: {output[0].shape}")
                print(f"DEBUG - Output[1] type: {type(output[1])}")
                if isinstance(output[1], list) or isinstance(output[1], tuple):
                    print(f"DEBUG - Output[1] length: {len(output[1])}")
                    for j, item in enumerate(output[1]):
                        print(f"DEBUG - Output[1][{j}] shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
                else:
                    print(f"DEBUG - Output[1] shape: {output[1].shape if hasattr(output[1], 'shape') else 'no shape'}")
            else:
                print(f"DEBUG - Output shape: {output.shape}")
        
        if isinstance(output, tuple):
            states = output[0]
            last_hidden = output[1]
        else:
            states = output
            last_hidden = states[:, -1, :]
        
        # Debug information about states and last_hidden
        if i == 0:
            print(f"DEBUG - States shape: {states.shape}")
            print(f"DEBUG - States min/max/mean/std: {states.min().item():.4f}/{states.max().item():.4f}/{states.mean().item():.4f}/{states.std().item():.4f}")
            
            print(f"DEBUG - Last hidden type: {type(last_hidden)}")
            if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                print(f"DEBUG - Last hidden length: {len(last_hidden)}")
                print(f"DEBUG - Last hidden[-1] shape: {last_hidden[-1].shape if hasattr(last_hidden[-1], 'shape') else 'no shape'}")
                print(f"DEBUG - Last hidden[-1] stats: {last_hidden[-1].min().item():.4f}/{last_hidden[-1].max().item():.4f}/{last_hidden[-1].mean().item():.4f}/{last_hidden[-1].std().item():.4f}")
            else:
                print(f"DEBUG - Last hidden shape: {last_hidden.shape}")
                print(f"DEBUG - Last hidden stats: {last_hidden.min().item():.4f}/{last_hidden.max().item():.4f}/{last_hidden.mean().item():.4f}/{last_hidden.std().item():.4f}")
        
        if use_last_state:
            if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                final_state = last_hidden[-1].cpu().numpy()
            else:
                final_state = last_hidden.cpu().numpy()
            activations.append(final_state)
        else:
            pooled_states = states.mean(dim=1).cpu().numpy()
            activations.append(pooled_states)
    
    activations = np.concatenate(activations, axis=0)
    
    # Debug information about activations
    print(f"DEBUG - Activations shape: {activations.shape}")
    print(f"DEBUG - Activations min/max/mean/std: {np.min(activations):.4f}/{np.max(activations):.4f}/{np.mean(activations):.4f}/{np.std(activations):.4f}")
    
    activations = scaler.transform(activations)
    
    # Debug information about scaled activations
    print(f"DEBUG - Scaled activations min/max/mean/std: {np.min(activations):.4f}/{np.max(activations):.4f}/{np.mean(activations):.4f}/{np.std(activations):.4f}")
    
    return classifier.score(activations, y_data)


device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)
print("Using device:", device)

# Load speech data
print("Loading speech data...")
X_train, y_train, X_test, y_test, class_names = load_speech_data(args.dataroot)
n_inp = X_train.shape[2]  # Feature dimension
n_out = len(class_names)  # Number of classes
print(f"Loaded data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
print(f"Input dimension: {n_inp}, Number of classes: {n_out}")

if args.ron_leaky:
    epsilon = 1/args.dt
    gamma = 1
else:
    gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
    epsilon = (
        args.epsilon - args.epsilon_range / 2.0,
        args.epsilon + args.epsilon_range / 2.0,
    )

train_accs, test_accs = [], []
for i in range(args.trials):
    print(f"Trial {i+1}/{args.trials}")
    
    if args.esn:
        units_per_layer = args.n_hid // args.n_layers
        model = DeepReservoir(
            input_size=n_inp,
            tot_units=args.n_hid,
            spectral_radius=args.rho,
            input_scaling=args.inp_scaling,
            inter_scaling=args.inp_scaling,
            connectivity_recurrent=units_per_layer,
            connectivity_input=units_per_layer,
            connectivity_inter=units_per_layer,
            leaky=args.leaky,
            cycle=args.cycle,
            concat=args.concat,
            n_layers=args.n_layers
        ).to(device)
    elif args.ron:
        model = RandomizedOscillatorsNetwork(
            n_inp=n_inp,
            n_hid=args.n_hid,
            dt=args.dt,
            gamma=gamma,
            epsilon=epsilon,
            diffusive_gamma=args.diffusive_gamma,
            rho=args.rho,
            input_scaling=args.inp_scaling,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
            cycle=args.cycle,
            #concat=args.concat,
        ).to(device)
    elif args.deepron:
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=n_inp,
            total_units=args.n_hid,
            dt=args.dt,
            gamma=gamma,
            epsilon=epsilon,
            n_layers=args.n_layers,
            rho=args.rho,
            input_scaling=args.inp_scaling,
            topology=args.topology,
        ).to(device)
    else:
        raise ValueError("No model type specified. Please use --esn, --ron, --pron, --mspron, or --deepron.")

    model.eval()  # Set model to evaluation mode
    
    # After model creation:
    if args.ron:
        print(f"RON Parameters - dt: {args.dt}, gamma: {gamma}, epsilon: {epsilon}")
        # Add parameter validation
        if isinstance(gamma, tuple) and (gamma[0] <= 0 or gamma[1] <= 0):
            print("WARNING: Gamma values should be positive for stability")
        if isinstance(epsilon, tuple) and (epsilon[0] <= 0 or epsilon[1] <= 0):
            print("WARNING: Epsilon values should be positive for stability")
    elif args.deepron:
        print(f"DeepRON Parameters - dt: {args.dt}, gamma: {gamma}, epsilon: {epsilon}, n_layers: {args.n_layers}")
        if isinstance(gamma, tuple) and (gamma[0] <= 0 or gamma[1] <= 0):
            print("WARNING: Gamma values should be positive for stability")
        if isinstance(epsilon, tuple) and (epsilon[0] <= 0 or epsilon[1] <= 0):
            print("WARNING: Epsilon values should be positive for stability")
            
    # Process training data
    print("Processing training data...")
    activations = []

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    batch_size = args.batch
    
    for j in tqdm(range(0, X_train.shape[0], batch_size), desc="Processing train data"):
        batch_X = X_train_tensor[j:j+batch_size].to(device)
        output = model(batch_X)
        
        # Debug the first batch output
        if j == 0:
            print(f"DEBUG - Training output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"DEBUG - Training output[0] shape: {output[0].shape}")
                print(f"DEBUG - Training output[1] type: {type(output[1])}")
                if isinstance(output[1], list) or isinstance(output[1], tuple):
                    print(f"DEBUG - Training output[1] length: {len(output[1])}")
                    for k, item in enumerate(output[1]):
                        print(f"DEBUG - Training output[1][{k}] shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
                else:
                    print(f"DEBUG - Training output[1] shape: {output[1].shape if hasattr(output[1], 'shape') else 'no shape'}")
            else:
                print(f"DEBUG - Training output shape: {output.shape}")
                
        if isinstance(output, tuple):
            states = output[0]
            last_hidden = output[1]
        else:
            states = output
            last_hidden = states[:, -1, :]
        
        if args.use_last_state:
            if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
                final_state = last_hidden[-1].cpu().numpy()
            else:
                final_state = last_hidden.cpu().numpy()
            activations.append(final_state)
        else:
            pooled_states = states.mean(dim=1).cpu().numpy()
            activations.append(pooled_states)
    
    activations = np.concatenate(activations, axis=0)
    
    # Scale data and train classifier
    print("Fitting scaler...")
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    print("Training logistic regression classifier...")
    
    #classifier = RidgeClassifier(max_iter=1000, alpha=1e-7)
    classifier = LogisticRegression(max_iter=10000, penalty='l2')
    classifier.fit(activations, y_train)
    
    
     
    # Evaluate
    print("Evaluating on training set...")
    train_acc = test(model, X_train, y_train, scaler, classifier, args.use_last_state, args.batch)
    print("Evaluating on test set...")
    test_acc = test(model, X_test, y_test, scaler, classifier, args.use_last_state, args.batch)
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

# Save results
if args.ron:
    f = open(os.path.join(args.resultroot, f"Speech_log_RON_{args.topology}{args.resultsuffix}.txt"), "a")
elif args.pron:
    f = open(os.path.join(args.resultroot, f"Speech_log_PRON{args.resultsuffix}.txt"), "a")
elif args.mspron:
    f = open(os.path.join(args.resultroot, f"Speech_log_MSPRON{args.resultsuffix}.txt"), "a")
elif args.esn:
    f = open(os.path.join(args.resultroot, f"Speech_log_ESN{args.resultsuffix}.txt"), "a")
elif args.deepron:
    f = open(os.path.join(args.resultroot, f"Speech_log_DEEPRON{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 2)) for train_acc in train_accs]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_accs]} "
    f"mean/std train: {np.mean(train_accs):.4f}, {np.std(train_accs):.4f} "
    f"mean/std test: {np.mean(test_accs):.4f}, {np.std(test_accs):.4f}"
)
f.write(ar + "\n")
f.close()