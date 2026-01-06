import argparse
import warnings
from typing import List
import os
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from experiments.utils import set_seed

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks.ucr import get_ucr_data


parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--dataroot", type=str)
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=256, help="hidden size of recurrent net"
)
parser.add_argument("--batch", type=int, default=30, help="batch size")
parser.add_argument(
    "--dt", type=float, default=0.042, help="step size <dt> of the coRNN"
)
parser.add_argument(
    "--gamma", type=float, default=2.7, help="y control parameter <gamma> of the coRNN"
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
parser.add_argument('--deepron', action="store_true")
parser.add_argument('--pron', action="store_true")
parser.add_argument('--mspron', action="store_true")
parser.add_argument("--antisymmetric", action="store_true", help="Enable antisymmetric coupling between layers (not topology)")
parser.add_argument("--coupling_epsilon", type=float, default=20.0, help="Coupling strength for antisymmetric inter-layer connections")

parser.add_argument("--matrix_friction", action="store_true")
parser.add_argument("--input_fn", type=str, default="linear", choices=["linear", "mlp"],
                    help="input preprocessing modality")

parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
parser.add_argument("--inp_scaling", type=float, default=1.0, help="ESN input scaling")
parser.add_argument("--rho", type=float, default=0.99, help="ESN spectral radius")
parser.add_argument("--leaky", type=float, default=1.0, help="ESN spectral radius")
parser.add_argument("--n_layers", type=int, default=1, help="Number of layers")
parser.add_argument("--concat", action="store_true")
parser.add_argument("--cycle", action="store_true", help="Enable cycle connections")
parser.add_argument("--use_test", action="store_true")
parser.add_argument(
    "--trials", type=int, default=1, help="How many times to run the experiment"
)
parser.add_argument(
    "--topology",
    type=str,
    default="full",
    choices=["full", "ring", "band", "lower", "toeplitz", "orthogonal"],
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

args = parser.parse_args()

# Set random seed for reproducibility
SEED = 42
set_seed(SEED)

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


@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        states, _ = model(x)
        output = states[:, -1, :]
        activations.append(output.cpu())
        ys.append(y)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)


n_inp = 1
n_out = 2  # FordA classes (not used directly in RC pipeline)
gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

bs_test = max(args.batch, 256)
if args.trials > 1:
    assert args.use_test, "Multiple runs are only for the final test phase with the test set."
    train_loader, valid_loader, test_loader = get_ucr_data(
        args.dataroot, args.batch, bs_test, whole_train=True
    )
else:
    train_loader, valid_loader, test_loader = get_ucr_data(
        args.dataroot, args.batch, bs_test
    )

train_accs, valid_accs, test_accs = [], [], []

for i in range(args.trials):
    # Set seed for reproducible model initialization
    set_seed(42 + i)
    
    if args.esn:
        units_per_layer = int(args.n_hid // args.n_layers)
        model = DeepReservoir(
            n_inp,
            tot_units=args.n_hid,
            n_layers=args.n_layers,
            concat=True,
            spectral_radius=args.rho,
            input_scaling=args.inp_scaling,
            inter_scaling=args.inp_scaling,
            leaky=args.leaky,
            connectivity_input=units_per_layer,
            connectivity_recurrent=units_per_layer,
            connectivity_inter=units_per_layer,
            cycle=args.cycle,
            antisymmetric=args.antisymmetric,
            epsilon=args.coupling_epsilon,
            gamma=args.diffusive_gamma,
        ).to(device)
    elif args.ron:
        model = RandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.diffusive_gamma,
            args.rho,
            args.inp_scaling,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
            antisymmetric_coupling=args.antisymmetric,
            coupling_epsilon=args.coupling_epsilon,
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
            inter_scaling=args.inp_scaling,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
            connectivity_input=int((1-args.sparsity *n_inp)),
            connectivity_inter=int(args.n_hid / args.n_layers),
            concat=True,
            topology=args.topology,
            antisymmetric_coupling=args.antisymmetric,
            coupling_epsilon=args.coupling_epsilon,
        ).to(device)
    elif args.pron:
        model = PhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device,
            input_function=args.input_fn,
            matrix_friction=args.matrix_friction
        ).to(device)
    elif args.mspron:
        model = MultistablePhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device
        ).to(device)
    else:
        raise ValueError("Wrong model choice.")

    activations, ys = [], []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        states, _ = model(x)
        output = states[:, -1, :]
        activations.append(output.cpu())
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

suffix_ac = "_AC" if args.antisymmetric else ""
if args.ron:
    f = open(os.path.join(args.resultroot, f"FordA_log_RON_{args.topology}{suffix_ac}{args.resultsuffix}.txt"), "a")
elif args.deepron:
    f = open(os.path.join(args.resultroot, f"FordA_log_DeepRON_{args.topology}{suffix_ac}{args.resultsuffix}.txt"), "a")
elif args.pron:
    f = open(os.path.join(args.resultroot, f"FordA_log_PRON{suffix_ac}{args.resultsuffix}.txt"), "a")
elif args.mspron:
    f = open(os.path.join(args.resultroot, f"FordA_log_MSPRON{suffix_ac}{args.resultsuffix}.txt"), "a")
elif args.esn:
    f = open(os.path.join(args.resultroot, f"FordA_log_ESN{suffix_ac}{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 2)) for train_acc in train_accs]} "
    f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_accs]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_accs]}"
    f"mean/std train: {np.mean(train_accs), np.std(train_accs)} "
    f"mean/std valid: {np.mean(valid_accs), np.std(valid_accs)} "
    f"mean/std test: {np.mean(test_accs), np.std(test_accs)}"
)
f.write(ar + "\n")
f.close()

# Print results to console
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Train Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
print(f"Valid Accuracy: {np.mean(valid_accs):.4f} ± {np.std(valid_accs):.4f}")
print(f"Test Accuracy:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
print("="*80)
