import argparse
import os
import warnings
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
from acds.benchmarks import get_psmnist_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_logreg_params(clf):
	n_weights = clf.coef_.size
	n_bias = clf.intercept_.size
	return n_weights + n_bias


def count_parameters(model):
	"""Return total, reservoir (non-trainable), and trainable parameter counts."""

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	reservoir_params = total_params - trainable_params
	return total_params, reservoir_params, trainable_params


@torch.no_grad()
def evaluate(loader, model, classifier, scaler, device):
	activations, ys = [], []
	for images, labels in tqdm(loader):
		images = images.to(device)
		images = images.unsqueeze(-1)  # (B, 784) -> (B, 784, 1)
		states, _ = model(images)
		output = states[:, -1, :]
		activations.append(output.cpu())
		ys.append(labels)

	activations = torch.cat(activations, dim=0).numpy()
	activations = scaler.transform(activations)
	ys = torch.cat(ys, dim=0).numpy()
	return classifier.score(activations, ys)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run a single psMNIST experiment")
parser.add_argument("--dataroot", type=str, help="Root folder containing psMNIST data")
parser.add_argument("--resultroot", type=str, help="Directory to append log files")
parser.add_argument("--resultsuffix", type=str, default="", help="Suffix for log filenames")
parser.add_argument("--n_hid", type=int, default=256, help="Total hidden units across layers")
parser.add_argument("--n_layers", type=int, default=1, help="Number of reservoir layers")
parser.add_argument("--batch", type=int, default=1000, help="Batch size")
parser.add_argument("--dt", type=float, default=0.042, help="coRNN time step")
parser.add_argument("--gamma", type=float, default=2.7, help="coRNN gamma center")
parser.add_argument("--epsilon", type=float, default=4.7, help="coRNN epsilon center")
parser.add_argument("--gamma_range", type=float, default=2.7, help="Range for gamma")
parser.add_argument("--epsilon_range", type=float, default=4.7, help="Range for epsilon")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--cpu", action="store_true", help="Force CPU")
parser.add_argument("--esn", action="store_true", help="Use ESN/DeepESN")
parser.add_argument("--ron", action="store_true", help="Use RON")
parser.add_argument("--pron", action="store_true", help="Use PRON")
parser.add_argument("--mspron", action="store_true", help="Use MSPRON")
parser.add_argument("--deepron", action="store_true", help="Use DeepRON")
parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="Diffusive coupling strength")
parser.add_argument("--inp_scaling", type=float, default=1.0, help="Input scaling")
parser.add_argument("--rho", type=float, default=0.99, help="Spectral radius")
parser.add_argument("--leaky", type=float, default=1.0, help="Leaky parameter for ESN")
parser.add_argument("--cycle", action="store_true", help="Use cycle reservoir")
parser.add_argument("--antisymmetric", action="store_true", help="Enable antisymmetric coupling")
parser.add_argument("--coupling_epsilon", type=float, default=0.4, help="Antisymmetric coupling epsilon")
parser.add_argument("--concat", action="store_true", help="Concatenate layer states for readout")
parser.add_argument("--use_test", action="store_true", help="Evaluate on test instead of validation")
parser.add_argument("--trials", type=int, default=1, help="Number of repeated runs")
parser.add_argument("--topology", type=str, default="full", choices=[
	"full",
	"ring",
	"band",
	"lower",
	"toeplitz",
	"orthogonal",
	"antisymmetric",
], help="Reservoir topology")
parser.add_argument("--sparsity", type=float, default=0.0, help="Reservoir sparsity")
parser.add_argument("--reservoir_scaler", type=float, default=1.0, help="Scaler for structured reservoirs")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
set_seed(args.seed)
print(f"Random seed set to: {args.seed}")

if args.dataroot is None:
	warnings.warn("No dataroot provided. Using current location as default.")
	args.dataroot = os.getcwd()
if args.resultroot is None:
	warnings.warn("No resultroot provided. Using current location as default.")
	args.resultroot = os.getcwd()

assert os.path.exists(args.resultroot), (
	f"{args.resultroot} folder does not exist, please create it and run the script again."
)
assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device:", device)

n_inp = 1
gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (args.epsilon - args.epsilon_range / 2.0, args.epsilon + args.epsilon_range / 2.0)

# Data loaders (psMNIST uses a fixed permutation; keep seed consistent with search)
train_loader, valid_loader, test_loader = get_psmnist_data(
	root=args.dataroot, bs_train=args.batch, bs_test=args.batch, seed=args.seed
)

train_accs, valid_accs, test_accs = [], [], []

for _ in range(args.trials):
	if args.esn:
		units_per_layer = args.n_hid // args.n_layers
		model = DeepReservoir(
			input_size=n_inp,
			tot_units=args.n_hid,
			n_layers=args.n_layers,
			spectral_radius=args.rho,
			input_scaling=args.inp_scaling,
			inter_scaling=args.inp_scaling,
			connectivity_recurrent=units_per_layer,
			connectivity_input=units_per_layer,
			connectivity_inter=units_per_layer,
			leaky=args.leaky,
			cycle=args.cycle,
			concat=args.concat,
			linear=False,
			epsilon=args.coupling_epsilon,
			antisymmetric=args.antisymmetric,
			gamma=args.diffusive_gamma,
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
		).to(device)
	elif args.mspron:
		model = MultistablePhysicallyImplementableRandomizedOscillatorsNetwork(
			n_inp,
			args.n_hid,
			args.dt,
			gamma,
			epsilon,
			args.inp_scaling,
			device=device,
		).to(device)
	elif args.deepron:
		model = DeepRandomizedOscillatorsNetwork(
			n_inp=n_inp,
			total_units=args.n_hid,
			dt=args.dt,
			gamma=gamma,
			epsilon=epsilon,
			n_layers=args.n_layers,
			diffusive_gamma=args.diffusive_gamma,
			rho=args.rho,
			input_scaling=args.inp_scaling,
			inter_scaling=args.inp_scaling,
			topology=args.topology,
			sparsity=args.sparsity,
			reservoir_scaler=args.reservoir_scaler,
			device=device,
			antisymmetric_coupling=args.antisymmetric,
			coupling_epsilon=args.coupling_epsilon,
			concat=args.concat,
			cycle=args.cycle,
		).to(device)
	else:
		raise ValueError("Select one model flag: --esn, --ron, --pron, --mspron, or --deepron")

	total_params, reservoir_params, trainable_params = count_parameters(model)
	print("\n" + "=" * 60)
	print("MODEL PARAMETERS")
	print("=" * 60)
	print(f"Total parameters:      {total_params:,}")
	print(f"Reservoir parameters:  {reservoir_params:,}")
	print(f"Trainable parameters:  {trainable_params:,}")
	print("=" * 60 + "\n")

	activations, ys = [], []
	for images, labels in tqdm(train_loader):
		images = images.to(device)
		images = images.unsqueeze(-1)
		states, _ = model(images)
		output = states[:, -1, :]
		activations.append(output.cpu())
		ys.append(labels)

	activations = torch.cat(activations, dim=0).numpy()
	ys = torch.cat(ys, dim=0).numpy()

	scaler = preprocessing.StandardScaler().fit(activations)
	activations = scaler.transform(activations)
	classifier = LogisticRegression(max_iter=1000).fit(activations, ys)

	readout_params = count_logreg_params(classifier)
	print(f"Readout (LogisticRegression) parameters: {readout_params}")

	train_acc = evaluate(train_loader, model, classifier, scaler, device)
	valid_acc = evaluate(valid_loader, model, classifier, scaler, device) if not args.use_test else 0.0
	test_acc = evaluate(test_loader, model, classifier, scaler, device) if args.use_test else 0.0

	train_accs.append(train_acc)
	valid_accs.append(valid_acc)
	test_accs.append(test_acc)


if args.ron:
	logfile = os.path.join(args.resultroot, f"psMNIST_log_RON_{args.topology}{args.resultsuffix}.txt")
elif args.pron:
	logfile = os.path.join(args.resultroot, f"psMNIST_log_PRON{args.resultsuffix}.txt")
elif args.mspron:
	logfile = os.path.join(args.resultroot, f"psMNIST_log_MSPRON{args.resultsuffix}.txt")
elif args.esn:
	logfile = os.path.join(args.resultroot, f"psMNIST_log_ESN{args.resultsuffix}.txt")
elif args.deepron:
	logfile = os.path.join(args.resultroot, f"psMNIST_log_DEEPRON{args.resultsuffix}.txt")
else:
	raise ValueError("Wrong model choice.")

ar = ""
for k, v in vars(args).items():
	ar += f"{str(k)}: {str(v)}, "
ar += (
	f"train: {[str(round(train_acc, 4)) for train_acc in train_accs]} "
	f"valid: {[str(round(valid_acc, 4)) for valid_acc in valid_accs]} "
	f"test: {[str(round(test_acc, 4)) for test_acc in test_accs]} "
	f"mean/std train: {(np.mean(train_accs), np.std(train_accs))} "
	f"mean/std valid: {(np.mean(valid_accs), np.std(valid_accs))} "
	f"mean/std test: {(np.mean(test_accs), np.std(test_accs))}"
)

with open(logfile, "a") as f:
	f.write(ar + "\n")

print("\n" + "=" * 60)
print("Final Results Summary:")
print("=" * 60)
print(f"Train: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
print(f"Valid: {np.mean(valid_accs):.4f} ± {np.std(valid_accs):.4f}")
print(f"Test:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
print("=" * 60)
