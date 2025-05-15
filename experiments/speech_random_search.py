import argparse
import os
import subprocess
import json
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from scipy.stats import loguniform

# This script performs random search by calling speech.py as a subprocess with generated parameters

parser = argparse.ArgumentParser(description="Speech Random Search (calls speech.py)")
parser.add_argument("--dataroot", type=str, required=True, help="Directory containing speech .npy files (one per class)")
parser.add_argument("--resultroot", type=str, help="Directory to save results")
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument("--n_trials", type=int, default=10, help="Number of random trials")
parser.add_argument("--model", type=str, required=True, choices=["esn", "ron", "deepron", "pron"], help="Model type to use")
parser.add_argument("--use_last_state", action="store_true", help="Use last hidden state instead of all states")
parser.add_argument("--batch", type=int, default=64, help="Batch size")
parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_args()

if args.resultroot is None:
    args.resultroot = os.getcwd()
if not os.path.exists(args.resultroot):
    os.makedirs(args.resultroot, exist_ok=True)

# Random config generator (similar to speech_random_search.py)
def get_random_config(model, seed):
    rng = np.random.RandomState(seed)
    config = {}
    if model in ["esn", "deepron"]:
        config["rho"] = 0.99
        config["leaky"] = 1
    if model in ["ron", "deepron", "pron"]:
        config["dt"] = loguniform.rvs(1e-5, 1, size=1, random_state=rng)[0]
        config["gamma"] = rng.uniform(0.5, 5.0)
        config["gamma_range"] = rng.uniform(0.0, 2.0)
        config["epsilon"] = rng.uniform(0.5, 5.0)
        config["epsilon_range"] = rng.uniform(0.0, 2.0)
    
    # return an integer
    config["n_layers"] = int(rng.uniform(1, 10))
    config["n_hid"] = 100
    config["inp_scaling"] = 1.0
    #config["n_layers"] = 1
    config["sparsity"] = 0.0
    
    return config

# Main random search loop
results = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for trial in tqdm(range(args.n_trials)):
    trial_seed = args.seed + trial
    config = get_random_config(args.model, trial_seed) 
    if args.model == "ron":
        resultsuffix = "_full"
    else:
        resultsuffix = args.resultsuffix
    cmd = [
        "python", "experiments/speech.py",
        "--dataroot", args.dataroot,
        "--resultroot", args.resultroot,
        f"--{args.model}",
        "--n_hid", str(config["n_hid"]),
        "--n_layers", str(config["n_layers"]),
        "--batch", str(args.batch),
        "--inp_scaling", str(config["inp_scaling"]),
        "--sparsity", str(config["sparsity"]),
        "--resultsuffix", resultsuffix,
        "--concat",
        "--use_test",
        "--wandb",
        "--seed", str(trial_seed)
    ]
    if args.cpu:
        cmd.append("--cpu")
    if args.use_last_state:
        cmd.append("--use_last_state")
    if args.model in ["esn", "deepron"]:
        cmd += ["--rho", str(config["rho"]), "--leaky", str(config["leaky"])]
    if args.model in ["ron", "deepron", "pron"]:
        cmd += [
            "--dt", str(config["dt"]),
            "--gamma", str(config["gamma"]),
            "--gamma_range", str(config["gamma_range"]),
            "--epsilon", str(config["epsilon"]),
            "--epsilon_range", str(config["epsilon_range"])
        ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        status = "success" if result.returncode == 0 else f"error: {result.stderr.strip()}"
    except Exception as e:
        status = f"exception: {str(e)}"
        result = None

    # After subprocess run, try to extract the accuracy and parameters from speech.py output
    log_filename = os.path.join(args.resultroot, f"Speech_log_{args.model.upper()}{resultsuffix}.txt")

    last_log = None
    if result and result.stdout:
        try:
            with open(log_filename, "r") as flog:
                lines = flog.readlines()
                if lines:
                    last_log = lines[-1].strip()
                    summary_log = os.path.join(args.resultroot, f"SpeechRandomSearch_summary_{args.model}{resultsuffix}.txt")
                    with open(summary_log, "a") as fsum:
                        fsum.write(last_log + "\n")
        except Exception as e:
            print(f"Could not read or write log: {e}")


