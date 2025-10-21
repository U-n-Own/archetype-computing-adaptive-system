"""
Results aggregator for sMNIST and Speech epsilon search experiments.
Runs both test scripts and generates formatted tables for reporting.
"""
import subprocess
import re
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

print("=" * 80)
print("ANTISYMMETRIC COUPLING EPSILON SEARCH - RESULTS AGGREGATOR")
print("=" * 80)
print()

# Configuration
RESULTS_DIR = "results_epsilon_search"
os.makedirs(RESULTS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_smnist_output(output: str) -> Dict[str, Any]:
    """Parse sMNIST test output and extract results."""
    results = {
        'dataset': 'sMNIST',
        'models': []
    }
    
    # Extract model results
    lines = output.split('\n')
    in_results = False
    
    for i, line in enumerate(lines):
        if 'Rank' in line and 'Model' in line and 'Epsilon' in line:
            in_results = True
            continue
        
        if in_results and line.strip().startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')):
            parts = line.split()
            if len(parts) >= 6:
                rank = int(parts[0])
                # Check if antisymmetric
                is_antisym = '✓' in line or 'Antisymmetric' in line
                
                # Extract model name
                if is_antisym:
                    # Remove the ✓ marker
                    name_start = line.find('✓') + 1
                else:
                    name_start = line.find(parts[0]) + len(parts[0])
                
                # Find epsilon value
                eps_match = re.search(r'ε=([0-9.]+)', line)
                epsilon = float(eps_match.group(1)) if eps_match else None
                
                # Extract accuracies (format: XX.XX%)
                acc_matches = re.findall(r'(\d+\.\d+)%', line)
                if len(acc_matches) >= 2:
                    test_acc = float(acc_matches[0])
                    valid_acc = float(acc_matches[1])
                    
                    # Extract saturation if present
                    if len(acc_matches) >= 3:
                        saturated = float(acc_matches[2])
                    else:
                        saturated = None
                    
                    model_name = "Antisymmetric 5-layer" if is_antisym else "Standard 1-layer"
                    
                    results['models'].append({
                        'rank': rank,
                        'model': model_name,
                        'n_layers': 5 if is_antisym else 1,
                        'epsilon': epsilon,
                        'test_acc': test_acc,
                        'valid_acc': valid_acc,
                        'saturated_pct': saturated
                    })
        
        if in_results and line.strip().startswith('='):
            in_results = False
    
    # Extract key findings
    for i, line in enumerate(lines):
        if 'Baseline (Standard 1-layer):' in line:
            if i + 1 < len(lines):
                match = re.search(r'Test Accuracy: (\d+\.\d+)%', lines[i+1])
                if match:
                    results['baseline_acc'] = float(match.group(1))
        
        if 'Best Antisymmetric (5-layer):' in line:
            if i + 1 < len(lines):
                match = re.search(r'Test Accuracy: (\d+\.\d+)%', lines[i+1])
                if match:
                    results['best_antisym_acc'] = float(match.group(1))
            if i + 2 < len(lines):
                match = re.search(r'Epsilon: ([0-9.]+)', lines[i+2])
                if match:
                    results['best_antisym_epsilon'] = float(match.group(1))
        
        if 'Antisymmetric vs Baseline:' in line:
            match = re.search(r'([+-]\d+\.\d+) percentage points', line)
            if match:
                results['improvement'] = float(match.group(1))
    
    return results


def parse_speech_output(output: str) -> Dict[str, Any]:
    """Parse Speech test output and extract results."""
    results = {
        'dataset': 'Speech Commands',
        'models': []
    }
    
    # Extract model results
    lines = output.split('\n')
    in_results = False
    
    for i, line in enumerate(lines):
        if 'Rank' in line and 'Model' in line and 'Epsilon' in line:
            in_results = True
            continue
        
        if in_results and line.strip().startswith(('1', '2', '3', '4', '5')):
            parts = line.split()
            if len(parts) >= 6:
                rank = int(parts[0])
                is_antisym = '✓' in line or 'Antisymmetric' in line
                
                # Extract epsilon value
                eps_match = re.search(r'ε=([0-9.]+)', line)
                epsilon = float(eps_match.group(1)) if eps_match else None
                
                # Extract accuracies
                acc_matches = re.findall(r'(\d+\.\d+)%', line)
                if len(acc_matches) >= 2:
                    test_acc = float(acc_matches[0])
                    train_acc = float(acc_matches[1])
                    
                    if len(acc_matches) >= 3:
                        saturated = float(acc_matches[2])
                    else:
                        saturated = None
                    
                    model_name = "Antisymmetric 5-layer" if is_antisym else "Standard 1-layer"
                    
                    results['models'].append({
                        'rank': rank,
                        'model': model_name,
                        'n_layers': 5 if is_antisym else 1,
                        'epsilon': epsilon,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'saturated_pct': saturated
                    })
        
        if in_results and line.strip().startswith('='):
            in_results = False
    
    # Extract key findings
    for i, line in enumerate(lines):
        if 'Baseline (Standard 1-layer):' in line:
            if i + 1 < len(lines):
                match = re.search(r'Test Accuracy: (\d+\.\d+)%', lines[i+1])
                if match:
                    results['baseline_acc'] = float(match.group(1))
        
        if 'Best Antisymmetric (5-layer):' in line:
            if i + 1 < len(lines):
                match = re.search(r'Test Accuracy: (\d+\.\d+)%', lines[i+1])
                if match:
                    results['best_antisym_acc'] = float(match.group(1))
            if i + 2 < len(lines):
                match = re.search(r'Epsilon: ([0-9.]+)', lines[i+2])
                if match:
                    results['best_antisym_epsilon'] = float(match.group(1))
        
        if 'Antisymmetric vs Baseline:' in line:
            match = re.search(r'([+-]\d+\.\d+) percentage points', line)
            if match:
                results['improvement'] = float(match.group(1))
    
    return results


def generate_summary_table(results_smnist: Dict, results_speech: Dict) -> str:
    """Generate simple text table for later parsing."""
    table = "# SUMMARY TABLE - Antisymmetric Coupling Performance\n"
    table += "# Format: Dataset | Model | Layers | Epsilon | Test_Acc | Improvement\n\n"
    
    # sMNIST rows
    table += f"sMNIST | Standard ESN | 1 | -- | {results_smnist.get('baseline_acc', 0):.2f} | --\n"
    table += f"sMNIST | Antisymmetric ESN | 5 | {results_smnist.get('best_antisym_epsilon', 0):.3f} | "
    table += f"{results_smnist.get('best_antisym_acc', 0):.2f} | {results_smnist.get('improvement', 0):+.2f}\n"
    
    # Speech rows
    table += f"Speech Commands | Standard ESN | 1 | -- | {results_speech.get('baseline_acc', 0):.2f} | --\n"
    table += f"Speech Commands | Antisymmetric ESN | 5 | {results_speech.get('best_antisym_epsilon', 0):.3f} | "
    table += f"{results_speech.get('best_antisym_acc', 0):.2f} | {results_speech.get('improvement', 0):+.2f}\n"
    
    return table


def generate_detailed_results(results_smnist: Dict, results_speech: Dict) -> str:
    """Generate detailed results in parseable text format."""
    text = "# DETAILED RESULTS - Antisymmetric Coupling Epsilon Search\n"
    text += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    text += "## EXPERIMENTAL CONFIGURATION\n"
    text += "Architecture: Echo State Network (ESN)\n"
    text += "Hidden Units: 500 total\n"
    text += "Spectral Radius: 0.9 (sMNIST: 0.999)\n"
    text += "Leaky Rate: 0.001\n"
    text += "Input Scaling: 1.0\n"
    text += "Antisymmetric Layers: 5 layers with 100 units each\n"
    text += "Epsilon Values Tested: 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5 (sMNIST); 0.001, 0.01, 0.5 (Speech)\n\n"
    
    # sMNIST detailed
    text += "## sMNIST RESULTS\n"
    text += "# Format: Rank | Model | Layers | Epsilon | Test_Acc | Valid_Acc | Saturated_Pct\n"
    if 'models' in results_smnist and results_smnist['models']:
        for m in results_smnist['models']:
            eps_str = f"{m['epsilon']:.3f}" if m['epsilon'] is not None else "--"
            sat_str = f"{m['saturated_pct']:.2f}" if m['saturated_pct'] is not None else "--"
            valid_str = f"{m['valid_acc']:.2f}" if 'valid_acc' in m else "--"
            text += f"{m['rank']} | {m['model']} | {m['n_layers']} | {eps_str} | "
            text += f"{m['test_acc']:.2f} | {valid_str} | {sat_str}\n"
    
    text += f"\nBaseline Accuracy: {results_smnist.get('baseline_acc', 0):.2f}%\n"
    text += f"Best Antisymmetric Accuracy: {results_smnist.get('best_antisym_acc', 0):.2f}%\n"
    text += f"Best Epsilon: {results_smnist.get('best_antisym_epsilon', 0):.3f}\n"
    text += f"Improvement: {results_smnist.get('improvement', 0):+.2f} percentage points\n\n"
    
    # Speech detailed
    text += "## SPEECH COMMANDS RESULTS\n"
    text += "# Format: Rank | Model | Layers | Epsilon | Test_Acc | Train_Acc | Saturated_Pct\n"
    if 'models' in results_speech and results_speech['models']:
        for m in results_speech['models']:
            eps_str = f"{m['epsilon']:.3f}" if m['epsilon'] is not None else "--"
            sat_str = f"{m['saturated_pct']:.2f}" if m['saturated_pct'] is not None else "--"
            train_str = f"{m['train_acc']:.2f}" if 'train_acc' in m else "--"
            text += f"{m['rank']} | {m['model']} | {m['n_layers']} | {eps_str} | "
            text += f"{m['test_acc']:.2f} | {train_str} | {sat_str}\n"
    
    text += f"\nBaseline Accuracy: {results_speech.get('baseline_acc', 0):.2f}%\n"
    text += f"Best Antisymmetric Accuracy: {results_speech.get('best_antisym_acc', 0):.2f}%\n"
    text += f"Best Epsilon: {results_speech.get('best_antisym_epsilon', 0):.3f}\n"
    text += f"Improvement: {results_speech.get('improvement', 0):+.2f} percentage points\n"
    
    return text


def generate_csv_table(results_smnist: Dict, results_speech: Dict) -> str:
    """Generate CSV for spreadsheet import."""
    csv = "Dataset,Model,Layers,Epsilon,Test_Accuracy,Improvement,Saturated_Pct\n"
    
    # sMNIST
    csv += f"sMNIST,Standard ESN,1,,{results_smnist.get('baseline_acc', 0):.2f},,\n"
    if 'models' in results_smnist:
        for m in results_smnist['models']:
            if 'Antisymmetric' in m['model']:
                eps_str = f"{m['epsilon']}" if m['epsilon'] is not None else ""
                sat_str = f"{m['saturated_pct']:.2f}" if m['saturated_pct'] is not None else ""
                csv += f"sMNIST,{m['model']},{m['n_layers']},{eps_str},{m['test_acc']:.2f},,{sat_str}\n"
    
    # Speech
    csv += f"Speech Commands,Standard ESN,1,,{results_speech.get('baseline_acc', 0):.2f},,\n"
    if 'models' in results_speech:
        for m in results_speech['models']:
            if 'Antisymmetric' in m['model']:
                eps_str = f"{m['epsilon']}" if m['epsilon'] is not None else ""
                sat_str = f"{m['saturated_pct']:.2f}" if m['saturated_pct'] is not None else ""
                csv += f"Speech Commands,{m['model']},{m['n_layers']},{eps_str},{m['test_acc']:.2f},,{sat_str}\n"
    
    return csv


if __name__ == "__main__":
    all_results = {}
    
    # Run sMNIST test
    print("Running sMNIST epsilon search test...")
    print("-" * 80)
    try:
        # Run with real-time output while also capturing
        result = subprocess.run(
            ["python", "test_smnist_epsilon_search.py"],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=True  # Raise exception on non-zero exit
        )
        smnist_output = result.stdout
        
        # Print output (in case it was captured silently)
        if smnist_output:
            print(smnist_output)
        
        # Also print stderr if any
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Save raw output
        with open(f"{RESULTS_DIR}/smnist_output_{timestamp}.txt", "w") as f:
            f.write(smnist_output)
            if result.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
        
        # Parse results
        all_results['smnist'] = parse_smnist_output(smnist_output)
        print("\n✓ sMNIST test completed successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ sMNIST test failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}\n")
        all_results['smnist'] = {'dataset': 'sMNIST', 'models': [], 'error': str(e)}
    except Exception as e:
        print(f"\n✗ sMNIST test failed: {e}\n")
        all_results['smnist'] = {'dataset': 'sMNIST', 'models': [], 'error': str(e)}
    
    # Run Speech test
    print("\n" + "="*80)
    print("Running Speech Commands epsilon search test...")
    print("-" * 80)
    try:
        # Run with real-time output while also capturing
        result = subprocess.run(
            ["python", "test_speech_epsilon_search.py"],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=True  # Raise exception on non-zero exit
        )
        speech_output = result.stdout
        
        # Print output (in case it was captured silently)
        if speech_output:
            print(speech_output)
        
        # Also print stderr if any
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Save raw output
        with open(f"{RESULTS_DIR}/speech_output_{timestamp}.txt", "w") as f:
            f.write(speech_output)
            if result.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
        
        # Parse results
        all_results['speech'] = parse_speech_output(speech_output)
        print("\n✓ Speech test completed successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Speech test failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}\n")
        all_results['speech'] = {'dataset': 'Speech Commands', 'models': [], 'error': str(e)}
    except Exception as e:
        print(f"\n✗ Speech test failed: {e}\n")
        all_results['speech'] = {'dataset': 'Speech Commands', 'models': [], 'error': str(e)}
    
    # Generate reports
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    # Save JSON results
    json_file = f"{RESULTS_DIR}/results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ JSON results saved to: {json_file}")
    
    # Generate summary table (parseable text format)
    summary_file = f"{RESULTS_DIR}/results_summary_{timestamp}.txt"
    summary_content = generate_summary_table(all_results['smnist'], all_results['speech'])
    with open(summary_file, "w") as f:
        f.write(summary_content)
    print(f"✓ Summary table saved to: {summary_file}")
    
    # Generate detailed results (parseable text format)
    detailed_file = f"{RESULTS_DIR}/results_detailed_{timestamp}.txt"
    detailed_content = generate_detailed_results(all_results['smnist'], all_results['speech'])
    with open(detailed_file, "w") as f:
        f.write(detailed_content)
    print(f"✓ Detailed results saved to: {detailed_file}")
    
    # Generate CSV
    csv_file = f"{RESULTS_DIR}/results_data_{timestamp}.csv"
    csv_content = generate_csv_table(all_results['smnist'], all_results['speech'])
    with open(csv_file, "w") as f:
        f.write(csv_content)
    print(f"✓ CSV data saved to: {csv_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nsMNIST:")
    print(f"  Baseline: {all_results['smnist'].get('baseline_acc', 'N/A')}%")
    print(f"  Best Antisymmetric: {all_results['smnist'].get('best_antisym_acc', 'N/A')}% (ε={all_results['smnist'].get('best_antisym_epsilon', 'N/A')})")
    print(f"  Improvement: {all_results['smnist'].get('improvement', 'N/A')}pp")
    
    print("\nSpeech Commands:")
    print(f"  Baseline: {all_results['speech'].get('baseline_acc', 'N/A')}%")
    print(f"  Best Antisymmetric: {all_results['speech'].get('best_antisym_acc', 'N/A')}% (ε={all_results['speech'].get('best_antisym_epsilon', 'N/A')})")
    print(f"  Improvement: {all_results['speech'].get('improvement', 'N/A')}pp")
    
    print("\n" + "="*80)
    print(f"All results saved to: {RESULTS_DIR}/")
    print("="*80)
