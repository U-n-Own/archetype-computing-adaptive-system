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


def generate_latex_table(results_smnist: Dict, results_speech: Dict) -> str:
    """Generate LaTeX table for paper."""
    latex = r"""\begin{table}[h]
\centering
\caption{Antisymmetric Coupling Performance on Sequential Tasks}
\label{tab:antisym_epsilon_search}
\begin{tabular}{lcccccc}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{Layers} & \textbf{$\epsilon$} & \textbf{Test Acc (\%)} & \textbf{Improvement} \\
\midrule
"""
    
    # sMNIST rows
    latex += "\\multirow{2}{*}{sMNIST} & Standard ESN & 1 & -- & "
    latex += f"{results_smnist.get('baseline_acc', 0):.2f} & -- \\\\\n"
    latex += " & Antisymmetric ESN & 5 & "
    latex += f"{results_smnist.get('best_antisym_epsilon', 0):.3f} & "
    latex += f"{results_smnist.get('best_antisym_acc', 0):.2f} & "
    latex += f"{results_smnist.get('improvement', 0):+.2f} \\\\\n"
    latex += "\\midrule\n"
    
    # Speech rows
    latex += "\\multirow{2}{*}{Speech Cmd.} & Standard ESN & 1 & -- & "
    latex += f"{results_speech.get('baseline_acc', 0):.2f} & -- \\\\\n"
    latex += " & Antisymmetric ESN & 5 & "
    latex += f"{results_speech.get('best_antisym_epsilon', 0):.3f} & "
    latex += f"{results_speech.get('best_antisym_acc', 0):.2f} & "
    latex += f"{results_speech.get('improvement', 0):+.2f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_markdown_table(results_smnist: Dict, results_speech: Dict) -> str:
    """Generate Markdown table for documentation."""
    md = "# Antisymmetric Coupling Epsilon Search Results\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    md += "## Summary Table\n\n"
    md += "| Dataset | Model | Layers | Epsilon | Test Accuracy | Improvement |\n"
    md += "|---------|-------|--------|---------|---------------|-------------|\n"
    
    # sMNIST
    md += f"| sMNIST | Standard ESN | 1 | -- | {results_smnist.get('baseline_acc', 0):.2f}% | -- |\n"
    md += f"| sMNIST | Antisymmetric ESN | 5 | {results_smnist.get('best_antisym_epsilon', 0):.3f} | "
    md += f"{results_smnist.get('best_antisym_acc', 0):.2f}% | {results_smnist.get('improvement', 0):+.2f}pp |\n"
    
    # Speech
    md += f"| Speech Cmd. | Standard ESN | 1 | -- | {results_speech.get('baseline_acc', 0):.2f}% | -- |\n"
    md += f"| Speech Cmd. | Antisymmetric ESN | 5 | {results_speech.get('best_antisym_epsilon', 0):.3f} | "
    md += f"{results_speech.get('best_antisym_acc', 0):.2f}% | {results_speech.get('improvement', 0):+.2f}pp |\n"
    
    md += "\n## Detailed Results\n\n"
    
    # sMNIST detailed
    md += "### sMNIST\n\n"
    if 'models' in results_smnist and results_smnist['models']:
        md += "| Rank | Model | Layers | Epsilon | Test Acc | Valid Acc | Saturated % |\n"
        md += "|------|-------|--------|---------|----------|-----------|-------------|\n"
        for m in results_smnist['models']:
            eps_str = f"{m['epsilon']:.3f}" if m['epsilon'] is not None else "--"
            sat_str = f"{m['saturated_pct']:.2f}" if m['saturated_pct'] is not None else "--"
            valid_str = f"{m['valid_acc']:.2f}%" if 'valid_acc' in m else "--"
            md += f"| {m['rank']} | {m['model']} | {m['n_layers']} | {eps_str} | "
            md += f"{m['test_acc']:.2f}% | {valid_str} | {sat_str}% |\n"
    
    md += "\n### Speech Commands\n\n"
    if 'models' in results_speech and results_speech['models']:
        md += "| Rank | Model | Layers | Epsilon | Test Acc | Train Acc | Saturated % |\n"
        md += "|------|-------|--------|---------|----------|-----------|-------------|\n"
        for m in results_speech['models']:
            eps_str = f"{m['epsilon']:.3f}" if m['epsilon'] is not None else "--"
            sat_str = f"{m['saturated_pct']:.2f}" if m['saturated_pct'] is not None else "--"
            train_str = f"{m['train_acc']:.2f}%" if 'train_acc' in m else "--"
            md += f"| {m['rank']} | {m['model']} | {m['n_layers']} | {eps_str} | "
            md += f"{m['test_acc']:.2f}% | {train_str} | {sat_str}% |\n"
    
    md += "\n## Experimental Configuration\n\n"
    md += "- **Architecture**: Echo State Network (ESN)\n"
    md += "- **Hidden Units**: 500 total\n"
    md += "- **Spectral Radius**: 0.9 (sMNIST: 0.999)\n"
    md += "- **Leaky Rate**: 0.001\n"
    md += "- **Input Scaling**: 1.0\n"
    md += "- **Antisymmetric Layers**: 5 layers with 100 units each\n"
    md += "- **Epsilon Values Tested**: 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5 (sMNIST); 0.001, 0.01, 0.5 (Speech)\n"
    
    return md


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
        result = subprocess.run(
            ["python", "test_smnist_epsilon_search.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        smnist_output = result.stdout
        print(smnist_output)
        
        # Save raw output
        with open(f"{RESULTS_DIR}/smnist_output_{timestamp}.txt", "w") as f:
            f.write(smnist_output)
        
        # Parse results
        all_results['smnist'] = parse_smnist_output(smnist_output)
        print("\n✓ sMNIST test completed successfully\n")
    except Exception as e:
        print(f"\n✗ sMNIST test failed: {e}\n")
        all_results['smnist'] = {'dataset': 'sMNIST', 'models': [], 'error': str(e)}
    
    # Run Speech test
    print("\n" + "="*80)
    print("Running Speech Commands epsilon search test...")
    print("-" * 80)
    try:
        result = subprocess.run(
            ["python", "test_speech_epsilon_search.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        speech_output = result.stdout
        print(speech_output)
        
        # Save raw output
        with open(f"{RESULTS_DIR}/speech_output_{timestamp}.txt", "w") as f:
            f.write(speech_output)
        
        # Parse results
        all_results['speech'] = parse_speech_output(speech_output)
        print("\n✓ Speech test completed successfully\n")
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
    
    # Generate LaTeX table
    latex_file = f"{RESULTS_DIR}/results_table_{timestamp}.tex"
    latex_content = generate_latex_table(all_results['smnist'], all_results['speech'])
    with open(latex_file, "w") as f:
        f.write(latex_content)
    print(f"✓ LaTeX table saved to: {latex_file}")
    
    # Generate Markdown table
    md_file = f"{RESULTS_DIR}/results_report_{timestamp}.md"
    md_content = generate_markdown_table(all_results['smnist'], all_results['speech'])
    with open(md_file, "w") as f:
        f.write(md_content)
    print(f"✓ Markdown report saved to: {md_file}")
    
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
