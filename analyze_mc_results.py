"""
Additional analysis: Compute relative improvements and create detailed comparison table
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
results = {
    'Configuration': [
        '1×100 units',
        '2×50 units', '2×50 units', '2×50 units',
        '4×25 units', '4×25 units', '4×25 units',
        '50×2 units', '50×2 units', '50×2 units',
        '100×1 unit', '100×1 unit', '100×1 unit'
    ],
    'Type': [
        'Standard',
        'Standard', 'Antisym (ε=0.1)', 'Antisym (ε=10)',
        'Standard', 'Antisym (ε=0.1)', 'Antisym (ε=10)',
        'Standard', 'Antisym (ε=0.1)', 'Antisym (ε=10)',
        'Standard', 'Antisym (ε=0.1)', 'Antisym (ε=10)'
    ],
    'Test_MC': [
        24.0155,
        16.3253, 17.6437, 0.0972,
        14.8140, 16.2822, 0.1029,
        11.1210, 9.8038, 0.4775,
        20.5284, 22.4307, 0.1397
    ],
    'Test_Std': [
        3.4442,
        3.3427, 0.9469, 0.0147,
        2.6448, 0.6888, 0.0126,
        5.4344, 1.0597, 0.3431,
        1.5833, 3.7230, 0.0457
    ],
    'N_Layers': [1, 2, 2, 2, 4, 4, 4, 50, 50, 50, 100, 100, 100]
}

df = pd.DataFrame(results)

# Compute relative improvement for antisymmetric configurations
improvements = []
baseline_configs = {
    '2×50 units': 16.3253,
    '4×25 units': 14.8140,
    '50×2 units': 11.1210,
    '100×1 unit': 20.5284
}

for idx, row in df.iterrows():
    config = row['Configuration']
    if config in baseline_configs and 'Antisym' in row['Type']:
        baseline = baseline_configs[config]
        improvement = ((row['Test_MC'] - baseline) / baseline) * 100
        improvements.append(improvement)
    else:
        improvements.append(None)

df['Improvement_%'] = improvements

# Print detailed table
print("="*100)
print("DETAILED COMPARISON TABLE")
print("="*100)
print(df.to_string(index=False))

# Create improvement comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Improvement percentage for ε=0.1
configs = ['2×50', '4×25', '50×2', '100×1']
improvements_low = []
for config in configs:
    config_full = config + ' units'
    matching_rows = df[(df['Configuration'] == config_full) & (df['Type'] == 'Antisym (ε=0.1)')]
    if len(matching_rows) > 0:
        row = matching_rows.iloc[0]
        improvements_low.append(row['Improvement_%'])
    else:
        improvements_low.append(0)

colors = ['green' if x > 0 else 'red' for x in improvements_low]
bars = axes[0].bar(configs, improvements_low, color=colors, alpha=0.7, edgecolor='black')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0].set_xlabel('Configuration', fontsize=12)
axes[0].set_ylabel('Improvement over Standard (%)', fontsize=12)
axes[0].set_title('Antisymmetric Coupling (ε_c=0.1) - Performance Change', fontsize=14)
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, improvements_low):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

# Plot 2: Variance reduction for ε=0.1
variance_reduction = []
for config in configs:
    config_full = config + ' units'
    std_standard_rows = df[(df['Configuration'] == config_full) & (df['Type'] == 'Standard')]
    std_antisym_rows = df[(df['Configuration'] == config_full) & (df['Type'] == 'Antisym (ε=0.1)')]
    if len(std_standard_rows) > 0 and len(std_antisym_rows) > 0:
        std_standard = std_standard_rows.iloc[0]['Test_Std']
        std_antisym = std_antisym_rows.iloc[0]['Test_Std']
        reduction = ((std_standard - std_antisym) / std_standard) * 100
        variance_reduction.append(reduction)
    else:
        variance_reduction.append(0)

bars2 = axes[1].bar(configs, variance_reduction, color='blue', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Configuration', fontsize=12)
axes[1].set_ylabel('Variance Reduction (%)', fontsize=12)
axes[1].set_title('Antisymmetric Coupling (ε_c=0.1) - Stability Improvement', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars2, variance_reduction):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results_mc_antisymmetric/improvement_analysis.png', dpi=300, bbox_inches='tight')
print("\n" + "="*100)
print("Saved: results_mc_antisymmetric/improvement_analysis.png")
print("="*100)

# Print key insights
print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

print("\n1. PERFORMANCE IMPROVEMENTS (ε_c=0.1):")
for config, improvement in zip(configs, improvements_low):
    status = "✓ IMPROVEMENT" if improvement > 0 else "✗ DEGRADATION"
    print(f"   {config:10s}: {improvement:+6.2f}% {status}")

print("\n2. VARIANCE REDUCTION (ε_c=0.1):")
for config, reduction in zip(configs, variance_reduction):
    print(f"   {config:10s}: {reduction:6.2f}% more stable")

print("\n3. CATASTROPHIC FAILURE (ε_c=10.0):")
for config in configs:
    config_full = config + ' units'
    mc_high_rows = df[(df['Configuration'] == config_full) & (df['Type'] == 'Antisym (ε=10)')]
    if len(mc_high_rows) > 0 and config_full in baseline_configs:
        mc_high = mc_high_rows.iloc[0]['Test_MC']
        baseline = baseline_configs[config_full]
        loss = ((baseline - mc_high) / baseline) * 100
        print(f"   {config:10s}: {loss:6.2f}% capacity loss (MC: {mc_high:.4f})")

print("\n4. BEST CONFIGURATION:")
best_row = df[df['Test_MC'] == df['Test_MC'].max()].iloc[0]
print(f"   Configuration: {best_row['Configuration']}")
print(f"   Type: {best_row['Type']}")
print(f"   Test MC: {best_row['Test_MC']:.4f} ± {best_row['Test_Std']:.4f}")

print("\n5. BEST WITH ANTISYMMETRIC COUPLING:")
best_antisym = df[df['Type'].str.contains('Antisym \(ε=0.1\)')].sort_values('Test_MC', ascending=False).iloc[0]
print(f"   Configuration: {best_antisym['Configuration']}")
print(f"   Test MC: {best_antisym['Test_MC']:.4f} ± {best_antisym['Test_Std']:.4f}")
if best_antisym['Improvement_%'] is not None:
    print(f"   Improvement: {best_antisym['Improvement_%']:+.2f}%")

print("\n" + "="*100)

# Create a heatmap showing MC across all configurations
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for heatmap
configs_ordered = ['1×100 units', '2×50 units', '4×25 units', '50×2 units', '100×1 unit']
types_ordered = ['Standard', 'Antisym (ε=0.1)', 'Antisym (ε=10)']

heatmap_data = np.zeros((len(configs_ordered), len(types_ordered)))
annotations = []

for i, config in enumerate(configs_ordered):
    row_annotations = []
    for j, type_ in enumerate(types_ordered):
        matches = df[(df['Configuration'] == config) & (df['Type'] == type_)]
        if len(matches) > 0:
            mc = matches.iloc[0]['Test_MC']
            std = matches.iloc[0]['Test_Std']
            heatmap_data[i, j] = mc
            row_annotations.append(f'{mc:.2f}\n±{std:.2f}')
        else:
            heatmap_data[i, j] = np.nan
            row_annotations.append('N/A')
    annotations.append(row_annotations)

# Create heatmap
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=25)

# Set ticks and labels
ax.set_xticks(np.arange(len(types_ordered)))
ax.set_yticks(np.arange(len(configs_ordered)))
ax.set_xticklabels(types_ordered, fontsize=11)
ax.set_yticklabels(configs_ordered, fontsize=11)

# Rotate the tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add annotations
for i in range(len(configs_ordered)):
    for j in range(len(types_ordered)):
        if annotations[i][j] != 'N/A':
            text = ax.text(j, i, annotations[i][j],
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Total Memory Capacity", rotation=-90, va="bottom", fontsize=11)

ax.set_title("Memory Capacity Heatmap - All Configurations", fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results_mc_antisymmetric/mc_heatmap.png', dpi=300, bbox_inches='tight')
print("\nSaved: results_mc_antisymmetric/mc_heatmap.png")
print("="*100)
