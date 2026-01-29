#!/usr/bin/env python3
"""
Generate separate graphs for threshold vs precision and threshold vs count accuracy for Walnut Research and Glenn
datasets using all_threshold_results.json.

How to run:
  python generate_threshold_graphs.py

Expects all_threshold_results.json in project root. Writes PNGs to paths defined inside the script.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Load results
results_file = Path("/Users/kalpit/TuesdayPresentation/all_threshold_results.json")
with open(results_file, 'r') as f:
    data = json.load(f)

# Organize by dataset
walnut_results = [r for r in data['all_results'] if r['dataset'] == 'walnut_research']
glenn_results = [r for r in data['all_results'] if r['dataset'] == 'glenn']

# Group by threshold (take best count accuracy for each threshold)
def get_best_by_threshold(results):
    """Get the result with best count accuracy for each threshold"""
    by_threshold = defaultdict(list)
    for r in results:
        if r['threshold'] is not None:
            by_threshold[r['threshold']].append(r)
    
    best_results = []
    for threshold, results_list in by_threshold.items():
        # Get result with best count accuracy
        best = max(results_list, key=lambda x: x['count_accuracy'] if x['count_accuracy'] else -999)
        best_results.append(best)
    
    return sorted(best_results, key=lambda x: x['threshold'])

walnut_best = get_best_by_threshold(walnut_results)
glenn_best = get_best_by_threshold(glenn_results)

# Create comprehensive figure for Walnut Research
fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('Walnut Research Dataset: Threshold Analysis (Ground Truth: 487 walnuts)', 
              fontsize=16, fontweight='bold')

# Extract data
thresholds_w = [r['threshold'] for r in walnut_best]
count_acc_w = [r['count_accuracy'] if r['count_accuracy'] else 0 for r in walnut_best]
precision_w = [r['precision'] for r in walnut_best]
recall_w = [r['recall'] for r in walnut_best]
f1_w = [r['f1'] for r in walnut_best]
predicted_w = [r['total_predicted'] for r in walnut_best]

# 1. Count Accuracy vs Threshold
ax1 = axes[0, 0]
ax1.plot(thresholds_w, count_acc_w, 'o-', color='green', linewidth=2.5, markersize=10, label='Count Accuracy')
ax1.axhline(y=95, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='95% Threshold')
ax1.axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.7, label='90% Threshold')
ax1.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Count Accuracy vs Threshold', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 105])
# Highlight best
best_idx = count_acc_w.index(max(count_acc_w))
ax1.plot(thresholds_w[best_idx], count_acc_w[best_idx], 'o', color='red', markersize=15, 
         markeredgecolor='black', markeredgewidth=2, label='Best (99.18%)')
ax1.legend(fontsize=10)

# 2. Precision vs Threshold
ax2 = axes[0, 1]
ax2.plot(thresholds_w, precision_w, 's-', color='blue', linewidth=2.5, markersize=10, label='Precision')
ax2.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax2.set_title('Precision vs Threshold', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

# 3. Precision, Recall, F1 vs Threshold
ax3 = axes[1, 0]
ax3.plot(thresholds_w, precision_w, 'o-', label='Precision', linewidth=2, markersize=8)
ax3.plot(thresholds_w, recall_w, 's-', label='Recall', linewidth=2, markersize=8)
ax3.plot(thresholds_w, f1_w, '^-', label='F1 Score', linewidth=2, markersize=8)
ax3.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax3.set_title('Precision, Recall, and F1 Score vs Threshold', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 100])

# 4. Precision vs Count Accuracy (Trade-off)
ax4 = axes[1, 1]
scatter = ax4.scatter(precision_w, count_acc_w, c=thresholds_w, cmap='viridis', 
                     s=150, edgecolors='black', linewidth=1.5, alpha=0.7)
for i, t in enumerate(thresholds_w):
    ax4.annotate(f'{t:.2f}', (precision_w[i], count_acc_w[i]), 
                fontsize=9, ha='center', va='bottom', fontweight='bold')
ax4.set_xlabel('Precision (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Count Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Precision vs Count Accuracy Trade-off', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Threshold')
# Highlight best
ax4.plot(precision_w[best_idx], count_acc_w[best_idx], 'o', color='red', markersize=20, 
        markeredgecolor='black', markeredgewidth=2)

plt.tight_layout()
output1 = Path("/Users/kalpit/TuesdayPresentation/threshold_analysis_walnut_research.png")
fig1.savefig(output1, dpi=300, bbox_inches='tight')
print(f"✅ Saved Walnut Research graphs to: {output1}")
plt.close(fig1)

# Create comprehensive figure for Glenn
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Glenn Dataset: Threshold Analysis (Ground Truth: 1782 walnuts)', 
              fontsize=16, fontweight='bold')

# Extract data
thresholds_g = [r['threshold'] for r in glenn_best]
count_acc_g = [r['count_accuracy'] if r['count_accuracy'] else 0 for r in glenn_best]
precision_g = [r['precision'] for r in glenn_best]
recall_g = [r['recall'] for r in glenn_best]
f1_g = [r['f1'] for r in glenn_best]
predicted_g = [r['total_predicted'] for r in glenn_best]

# 1. Count Accuracy vs Threshold
ax1 = axes[0, 0]
ax1.plot(thresholds_g, count_acc_g, 'o-', color='green', linewidth=2.5, markersize=10, label='Count Accuracy')
ax1.axhline(y=95, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='95% Threshold')
ax1.axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.7, label='90% Threshold')
ax1.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Count Accuracy vs Threshold', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 105])
# Highlight best
best_idx_g = count_acc_g.index(max(count_acc_g))
ax1.plot(thresholds_g[best_idx_g], count_acc_g[best_idx_g], 'o', color='red', markersize=15, 
         markeredgecolor='black', markeredgewidth=2, label='Best (98.93%)')
ax1.legend(fontsize=10)

# 2. Precision vs Threshold
ax2 = axes[0, 1]
ax2.plot(thresholds_g, precision_g, 's-', color='blue', linewidth=2.5, markersize=10, label='Precision')
ax2.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax2.set_title('Precision vs Threshold', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

# 3. Precision, Recall, F1 vs Threshold
ax3 = axes[1, 0]
ax3.plot(thresholds_g, precision_g, 'o-', label='Precision', linewidth=2, markersize=8)
ax3.plot(thresholds_g, recall_g, 's-', label='Recall', linewidth=2, markersize=8)
ax3.plot(thresholds_g, f1_g, '^-', label='F1 Score', linewidth=2, markersize=8)
ax3.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax3.set_title('Precision, Recall, and F1 Score vs Threshold', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 100])

# 4. Precision vs Count Accuracy (Trade-off)
ax4 = axes[1, 1]
scatter = ax4.scatter(precision_g, count_acc_g, c=thresholds_g, cmap='viridis', 
                     s=150, edgecolors='black', linewidth=1.5, alpha=0.7)
for i, t in enumerate(thresholds_g):
    ax4.annotate(f'{t:.2f}', (precision_g[i], count_acc_g[i]), 
                fontsize=9, ha='center', va='bottom', fontweight='bold')
ax4.set_xlabel('Precision (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Count Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Precision vs Count Accuracy Trade-off', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Threshold')
# Highlight best
ax4.plot(precision_g[best_idx_g], count_acc_g[best_idx_g], 'o', color='red', markersize=20, 
        markeredgecolor='black', markeredgewidth=2)

plt.tight_layout()
output2 = Path("/Users/kalpit/TuesdayPresentation/threshold_analysis_glenn.png")
fig2.savefig(output2, dpi=300, bbox_inches='tight')
print(f"✅ Saved Glenn dataset graphs to: {output2}")
plt.close(fig2)

# Create comparison figure showing both datasets
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle('Threshold Comparison: Count Accuracy and Precision', fontsize=16, fontweight='bold')

# Count Accuracy comparison
ax1 = axes[0]
ax1.plot(thresholds_w, count_acc_w, 'o-', color='blue', linewidth=2.5, markersize=10, 
         label='Walnut Research (GT: 487)', alpha=0.8)
ax1.plot(thresholds_g, count_acc_g, 's-', color='green', linewidth=2.5, markersize=10, 
         label='Glenn (GT: 1782)', alpha=0.8)
ax1.axhline(y=95, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='95% Threshold')
ax1.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Count Accuracy Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 105])

# Precision comparison
ax2 = axes[1]
ax2.plot(thresholds_w, precision_w, 'o-', color='blue', linewidth=2.5, markersize=10, 
         label='Walnut Research', alpha=0.8)
ax2.plot(thresholds_g, precision_g, 's-', color='green', linewidth=2.5, markersize=10, 
         label='Glenn', alpha=0.8)
ax2.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax2.set_title('Precision Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

plt.tight_layout()
output3 = Path("/Users/kalpit/TuesdayPresentation/threshold_comparison_both_datasets.png")
fig3.savefig(output3, dpi=300, bbox_inches='tight')
print(f"✅ Saved comparison graphs to: {output3}")
plt.close(fig3)

print("\n📊 Summary:")
print(f"  Walnut Research: {len(walnut_best)} threshold configurations")
print(f"  Glenn Dataset: {len(glenn_best)} threshold configurations")
print(f"\n✅ All graphs saved successfully!")
