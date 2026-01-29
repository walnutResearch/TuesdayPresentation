#!/usr/bin/env python3
"""
Generate a comprehensive single-page visualization of threshold metrics (count accuracy, precision, recall, F1)
for Walnut Research and Glenn datasets from all_threshold_results.json.

How to run:
  python generate_comprehensive_threshold_graph.py

Expects all_threshold_results.json in project root. Writes PNG to project root (path set inside script).
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

# Extract data
thresholds_w = [r['threshold'] for r in walnut_best]
count_acc_w = [r['count_accuracy'] if r['count_accuracy'] else 0 for r in walnut_best]
precision_w = [r['precision'] for r in walnut_best]
recall_w = [r['recall'] for r in walnut_best]
f1_w = [r['f1'] for r in walnut_best]
predicted_w = [r['total_predicted'] for r in walnut_best]

thresholds_g = [r['threshold'] for r in glenn_best]
count_acc_g = [r['count_accuracy'] if r['count_accuracy'] else 0 for r in glenn_best]
precision_g = [r['precision'] for r in glenn_best]
recall_g = [r['recall'] for r in glenn_best]
f1_g = [r['f1'] for r in glenn_best]
predicted_g = [r['total_predicted'] for r in glenn_best]

# Create comprehensive single-page figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('Comprehensive Threshold Analysis: Precision & Count Accuracy', 
             fontsize=18, fontweight='bold', y=0.98)

# Row 1: Walnut Research Dataset
# 1. Count Accuracy
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(thresholds_w, count_acc_w, 'o-', color='#2E86AB', linewidth=3, markersize=12, label='Count Accuracy')
ax1.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
ax1.axhline(y=90, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
best_idx_w = count_acc_w.index(max(count_acc_w))
ax1.plot(thresholds_w[best_idx_w], count_acc_w[best_idx_w], 'o', color='red', markersize=18, 
         markeredgecolor='black', markeredgewidth=2, zorder=5)
ax1.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax1.set_ylabel('Count Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Walnut Research: Count Accuracy\n(GT: 487, Best: 99.18% @ 0.5)', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 105])

# 2. Precision
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(thresholds_w, precision_w, 's-', color='#A23B72', linewidth=3, markersize=12, label='Precision')
ax2.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax2.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
ax2.set_title('Walnut Research: Precision', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

# 3. Precision vs Count Accuracy
ax3 = fig.add_subplot(gs[0, 2])
scatter = ax3.scatter(precision_w, count_acc_w, c=thresholds_w, cmap='viridis', 
                     s=200, edgecolors='black', linewidth=2, alpha=0.8, zorder=3)
for i, t in enumerate(thresholds_w):
    ax3.annotate(f'{t:.2f}', (precision_w[i], count_acc_w[i]), 
                fontsize=10, ha='center', va='bottom', fontweight='bold', zorder=4)
ax3.plot(precision_w[best_idx_w], count_acc_w[best_idx_w], 'o', color='red', markersize=25, 
        markeredgecolor='black', markeredgewidth=3, zorder=5)
ax3.set_xlabel('Precision (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Count Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('Walnut Research: Trade-off Analysis', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Threshold', shrink=0.8)

# Row 2: Glenn Dataset
# 4. Count Accuracy
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(thresholds_g, count_acc_g, 'o-', color='#2E86AB', linewidth=3, markersize=12, label='Count Accuracy')
ax4.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
ax4.axhline(y=90, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
best_idx_g = count_acc_g.index(max(count_acc_g))
ax4.plot(thresholds_g[best_idx_g], count_acc_g[best_idx_g], 'o', color='red', markersize=18, 
         markeredgecolor='black', markeredgewidth=2, zorder=5)
ax4.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax4.set_ylabel('Count Accuracy (%)', fontsize=11, fontweight='bold')
ax4.set_title('Glenn Dataset: Count Accuracy\n(GT: 1782, Best: 98.93% @ 0.42)', 
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 105])

# 5. Precision
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(thresholds_g, precision_g, 's-', color='#A23B72', linewidth=3, markersize=12, label='Precision')
ax5.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax5.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
ax5.set_title('Glenn Dataset: Precision', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 100])

# 6. Precision vs Count Accuracy
ax6 = fig.add_subplot(gs[1, 2])
scatter2 = ax6.scatter(precision_g, count_acc_g, c=thresholds_g, cmap='viridis', 
                      s=200, edgecolors='black', linewidth=2, alpha=0.8, zorder=3)
for i, t in enumerate(thresholds_g):
    ax6.annotate(f'{t:.2f}', (precision_g[i], count_acc_g[i]), 
                fontsize=10, ha='center', va='bottom', fontweight='bold', zorder=4)
ax6.plot(precision_g[best_idx_g], count_acc_g[best_idx_g], 'o', color='red', markersize=25, 
        markeredgecolor='black', markeredgewidth=3, zorder=5)
ax6.set_xlabel('Precision (%)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Count Accuracy (%)', fontsize=11, fontweight='bold')
ax6.set_title('Glenn Dataset: Trade-off Analysis', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax6, label='Threshold', shrink=0.8)

# Row 3: Combined Metrics
# 7. All Metrics - Walnut Research
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(thresholds_w, count_acc_w, 'o-', label='Count Acc', linewidth=2.5, markersize=10, color='green')
ax7.plot(thresholds_w, precision_w, 's-', label='Precision', linewidth=2.5, markersize=10, color='blue')
ax7.plot(thresholds_w, recall_w, '^-', label='Recall', linewidth=2.5, markersize=10, color='orange')
ax7.plot(thresholds_w, f1_w, 'd-', label='F1', linewidth=2.5, markersize=10, color='purple')
ax7.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax7.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax7.set_title('Walnut Research: All Metrics', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9, loc='best')
ax7.grid(True, alpha=0.3)
ax7.set_ylim([0, 105])

# 8. All Metrics - Glenn
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(thresholds_g, count_acc_g, 'o-', label='Count Acc', linewidth=2.5, markersize=10, color='green')
ax8.plot(thresholds_g, precision_g, 's-', label='Precision', linewidth=2.5, markersize=10, color='blue')
ax8.plot(thresholds_g, recall_g, '^-', label='Recall', linewidth=2.5, markersize=10, color='orange')
ax8.plot(thresholds_g, f1_g, 'd-', label='F1', linewidth=2.5, markersize=10, color='purple')
ax8.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax8.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax8.set_title('Glenn Dataset: All Metrics', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9, loc='best')
ax8.grid(True, alpha=0.3)
ax8.set_ylim([0, 105])

# 9. Comparison - Count Accuracy
ax9 = fig.add_subplot(gs[2, 2])
ax9.plot(thresholds_w, count_acc_w, 'o-', color='#2E86AB', linewidth=3, markersize=12, 
         label='Walnut Research', alpha=0.8)
ax9.plot(thresholds_g, count_acc_g, 's-', color='#06A77D', linewidth=3, markersize=12, 
         label='Glenn', alpha=0.8)
ax9.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
ax9.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax9.set_ylabel('Count Accuracy (%)', fontsize=11, fontweight='bold')
ax9.set_title('Count Accuracy: Both Datasets', fontsize=12, fontweight='bold')
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)
ax9.set_ylim([0, 105])

plt.savefig("/Users/kalpit/TuesdayPresentation/comprehensive_threshold_analysis.png", 
            dpi=300, bbox_inches='tight')
print("✅ Saved comprehensive analysis to: comprehensive_threshold_analysis.png")
plt.close()




