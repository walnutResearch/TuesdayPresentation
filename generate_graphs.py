#!/usr/bin/env python3
"""
Generate bar/line graphs for per-image count error, error percent, and count accuracy percent from
per_image_count_accuracy.json. Writes PNGs to the script directory.

How to run:
  python generate_graphs.py

Expects per_image_count_accuracy.json in the same directory as this script (Path(__file__).parent).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the JSON file
json_file = Path(__file__).parent / "per_image_count_accuracy.json"
with open(json_file, 'r') as f:
    data = json.load(f)

per_image = data['per_image']
summary = data['summary']

# Extract data for plotting
images = [entry['image'] for entry in per_image]
count_errors = [entry['count_error'] for entry in per_image]
error_percents = [entry['error_percent'] for entry in per_image]
count_accuracy_percents = [entry['count_accuracy_percent'] for entry in per_image]

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Count Error (signed) - Bar plot
ax1 = plt.subplot(3, 1, 1)
colors = ['red' if e < 0 else 'green' if e > 0 else 'gray' for e in count_errors]
bars = ax1.bar(range(len(images)), count_errors, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Image Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count Error', fontsize=12, fontweight='bold')
ax1.set_title('Count Error per Image (Positive = Over-counting, Negative = Under-counting)', 
               fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(range(0, len(images), 5))
ax1.set_xticklabels([f"{i}" for i in range(0, len(images), 5)], rotation=0)
ax1.legend([plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7),
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7)],
           ['Over-counting', 'Under-counting'], loc='upper right')

# Add summary text
textstr = f"Total Error: {summary['total_count_error']}\nAvg Error: {np.mean(count_errors):.2f}"
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Error Percentage (signed) - Bar plot
ax2 = plt.subplot(3, 1, 2)
colors2 = ['red' if e < 0 else 'green' if e > 0 else 'gray' for e in error_percents]
bars2 = ax2.bar(range(len(images)), error_percents, color=colors2, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Image Index', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Error Percentage per Image (Positive = Over-counting, Negative = Under-counting)', 
               fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(range(0, len(images), 5))
ax2.set_xticklabels([f"{i}" for i in range(0, len(images), 5)], rotation=0)
ax2.legend([plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7),
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7)],
           ['Over-counting', 'Under-counting'], loc='upper right')

# Add summary text
textstr2 = f"Overall Error: {summary['overall_error_percent']:.2f}%\nAvg Error: {np.mean([abs(e) for e in error_percents]):.2f}%"
ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Count Accuracy Percentage - Line plot with bars
ax3 = plt.subplot(3, 1, 3)
bars3 = ax3.bar(range(len(images)), count_accuracy_percents, 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.axhline(y=summary['overall_count_accuracy_percent'], color='red', 
            linestyle='--', linewidth=2, label=f"Overall: {summary['overall_count_accuracy_percent']:.2f}%")
ax3.axhline(y=summary['average_per_image_count_accuracy_percent'], color='orange', 
            linestyle='--', linewidth=2, label=f"Average: {summary['average_per_image_count_accuracy_percent']:.2f}%")
ax3.set_xlabel('Image Index', fontsize=12, fontweight='bold')
ax3.set_ylabel('Count Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Count Accuracy Percentage per Image', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(range(0, len(images), 5))
ax3.set_xticklabels([f"{i}" for i in range(0, len(images), 5)], rotation=0)
ax3.set_ylim([0, 105])
ax3.legend(loc='upper right')

# Add summary text
textstr3 = f"Overall Accuracy: {summary['overall_count_accuracy_percent']:.2f}%\nAvg Accuracy: {summary['average_per_image_count_accuracy_percent']:.2f}%"
ax3.text(0.02, 0.98, textstr3, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()

# Save the figure
output_file = Path(__file__).parent / "count_metrics_graphs.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved graphs to: {output_file}")

# Also create individual graphs
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

# Count Error
axes[0].bar(range(len(images)), count_errors, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0].set_xlabel('Image Index', fontsize=11)
axes[0].set_ylabel('Count Error', fontsize=11, fontweight='bold')
axes[0].set_title('Count Error per Image', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticks(range(0, len(images), 10))
axes[0].set_xticklabels([f"{i}" for i in range(0, len(images), 10)], rotation=0)

# Error Percentage
axes[1].bar(range(len(images)), error_percents, color=colors2, alpha=0.7, edgecolor='black', linewidth=0.5)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Image Index', fontsize=11)
axes[1].set_ylabel('Error Percentage (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Error Percentage per Image', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_xticks(range(0, len(images), 10))
axes[1].set_xticklabels([f"{i}" for i in range(0, len(images), 10)], rotation=0)

# Count Accuracy
axes[2].bar(range(len(images)), count_accuracy_percents, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[2].axhline(y=summary['overall_count_accuracy_percent'], color='red', 
                linestyle='--', linewidth=2, label=f"Overall: {summary['overall_count_accuracy_percent']:.2f}%")
axes[2].set_xlabel('Image Index', fontsize=11)
axes[2].set_ylabel('Count Accuracy (%)', fontsize=11, fontweight='bold')
axes[2].set_title('Count Accuracy per Image', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].set_xticks(range(0, len(images), 10))
axes[2].set_xticklabels([f"{i}" for i in range(0, len(images), 10)], rotation=0)
axes[2].set_ylim([0, 105])
axes[2].legend()

plt.tight_layout()
output_file2 = Path(__file__).parent / "count_metrics_individual.png"
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✅ Saved individual graphs to: {output_file2}")

# Create scatter plots
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: Predicted vs Ground Truth
predicted = [entry['num_preds'] for entry in per_image]
ground_truth = [entry['num_gts'] for entry in per_image]
axes[0].scatter(ground_truth, predicted, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
# Add diagonal line (perfect prediction)
max_val = max(max(predicted), max(ground_truth))
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Ground Truth Count', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Predicted Count', fontsize=11, fontweight='bold')
axes[0].set_title('Predicted vs Ground Truth Counts', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_aspect('equal', adjustable='box')

# Scatter: Error Percentage vs Ground Truth
axes[1].scatter(ground_truth, error_percents, alpha=0.6, s=50, 
               c=['red' if e < 0 else 'green' if e > 0 else 'gray' for e in error_percents],
               edgecolors='black', linewidth=0.5)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Ground Truth Count', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Error Percentage (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Error Percentage vs Ground Truth Count', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend([plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.6),
                plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.6)],
               ['Over-counting', 'Under-counting'], loc='upper right')

plt.tight_layout()
output_file3 = Path(__file__).parent / "count_scatter_plots.png"
plt.savefig(output_file3, dpi=300, bbox_inches='tight')
print(f"✅ Saved scatter plots to: {output_file3}")

print(f"\n📊 Summary Statistics:")
print(f"  Total Images: {summary['total_images']}")
print(f"  Total Predicted: {summary['total_predicted']}")
print(f"  Total Ground Truth: {summary['total_ground_truth']}")
print(f"  Total Count Error: {summary['total_count_error']}")
print(f"  Overall Error Percentage: {summary['overall_error_percent']:.2f}%")
print(f"  Overall Count Accuracy: {summary['overall_count_accuracy_percent']:.2f}%")

plt.close('all')

