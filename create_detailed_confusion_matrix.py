#!/usr/bin/env python3
"""
Create a detailed confusion matrix from detector results using finer count bins (e.g. 0, 1-5, 6-15, ...).
Reads per_image_results (pred_count, true_count) from a results JSON and saves a confusion matrix plot.

How to run:
  python create_detailed_confusion_matrix.py

Results file and output_dir are set inside create_detailed_confusion_matrix(); edit as needed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def create_detailed_confusion_matrix(results_file, output_dir):
    """Create detailed confusion matrix with individual patch analysis"""
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("Creating detailed confusion matrix...")
    print("This will analyze the count-based performance in more detail")
    
    # Extract per-image results
    per_image_results = data['per_image_results']
    
    # Create more granular count ranges
    def get_detailed_category(count):
        if count == 0:
            return "0"
        elif count <= 5:
            return "1-5"
        elif count <= 15:
            return "6-15"
        elif count <= 30:
            return "16-30"
        elif count <= 50:
            return "31-50"
        elif count <= 80:
            return "51-80"
        elif count <= 120:
            return "81-120"
        else:
            return "120+"
    
    # Get detailed categories
    true_categories = [get_detailed_category(r['true_count']) for r in per_image_results]
    pred_categories = [get_detailed_category(r['pred_count']) for r in per_image_results]
    
    # Create detailed confusion matrix
    detailed_categories = ["0", "1-5", "6-15", "16-30", "31-50", "51-80", "81-120", "120+"]
    cm_detailed = confusion_matrix(true_categories, pred_categories, labels=detailed_categories)
    
    # Create visualization
    plt.figure(figsize=(14, 12))
    
    # Plot detailed confusion matrix
    sns.heatmap(cm_detailed, annot=True, fmt='d', cmap='Blues', 
                xticklabels=detailed_categories, yticklabels=detailed_categories,
                cbar_kws={'label': 'Number of Images'})
    
    plt.title('Detailed Binary Classifier Confusion Matrix\n(Patch Size: 32x32, Stride: 16, Threshold: 0.1)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Count Range', fontsize=12)
    plt.ylabel('True Count Range', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy = np.trace(cm_detailed) / np.sum(cm_detailed)
    plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.3f}', 
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create count error analysis
    plt.figure(figsize=(15, 12))
    
    # Extract counts
    true_counts = [r['true_count'] for r in per_image_results]
    pred_counts = [r['pred_count'] for r in per_image_results]
    errors = [abs(p - t) for p, t in zip(pred_counts, true_counts)]
    relative_errors = [abs(p - t) / max(t, 1) * 100 for p, t in zip(pred_counts, true_counts)]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot with color coding by error
    scatter = ax1.scatter(true_counts, pred_counts, c=errors, cmap='Reds', alpha=0.7, s=60)
    min_val = min(min(true_counts), min(pred_counts))
    max_val = max(max(true_counts), max(pred_counts))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('True Count')
    ax1.set_ylabel('Predicted Count')
    ax1.set_title('Predicted vs True Counts\n(Color = Absolute Error)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Absolute Error')
    
    # 2. Error vs True Count
    ax2.scatter(true_counts, errors, alpha=0.7, s=50)
    ax2.set_xlabel('True Count')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error vs True Count')
    ax2.grid(True, alpha=0.3)
    
    # 3. Relative Error vs True Count
    ax3.scatter(true_counts, relative_errors, alpha=0.7, s=50)
    ax3.set_xlabel('True Count')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Relative Error vs True Count')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(errors), color='r', linestyle='--', linewidth=2, 
                label=f'Mean Error: {np.mean(errors):.1f}')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Binary Classifier Error Analysis (32x32 patches, stride 16, threshold 0.1)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create performance summary
    plt.figure(figsize=(12, 8))
    
    # Calculate metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    mape = np.mean(relative_errors)
    
    # Create metrics table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Mean Absolute Error (MAE)', f'{mae:.2f}'],
        ['Root Mean Square Error (RMSE)', f'{rmse:.2f}'],
        ['Mean Absolute Percentage Error (MAPE)', f'{mape:.1f}%'],
        ['Overall Accuracy', f'{accuracy:.3f}'],
        ['True Count Mean', f'{np.mean(true_counts):.1f}'],
        ['Predicted Count Mean', f'{np.mean(pred_counts):.1f}'],
        ['True Count Std', f'{np.std(true_counts):.1f}'],
        ['Predicted Count Std', f'{np.std(pred_counts):.1f}'],
    ]
    
    table = ax.table(cellText=metrics_data, colLabels=None, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(metrics_data)):
        if i == 0:  # Header row
            table[(i, 0)].set_facecolor('#40466e')
            table[(i, 1)].set_facecolor('#40466e')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#ffffff')
    
    plt.title('Binary Classifier Performance Summary\n(Patch Size: 32x32, Stride: 16, Threshold: 0.1)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed confusion matrix and analysis saved to {output_dir}")
    
    # Print detailed summary
    print("\n" + "="*70)
    print("DETAILED BINARY CLASSIFIER CONFUSION MATRIX")
    print("="*70)
    print(f"Patch Size: 32x32, Stride: 16, Threshold: 0.1")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")
    print("\nDetailed Confusion Matrix:")
    print("Rows = True Count Ranges, Columns = Predicted Count Ranges")
    print()
    
    # Print detailed confusion matrix
    print(" " * 12, end="")
    for cat in detailed_categories:
        print(f"{cat:>8}", end="")
    print()
    
    for i, true_cat in enumerate(detailed_categories):
        print(f"{true_cat:>12}", end="")
        for j, pred_cat in enumerate(detailed_categories):
            print(f"{cm_detailed[i,j]:>8}", end="")
        print()
    
    # Print best and worst predictions
    print(f"\nBest Predictions (Error ≤ 5):")
    for i, result in enumerate(per_image_results):
        error = abs(result['pred_count'] - result['true_count'])
        if error <= 5:
            print(f"  Image {i+1}: True={result['true_count']}, Pred={result['pred_count']}, Error={error}")
    
    print(f"\nWorst Predictions (Error > 50):")
    for i, result in enumerate(per_image_results):
        error = abs(result['pred_count'] - result['true_count'])
        if error > 50:
            print(f"  Image {i+1}: True={result['true_count']}, Pred={result['pred_count']}, Error={error}")

def main():
    results_file = "binary_test_32x16_0.1/binary_test_results.json"
    output_dir = "binary_test_32x16_0.1/confusion_matrix"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    create_detailed_confusion_matrix(results_file, output_dir)

if __name__ == "__main__":
    main()

