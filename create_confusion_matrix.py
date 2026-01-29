#!/usr/bin/env python3
"""
Create a confusion matrix and classification metrics from binary classifier (detector) results. Uses per-image
predicted vs true counts binned into count ranges (e.g. Low/Medium/High). Saves confusion matrix plot and metrics.

How to run:
  python create_confusion_matrix.py

Results file and output_dir are set inside create_confusion_matrix_from_results(); typically reads a JSON with per_image_results (pred_count, true_count).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

def create_confusion_matrix_from_results(results_file, output_dir):
    """Create confusion matrix from binary classifier results"""
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract per-image results
    per_image_results = data['per_image_results']
    
    # For binary classifier, we need to think about this differently
    # Each image has predicted count vs true count
    # We can create a confusion matrix based on count ranges or individual detections
    
    print("Creating confusion matrix for binary classifier...")
    print("Note: This will be based on count ranges since we don't have individual patch classifications")
    
    # Define count ranges for classification
    def get_count_category(count):
        if count == 0:
            return "None"
        elif count <= 10:
            return "Low (1-10)"
        elif count <= 30:
            return "Medium (11-30)"
        elif count <= 60:
            return "High (31-60)"
        else:
            return "Very High (60+)"
    
    # Get categories for true and predicted counts
    true_categories = [get_count_category(r['true_count']) for r in per_image_results]
    pred_categories = [get_count_category(r['pred_count']) for r in per_image_results]
    
    # Create confusion matrix
    categories = ["None", "Low (1-10)", "Medium (11-30)", "High (31-60)", "Very High (60+)"]
    cm = confusion_matrix(true_categories, pred_categories, labels=categories)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories,
                cbar_kws={'label': 'Number of Images'})
    
    plt.title('Binary Classifier Confusion Matrix\n(Patch Size: 32x32, Stride: 16, Threshold: 0.1)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Count Category', fontsize=12)
    plt.ylabel('True Count Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.3f}', 
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed metrics table
    plt.figure(figsize=(14, 8))
    
    # Calculate per-category metrics
    category_metrics = []
    for i, category in enumerate(categories):
        true_positives = cm[i, i]
        false_positives = np.sum(cm[:, i]) - true_positives
        false_negatives = np.sum(cm[i, :]) - true_positives
        true_negatives = np.sum(cm) - true_positives - false_positives - false_negatives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        category_metrics.append({
            'Category': category,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': np.sum(cm[i, :])
        })
    
    # Create table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Category', 'Precision', 'Recall', 'F1-Score', 'Support']
    
    for metric in category_metrics:
        table_data.append([
            metric['Category'],
            f"{metric['Precision']:.3f}",
            f"{metric['Recall']:.3f}",
            f"{metric['F1-Score']:.3f}",
            f"{int(metric['Support'])}"
        ])
    
    # Add overall metrics
    overall_accuracy = np.trace(cm) / np.sum(cm)
    table_data.append(['', '', '', '', ''])
    table_data.append(['Overall Accuracy', f"{overall_accuracy:.3f}", '', '', ''])
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Binary Classifier Performance Metrics\n(Patch Size: 32x32, Stride: 16, Threshold: 0.1)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create count comparison plot
    plt.figure(figsize=(15, 10))
    
    # Extract counts
    true_counts = [r['true_count'] for r in per_image_results]
    pred_counts = [r['pred_count'] for r in per_image_results]
    image_names = [f"Image {i+1}" for i in range(len(per_image_results))]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Scatter plot
    ax1.scatter(true_counts, pred_counts, alpha=0.7, s=50)
    min_val = min(min(true_counts), min(pred_counts))
    max_val = max(max(true_counts), max(pred_counts))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('True Count')
    ax1.set_ylabel('Predicted Count')
    ax1.set_title('Predicted vs True Counts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    residuals = np.array(pred_counts) - np.array(true_counts)
    ax2.scatter(true_counts, residuals, alpha=0.7, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('True Count')
    ax2.set_ylabel('Residuals (Predicted - True)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-image comparison
    x_pos = np.arange(len(image_names))
    width = 0.35
    ax3.bar(x_pos - width/2, true_counts, width, label='True Count', alpha=0.8)
    ax3.bar(x_pos + width/2, pred_counts, width, label='Predicted Count', alpha=0.8)
    ax3.set_xlabel('Image')
    ax3.set_ylabel('Count')
    ax3.set_title('Per-Image Count Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_pos[::5])  # Show every 5th label
    ax3.set_xticklabels([f"Img {i+1}" for i in range(0, len(image_names), 5)])
    
    # 4. Error distribution
    ax4.hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel('Residuals (Predicted - True)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Binary Classifier Analysis (32x32 patches, stride 16, threshold 0.1)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix and analysis plots saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("BINARY CLASSIFIER CONFUSION MATRIX SUMMARY")
    print("="*60)
    print(f"Patch Size: 32x32, Stride: 16, Threshold: 0.1")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print("\nConfusion Matrix:")
    print("Rows = True Categories, Columns = Predicted Categories")
    print()
    
    # Print confusion matrix
    print(" " * 15, end="")
    for cat in categories:
        print(f"{cat:>12}", end="")
    print()
    
    for i, true_cat in enumerate(categories):
        print(f"{true_cat:>15}", end="")
        for j, pred_cat in enumerate(categories):
            print(f"{cm[i,j]:>12}", end="")
        print()
    
    print("\nPer-Category Metrics:")
    for metric in category_metrics:
        print(f"{metric['Category']:>15}: Precision={metric['Precision']:.3f}, "
              f"Recall={metric['Recall']:.3f}, F1={metric['F1-Score']:.3f}")

def main():
    results_file = "binary_test_32x16_0.1/binary_test_results.json"
    output_dir = "binary_test_32x16_0.1/confusion_matrix"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    create_confusion_matrix_from_results(results_file, output_dir)

if __name__ == "__main__":
    main()

