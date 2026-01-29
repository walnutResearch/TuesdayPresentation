#!/usr/bin/env python3
"""
Calculate count accuracy (per-image and total) from per_image_metrics.json files. Can scan multiple metric files
(e.g. for different thresholds) and print or compare count accuracy. Used for threshold/dataset comparison.

How to run:
  python calculate_count_accuracy.py

Or import: from calculate_count_accuracy import calculate_count_accuracy, calculate_count_accuracy_total
File paths are typically passed as arguments or set in a main block.
"""

import json
import os
import glob

def calculate_count_accuracy(metrics_file):
    """Calculate count accuracy from per_image_metrics.json"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    total_error = 0
    total_gt = 0
    num_images = 0
    
    for image_metrics in metrics:
        gt_count = image_metrics.get('num_gts', 0)
        pred_count = image_metrics.get('num_preds', 0)
        
        if gt_count > 0:
            error = abs(pred_count - gt_count) / gt_count
            total_error += error
            total_gt += gt_count
            num_images += 1
    
    if num_images == 0:
        return 0.0
    
    # Count accuracy = 1 - average relative error
    avg_error = total_error / num_images
    count_accuracy = 1 - avg_error
    
    return count_accuracy

def calculate_count_accuracy_total(metrics_file):
    """Calculate count accuracy based on total counts"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    total_gt = 0
    total_pred = 0
    
    for image_metrics in metrics:
        gt_count = image_metrics.get('num_gts', 0)
        pred_count = image_metrics.get('num_preds', 0)
        total_gt += gt_count
        total_pred += pred_count
    
    if total_gt == 0:
        return 0.0
    
    # Count accuracy = 1 - |total_pred - total_gt| / total_gt
    error = abs(total_pred - total_gt) / total_gt
    count_accuracy = 1 - error
    
    return count_accuracy

def main():
    # Find all evaluation directories
    eval_dirs = glob.glob('glenn_eval_th*')
    
    results = []
    
    for eval_dir in sorted(eval_dirs):
        metrics_file = os.path.join(eval_dir, 'per_image_metrics.json')
        
        if os.path.exists(metrics_file):
            # Extract threshold from directory name (handle both th0.40 and th040 formats)
            threshold_str = eval_dir.replace('glenn_eval_th', '')
            # If it's like "040", convert to "0.40"
            if len(threshold_str) == 3 and threshold_str.isdigit():
                threshold = float(threshold_str[0] + '.' + threshold_str[1:])
            else:
                try:
                    threshold = float(threshold_str)
                except ValueError:
                    continue
            
            # Calculate both per-image and total count accuracy
            count_acc_per_image = calculate_count_accuracy(metrics_file)
            count_acc_total = calculate_count_accuracy_total(metrics_file)
            
            results.append((threshold, count_acc_per_image, count_acc_total))
            print(f"Threshold {threshold:.2f}: Per-Image Count Acc = {count_acc_per_image:.4f}, Total Count Acc = {count_acc_total:.4f}")
    
    # Sort by total count accuracy (which seems to match user's definition)
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("\n" + "="*60)
    print("Best Thresholds by Total Count Accuracy:")
    print("="*60)
    for threshold, per_img, total in results[:10]:
        print(f"Threshold {threshold:.2f}: Total Count Accuracy = {total:.4f} (Per-Image: {per_img:.4f})")

if __name__ == '__main__':
    main()

