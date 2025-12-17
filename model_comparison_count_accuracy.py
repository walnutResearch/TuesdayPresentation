#!/usr/bin/env python3
"""Compare models based on total count accuracy and precision"""

import json
from pathlib import Path

# Ground truth total
total_gt = 487

# Load results for each threshold
thresholds = [0.5, 0.6, 0.7, 0.8]
results = {}

for thresh in thresholds:
    if thresh == 0.5:
        summary_path = Path("detector_evaluation_results_new/summary.json")
        per_image_path = Path("detector_evaluation_results_new/per_image_metrics.json")
    else:
        thresh_str = f"{int(thresh*10):02d}"  # Format as 06, 07, 08
        summary_path = Path(f"detector_evaluation_results_new_th{thresh_str}/summary.json")
        per_image_path = Path(f"detector_evaluation_results_new_th{thresh_str}/per_image_metrics.json")
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    with open(per_image_path) as f:
        per_image = json.load(f)
    
    # Calculate total predicted count
    total_predicted = sum(item['num_preds'] for item in per_image)
    
    # Count accuracy (how close predicted is to ground truth)
    count_error = abs(total_predicted - total_gt)
    count_accuracy = (1 - count_error / total_gt) * 100 if total_gt > 0 else 0
    
    results[thresh] = {
        'total_predicted': total_predicted,
        'total_gt': total_gt,
        'count_error': count_error,
        'count_accuracy': count_accuracy,
        'precision': summary['precision'] * 100,
        'recall': summary['recall'] * 100,
        'f1': summary['f1'] * 100,
        'tp': summary['TP'],
        'fp': summary['FP'],
        'fn': summary['FN']
    }

# Print comparison
print("=" * 80)
print("MODEL COMPARISON: Total Count Accuracy & Precision")
print("=" * 80)
print(f"\nGround Truth Total: {total_gt} walnuts\n")
print(f"{'Threshold':<12} {'Predicted':<12} {'Count Error':<15} {'Count Acc %':<15} {'Precision %':<15} {'Recall %':<15} {'F1 %':<15}")
print("-" * 80)

best_count_acc = -1
best_precision = -1
best_combined = -1
best_thresh_count = None
best_thresh_precision = None
best_thresh_combined = None

for thresh in sorted(thresholds):
    r = results[thresh]
    print(f"{thresh:<12} {r['total_predicted']:<12} {r['count_error']:<15} {r['count_accuracy']:<15.2f} {r['precision']:<15.2f} {r['recall']:<15.2f} {r['f1']:<15.2f}")
    
    # Track best for count accuracy
    if r['count_accuracy'] > best_count_acc:
        best_count_acc = r['count_accuracy']
        best_thresh_count = thresh
    
    # Track best for precision
    if r['precision'] > best_precision:
        best_precision = r['precision']
        best_thresh_precision = thresh
    
    # Combined score: balance count accuracy and precision
    combined_score = (r['count_accuracy'] * 0.4) + (r['precision'] * 0.6)
    if combined_score > best_combined:
        best_combined = combined_score
        best_thresh_combined = thresh

print("\n" + "=" * 80)
print("BEST PERFORMANCE:")
print("=" * 80)
print(f"🏆 Best Count Accuracy: Threshold {best_thresh_count} ({results[best_thresh_count]['count_accuracy']:.2f}%)")
print(f"   - Predicted: {results[best_thresh_count]['total_predicted']} vs GT: {total_gt}")
print(f"   - Count Error: {results[best_thresh_count]['count_error']}")
print(f"   - Precision: {results[best_thresh_count]['precision']:.2f}%")

print(f"\n🎯 Best Precision: Threshold {best_thresh_precision} ({results[best_thresh_precision]['precision']:.2f}%)")
print(f"   - Predicted: {results[best_thresh_precision]['total_predicted']} vs GT: {total_gt}")
print(f"   - Count Error: {results[best_thresh_precision]['count_error']}")
print(f"   - Count Accuracy: {results[best_thresh_precision]['count_accuracy']:.2f}%")

print(f"\n⭐ Best Combined (Count Acc 40% + Precision 60%): Threshold {best_thresh_combined}")
print(f"   - Count Accuracy: {results[best_thresh_combined]['count_accuracy']:.2f}%")
print(f"   - Precision: {results[best_thresh_combined]['precision']:.2f}%")
print(f"   - Combined Score: {best_combined:.2f}")
print(f"   - Predicted: {results[best_thresh_combined]['total_predicted']} vs GT: {total_gt}")
print(f"   - Count Error: {results[best_thresh_combined]['count_error']}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
if best_thresh_combined == best_thresh_precision:
    print(f"✅ Use Threshold {best_thresh_combined} - Best balance of count accuracy and precision!")
else:
    print(f"✅ For best count accuracy: Use Threshold {best_thresh_count}")
    print(f"✅ For best precision: Use Threshold {best_thresh_precision}")
    print(f"✅ For balanced performance: Use Threshold {best_thresh_combined}")

