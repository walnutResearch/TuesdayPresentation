#!/usr/bin/env python3
"""
Generate per-image count accuracy JSON from per_image_metrics.json: for each image compute count error,
error percent, count_accuracy_percent, and keep TP/FP/FN. Also computes overall summary stats.

How to run:
  python generate_count_accuracy.py

Reads per_image_metrics.json from the same directory as this script (Path(__file__).parent). Writes per_image_count_accuracy.json and summary to the same directory (paths set in script).
"""

import json
from pathlib import Path

# Read the per_image_metrics.json file
metrics_file = Path(__file__).parent / "per_image_metrics.json"
with open(metrics_file, 'r') as f:
    per_image_metrics = json.load(f)

# Calculate count accuracy for each image
count_accuracy_results = []

for metric in per_image_metrics:
    num_preds = metric['num_preds']
    num_gts = metric['num_gts']
    
    # Calculate count error and accuracy
    # Signed error: positive = over-counting, negative = under-counting
    if num_gts > 0:
        count_error = num_preds - num_gts  # Signed: + if more predicted, - if less predicted
        count_error_abs = abs(count_error)
        count_accuracy = (1 - count_error_abs / num_gts) * 100
        # Signed error percent: positive = over-counting, negative = under-counting
        relative_error = (count_error / num_gts) * 100  # Signed percentage
    else:
        count_error = num_preds  # If no ground truth, error is just the predictions
        count_error_abs = abs(count_error)
        count_accuracy = 0.0 if num_preds > 0 else 100.0
        relative_error = 100.0 if num_preds > 0 else 0.0
    
    count_accuracy_results.append({
        "image": metric['image'],
        "num_preds": num_preds,
        "num_gts": num_gts,
        "count_error": count_error,
        "error_percent": round(relative_error, 2),
        "count_accuracy_percent": round(count_accuracy, 2),
        # Keep original detection metrics for reference
        "TP": metric['TP'],
        "FP": metric['FP'],
        "FN": metric['FN']
    })

# Calculate overall statistics
total_preds = sum(m['num_preds'] for m in count_accuracy_results)
total_gts = sum(m['num_gts'] for m in count_accuracy_results)
total_count_error = total_preds - total_gts  # Signed error
total_count_error_abs = abs(total_count_error)
overall_count_accuracy = (1 - total_count_error_abs / total_gts) * 100 if total_gts > 0 else 0.0
overall_error_percent = (total_count_error_abs / total_gts) * 100 if total_gts > 0 else 0.0

# Calculate average per-image metrics
avg_count_accuracy = sum(m['count_accuracy_percent'] for m in count_accuracy_results) / len(count_accuracy_results)
# For average error percent, use absolute values to get average magnitude
avg_error_percent = sum(abs(m['error_percent']) for m in count_accuracy_results) / len(count_accuracy_results)

# Create output structure
output = {
    "summary": {
        "total_images": len(count_accuracy_results),
        "total_predicted": total_preds,
        "total_ground_truth": total_gts,
        "total_count_error": total_count_error,
        "total_count_error_abs": total_count_error_abs,
        "overall_error_percent": round(overall_error_percent, 2),
        "overall_count_accuracy_percent": round(overall_count_accuracy, 2),
        "average_per_image_error_percent": round(avg_error_percent, 2),
        "average_per_image_count_accuracy_percent": round(avg_count_accuracy, 2)
    },
    "per_image": count_accuracy_results
}

# Save to new file
output_file = Path(__file__).parent / "per_image_count_accuracy.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ Generated per-image count error percentage JSON")
print(f"   File: {output_file}")
print(f"\nSummary:")
print(f"  Total Images: {len(count_accuracy_results)}")
print(f"  Total Predicted: {total_preds}")
print(f"  Total Ground Truth: {total_gts}")
print(f"  Total Count Error: {total_count_error} ({'+' if total_count_error >= 0 else ''}{total_count_error})")
print(f"    → Positive = over-counting, Negative = under-counting")
print(f"  Overall Error Percentage: {overall_error_percent:.2f}%")
print(f"  Overall Count Accuracy: {overall_count_accuracy:.2f}%")
print(f"  Average Per-Image Error Percentage: {avg_error_percent:.2f}%")
print(f"  Average Per-Image Count Accuracy: {avg_count_accuracy:.2f}%")

