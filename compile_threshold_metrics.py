#!/usr/bin/env python3
"""
Scan evaluation output directories for summary.json and per_image_metrics.json, extract precision/recall/F1
and (optionally) count accuracy per threshold, and write a compiled JSON. Used to build threshold comparison data.

How to run:
  python compile_threshold_metrics.py

Paths and ground-truth totals are configured inside the script.
"""

import json
from pathlib import Path
import re

def extract_threshold_from_path(path_str):
    """Extract threshold value from directory name"""
    # Patterns: glenn_eval_th042, detector_eval_precision_th05, detector_evaluation_results_new_th06
    patterns = [
        (r'glenn_eval_th(\d+)', lambda m: float(m.group(1)) / 100),  # th042 -> 0.42
        (r'th(\d+)', lambda m: float(m.group(1)) / 10),  # th05 -> 0.5, th06 -> 0.6
        (r'th0(\d)', lambda m: float(m.group(1)) / 10),  # th05 -> 0.5
    ]
    
    for pattern, converter in patterns:
        match = re.search(pattern, path_str)
        if match:
            return converter(match)
    return None

def load_summary_and_calculate_metrics(summary_path, per_image_path, ground_truth_total=None):
    """Load summary and calculate count accuracy"""
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        with open(per_image_path, 'r') as f:
            per_image = json.load(f)
        
        # Calculate total predicted
        total_predicted = sum(item.get('num_preds', 0) for item in per_image)
        
        # Calculate count accuracy if ground truth is provided
        count_accuracy = None
        count_error = None
        if ground_truth_total is not None and ground_truth_total > 0:
            count_error = abs(total_predicted - ground_truth_total)
            count_accuracy = (1 - count_error / ground_truth_total) * 100
        
        return {
            'precision': summary.get('precision', 0) * 100,
            'recall': summary.get('recall', 0) * 100,
            'f1': summary.get('f1', 0) * 100,
            'TP': summary.get('TP', 0),
            'FP': summary.get('FP', 0),
            'FN': summary.get('FN', 0),
            'total_predicted': total_predicted,
            'count_error': count_error,
            'count_accuracy': count_accuracy,
            'threshold': summary.get('threshold', None)
        }
    except Exception as e:
        print(f"Error loading {summary_path}: {e}")
        return None

# Find all evaluation directories
base_dir = Path("/Users/kalpit/TuesdayPresentation")
results = {}

# Glenn evaluations (different dataset, 1782 ground truth)
glenn_dirs = list(base_dir.glob("glenn_eval_th*"))
for glenn_dir in glenn_dirs:
    summary_path = glenn_dir / "summary.json"
    per_image_path = glenn_dir / "per_image_metrics.json"
    
    if summary_path.exists() and per_image_path.exists():
        threshold = extract_threshold_from_path(str(glenn_dir))
        if threshold is None:
            threshold = 0.42  # Default for glenn_eval_th042
        
        metrics = load_summary_and_calculate_metrics(summary_path, per_image_path, ground_truth_total=1782)
        if metrics:
            metrics['threshold'] = threshold
            metrics['dataset'] = 'glenn'
            results[f"glenn_th{threshold:.2f}"] = metrics

# Detector evaluations (Walnut_Research test dataset, 487 ground truth)
detector_dirs = list(base_dir.glob("detector_eval*")) + list(base_dir.glob("detector_evaluation_results*"))
for detector_dir in detector_dirs:
    if not detector_dir.is_dir():
        continue
    
    summary_path = detector_dir / "summary.json"
    per_image_path = detector_dir / "per_image_metrics.json"
    
    if summary_path.exists() and per_image_path.exists():
        threshold = extract_threshold_from_path(str(detector_dir))
        if threshold is None:
            # Try to get from summary.json
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    threshold = summary.get('threshold', 0.5)
            except:
                threshold = 0.5
        
        metrics = load_summary_and_calculate_metrics(summary_path, per_image_path, ground_truth_total=487)
        if metrics:
            metrics['threshold'] = threshold
            metrics['dataset'] = 'walnut_research'
            results[f"walnut_th{threshold:.2f}"] = metrics

# Compile results by threshold
threshold_comparison = {}
for key, metrics in results.items():
    threshold = metrics['threshold']
    dataset = metrics['dataset']
    
    if threshold not in threshold_comparison:
        threshold_comparison[threshold] = {}
    
    threshold_comparison[threshold][dataset] = metrics

# Create comprehensive output
output = {
    "summary": {
        "total_evaluations": len(results),
        "datasets": {
            "glenn": {
                "ground_truth_total": 1782,
                "description": "Glenn dataset evaluations"
            },
            "walnut_research": {
                "ground_truth_total": 487,
                "description": "Walnut Research test dataset"
            }
        }
    },
    "by_threshold": {},
    "by_dataset": {
        "glenn": {},
        "walnut_research": {}
    }
}

# Organize by threshold
for threshold in sorted(threshold_comparison.keys()):
    output["by_threshold"][f"{threshold:.2f}"] = threshold_comparison[threshold]

# Organize by dataset
for key, metrics in results.items():
    dataset = metrics['dataset']
    threshold = metrics['threshold']
    output["by_dataset"][dataset][f"{threshold:.2f}"] = metrics

# Save comprehensive results
output_file = base_dir / "threshold_metrics_comparison.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print("=" * 80)
print("THRESHOLD METRICS COMPILATION")
print("=" * 80)
print(f"\n✅ Compiled metrics for {len(results)} evaluations")
print(f"   Saved to: {output_file}\n")

# Print summary tables
print("=" * 80)
print("GLENN DATASET (Ground Truth: 1782 walnuts)")
print("=" * 80)
print(f"{'Threshold':<12} {'Predicted':<12} {'Count Acc %':<15} {'Precision %':<15} {'Recall %':<15} {'F1 %':<15}")
print("-" * 80)
for threshold in sorted([t for t in threshold_comparison.keys() if 'glenn' in str(threshold_comparison[t])]):
    if 'glenn' in threshold_comparison[threshold]:
        m = threshold_comparison[threshold]['glenn']
        print(f"{threshold:<12.2f} {m['total_predicted']:<12} {m['count_accuracy']:<15.2f} {m['precision']:<15.2f} {m['recall']:<15.2f} {m['f1']:<15.2f}")

print("\n" + "=" * 80)
print("WALNUT RESEARCH DATASET (Ground Truth: 487 walnuts)")
print("=" * 80)
print(f"{'Threshold':<12} {'Predicted':<12} {'Count Acc %':<15} {'Precision %':<15} {'Recall %':<15} {'F1 %':<15}")
print("-" * 80)
for threshold in sorted([t for t in threshold_comparison.keys() if 'walnut_research' in str(threshold_comparison[t])]):
    if 'walnut_research' in threshold_comparison[threshold]:
        m = threshold_comparison[threshold]['walnut_research']
        print(f"{threshold:<12.2f} {m['total_predicted']:<12} {m['count_accuracy']:<15.2f} {m['precision']:<15.2f} {m['recall']:<15.2f} {m['f1']:<15.2f}")

