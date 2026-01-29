#!/usr/bin/env python3
"""
Scan detector evaluation directories (e.g. detector_eval_*, glenn_eval_*) for summary.json and per_image_metrics.json,
compile precision/recall/F1 and count accuracy per threshold into a single all_threshold_results.json.

How to run:
  python compile_all_threshold_results.py

Paths and GROUND_TRUTH dict are set inside the script. Writes to all_threshold_results.json in project root.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Ground truth totals for different datasets
GROUND_TRUTH = {
    'walnut_research': 487,  # Walnut_Research/test dataset
    'glenn': 1782,  # Glenn dataset
}

def extract_threshold_from_path(path_str):
    """Extract threshold value from directory name"""
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

def determine_dataset(summary_path):
    """Determine which dataset based on path"""
    path_str = str(summary_path)
    if 'glenn' in path_str.lower():
        return 'glenn'
    elif 'detector_eval' in path_str.lower() or 'walnut_research' in path_str.lower():
        return 'walnut_research'
    return None

def load_evaluation_results(summary_path, per_image_path=None):
    """Load evaluation results and calculate metrics"""
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Try to get threshold from summary or path
        threshold = summary.get('threshold')
        if threshold is None:
            threshold = extract_threshold_from_path(str(summary_path))
        
        # Determine dataset
        dataset = determine_dataset(summary_path)
        gt_total = GROUND_TRUTH.get(dataset) if dataset else None
        
        # Calculate total predicted
        total_predicted = None
        if per_image_path and Path(per_image_path).exists():
            with open(per_image_path, 'r') as f:
                per_image = json.load(f)
            if isinstance(per_image, list):
                total_predicted = sum(item.get('num_preds', 0) for item in per_image)
            elif isinstance(per_image, dict) and 'per_image' in per_image:
                total_predicted = sum(item.get('num_preds', 0) for item in per_image['per_image'])
        
        # If we can't get from per_image, estimate from TP + FP
        if total_predicted is None:
            total_predicted = summary.get('TP', 0) + summary.get('FP', 0)
        
        # Calculate count accuracy
        count_accuracy = None
        count_error = None
        if gt_total is not None and gt_total > 0:
            count_error = abs(total_predicted - gt_total)
            count_accuracy = (1 - count_error / gt_total) * 100
        
        return {
            'threshold': threshold,
            'dataset': dataset,
            'precision': summary.get('precision', 0) * 100,
            'recall': summary.get('recall', 0) * 100,
            'f1': summary.get('f1', 0) * 100,
            'TP': summary.get('TP', 0),
            'FP': summary.get('FP', 0),
            'FN': summary.get('FN', 0),
            'total_predicted': total_predicted,
            'ground_truth': gt_total,
            'count_error': count_error,
            'count_accuracy': count_accuracy,
            'images': summary.get('images', 0),
            'patch_size': summary.get('patch_size'),
            'stride': summary.get('stride'),
            'match_distance': summary.get('match_distance_px'),
            'source_dir': str(summary_path.parent),
        }
    except Exception as e:
        print(f"Error loading {summary_path}: {e}")
        return None

# Find all summary.json files
base_dir = Path("/Users/kalpit/TuesdayPresentation")
all_results = []

# List of evaluation directories to check
eval_dirs = [
    "detector_eval_model1",
    "detector_eval_precision_model",
    "detector_eval_precision_th05",
    "detector_eval_precision_th07",
    "detector_eval_precision_th08",
    "detector_evaluation_results",
    "detector_evaluation_results_new",
    "detector_evaluation_results_new_th06",
    "detector_evaluation_results_new_th07",
    "detector_evaluation_results_new_th08",
    "glenn_eval_th038",
    "glenn_eval_th04",
    "glenn_eval_th042",
    "glenn_eval_th043",
    "glenn_eval_th045",
    "glenn_eval_th05",
    "glenn_eval_th06",
]

for eval_dir in eval_dirs:
    summary_path = base_dir / eval_dir / "summary.json"
    per_image_path = base_dir / eval_dir / "per_image_metrics.json"
    
    if summary_path.exists():
        result = load_evaluation_results(summary_path, per_image_path if per_image_path.exists() else None)
        if result:
            all_results.append(result)

# Also check for any other summary.json files
for summary_path in base_dir.rglob("summary.json"):
    if summary_path.parent.name not in ['data', ''] and 'report_results' not in str(summary_path):
        if summary_path not in [Path(base_dir / d / "summary.json") for d in eval_dirs]:
            result = load_evaluation_results(summary_path)
            if result and result not in all_results:
                all_results.append(result)

# Organize results
output = {
    "summary": {
        "total_evaluations": len(all_results),
        "datasets": GROUND_TRUTH,
    },
    "all_results": [],
    "by_dataset": defaultdict(list),
    "by_threshold": defaultdict(list),
}

for result in all_results:
    output["all_results"].append(result)
    if result['dataset']:
        output["by_dataset"][result['dataset']].append(result)
    if result['threshold'] is not None:
        output["by_threshold"][result['threshold']].append(result)

# Sort results
for dataset in output["by_dataset"]:
    output["by_dataset"][dataset].sort(key=lambda x: x['threshold'] if x['threshold'] else 0)

# Save comprehensive results
output_file = base_dir / "all_threshold_results.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print("=" * 100)
print("ALL HISTORICAL THRESHOLD RESULTS - PRECISION & ACCURACY")
print("=" * 100)
print(f"\n✅ Compiled {len(all_results)} evaluation results")
print(f"   Saved to: {output_file}\n")

# Print detailed tables
for dataset_name, gt_total in GROUND_TRUTH.items():
    dataset_results = output["by_dataset"][dataset_name]
    if not dataset_results:
        continue
    
    print("=" * 100)
    print(f"{dataset_name.upper().replace('_', ' ')} DATASET (Ground Truth: {gt_total} walnuts)")
    print("=" * 100)
    print(f"{'Threshold':<12} {'Predicted':<12} {'Count Acc %':<15} {'Precision %':<15} {'Recall %':<15} {'F1 %':<15} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 100)
    
    for r in sorted(dataset_results, key=lambda x: x['threshold'] if x['threshold'] else 0):
        threshold = f"{r['threshold']:.2f}" if r['threshold'] else "N/A"
        count_acc = f"{r['count_accuracy']:.2f}" if r['count_accuracy'] else "N/A"
        print(f"{threshold:<12} {r['total_predicted']:<12} {count_acc:<15} {r['precision']:<15.2f} {r['recall']:<15.2f} {r['f1']:<15.2f} {r['TP']:<8} {r['FP']:<8} {r['FN']:<8}")
    
    print()

# Print best results
print("=" * 100)
print("BEST RESULTS BY METRIC")
print("=" * 100)

for dataset_name in GROUND_TRUTH.keys():
    dataset_results = output["by_dataset"][dataset_name]
    if not dataset_results:
        continue
    
    # Best count accuracy
    best_count_acc = max([r for r in dataset_results if r['count_accuracy']], 
                         key=lambda x: x['count_accuracy'], default=None)
    if best_count_acc:
        print(f"\n{dataset_name.upper().replace('_', ' ')} - Best Count Accuracy:")
        print(f"  Threshold: {best_count_acc['threshold']:.2f}")
        print(f"  Count Accuracy: {best_count_acc['count_accuracy']:.2f}%")
        print(f"  Precision: {best_count_acc['precision']:.2f}%")
        print(f"  Recall: {best_count_acc['recall']:.2f}%")
        print(f"  F1: {best_count_acc['f1']:.2f}%")
        print(f"  Predicted: {best_count_acc['total_predicted']} vs Ground Truth: {best_count_acc['ground_truth']}")
    
    # Best precision
    best_precision = max(dataset_results, key=lambda x: x['precision'], default=None)
    if best_precision:
        print(f"\n{dataset_name.upper().replace('_', ' ')} - Best Precision:")
        print(f"  Threshold: {best_precision['threshold']:.2f}")
        print(f"  Precision: {best_precision['precision']:.2f}%")
        print(f"  Count Accuracy: {best_precision['count_accuracy']:.2f}%" if best_precision['count_accuracy'] else "  Count Accuracy: N/A")
        print(f"  Recall: {best_precision['recall']:.2f}%")
        print(f"  F1: {best_precision['f1']:.2f}%")
    
    # Best F1
    best_f1 = max(dataset_results, key=lambda x: x['f1'], default=None)
    if best_f1:
        print(f"\n{dataset_name.upper().replace('_', ' ')} - Best F1 Score:")
        print(f"  Threshold: {best_f1['threshold']:.2f}")
        print(f"  F1: {best_f1['f1']:.2f}%")
        print(f"  Precision: {best_f1['precision']:.2f}%")
        print(f"  Recall: {best_f1['recall']:.2f}%")
        print(f"  Count Accuracy: {best_f1['count_accuracy']:.2f}%" if best_f1['count_accuracy'] else "  Count Accuracy: N/A")

