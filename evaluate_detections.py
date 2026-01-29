#!/usr/bin/env python3
"""
Evaluate detection accuracy: load detection results and ground-truth annotations, match detections to GT by
distance (match_distance), and compute TP/FP/FN, precision, recall, F1. Used by evaluation pipelines.

How to run:
  Use as a library: from evaluate_detections import load_annotations, match_detections_to_ground_truth, ...
  Or run the main block with paths to detection results JSON and annotation directory.
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

def load_annotations(annotation_path):
    """Load ground truth annotations from text file"""
    gt_points = []
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    x, y = int(parts[0]), int(parts[1])
                    gt_points.append([x, y])
    return np.array(gt_points) if gt_points else np.array([]).reshape(0, 2)

def match_detections_to_ground_truth(detections, ground_truth, match_distance=20.0):
    """
    Match detections to ground truth using distance threshold
    Returns: (true_positives, false_positives, false_negatives)
    """
    if len(ground_truth) == 0:
        return 0, len(detections), 0
    
    if len(detections) == 0:
        return 0, 0, len(ground_truth)
    
    # Calculate distance matrix
    distances = cdist(detections, ground_truth)
    
    # Find matches (greedy assignment)
    matched_gt = set()
    matched_det = set()
    true_positives = 0
    
    # Sort by distance and match
    matches = []
    for i in range(len(detections)):
        for j in range(len(ground_truth)):
            if distances[i, j] <= match_distance:
                matches.append((i, j, distances[i, j]))
    
    # Sort by distance and assign greedily
    matches.sort(key=lambda x: x[2])
    
    for det_idx, gt_idx, dist in matches:
        if det_idx not in matched_det and gt_idx not in matched_gt:
            matched_det.add(det_idx)
            matched_gt.add(gt_idx)
            true_positives += 1
    
    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    return true_positives, false_positives, false_negatives

def evaluate_detection_results(detection_results_path, annotations_dir, match_distance=20.0):
    """Evaluate detection results against ground truth"""
    
    # Load detection results
    with open(detection_results_path, 'r') as f:
        results = json.load(f)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_det = 0
    
    per_image_metrics = []
    
    for result in results:
        image_path = Path(result['image_path'])
        image_name = image_path.stem
        
        # Load ground truth
        annotation_path = Path(annotations_dir) / f"{image_name}.txt"
        
        if not annotation_path.exists():
            print(f"Warning: No annotation found for {image_name}")
            continue
        
        gt_points = load_annotations(annotation_path)
        detections = np.array(result['centers']) if result['centers'] else np.array([]).reshape(0, 2)
        
        # Match detections
        tp, fp, fn = match_detections_to_ground_truth(detections, gt_points, match_distance)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_image_metrics.append({
            'image': image_name,
            'gt_count': len(gt_points),
            'det_count': len(detections),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt_points)
        total_det += len(detections)
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    return {
        'overall': {
            'total_gt': total_gt,
            'total_det': total_det,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        },
        'per_image': per_image_metrics
    }

def main():
    import sys
    
    # Paths
    test_dir = "/Users/kalpit/TuesdayPresentation/Walnut_Research/test"
    annotations_dir = f"{test_dir}/annotations"
    
    # Evaluate both runs
    print("=" * 70)
    print("DETECTION ACCURACY EVALUATION")
    print("=" * 70)
    
    # Run 1: Original (patch_size=32, stride=16, threshold=0.5)
    print("\n📊 Run 1: patch_size=32, stride=16, threshold=0.5")
    print("-" * 70)
    results1 = evaluate_detection_results(
        f"{test_dir}/detections/detection_results.json",
        annotations_dir,
        match_distance=20.0
    )
    
    overall1 = results1['overall']
    print(f"Ground Truth Walnuts: {overall1['total_gt']}")
    print(f"Detected Walnuts: {overall1['total_det']}")
    print(f"True Positives: {overall1['true_positives']}")
    print(f"False Positives: {overall1['false_positives']}")
    print(f"False Negatives: {overall1['false_negatives']}")
    print(f"\nPrecision: {overall1['precision']:.3f} ({overall1['precision']*100:.1f}%)")
    print(f"Recall: {overall1['recall']:.3f} ({overall1['recall']*100:.1f}%)")
    print(f"F1 Score: {overall1['f1']:.3f}")
    
    # Run 2: New (patch_size=48, stride=24, threshold=0.4)
    print("\n📊 Run 2: patch_size=48, stride=24, threshold=0.4")
    print("-" * 70)
    results2 = evaluate_detection_results(
        f"{test_dir}/detections_v2/detection_results.json",
        annotations_dir,
        match_distance=20.0
    )
    
    overall2 = results2['overall']
    print(f"Ground Truth Walnuts: {overall2['total_gt']}")
    print(f"Detected Walnuts: {overall2['total_det']}")
    print(f"True Positives: {overall2['true_positives']}")
    print(f"False Positives: {overall2['false_positives']}")
    print(f"False Negatives: {overall2['false_negatives']}")
    print(f"\nPrecision: {overall2['precision']:.3f} ({overall2['precision']*100:.1f}%)")
    print(f"Recall: {overall2['recall']:.3f} ({overall2['recall']*100:.1f}%)")
    print(f"F1 Score: {overall2['f1']:.3f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'Run 1':<15} {'Run 2':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Precision':<20} {overall1['precision']:.3f} ({overall1['precision']*100:.1f}%){'':<5} {overall2['precision']:.3f} ({overall2['precision']*100:.1f}%){'':<5} {overall2['precision'] - overall1['precision']:+.3f}")
    print(f"{'Recall':<20} {overall1['recall']:.3f} ({overall1['recall']*100:.1f}%){'':<5} {overall2['recall']:.3f} ({overall2['recall']*100:.1f}%){'':<5} {overall2['recall'] - overall1['recall']:+.3f}")
    print(f"{'F1 Score':<20} {overall1['f1']:.3f}{'':<15} {overall2['f1']:.3f}{'':<15} {overall2['f1'] - overall1['f1']:+.3f}")
    print(f"{'False Positives':<20} {overall1['false_positives']:<15} {overall2['false_positives']:<15} {overall2['false_positives'] - overall1['false_positives']:+d}")
    print(f"{'False Negatives':<20} {overall1['false_negatives']:<15} {overall2['false_negatives']:<15} {overall2['false_negatives'] - overall1['false_negatives']:+d}")

if __name__ == "__main__":
    main()

