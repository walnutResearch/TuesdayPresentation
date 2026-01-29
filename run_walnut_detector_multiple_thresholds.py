#!/usr/bin/env python3
"""
Run the walnut detector at multiple confidence thresholds and evaluate against ground-truth annotations.
Computes count accuracy, precision, recall, F1 per threshold and can report the best threshold.

How to run:
  python run_walnut_detector_multiple_thresholds.py

Configure model path, image dir, annotations dir, and threshold list inside the script or via its logic.
"""

import os
import sys
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

def load_annotations(annotation_path: Path) -> List[Tuple[int, int]]:
    """Load annotation coordinates from text file"""
    coords = []
    if not annotation_path.exists():
        return coords
    
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse x y coordinates
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = int(float(parts[0]))
                    y = int(float(parts[1]))
                    coords.append((x, y))
                except ValueError:
                    continue
    
    return coords

def evaluate_against_annotations(detection_results_file: str, annotations_dir: str, image_dir: str) -> dict:
    """Evaluate detection results against ground truth annotations"""
    # Load detection results
    with open(detection_results_file, 'r') as f:
        detection_results = json.load(f)
    
    pred_counts = []
    true_counts = []
    per_image_errors = []
    missing_annotations = []
    
    print(f"   Comparing {len(detection_results)} images with annotations...")
    
    for result in detection_results:
        image_path = Path(result['image_path'])
        image_name = image_path.stem
        
        # Load ground truth annotation
        annotation_path = Path(annotations_dir) / f"{image_name}.txt"
        gt_coords = load_annotations(annotation_path)
        
        if len(gt_coords) == 0 and not annotation_path.exists():
            missing_annotations.append(image_name)
        
        pred_count = result['num_walnuts']
        true_count = len(gt_coords)
        
        pred_counts.append(pred_count)
        true_counts.append(true_count)
        per_image_errors.append(abs(pred_count - true_count))
    
    if missing_annotations:
        print(f"   ⚠️  Warning: {len(missing_annotations)} images missing annotations")
    
    # Calculate metrics
    pred_counts = np.array(pred_counts)
    true_counts = np.array(true_counts)
    
    total_pred = np.sum(pred_counts)
    total_true = np.sum(true_counts)
    count_error = abs(total_pred - total_true)
    
    # Count accuracy (percentage)
    if total_true > 0:
        count_accuracy = (1 - count_error / total_true) * 100
    else:
        count_accuracy = 0.0
    
    # Mean Absolute Error
    mae = np.mean(per_image_errors)
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((pred_counts - true_counts) ** 2))
    
    # Correlation
    if len(pred_counts) > 1 and np.std(pred_counts) > 0 and np.std(true_counts) > 0:
        correlation = np.corrcoef(pred_counts, true_counts)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'total_predicted': int(total_pred),
        'total_true': int(total_true),
        'count_error': int(count_error),
        'count_accuracy': float(count_accuracy),
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(correlation),
        'num_images': len(pred_counts)
    }

def run_detector_for_threshold(model_path, image_dir, annotations_dir, output_base_dir, threshold, patch_size=32, stride=16, cluster=True):
    """Run walnut_detector.py for a specific threshold and evaluate against annotations"""
    # Use threshold value directly in directory name to avoid rounding issues
    threshold_str = str(threshold).replace('.', '_')
    output_dir = os.path.join(output_base_dir, f"threshold_{threshold_str}")
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        "walnut_detector.py",
        "--model_path", model_path,
        "--image_dir", image_dir,
        "--output_dir", output_dir,
        "--patch_size", str(patch_size),
        "--stride", str(stride),
        "--threshold", str(threshold)
    ]
    
    if cluster:
        cmd.append("--cluster")
    
    print(f"\n{'='*70}")
    print(f"Running with threshold: {threshold:.1f}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run with real-time output
        result = subprocess.run(cmd, check=True)
        
        # Load results
        results_file = os.path.join(output_dir, "detection_results.json")
        if os.path.exists(results_file):
            print(f"\n📊 Loading detection results from {results_file}...")
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"✅ Processed {len(results)} images")
            print(f"📈 Evaluating against ground truth annotations...")
            
            # Evaluate against annotations
            eval_metrics = evaluate_against_annotations(results_file, annotations_dir, image_dir)
            
            print(f"✅ Evaluation complete for threshold {threshold:.1f}")
            print(f"   Predicted: {eval_metrics['total_predicted']} walnuts")
            print(f"   Ground Truth: {eval_metrics['total_true']} walnuts")
            print(f"   Count Accuracy: {eval_metrics['count_accuracy']:.2f}%")
            print(f"   MAE: {eval_metrics['mae']:.2f}, RMSE: {eval_metrics['rmse']:.2f}")
            
            total_walnuts = sum(r['num_walnuts'] for r in results)
            mean_confidence = sum(r['mean_confidence'] for r in results) / len(results) if results else 0
            
            return {
                'threshold': threshold,
                'total_walnuts': total_walnuts,
                'num_images': len(results),
                'mean_confidence': mean_confidence,
                'results_file': results_file,
                'output_dir': output_dir,
                'evaluation': eval_metrics,
                'success': True
            }
        else:
            return {
                'threshold': threshold,
                'success': False,
                'error': 'Results file not found'
            }
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running detector for threshold {threshold:.1f}")
        print(f"Error: {e}")
        return {
            'threshold': threshold,
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        print(f"\n❌ Unexpected error for threshold {threshold:.1f}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'threshold': threshold,
            'success': False,
            'error': str(e)
        }

def main():
    # Configuration
    model_path = "/Users/kalpit/TuesdayPresentation/models_new/walnut_classifier.pth"
    image_dir = "/Users/kalpit/TuesdayPresentation/WalnutVarietyTrial annotated/cropped_images"
    annotations_dir = "/Users/kalpit/TuesdayPresentation/WalnutVarietyTrial annotated/annotated_images"
    output_base_dir = "/Users/kalpit/TuesdayPresentation/walnut_variety_detector_results"
    thresholds = [0.55]
    patch_size = 32
    stride = 16
    cluster = True
    
    print("🥜 Running Walnut Detector with Multiple Thresholds")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Image Directory: {image_dir}")
    print(f"Annotations Directory: {annotations_dir}")
    print(f"Thresholds: {thresholds}")
    print(f"Patch Size: {patch_size}, Stride: {stride}")
    print(f"Clustering: {cluster}")
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Run for each threshold
    all_results = []
    total_thresholds = len(thresholds)
    
    for idx, threshold in enumerate(thresholds, 1):
        print(f"\n{'#'*70}")
        print(f"# Processing Threshold {idx}/{total_thresholds}: {threshold:.1f}")
        print(f"{'#'*70}")
        
        result = run_detector_for_threshold(
            model_path, image_dir, annotations_dir, output_base_dir, threshold,
            patch_size, stride, cluster
        )
        all_results.append(result)
        
        if result['success']:
            print(f"\n✅ Threshold {threshold:.1f} completed successfully!")
        else:
            print(f"\n❌ Threshold {threshold:.1f} failed: {result.get('error', 'Unknown error')}")
    
    # Print summary with evaluation metrics
    print("\n" + "=" * 70)
    print("📊 SUMMARY - ALL THRESHOLDS (Compared with Ground Truth)")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Predicted':<12} {'Ground Truth':<15} {'Count Acc %':<15} {'MAE':<10} {'RMSE':<10} {'Status':<10}")
    print("-" * 70)
    
    for result in all_results:
        if result['success'] and 'evaluation' in result:
            eval_metrics = result['evaluation']
            print(f"{result['threshold']:<12.1f} {eval_metrics['total_predicted']:<12} {eval_metrics['total_true']:<15} {eval_metrics['count_accuracy']:<15.2f} {eval_metrics['mae']:<10.2f} {eval_metrics['rmse']:<10.2f} {'✅':<10}")
        else:
            print(f"{result['threshold']:<12.1f} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'N/A':<10} {'❌':<10}")
    
    # Find best threshold
    successful_results = [r for r in all_results if r['success'] and 'evaluation' in r]
    
    if successful_results:
        # Best by count accuracy
        best_count_acc = max(successful_results, key=lambda x: x['evaluation']['count_accuracy'])
        
        # Best by MAE (lower is better)
        best_mae = min(successful_results, key=lambda x: x['evaluation']['mae'])
        
        # Best by correlation
        best_corr = max(successful_results, key=lambda x: x['evaluation']['correlation'])
        
        print("\n" + "=" * 70)
        print("🏆 BEST THRESHOLDS")
        print("=" * 70)
        
        print(f"\n🥇 Best Count Accuracy:")
        print(f"   Threshold: {best_count_acc['threshold']:.1f}")
        print(f"   Count Accuracy: {best_count_acc['evaluation']['count_accuracy']:.2f}%")
        print(f"   Predicted: {best_count_acc['evaluation']['total_predicted']} vs Ground Truth: {best_count_acc['evaluation']['total_true']}")
        print(f"   MAE: {best_count_acc['evaluation']['mae']:.2f}, RMSE: {best_count_acc['evaluation']['rmse']:.2f}")
        
        print(f"\n🥇 Best MAE (Mean Absolute Error):")
        print(f"   Threshold: {best_mae['threshold']:.1f}")
        print(f"   MAE: {best_mae['evaluation']['mae']:.2f}")
        print(f"   Count Accuracy: {best_mae['evaluation']['count_accuracy']:.2f}%")
        print(f"   Predicted: {best_mae['evaluation']['total_predicted']} vs Ground Truth: {best_mae['evaluation']['total_true']}")
        
        print(f"\n🥇 Best Correlation:")
        print(f"   Threshold: {best_corr['threshold']:.1f}")
        print(f"   Correlation: {best_corr['evaluation']['correlation']:.3f}")
        print(f"   Count Accuracy: {best_corr['evaluation']['count_accuracy']:.2f}%")
    
    # Save summary
    summary_file = os.path.join(output_base_dir, "threshold_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'model_path': model_path,
            'image_dir': image_dir,
            'annotations_dir': annotations_dir,
            'parameters': {
                'patch_size': patch_size,
                'stride': stride,
                'cluster': cluster
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✅ Summary saved to: {summary_file}")
    print(f"📁 Individual results in: {output_base_dir}/threshold_*/")

if __name__ == "__main__":
    main()

