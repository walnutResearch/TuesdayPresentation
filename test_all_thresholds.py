#!/usr/bin/env python3
"""
Test the walnut detector at thresholds 0.1–0.9 on the test set, then compile precision and count accuracy
per threshold. Uses evaluate_detector.evaluate(); writes results to threshold_evaluations/ and all_threshold_results.json.

How to run:
  python test_all_thresholds.py

Edit MODEL_PATH, TEST_IMAGES_DIR, TEST_LABELS_DIR, OUTPUT_BASE_DIR, THRESHOLDS, PATCH_SIZE, STRIDE, MATCH_DISTANCE at top of file.
"""

import json
import os
from pathlib import Path
from evaluate_detector import evaluate

# Configuration
MODEL_PATH = "models_new/walnut_classifier.pth"  # Using the main model
TEST_IMAGES_DIR = "/Users/kalpit/TuesdayPresentation/Walnut_Research/test/images"
TEST_LABELS_DIR = "/Users/kalpit/TuesdayPresentation/Walnut_Research/test/annotations"
OUTPUT_BASE_DIR = "/Users/kalpit/TuesdayPresentation/threshold_evaluations"

# Thresholds to test
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Detection parameters
PATCH_SIZE = 32
STRIDE = 16
MATCH_DISTANCE = 20.0
CLUSTER = True

def calculate_count_accuracy(total_predicted, ground_truth):
    """Calculate count accuracy percentage"""
    if ground_truth == 0:
        return None
    count_error = abs(total_predicted - ground_truth)
    count_accuracy = (1 - count_error / ground_truth) * 100
    return count_accuracy

def main():
    print("=" * 100)
    print("TESTING ALL THRESHOLDS (0.1 to 0.9)")
    print("=" * 100)
    print(f"Model: {MODEL_PATH}")
    print(f"Test Images: {TEST_IMAGES_DIR}")
    print(f"Thresholds: {THRESHOLDS}")
    print()
    
    all_results = {}
    
    # Get ground truth total first
    print("📊 Calculating ground truth total...")
    image_files = (list(Path(TEST_IMAGES_DIR).glob("*.png")) + 
                   list(Path(TEST_IMAGES_DIR).glob("*.jpg")) +
                   list(Path(TEST_IMAGES_DIR).glob("*.JPG")) +
                   list(Path(TEST_IMAGES_DIR).glob("*.PNG")))
    
    print(f"Found {len(image_files)} image files")
    
    total_ground_truth = 0
    for img_path in image_files:
        label_path = Path(TEST_LABELS_DIR) / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                total_ground_truth += len(lines)
        else:
            print(f"Warning: No annotation file for {img_path.name}")
    
    print(f"✅ Total Ground Truth: {total_ground_truth} walnuts\n")
    
    # Test each threshold
    for threshold in THRESHOLDS:
        print(f"\n{'='*100}")
        print(f"Testing Threshold: {threshold:.1f}")
        print(f"{'='*100}")
        
        output_dir = Path(OUTPUT_BASE_DIR) / f"th{threshold:.1f}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run evaluation
            results = evaluate(
                model_path=MODEL_PATH,
                images_dir=TEST_IMAGES_DIR,
                labels_dir=TEST_LABELS_DIR,
                output_dir=str(output_dir),
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                threshold=threshold,
                match_dist=MATCH_DISTANCE,
                cluster=CLUSTER,
            )
            
            # Load per_image metrics from saved file
            per_image_file = output_dir / "per_image_metrics.json"
            if per_image_file.exists():
                with open(per_image_file, 'r') as f:
                    per_image = json.load(f)
                total_predicted = sum(item['num_preds'] for item in per_image)
            else:
                # Fallback: calculate from TP + FP
                total_predicted = results['TP'] + results['FP']
            
            # Calculate count accuracy
            count_accuracy = calculate_count_accuracy(total_predicted, total_ground_truth)
            count_error = abs(total_predicted - total_ground_truth)
            
            # Store results
            all_results[f"{threshold:.1f}"] = {
                "threshold": threshold,
                "total_predicted": total_predicted,
                "total_ground_truth": total_ground_truth,
                "count_error": count_error,
                "count_accuracy": count_accuracy,
                "precision": results['precision'] * 100,
                "recall": results['recall'] * 100,
                "f1": results['f1'] * 100,
                "TP": results['TP'],
                "FP": results['FP'],
                "FN": results['FN'],
                "images": results['images'],
            }
            
            print(f"✅ Threshold {threshold:.1f} Results:")
            print(f"   Predicted: {total_predicted} | Ground Truth: {total_ground_truth}")
            print(f"   Count Accuracy: {count_accuracy:.2f}%")
            print(f"   Precision: {results['precision']*100:.2f}%")
            print(f"   Recall: {results['recall']*100:.2f}%")
            print(f"   F1: {results['f1']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error testing threshold {threshold:.1f}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save comprehensive results
    output_file = Path(OUTPUT_BASE_DIR) / "all_thresholds_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "total_ground_truth": total_ground_truth,
                "model": MODEL_PATH,
                "test_dataset": TEST_IMAGES_DIR,
                "parameters": {
                    "patch_size": PATCH_SIZE,
                    "stride": STRIDE,
                    "match_distance": MATCH_DISTANCE,
                    "cluster": CLUSTER,
                }
            },
            "results": all_results
        }, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: PRECISION & COUNT ACCURACY BY THRESHOLD")
    print("=" * 100)
    print(f"{'Threshold':<12} {'Predicted':<12} {'Count Acc %':<15} {'Precision %':<15} {'Recall %':<15} {'F1 %':<15} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 100)
    
    for thresh_str in sorted(all_results.keys(), key=float):
        r = all_results[thresh_str]
        count_acc = f"{r['count_accuracy']:.2f}" if r['count_accuracy'] else "N/A"
        print(f"{r['threshold']:<12.1f} {r['total_predicted']:<12} {count_acc:<15} {r['precision']:<15.2f} {r['recall']:<15.2f} {r['f1']:<15.2f} {r['TP']:<8} {r['FP']:<8} {r['FN']:<8}")
    
    # Find best results
    print("\n" + "=" * 100)
    print("BEST RESULTS")
    print("=" * 100)
    
    best_count_acc = max([r for r in all_results.values() if r['count_accuracy']], 
                         key=lambda x: x['count_accuracy'], default=None)
    if best_count_acc:
        print(f"\n🏆 Best Count Accuracy:")
        print(f"   Threshold: {best_count_acc['threshold']:.1f}")
        print(f"   Count Accuracy: {best_count_acc['count_accuracy']:.2f}%")
        print(f"   Precision: {best_count_acc['precision']:.2f}%")
        print(f"   Predicted: {best_count_acc['total_predicted']} vs Ground Truth: {best_count_acc['total_ground_truth']}")
    
    best_precision = max(all_results.values(), key=lambda x: x['precision'])
    print(f"\n🎯 Best Precision:")
    print(f"   Threshold: {best_precision['threshold']:.1f}")
    print(f"   Precision: {best_precision['precision']:.2f}%")
    print(f"   Count Accuracy: {best_precision['count_accuracy']:.2f}%" if best_precision['count_accuracy'] else "   Count Accuracy: N/A")
    
    best_f1 = max(all_results.values(), key=lambda x: x['f1'])
    print(f"\n⚖️  Best F1 Score:")
    print(f"   Threshold: {best_f1['threshold']:.1f}")
    print(f"   F1: {best_f1['f1']:.2f}%")
    print(f"   Precision: {best_f1['precision']:.2f}% | Recall: {best_f1['recall']:.2f}%")
    
    print(f"\n✅ All results saved to: {output_file}")

if __name__ == "__main__":
    main()

