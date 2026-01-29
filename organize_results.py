#!/usr/bin/env python3
"""
Copy detection overlays, confidence maps, confusion matrices, and plots into report_results/images/ and
copy/compile result JSONs into report_results/data/ for a tidy report layout.

How to run:
  python organize_results.py

Paths point to Walnut_Research/test/detections, report_results, etc.; edit at top of functions if needed.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def copy_result_images():
    """Copy all result images to organized folders"""
    
    base_dir = Path("/Users/kalpit/TuesdayPresentation")
    report_dir = base_dir / "report_results"
    
    # Create folder structure
    folders = {
        'detections_run1': report_dir / "images" / "detections" / "run1",
        'detections_run2': report_dir / "images" / "detections" / "run2",
        'confidence_run1': report_dir / "images" / "confidence_maps" / "run1",
        'confidence_run2': report_dir / "images" / "confidence_maps" / "run2",
        'confusion': report_dir / "images" / "confusion_matrices",
        'plots': report_dir / "images" / "plots",
        'other': report_dir / "images" / "other"
    }
    
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    # Copy Run 1 detections
    run1_dir = base_dir / "Walnut_Research" / "test" / "detections"
    if run1_dir.exists():
        for img_file in run1_dir.glob("*_detections.jpg"):
            shutil.copy2(img_file, folders['detections_run1'] / img_file.name)
        for img_file in run1_dir.glob("*_confidence.png"):
            shutil.copy2(img_file, folders['confidence_run1'] / img_file.name)
    
    # Copy Run 2 detections
    run2_dir = base_dir / "Walnut_Research" / "test" / "detections_v2"
    if run2_dir.exists():
        for img_file in run2_dir.glob("*_detections.jpg"):
            shutil.copy2(img_file, folders['detections_run2'] / img_file.name)
        for img_file in run2_dir.glob("*_confidence.png"):
            shutil.copy2(img_file, folders['confidence_run2'] / img_file.name)
    
    # Copy confusion matrices
    for conf_dir in base_dir.glob("*/confusion_summary.png"):
        shutil.copy2(conf_dir, folders['confusion'] / conf_dir.name)
    
    # Copy plots
    plots_dirs = [
        base_dir / "binary_test_32x16_0.1" / "plots",
        base_dir / "density_test_all_25" / "plots",
        base_dir / "viz"
    ]
    for plots_dir in plots_dirs:
        if plots_dir.exists():
            for img_file in plots_dir.glob("*.png"):
                shutil.copy2(img_file, folders['plots'] / img_file.name)
    
    # Copy other detection images
    other_dirs = [
        base_dir / "detector_direct_results",
        base_dir / "test_single_image"
    ]
    for other_dir in other_dirs:
        if other_dir.exists():
            for img_file in other_dir.glob("*_detections.jpg"):
                shutil.copy2(img_file, folders['other'] / img_file.name)
    
    print(f"✅ Copied result images to {report_dir / 'images'}")

def compile_data_files():
    """Compile all result data files"""
    
    base_dir = Path("/Users/kalpit/TuesdayPresentation")
    report_dir = base_dir / "report_results" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile Run 1 results
    run1_file = base_dir / "Walnut_Research" / "test" / "detections" / "detection_results.json"
    if run1_file.exists():
        shutil.copy2(run1_file, report_dir / "run1_results.json")
    
    # Compile Run 2 results
    run2_file = base_dir / "Walnut_Research" / "test" / "detections_v2" / "detection_results.json"
    if run2_file.exists():
        shutil.copy2(run2_file, report_dir / "run2_results.json")
    
    # Calculate and save accuracy metrics
    from evaluate_detections import evaluate_detection_results
    
    test_dir = base_dir / "Walnut_Research" / "test"
    annotations_dir = test_dir / "annotations"
    
    metrics = {}
    
    # Run 1 metrics
    if run1_file.exists():
        results1 = evaluate_detection_results(
            str(run1_file),
            str(annotations_dir),
            match_distance=40.0
        )
        metrics['run1'] = results1['overall']
        metrics['run1']['per_image'] = results1['per_image']
    
    # Run 2 metrics
    if run2_file.exists():
        results2 = evaluate_detection_results(
            str(run2_file),
            str(annotations_dir),
            match_distance=40.0
        )
        metrics['run2'] = results2['overall']
        metrics['run2']['per_image'] = results2['per_image']
    
    # Save metrics
    with open(report_dir / "accuracy_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create summary
    summary = {
        'generated': datetime.now().isoformat(),
        'test_dataset': {
            'total_images': 10,
            'total_ground_truth': 487
        },
        'run1': {
            'parameters': {
                'patch_size': 32,
                'stride': 16,
                'threshold': 0.5
            },
            'results': metrics.get('run1', {})
        },
        'run2': {
            'parameters': {
                'patch_size': 48,
                'stride': 24,
                'threshold': 0.4
            },
            'results': metrics.get('run2', {})
        }
    }
    
    with open(report_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Compiled data files to {report_dir}")

if __name__ == "__main__":
    print("Organizing results for report...")
    copy_result_images()
    compile_data_files()
    print("✅ Results organization complete!")

