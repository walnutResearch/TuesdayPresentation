#!/usr/bin/env python3
"""
Process all original (full) images: use existing per-quadrant evaluation metrics where available to sum counts
per original image; for any originals not covered, run the detector on full images. Writes glenn_original_image_counts.json.

How to run:
  python process_all_original_images.py

Paths (original_dir, metrics_file, model_path, output_file) are set inside main().
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from walnut_detector import WalnutDetector

def extract_base_name(cropped_image_name):
    """Extract base name from cropped image name (remove _qXX suffix)"""
    pattern = r'_q\d{2}\.JPG$'
    base_name = re.sub(pattern, '.JPG', cropped_image_name)
    return base_name

def main():
    # Paths
    original_dir = '/Users/kalpit/TuesdayPresentation/GlennDormancyRootstock/GlennDormancyBreakingRGB'
    metrics_file = '/Users/kalpit/TuesdayPresentation/glenn_eval_th042/per_image_metrics.json'
    model_path = '/Users/kalpit/TuesdayPresentation/models_precision/walnut_classifier_best_precision.pth'
    output_file = '/Users/kalpit/TuesdayPresentation/glenn_original_image_counts.json'
    
    # Get all original images
    original_images = sorted([f for f in os.listdir(original_dir) if f.endswith('.JPG')])
    print(f"Found {len(original_images)} original images")
    
    # Load existing evaluation results
    with open(metrics_file, 'r') as f:
        cropped_metrics = json.load(f)
    
    # Group cropped images by their base name and sum predicted counts
    evaluated_counts = defaultdict(int)
    evaluated_images = set()
    
    for item in cropped_metrics:
        cropped_image = item['image']
        predicted_count = item['num_preds']
        base_name = extract_base_name(cropped_image)
        evaluated_counts[base_name] += predicted_count
        evaluated_images.add(base_name)
    
    print(f"Found evaluation results for {len(evaluated_images)} images")
    
    # Initialize detector for processing remaining images
    print("\nInitializing detector...")
    detector = WalnutDetector(
        model_path=model_path,
        patch_size=32,
        stride=16,
        confidence_threshold=0.42  # Using the best threshold we found
    )
    
    # Process all images
    results = []
    processed_count = 0
    new_count = 0
    
    print(f"\nProcessing {len(original_images)} images...")
    for image_name in original_images:
        image_path = os.path.join(original_dir, image_name)
        
        if image_name in evaluated_images:
            # Use existing evaluation results
            predicted_count = evaluated_counts[image_name]
            processed_count += 1
            print(f"  [{processed_count}/{len(original_images)}] {image_name}: {predicted_count} (from evaluation)")
        else:
            # Process with detector
            try:
                result = detector.process_image(image_path, output_dir=None, cluster=True)
                predicted_count = result['num_walnuts']
                new_count += 1
                processed_count += 1
                print(f"  [{processed_count}/{len(original_images)}] {image_name}: {predicted_count} (new detection)")
            except Exception as e:
                print(f"  Error processing {image_name}: {e}")
                predicted_count = 0
        
        results.append({
            "image": image_name,
            "predicted_count": predicted_count
        })
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    print(f"Images with existing results: {len(evaluated_images)}")
    print(f"Images newly processed: {new_count}")
    print(f"Output saved to: {output_file}")
    
    # Summary statistics
    total_predicted = sum(item['predicted_count'] for item in results)
    avg_per_image = total_predicted / len(results) if results else 0
    print(f"\nSummary Statistics:")
    print(f"  Total predicted walnuts: {total_predicted}")
    print(f"  Average per image: {avg_per_image:.2f}")

if __name__ == '__main__':
    main()


