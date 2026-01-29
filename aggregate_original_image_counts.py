#!/usr/bin/env python3
"""
Aggregate predicted walnut counts from cropped/quadrant images (e.g. *_q00.JPG) back to original image names
by stripping _qXX and summing counts. Reads glenn_eval_th042/per_image_metrics.json, writes glenn_original_image_counts.json.

How to run:
  python aggregate_original_image_counts.py

Input and output paths are set inside main().
"""

import json
import os
import re
from collections import defaultdict

def extract_base_name(cropped_image_name):
    """Extract base name from cropped image name (remove _qXX suffix)"""
    # Pattern: DJI_20250924120521_0001_D_q00.JPG -> DJI_20250924120521_0001_D.JPG
    # Remove _q followed by 2 digits before .JPG
    pattern = r'_q\d{2}\.JPG$'
    base_name = re.sub(pattern, '.JPG', cropped_image_name)
    return base_name

def main():
    # Read the per_image_metrics.json from evaluation results
    metrics_file = '/Users/kalpit/TuesdayPresentation/glenn_eval_th042/per_image_metrics.json'
    
    with open(metrics_file, 'r') as f:
        cropped_metrics = json.load(f)
    
    # Group cropped images by their base name and sum predicted counts
    original_counts = defaultdict(int)
    
    for item in cropped_metrics:
        cropped_image = item['image']
        predicted_count = item['num_preds']
        
        # Extract base name (original image name)
        base_name = extract_base_name(cropped_image)
        
        # Sum the predicted counts for all cropped parts of the same original image
        original_counts[base_name] += predicted_count
    
    # Convert to list of dictionaries for JSON output
    results = []
    for image_name in sorted(original_counts.keys()):
        results.append({
            "image": image_name,
            "predicted_count": original_counts[image_name]
        })
    
    # Save to JSON file
    output_file = '/Users/kalpit/TuesdayPresentation/glenn_original_image_counts.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Created JSON file with {len(results)} original images")
    print(f"Output saved to: {output_file}")
    print("\nFirst 10 results:")
    for item in results[:10]:
        print(f"  {item['image']}: {item['predicted_count']} walnuts")
    
    # Print summary statistics
    total_predicted = sum(item['predicted_count'] for item in results)
    avg_per_image = total_predicted / len(results) if results else 0
    print(f"\nSummary:")
    print(f"  Total images: {len(results)}")
    print(f"  Total predicted walnuts: {total_predicted}")
    print(f"  Average per image: {avg_per_image:.2f}")

if __name__ == '__main__':
    main()


