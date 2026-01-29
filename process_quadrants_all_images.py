#!/usr/bin/env python3
"""
Process all original (full) images by splitting each into 4 quadrants, running the walnut detector on each quadrant,
and streaming/writing per-image predicted counts to JSON/NDJSON. Paths and detector settings are hardcoded.

How to run:
  python process_quadrants_all_images.py

Edit original_dir, model_path, output_file, stream_file, and detector args inside main() as needed.
"""

import json
import os
import warnings
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path
from walnut_detector import WalnutDetector
from tqdm import tqdm

# Suppress multiprocessing warnings (harmless semaphore cleanup warnings)
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

def load_image(image_path):
    """Load a full image and return it as numpy array"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image

def process_full_image(detector, image):
    """Process a full image and return count"""
    if isinstance(image, np.ndarray):
        centers, confidences, _ = detector.detect_walnuts(image)
        if len(centers) > 0:
            centers, confidences = detector.cluster_detections(centers, confidences)
        return len(centers)
    return 0

def main():
    # Paths
    original_dir = '/Volumes/Samsung USB/OneDrive_2026-01-17/WVT- September/'
    model_path = '/Users/kalpit/TuesdayPresentation/models/walnut_classifier.pth'
    # Create unique filenames per run to avoid overwriting previous results
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output = '/Volumes/Samsung USB/OneDrive_2026-01-17/detections'
    output_file = f'{base_output}_{run_id}.json'
    stream_file = f'{base_output}_{run_id}_stream.ndjson'
    
    # Get all original images (exclude macOS metadata files starting with ._)
    original_images = sorted([f for f in os.listdir(original_dir) 
                             if f.endswith('.JPG') and not f.startswith('._')])
    print(f"Found {len(original_images)} original images")
    
    # Initialize detector
    print("\nInitializing detector with threshold 0.5...")
    detector = WalnutDetector(
        model_path=model_path,
        patch_size=32,
        stride=24,
        confidence_threshold=0.6
    )
    
    # Process all images (stream results to disk to avoid RAM growth)
    total_images = len(original_images)
    total_predicted = 0
    processed = 0
    first_five = []
    
    print(f"\nProcessing {total_images} images (full images, no quadrants)...")
    
    # Batch size to reduce run length and resource pressure
    batch_size = 70
    total_batches = (total_images + batch_size - 1) // batch_size
    
    # Open NDJSON stream file (append-safe if re-run with same run_id)
    with open(stream_file, 'a') as sf:
        for batch_idx, start in enumerate(range(0, total_images, batch_size), 1):
            end = min(start + batch_size, total_images)
            batch_images = original_images[start:end]
            print(f"\nBatch {batch_idx}/{total_batches}: images {start+1}-{end}")
            
            for image_name in tqdm(batch_images, desc="Processing batch images", leave=False):
                image_path = os.path.join(original_dir, image_name)
                
                try:
                    # Load full image
                    image = load_image(image_path)
                    
                    # Process full image
                    total_count = process_full_image(detector, image)
                    quadrant_counts = {}  # No quadrants now
                    
                    result_obj = {
                        "image": image_name,
                        "predicted_count": total_count,
                        "quadrant_counts": quadrant_counts
                    }
                    
                except Exception as e:
                    print(f"\nError processing {image_name}: {e}")
                    result_obj = {
                        "image": image_name,
                        "predicted_count": 0,
                        "error": str(e)
                    }
                
                # Stream write one JSON object per line and flush
                sf.write(json.dumps(result_obj) + "\n")
                sf.flush()
                
                # Update counters
                processed += 1
                total_predicted += result_obj["predicted_count"]
                
                # Keep small sample for preview
                if len(first_five) < 5:
                    first_five.append(result_obj)
    
    # Save summary (small JSON) to the original output_file
    summary = {
        "total_images": processed,
        "total_predicted": total_predicted,
        "average_per_image": (total_predicted / processed) if processed else 0,
        "stream_file": stream_file
    }
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {processed}")
    print(f"Total predicted walnuts: {total_predicted}")
    print(f"Average per image: {summary['average_per_image']:.2f}")
    print(f"Results (NDJSON stream): {stream_file}")
    print(f"Summary saved to: {output_file}")
    
    # Show first few results
    print(f"\nFirst 5 results (from stream):")
    for item in first_five:
        print(f"  {item['image']}: {item['predicted_count']} walnuts")

if __name__ == '__main__':
    main()


