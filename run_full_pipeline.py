#!/usr/bin/env python3
"""
Complete pipeline: (1) count walnuts per image by dividing each image into quadrants and running the detector
on each quadrant with threshold 0.42; (2) generate a GeoJSON with one point per walnut at each image's GPS.
Saves per-image counts JSON and a single GeoJSON. Also provides get_image_metadata() for GPS from EXIF.

How to run:
  python run_full_pipeline.py [--image_dir path/to/images]

Default image_dir is set in argparse. Counts JSON and GeoJSON paths are derived from the directory name.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from walnut_detector import WalnutDetector
from tqdm import tqdm

# ============================================================================
# GPS Extraction Functions
# ============================================================================

def get_decimal_from_dms(dms, ref):
    """Convert degrees, minutes, seconds to decimal degrees"""
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    
    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds
        
    return degrees + minutes + seconds

def get_exif_location(exifdata):
    """Extract GPS coordinates from EXIF data"""
    # GPSInfo tag ID is 34853
    gps_tag_id = 34853
    
    if gps_tag_id not in exifdata:
        return None, None
    
    try:
        gps_data = exifdata.get_ifd(gps_tag_id)
    except Exception:
        return None, None
    
    if not gps_data:
        return None, None
    
    # Extract latitude
    lat_ref = gps_data.get(1)
    lat_dms = gps_data.get(2)
    
    lat = None
    if lat_dms and lat_ref:
        if isinstance(lat_dms, tuple) and len(lat_dms) == 3:
            lat = get_decimal_from_dms(lat_dms, lat_ref)
    
    # Extract longitude
    lon_ref = gps_data.get(3)
    lon_dms = gps_data.get(4)
    
    lon = None
    if lon_dms and lon_ref:
        if isinstance(lon_dms, tuple) and len(lon_dms) == 3:
            lon = get_decimal_from_dms(lon_dms, lon_ref)
    
    return lon, lat

def get_image_metadata(image_path):
    """Extract GPS coordinates from image"""
    try:
        image = Image.open(image_path)
        exifdata = image.getexif()
        
        if exifdata is None:
            return None, None
        
        lon, lat = get_exif_location(exifdata)
        return lon, lat
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None, None

# ============================================================================
# Quadrant Processing Functions
# ============================================================================

def divide_into_quadrants(image_path):
    """Divide an image into 4 quadrants and return them"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    
    # Calculate quadrant boundaries
    mid_h = h // 2
    mid_w = w // 2
    
    # Extract 4 quadrants
    q00 = image[0:mid_h, 0:mid_w]  # Top-left
    q01 = image[0:mid_h, mid_w:w]   # Top-right
    q10 = image[mid_h:h, 0:mid_w]   # Bottom-left
    q11 = image[mid_h:h, mid_w:w]   # Bottom-right
    
    return {
        'q00': q00,
        'q01': q01,
        'q10': q10,
        'q11': q11
    }

def process_quadrant(detector, quadrant_image, quadrant_name):
    """Process a single quadrant and return count"""
    if isinstance(quadrant_image, np.ndarray):
        centers, confidences, _ = detector.detect_walnuts(quadrant_image)
        
        # Cluster detections
        if len(centers) > 0:
            centers, confidences = detector.cluster_detections(centers, confidences)
        
        return len(centers)
    return 0

# ============================================================================
# Main Pipeline
# ============================================================================

def step1_count_walnuts(original_dir=None):
    """Step 1: Count walnuts in all images by processing quadrants"""
    print("\n" + "="*70)
    print("STEP 1: Counting Walnuts")
    print("="*70)
    
    # Paths
    if original_dir is None:
        original_dir = '/Users/kalpit/TuesdayPresentation/GlennDormancyRootstock/GlennDormancyBreakingRGB'
    model_path = '/Users/kalpit/TuesdayPresentation/models_precision/walnut_classifier_best_precision.pth'
    
    # Generate output filename based on directory name
    dir_name = os.path.basename(original_dir.rstrip('/'))
    output_file = f'/Users/kalpit/TuesdayPresentation/{dir_name}_image_counts.json'
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Get all original images
    original_images = sorted([f for f in os.listdir(original_dir) if f.endswith('.JPG')])
    print(f"Found {len(original_images)} original images")
    
    # Initialize detector
    print("\nInitializing detector with threshold 0.42...")
    detector = WalnutDetector(
        model_path=model_path,
        patch_size=32,
        stride=16,
        confidence_threshold=0.42
    )
    
    # Process all images
    results = []
    
    print(f"\nProcessing {len(original_images)} images (dividing into 4 quadrants each)...")
    for image_name in tqdm(original_images, desc="Processing images"):
        image_path = os.path.join(original_dir, image_name)
        
        try:
            # Divide into quadrants
            quadrants = divide_into_quadrants(image_path)
            
            # Process each quadrant
            total_count = 0
            quadrant_counts = {}
            
            for q_name, q_image in quadrants.items():
                if q_image.size > 0:
                    count = process_quadrant(detector, q_image, q_name)
                    quadrant_counts[q_name] = count
                    total_count += count
                else:
                    quadrant_counts[q_name] = 0
            
            results.append({
                "image": image_name,
                "predicted_count": total_count,
                "quadrant_counts": quadrant_counts
            })
            
        except Exception as e:
            print(f"\nError processing {image_name}: {e}")
            results.append({
                "image": image_name,
                "predicted_count": 0,
                "quadrant_counts": {"q00": 0, "q01": 0, "q10": 0, "q11": 0},
                "error": str(e)
            })
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Step 1 Complete!")
    print(f"{'='*70}")
    print(f"Total images processed: {len(results)}")
    print(f"Output saved to: {output_file}")
    
    # Summary statistics
    total_predicted = sum(item['predicted_count'] for item in results)
    avg_per_image = total_predicted / len(results) if results else 0
    print(f"\nSummary Statistics:")
    print(f"  Total predicted walnuts: {total_predicted}")
    print(f"  Average per image: {avg_per_image:.2f}")
    
    return output_file

def step2_generate_geojson(counts_file, image_dir=None):
    """Step 2: Generate GeoJSON with one point per walnut"""
    print("\n" + "="*70)
    print("STEP 2: Generating GeoJSON")
    print("="*70)
    
    # Paths
    if image_dir is None:
        image_dir = '/Users/kalpit/TuesdayPresentation/GlennDormancyRootstock/GlennDormancyBreakingRGB'
    
    # Generate output filename based on directory name
    dir_name = os.path.basename(image_dir.rstrip('/'))
    output_file = f'/Users/kalpit/TuesdayPresentation/{dir_name}_walnut_counts.geojson'
    
    # Load walnut counts
    with open(counts_file, 'r') as f:
        counts_data = json.load(f)
    
    # Create a dictionary mapping image names to counts
    counts_dict = {item['image']: item['predicted_count'] for item in counts_data}
    
    # Get all images
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG')])
    
    print(f"Processing {len(image_files)} images for GPS extraction...")
    
    # Create GeoJSON features
    features = []
    missing_coords = []
    
    for image_name in tqdm(image_files, desc="Extracting GPS and creating points"):
        image_path = os.path.join(image_dir, image_name)
        
        # Get GPS coordinates
        lon, lat = get_image_metadata(image_path)
        
        if lon is None or lat is None:
            missing_coords.append(image_name)
            continue
        
        # Get walnut count
        walnut_count = counts_dict.get(image_name, 0)
        
        # Create one point for each walnut at this image's GPS location
        for walnut_num in range(1, walnut_count + 1):
            feature = {
                "type": "Feature",
                "properties": {
                    "attachmentName": [],
                    "attachmentPath": [],
                    "color": "#40c4ff",
                    "creationDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "description": f"Walnut {walnut_num} of {walnut_count} from {image_name}",
                    "fill": 1.0,
                    "name": f"{image_name.replace('.JPG', '')} - Walnut {walnut_num}",
                    "visualType": "ANNOTATION",
                    "walnut_number": walnut_num,
                    "total_walnuts_in_image": walnut_count,
                    "image_name": image_name
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            }
            
            features.append(feature)
    
    # Create GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "name": "Walnut Counts",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": features
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Step 2 Complete!")
    print(f"{'='*70}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Images with GPS coordinates: {len(image_files) - len(missing_coords)}")
    print(f"Images without GPS coordinates: {len(missing_coords)}")
    print(f"Output saved to: {output_file}")
    
    if missing_coords:
        print(f"\n⚠️  Images without GPS coordinates:")
        for img in missing_coords[:10]:
            print(f"  - {img}")
        if len(missing_coords) > 10:
            print(f"  ... and {len(missing_coords) - 10} more")
    
    # Summary
    total_walnuts = len(features)
    total_images = len(image_files) - len(missing_coords)
    print(f"\nSummary:")
    print(f"  Total walnut points: {total_walnuts}")
    print(f"  Total images with GPS: {total_images}")
    print(f"  Average walnuts per image: {total_walnuts / total_images:.2f}" if total_images > 0 else "  Average walnuts per image: 0")
    
    return output_file

def main():
    """Run the complete pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run walnut counting and GeoJSON generation pipeline")
    parser.add_argument("--image_dir", type=str, 
                       default='/Users/kalpit/TuesdayPresentation/GlennDormancyRootstock/GlennDormancyBreakingRGB',
                       help="Directory containing images to process")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("WALNUT COUNTING AND GEOJSON GENERATION PIPELINE")
    print("="*70)
    print(f"Processing images from: {args.image_dir}")
    
    try:
        # Step 1: Count walnuts
        counts_file = step1_count_walnuts(args.image_dir)
        
        # Step 2: Generate GeoJSON
        geojson_file = step2_generate_geojson(counts_file, args.image_dir)
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print(f"Counts JSON: {counts_file}")
        print(f"GeoJSON: {geojson_file}")
        print("\nYou can now use the GeoJSON file in GIS applications!")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

