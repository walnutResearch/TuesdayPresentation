#!/usr/bin/env python3
"""
Run WalnutDetector on all images in the WVT directory and create per-image GeoJSON files.
Each image gets its own GeoJSON file containing one point per detected walnut at the image's GPS coordinates.
Uses WalnutDetector (sliding window + clustering); extracts GPS from EXIF.

How to run:
  python run_wvt_detector_geojson.py

Edit model_path, image_dir, output_dir, patch_size, stride, threshold, and start_index at top of run_wvt_detector_geojson() if needed.
"""

import os
import json
from datetime import datetime

import cv2
from tqdm import tqdm

from walnut_detector import WalnutDetector
from run_full_pipeline import get_image_metadata


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------

def run_wvt_detector_geojson():
    # Configuration
    model_path = "/Users/kalpit/TuesdayPresentation/models/walnut_classifier.pth"
    image_dir = "/Volumes/Samsung USB/OneDrive_2026-01-17/WVT- September/"
    output_dir = "/Volumes/Samsung USB/OneDrive_2026-01-17/detections"

    patch_size = 32
    stride = 24
    threshold = 0.6

    os.makedirs(output_dir, exist_ok=True)

    # Collect images (ignore macOS metadata files)
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.endswith(".JPG") and not f.startswith("._")
    )

    print(f"Found {len(image_files)} images in {image_dir}")
    print(f"Model: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size: {patch_size}, stride: {stride}, threshold: {threshold}")

    # Initialize detector (device='auto' lets WalnutDetector pick best device)
    detector = WalnutDetector(
        model_path=model_path,
        patch_size=patch_size,
        stride=stride,
        confidence_threshold=threshold,
    )

    # Option: resume from a specific image index (1-based). Set to 1 to start from beginning.
    # Here we resume AFTER the 1169th image, i.e. start at 1170.
    start_index = 1170
    images_to_process = image_files[start_index - 1 :]

    print(f"\nResuming from image {start_index} of {len(image_files)} "
          f"({len(images_to_process)} images to process)...")

    # Process each image and write a per-image GeoJSON
    for idx, image_name in enumerate(
        tqdm(images_to_process, desc="Processing images"), start_index
    ):
        image_path = os.path.join(image_dir, image_name)

        # Get GPS coordinates
        lon, lat = get_image_metadata(image_path)
        if lon is None or lat is None:
            tqdm.write(f"[{idx}/{len(image_files)}] Skipping {image_name}: no GPS coordinates found")
            continue

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            tqdm.write(f"[{idx}/{len(image_files)}] Skipping {image_name}: could not load image")
            continue

        # Detect walnuts on the full image
        centers, confidences, _ = detector.detect_walnuts(img)
        if len(centers) > 0:
            centers, confidences = detector.cluster_detections(centers, confidences)

        walnut_count = len(centers)

        # Build per-image GeoJSON: one point per detected walnut at image GPS
        features = []
        for walnut_idx in range(1, walnut_count + 1):
            feature = {
                "type": "Feature",
                "properties": {
                    "attachmentName": [],
                    "attachmentPath": [],
                    "color": "#40c4ff",
                    "creationDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "description": f"Walnut {walnut_idx} of {walnut_count} from {image_name}",
                    "fill": 1.0,
                    "name": f"{image_name.replace('.JPG', '')} - Walnut {walnut_idx}",
                    "visualType": "ANNOTATION",
                    "walnut_number": walnut_idx,
                    "total_walnuts_in_image": walnut_count,
                    "image_name": image_name,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "name": image_name,
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                },
            },
            "features": features,
        }

        out_name = os.path.splitext(image_name)[0] + "_walnuts.geojson"
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, "w") as f:
            json.dump(geojson, f, indent=2)

    print("\nDone. Per-image GeoJSON files saved to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    run_wvt_detector_geojson()


