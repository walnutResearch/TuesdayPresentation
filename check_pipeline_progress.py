#!/usr/bin/env python3
"""
Check progress of the walnut counting + GeoJSON pipeline: reports if Step 1 (counts JSON) and Step 2 (GeoJSON)
are complete and shows recent log lines.

How to run:
  python check_pipeline_progress.py

Paths for counts file, GeoJSON file, and log file are set inside check_progress().
"""

import os
import json
from pathlib import Path

def check_progress():
    # Check for output files
    dir_name = "DJI_202509241201_009_GlennDormancyBreaking-2124"
    counts_file = f"/Users/kalpit/TuesdayPresentation/{dir_name}_image_counts.json"
    geojson_file = f"/Users/kalpit/TuesdayPresentation/{dir_name}_walnut_counts.geojson"
    
    print("="*70)
    print("PIPELINE PROGRESS CHECK")
    print("="*70)
    
    # Check counts file
    if os.path.exists(counts_file):
        with open(counts_file, 'r') as f:
            counts_data = json.load(f)
        total_images = len(counts_data)
        total_walnuts = sum(item['predicted_count'] for item in counts_data)
        print(f"\n✅ Step 1 (Counting) - COMPLETE")
        print(f"   Images processed: {total_images}")
        print(f"   Total walnuts: {total_walnuts}")
        print(f"   Average per image: {total_walnuts / total_images:.2f}")
    else:
        print(f"\n⏳ Step 1 (Counting) - IN PROGRESS")
        print(f"   Counts file not yet created: {counts_file}")
    
    # Check GeoJSON file
    if os.path.exists(geojson_file):
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
        total_points = len(geojson_data.get('features', []))
        print(f"\n✅ Step 2 (GeoJSON) - COMPLETE")
        print(f"   Total points: {total_points}")
        print(f"   Output file: {geojson_file}")
    else:
        print(f"\n⏳ Step 2 (GeoJSON) - WAITING or IN PROGRESS")
        print(f"   GeoJSON file not yet created: {geojson_file}")
    
    # Check log file
    log_file = "/Users/kalpit/TuesdayPresentation/pipeline_output.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        print(f"\n📋 Recent log output (last 5 lines):")
        for line in lines[-5:]:
            print(f"   {line.strip()}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    check_progress()

