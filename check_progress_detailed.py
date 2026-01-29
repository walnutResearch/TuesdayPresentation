#!/usr/bin/env python3
"""
Check detailed progress of the walnut counting/GeoJSON pipeline: whether the process is running, last progress
from pipeline_output.log, and status of counts JSON and GeoJSON output files.

How to run:
  python check_progress_detailed.py

Paths (counts file, geojson file, log file) are set inside check_progress().
"""

import os
import re
import json
from pathlib import Path

def parse_log_for_progress(log_file):
    """Parse log file to extract progress information"""
    if not os.path.exists(log_file):
        return None, None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Look for "Processing images:" progress
    image_progress = None
    for line in reversed(lines):
        if "Processing images:" in line:
            # Extract progress like "Processing images:  15%|█▌| 15/1384"
            match = re.search(r'Processing images:\s+(\d+)%\|.*?(\d+)/(\d+)', line)
            if match:
                percent = int(match.group(1))
                current = int(match.group(2))
                total = int(match.group(3))
                image_progress = (current, total, percent)
                break
    
    # Get last few lines for context
    recent_lines = lines[-10:] if len(lines) >= 10 else lines
    
    return image_progress, recent_lines

def check_progress():
    dir_name = "DJI_202509241201_009_GlennDormancyBreaking-2124"
    counts_file = f"/Users/kalpit/TuesdayPresentation/{dir_name}_image_counts.json"
    geojson_file = f"/Users/kalpit/TuesdayPresentation/{dir_name}_walnut_counts.geojson"
    log_file = "/Users/kalpit/TuesdayPresentation/pipeline_output.log"
    
    print("="*70)
    print("DETAILED PIPELINE PROGRESS")
    print("="*70)
    
    # Check if process is running
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'run_full_pipeline.py'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("\n✅ Pipeline process is RUNNING")
    else:
        print("\n⚠️  Pipeline process NOT running (may have completed or crashed)")
    
    # Check Step 1: Counting
    print("\n" + "-"*70)
    print("STEP 1: Counting Walnuts")
    print("-"*70)
    
    if os.path.exists(counts_file):
        with open(counts_file, 'r') as f:
            counts_data = json.load(f)
        total_images = len(counts_data)
        total_walnuts = sum(item['predicted_count'] for item in counts_data)
        print(f"✅ Status: COMPLETE")
        print(f"   Images processed: {total_images}/1384 ({total_images*100/1384:.1f}%)")
        print(f"   Total walnuts: {total_walnuts:,}")
        print(f"   Average per image: {total_walnuts / total_images:.2f}")
    else:
        print("⏳ Status: IN PROGRESS")
        # Try to parse log for progress
        image_progress, recent_lines = parse_log_for_progress(log_file)
        if image_progress:
            current, total, percent = image_progress
            print(f"   Progress: {current}/{total} images ({percent}%)")
            estimated_remaining = (total - current) * 7  # ~7 seconds per image
            hours = estimated_remaining // 3600
            minutes = (estimated_remaining % 3600) // 60
            print(f"   Estimated time remaining: ~{hours}h {minutes}m")
        else:
            print("   Processing first images...")
    
    # Check Step 2: GeoJSON
    print("\n" + "-"*70)
    print("STEP 2: Generating GeoJSON")
    print("-"*70)
    
    if os.path.exists(geojson_file):
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
        total_points = len(geojson_data.get('features', []))
        print(f"✅ Status: COMPLETE")
        print(f"   Total points: {total_points:,}")
        print(f"   Output file: {geojson_file}")
    else:
        if os.path.exists(counts_file):
            print("⏳ Status: WAITING (Step 1 complete, starting Step 2...)")
        else:
            print("⏳ Status: WAITING (Step 1 must complete first)")
    
    # Show recent log activity
    print("\n" + "-"*70)
    print("RECENT ACTIVITY (last 5 lines from log)")
    print("-"*70)
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        for line in lines[-5:]:
            # Clean up ANSI escape codes
            cleaned = re.sub(r'\x1b\[[0-9;]*m', '', line.strip())
            if cleaned:
                print(f"   {cleaned}")
    else:
        print("   No log file found")
    
    print("\n" + "="*70)
    print("💡 Tip: Run this script again to check updated progress")
    print("="*70)

if __name__ == '__main__':
    check_progress()

