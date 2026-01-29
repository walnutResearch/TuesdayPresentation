#!/usr/bin/env python3
"""
Combine annotations from 4 quadrants (q00, q01, q10, q11) into one annotation file for the original full image.
Maps quadrant (x,y) coordinates to full-image coordinates and writes a single TXT file.

How to run:
  Use as a module: from combine_quadrant_annotations import combine_quadrant_annotations
  Or run the main block at the bottom with paths set for main_image_path, quadrant_annotations dict, output_path.
"""

import os
from pathlib import Path

def parse_annotation_file(annotation_path):
    """Parse annotation file and return list of (x, y) coordinates."""
    walnuts = []
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
                    x = int(parts[0])
                    y = int(parts[1])
                    walnuts.append((x, y))
                except ValueError:
                    continue
    return walnuts

def combine_quadrant_annotations(main_image_path, quadrant_annotations, output_path):
    """
    Combine annotations from 4 quadrants into a single annotation file.
    
    Args:
        main_image_path: Path to main image (to get dimensions)
        quadrant_annotations: Dict with keys 'q00', 'q01', 'q10', 'q11' mapping to annotation file paths
        output_path: Path to save combined annotation file
    """
    from PIL import Image
    
    # Get main image dimensions
    img = Image.open(main_image_path)
    main_width = img.width
    main_height = img.height
    
    # Calculate quadrant offsets
    mid_w = main_width // 2
    mid_h = main_height // 2
    
    # Define coordinate transformations for each quadrant
    transformations = {
        'q00': (0, 0),           # Top-left: no offset
        'q01': (mid_w, 0),       # Top-right: add width offset
        'q10': (0, mid_h),       # Bottom-left: add height offset
        'q11': (mid_w, mid_h)    # Bottom-right: add both offsets
    }
    
    # Parse and transform annotations from each quadrant
    all_walnuts = []
    quadrant_counts = {}
    
    for quadrant, annotation_path in quadrant_annotations.items():
        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file not found: {annotation_path}")
            continue
        
        walnuts = parse_annotation_file(annotation_path)
        offset_x, offset_y = transformations[quadrant]
        
        # Transform coordinates
        transformed_walnuts = [(x + offset_x, y + offset_y) for x, y in walnuts]
        all_walnuts.extend(transformed_walnuts)
        quadrant_counts[quadrant] = len(walnuts)
        
        print(f"{quadrant}: {len(walnuts)} walnuts")
    
    # Sort by y coordinate, then x coordinate for consistent ordering
    all_walnuts.sort(key=lambda p: (p[1], p[0]))
    
    # Write combined annotation file
    with open(output_path, 'w') as f:
        f.write(f"# Walnut center annotations (x, y) coordinates\n")
        f.write(f"# Image: {os.path.basename(main_image_path)}\n")
        f.write(f"# Image size: {main_width}x{main_height}\n")
        f.write(f"# Total walnuts: {len(all_walnuts)}\n")
        f.write(f"# Quadrant counts: q00={quadrant_counts.get('q00', 0)}, "
                f"q01={quadrant_counts.get('q01', 0)}, "
                f"q10={quadrant_counts.get('q10', 0)}, "
                f"q11={quadrant_counts.get('q11', 0)}\n")
        f.write(f"# Format: x y\n")
        
        for x, y in all_walnuts:
            f.write(f"{x} {y}\n")
    
    print(f"\n✅ Combined annotation saved to: {output_path}")
    print(f"   Total walnuts: {len(all_walnuts)}")
    return output_path

if __name__ == "__main__":
    # Define paths
    main_image = "/Users/kalpit/TuesdayPresentation/DJI_202509241201_009_GlennDormancyBreaking-2124/DJI_20250924120521_0001_D.JPG"
    
    quadrant_annotations = {
        'q00': "/Users/kalpit/TuesdayPresentation/GlennDormancyRootstock/annotated_images/DJI_20250924120521_0001_D_q00.txt",
        'q01': "/Users/kalpit/TuesdayPresentation/GlennDormancyRootstock/annotated_images/DJI_20250924120521_0001_D_q01.txt",
        'q10': "/Users/kalpit/TuesdayPresentation/GlennDormancyRootstock/annotated_images/DJI_20250924120521_0001_D_q10.txt",
        'q11': "/Users/kalpit/TuesdayPresentation/GlennDormancyRootstock/annotated_images/DJI_20250924120521_0001_D_q11.txt"
    }
    
    # Determine output path (same directory as main image, with .txt extension)
    main_image_dir = os.path.dirname(main_image)
    main_image_basename = os.path.splitext(os.path.basename(main_image))[0]
    output_path = os.path.join(main_image_dir, f"{main_image_basename}.txt")
    
    combine_quadrant_annotations(main_image, quadrant_annotations, output_path)


