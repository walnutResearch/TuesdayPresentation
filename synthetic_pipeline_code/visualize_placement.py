#!/usr/bin/env python3
"""
Visualize placement masks for canopy plates (where walnuts can be placed). Loads canopy images,
builds the same placement mask used in synthesis, and saves overlay images for inspection.

How to run:
  python visualize_placement.py --canopy_plates path/to/canopy_plates [--output_dir placement_visualization]

Use --help for options. Run from synthetic_pipeline_code/.
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def visualize_placement_masks(canopy_plates_dir: str, output_dir: str = "placement_visualization"):
    """Visualize placement masks for canopy plates"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    canopy_files = list(Path(canopy_plates_dir).glob("*.png"))
    
    print(f"🔍 Visualizing placement masks for {len(canopy_files)} canopy plates...")
    
    for i, plate_file in enumerate(canopy_files[:5]):  # Show first 5 plates
        # Load canopy plate
        image = cv2.imread(str(plate_file))
        if image is None:
            continue
        
        # Create placement mask using the same logic as the pipeline
        placement_mask = create_placement_mask(image)
        
        # Create visualization
        vis_image = image.copy()
        
        # Overlay placement mask in green
        mask_overlay = np.zeros_like(image)
        mask_overlay[placement_mask > 128] = [0, 255, 0]  # Green for valid areas
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
        
        # Add text
        cv2.putText(vis_image, f"Green areas = Valid walnut placement", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"placement_mask_{i:03d}.png")
        cv2.imwrite(output_path, vis_image)
        
        print(f"  Saved: {output_path}")
    
    print(f"✅ Placement visualizations saved to: {output_dir}")

def create_placement_mask(image: np.ndarray) -> np.ndarray:
    """Create placement mask using the same logic as the pipeline"""
    
    # Convert to different color spaces for better analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask for green foliage areas
    lower_green = np.array([35, 40, 40])   # Lower bound for green
    upper_green = np.array([85, 255, 255]) # Upper bound for green
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Create mask for bright areas (avoid shadows)
    brightness_threshold = 80
    bright_mask = gray > brightness_threshold
    
    # Create mask for areas with good contrast
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
    local_std = np.sqrt(local_variance)
    texture_mask = local_std > 15
    
    # Combine all conditions
    placement_mask = green_mask & bright_mask & texture_mask
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    placement_mask = cv2.morphologyEx(placement_mask, cv2.MORPH_CLOSE, kernel)
    placement_mask = cv2.morphologyEx(placement_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small regions
    contours, _ = cv2.findContours(placement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    
    final_mask = np.zeros_like(placement_mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.fillPoly(final_mask, [contour], 255)
    
    return final_mask

def visualize_synthetic_placements(synthetic_images_dir: str, labels_dir: str, 
                                 output_dir: str = "synthetic_visualization"):
    """Visualize walnut placements in synthetic images"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = list(Path(synthetic_images_dir).glob("*.png"))
    
    print(f"🔍 Visualizing walnut placements in {len(image_files)} synthetic images...")
    
    for i, image_file in enumerate(image_files[:10]):  # Show first 10 images
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        # Load labels
        label_file = Path(labels_dir) / f"label_{image_file.stem.split('_')[-1]}.txt"
        if not label_file.exists():
            continue
        
        # Read walnut centers
        walnut_centers = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = int(float(parts[0])), int(float(parts[1]))
                        walnut_centers.append((x, y))
        
        # Create visualization
        vis_image = image.copy()
        
        # Draw walnut centers
        for j, (x, y) in enumerate(walnut_centers):
            cv2.circle(vis_image, (x, y), 8, (0, 255, 0), 2)  # Green circles
            cv2.circle(vis_image, (x, y), 2, (0, 0, 255), -1)  # Red centers
            cv2.putText(vis_image, str(j+1), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add info
        cv2.putText(vis_image, f"Walnuts: {len(walnut_centers)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"synthetic_placement_{i:03d}.png")
        cv2.imwrite(output_path, vis_image)
        
        print(f"  Saved: {output_path}")
    
    print(f"✅ Synthetic placement visualizations saved to: {output_dir}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Visualize walnut placement")
    parser.add_argument("--canopy_plates", default="./synthetic_data/canopy_plates",
                       help="Path to canopy plates directory")
    parser.add_argument("--synthetic_images", default="./synthetic_data/synthetic_images",
                       help="Path to synthetic images directory")
    parser.add_argument("--labels", default="./synthetic_data/labels",
                       help="Path to labels directory")
    parser.add_argument("--output", default="./placement_visualization",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    print("🔍 Walnut Placement Visualization")
    print("=" * 40)
    
    # Visualize placement masks
    visualize_placement_masks(args.canopy_plates, os.path.join(args.output, "masks"))
    
    # Visualize synthetic placements
    if os.path.exists(args.synthetic_images) and os.path.exists(args.labels):
        visualize_synthetic_placements(args.synthetic_images, args.labels, 
                                     os.path.join(args.output, "synthetic"))
    
    print("\n✅ Visualization complete!")
    print(f"📁 Check the '{args.output}' directory for visualizations")

if __name__ == "__main__":
    main()
