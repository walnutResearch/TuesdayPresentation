#!/usr/bin/env python3
"""
Synthetic walnut pipeline: extract walnut instances from annotated images, build canopy background
plates (walnut-free), and produce metadata/cutouts for synthetic_generator. Creates realistic training
data for the density model.

How to run:
  python synthetic_pipeline.py --train_data path/to/annotated_dataset --output path/to/output [--num_canopy_plates 30]

Use --help for all options. Run from synthetic_pipeline_code/ or ensure parent path is on PYTHONPATH.
"""

import os
import sys
import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from dataclasses import dataclass
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class WalnutInstance:
    """Data class to store walnut instance information"""
    image_path: str
    center_x: int
    center_y: int
    diameter: float
    brightness: float
    alpha_mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h

@dataclass
class CanopyPlate:
    """Data class to store canopy background plate information"""
    image_path: str
    image: np.ndarray
    mask: np.ndarray  # Areas where walnuts can be placed

class WalnutExtractor:
    """Extract individual walnut instances from annotated training data"""
    
    def __init__(self, train_data_dir: str, output_dir: str):
        self.train_data_dir = train_data_dir
        self.output_dir = output_dir
        self.walnut_instances: List[WalnutInstance] = []
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'walnut_cutouts'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
        
    def extract_walnuts(self, patch_size: int = 32, min_diameter: int = 4, max_diameter: int = 16):
        """
        Extract individual walnut instances from training images
        
        Args:
            patch_size: Size of patches to extract around each walnut
            min_diameter: Minimum walnut diameter to consider
            max_diameter: Maximum walnut diameter to consider
        """
        print("🔍 Extracting walnut instances from training data...")
        
        train_images_dir = os.path.join(self.train_data_dir, 'train', 'images')
        train_annotations_dir = os.path.join(self.train_data_dir, 'train', 'annotations')
        
        if not os.path.exists(train_images_dir) or not os.path.exists(train_annotations_dir):
            raise ValueError("Training data directories not found")
        
        # Get all image files
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            image_files.extend(Path(train_images_dir).glob(f'*{ext}'))
        
        total_walnuts = 0
        
        for image_file in image_files:
            image_name = image_file.stem
            annotation_file = os.path.join(train_annotations_dir, f"{image_name}.txt")
            
            if not os.path.exists(annotation_file):
                continue
                
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                continue
                
            # Load annotations
            walnuts = self._load_annotations(annotation_file)
            
            # Extract each walnut
            for i, (x, y) in enumerate(walnuts):
                walnut_instance = self._extract_single_walnut(
                    image, x, y, patch_size, min_diameter, max_diameter, 
                    f"{image_name}_walnut_{i:03d}"
                )
                
                if walnut_instance is not None:
                    self.walnut_instances.append(walnut_instance)
                    total_walnuts += 1
        
        print(f"✅ Extracted {total_walnuts} walnut instances")
        
        # Save metadata
        self._save_metadata()
        
        return self.walnut_instances
    
    def _load_annotations(self, annotation_file: str) -> List[Tuple[int, int]]:
        """Load annotation coordinates from file"""
        walnuts = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = int(float(parts[0])), int(float(parts[1]))
                        walnuts.append((x, y))
        return walnuts
    
    def _extract_single_walnut(self, image: np.ndarray, center_x: int, center_y: int, 
                              patch_size: int, min_diameter: int, max_diameter: int, 
                              name: str) -> Optional[WalnutInstance]:
        """Extract a single walnut instance with alpha mask"""
        
        h, w = image.shape[:2]
        
        # Calculate patch boundaries
        half_size = patch_size // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w, center_x + half_size)
        y2 = min(h, center_y + half_size)
        
        # Extract patch
        patch = image[y1:y2, x1:x2].copy()
        patch_h, patch_w = patch.shape[:2]
        
        # Create alpha mask using grabcut
        alpha_mask = self._create_alpha_mask(patch, center_x - x1, center_y - y1)
        
        # Calculate actual walnut diameter from mask
        diameter = self._calculate_diameter(alpha_mask)
        
        if diameter < min_diameter or diameter > max_diameter:
            return None
        
        # Calculate brightness
        brightness = self._calculate_brightness(patch, alpha_mask)
        
        # Save walnut cutout
        output_path = os.path.join(self.output_dir, 'walnut_cutouts', f"{name}.png")
        self._save_walnut_cutout(patch, alpha_mask, output_path)
        
        # Calculate bounding box
        bbox = self._calculate_bbox(alpha_mask)
        
        return WalnutInstance(
            image_path=output_path,
            center_x=center_x - x1,  # Relative to patch
            center_y=center_y - y1,  # Relative to patch
            diameter=diameter,
            brightness=brightness,
            alpha_mask=alpha_mask,
            bbox=bbox
        )
    
    def _create_alpha_mask(self, patch: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
        """Create alpha mask for walnut using grabcut"""
        
        h, w = patch.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        
        # Create initial mask with circle around center
        radius = min(15, min(h, w) // 4)
        cv2.circle(mask, (center_x, center_y), radius, cv2.GC_FGD, -1)
        cv2.circle(mask, (center_x, center_y), radius * 2, cv2.GC_PR_FGD, -1)
        
        # Set background
        mask[mask == 0] = cv2.GC_BGD
        
        # Apply grabcut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(patch, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        except:
            # Fallback to simple circle if grabcut fails
            mask = np.zeros((h, w), np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            return mask
        
        # Create final alpha mask
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        
        # Smooth the mask
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        return alpha
    
    def _calculate_diameter(self, alpha_mask: np.ndarray) -> float:
        """Calculate walnut diameter from alpha mask"""
        contours, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate equivalent diameter
        area = cv2.contourArea(largest_contour)
        diameter = 2 * math.sqrt(area / math.pi)
        
        return diameter
    
    def _calculate_brightness(self, patch: np.ndarray, alpha_mask: np.ndarray) -> float:
        """Calculate average brightness of walnut region"""
        mask_bool = alpha_mask > 128
        if not np.any(mask_bool):
            return 0
        
        # Convert to grayscale for brightness calculation
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray[mask_bool])
        
        return brightness
    
    def _calculate_bbox(self, alpha_mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate bounding box of walnut in alpha mask"""
        contours, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, 0, 0)
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
    
    def _save_walnut_cutout(self, patch: np.ndarray, alpha_mask: np.ndarray, output_path: str):
        """Save walnut cutout with alpha channel"""
        # Convert BGR to RGBA
        rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha_mask
        
        # Save as PNG with alpha
        cv2.imwrite(output_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    
    def _save_metadata(self):
        """Save metadata about extracted walnuts"""
        metadata = {
            'total_walnuts': len(self.walnut_instances),
            'diameters': [w.diameter for w in self.walnut_instances],
            'brightnesses': [w.brightness for w in self.walnut_instances],
            'instances': []
        }
        
        for i, walnut in enumerate(self.walnut_instances):
            instance_data = {
                'id': i,
                'image_path': walnut.image_path,
                'center_x': walnut.center_x,
                'center_y': walnut.center_y,
                'diameter': walnut.diameter,
                'brightness': walnut.brightness,
                'bbox': walnut.bbox
            }
            metadata['instances'].append(instance_data)
        
        # Calculate statistics
        if metadata['diameters']:
            metadata['diameter_stats'] = {
                'mean': np.mean(metadata['diameters']),
                'std': np.std(metadata['diameters']),
                'min': np.min(metadata['diameters']),
                'max': np.max(metadata['diameters']),
                'median': np.median(metadata['diameters'])
            }
        
        if metadata['brightnesses']:
            metadata['brightness_stats'] = {
                'mean': np.mean(metadata['brightnesses']),
                'std': np.std(metadata['brightnesses']),
                'min': np.min(metadata['brightnesses']),
                'max': np.max(metadata['brightnesses']),
                'median': np.median(metadata['brightnesses'])
            }
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata', 'walnut_instances.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Walnut Statistics:")
        if 'diameter_stats' in metadata:
            stats = metadata['diameter_stats']
            print(f"  Diameter: {stats['mean']:.1f} ± {stats['std']:.1f} px (range: {stats['min']:.1f}-{stats['max']:.1f})")
        if 'brightness_stats' in metadata:
            stats = metadata['brightness_stats']
            print(f"  Brightness: {stats['mean']:.1f} ± {stats['std']:.1f} (range: {stats['min']:.1f}-{stats['max']:.1f})")

class CanopyPlateGenerator:
    """Generate canopy background plates by removing walnuts from training images"""
    
    def __init__(self, train_data_dir: str, output_dir: str):
        self.train_data_dir = train_data_dir
        self.output_dir = output_dir
        self.canopy_plates: List[CanopyPlate] = []
        
        # Create output directory
        os.makedirs(os.path.join(output_dir, 'canopy_plates'), exist_ok=True)
    
    def generate_canopy_plates(self, num_plates: int = 30):
        """
        Generate canopy background plates by removing walnuts from training images
        
        Args:
            num_plates: Number of canopy plates to generate
        """
        print(f"🌳 Generating {num_plates} canopy background plates...")
        
        train_images_dir = os.path.join(self.train_data_dir, 'train', 'images')
        train_annotations_dir = os.path.join(self.train_data_dir, 'train', 'annotations')
        
        # Get all image files
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            image_files.extend(Path(train_images_dir).glob(f'*{ext}'))
        
        # Randomly select images for canopy plates
        selected_images = random.sample(image_files, min(num_plates, len(image_files)))
        
        for i, image_file in enumerate(selected_images):
            image_name = image_file.stem
            annotation_file = os.path.join(train_annotations_dir, f"{image_name}.txt")
            
            if not os.path.exists(annotation_file):
                continue
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            # Load annotations
            walnuts = self._load_annotations(annotation_file)
            
            # Remove walnuts and create canopy plate
            canopy_plate = self._create_canopy_plate(image, walnuts, f"canopy_plate_{i:03d}")
            
            if canopy_plate is not None:
                self.canopy_plates.append(canopy_plate)
        
        print(f"✅ Generated {len(self.canopy_plates)} canopy plates")
        
        return self.canopy_plates
    
    def _load_annotations(self, annotation_file: str) -> List[Tuple[int, int]]:
        """Load annotation coordinates from file"""
        walnuts = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = int(float(parts[0])), int(float(parts[1]))
                        walnuts.append((x, y))
        return walnuts
    
    def _create_canopy_plate(self, image: np.ndarray, walnuts: List[Tuple[int, int]], 
                           name: str) -> Optional[CanopyPlate]:
        """Create canopy plate by removing walnuts using inpainting"""
        
        # Create mask for walnut regions
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for x, y in walnuts:
            # Create circular mask around each walnut
            radius = 20  # Approximate walnut radius
            cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Dilate mask to ensure complete removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Inpaint to remove walnuts
        try:
            inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        except:
            # Fallback to simple blur if inpainting fails
            inpainted = image.copy()
            mask_bool = mask > 0
            inpainted[mask_bool] = cv2.GaussianBlur(image, (15, 15), 0)[mask_bool]
        
        # Save canopy plate
        output_path = os.path.join(self.output_dir, 'canopy_plates', f"{name}.png")
        cv2.imwrite(output_path, inpainted)
        
        # Create placement mask (areas where walnuts can be placed)
        placement_mask = self._create_placement_mask(inpainted)
        
        return CanopyPlate(
            image_path=output_path,
            image=inpainted,
            mask=placement_mask
        )
    
    def _create_placement_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask indicating where walnuts can be placed (green canopy areas only)"""
        
        # Convert to different color spaces for better analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for green foliage areas
        # HSV range for green colors
        lower_green = np.array([35, 40, 40])   # Lower bound for green
        upper_green = np.array([85, 255, 255]) # Upper bound for green
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Create mask for bright areas (avoid shadows)
        brightness_threshold = 80  # Minimum brightness to avoid shadows
        bright_mask = gray > brightness_threshold
        
        # Create mask for areas with good contrast (avoid uniform dark areas)
        # Calculate local standard deviation to find textured areas
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        local_std = np.sqrt(local_variance)
        texture_mask = local_std > 15  # Areas with sufficient texture
        
        # Combine all conditions
        placement_mask = green_mask & bright_mask & texture_mask
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        placement_mask = cv2.morphologyEx(placement_mask, cv2.MORPH_CLOSE, kernel)
        placement_mask = cv2.morphologyEx(placement_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small isolated regions
        contours, _ = cv2.findContours(placement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500  # Minimum area for walnut placement
        
        final_mask = np.zeros_like(placement_mask)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.fillPoly(final_mask, [contour], 255)
        
        return final_mask

def main():
    """Main function to run the synthetic data generation pipeline"""
    
    parser = argparse.ArgumentParser(description="Synthetic Walnut Generation Pipeline")
    parser.add_argument("--train_data", default="../walnut_annotated_dataset", 
                       help="Path to training dataset")
    parser.add_argument("--output", default="./synthetic_data", 
                       help="Output directory for synthetic data")
    parser.add_argument("--num_canopy_plates", type=int, default=30,
                       help="Number of canopy plates to generate")
    parser.add_argument("--patch_size", type=int, default=64,
                       help="Size of walnut patches to extract")
    
    args = parser.parse_args()
    
    print("🌰 Synthetic Walnut Generation Pipeline")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Step 1: Extract walnut instances
    print("\n📦 Step 1: Extracting walnut instances...")
    extractor = WalnutExtractor(args.train_data, args.output)
    walnut_instances = extractor.extract_walnuts(
        patch_size=32,  # Smaller patch size for smaller walnuts
        min_diameter=4,  # Much smaller minimum diameter
        max_diameter=16  # Much smaller maximum diameter
    )
    
    # Step 2: Generate canopy plates
    print("\n🌳 Step 2: Generating canopy plates...")
    canopy_generator = CanopyPlateGenerator(args.train_data, args.output)
    canopy_plates = canopy_generator.generate_canopy_plates(args.num_canopy_plates)
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📁 Output directory: {args.output}")
    print(f"🥜 Walnut instances: {len(walnut_instances)}")
    print(f"🌳 Canopy plates: {len(canopy_plates)}")

if __name__ == "__main__":
    main()
