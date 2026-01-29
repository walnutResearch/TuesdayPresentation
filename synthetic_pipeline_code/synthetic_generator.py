#!/usr/bin/env python3
"""
Generate synthetic walnut images: composite walnut cutouts onto canopy plates with photometric
matching and realistic placement. Produces synthetic images and density maps for training the density model.

How to run:
  python synthetic_generator.py --walnut_instances path/to/metadata --canopy_plates path/to/plates --output path/to/output [--num_images 2000]

Use --help for all options. Requires synthetic_pipeline output (WalnutInstance metadata). Run from synthetic_pipeline_code/.
"""

import os
import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
from dataclasses import dataclass
import argparse

# Import WalnutInstance from synthetic_pipeline
from synthetic_pipeline import WalnutInstance

@dataclass
class SyntheticImage:
    """Data class for synthetic image and its metadata"""
    image: np.ndarray
    density_map: np.ndarray
    walnut_centers: List[Tuple[int, int]]
    walnut_diameters: List[float]
    image_path: str
    label_path: str

class PhotometricMatcher:
    """Handles photometric matching between walnuts and background"""
    
    @staticmethod
    def match_walnut_to_background(walnut_patch: np.ndarray, background_patch: np.ndarray, 
                                 alpha_mask: np.ndarray) -> np.ndarray:
        """
        Match walnut appearance to background using conservative color preservation
        
        Args:
            walnut_patch: Walnut image patch
            background_patch: Background region around placement site
            alpha_mask: Alpha mask for walnut
            
        Returns:
            Photometrically matched walnut patch
        """
        
        # Convert to float for calculations
        walnut_float = walnut_patch.astype(np.float32)
        background_float = background_patch.astype(np.float32)
        
        # Calculate statistics for each channel
        matched_patch = walnut_float.copy()
        
        for channel in range(3):  # BGR channels
            walnut_channel = walnut_float[:, :, channel]
            background_channel = background_float[:, :, channel]
            
            # Calculate mean and std for walnut region
            walnut_mask = alpha_mask > 128
            if not np.any(walnut_mask):
                continue
                
            walnut_mean = np.mean(walnut_channel[walnut_mask])
            walnut_std = np.std(walnut_channel[walnut_mask])
            
            # Calculate mean and std for background region
            background_mean = np.mean(background_channel)
            background_std = np.std(background_channel)
            
            # Avoid division by zero
            if walnut_std < 1e-6:
                walnut_std = 1.0
            
            # More conservative matching - only adjust brightness, preserve color ratios
            # Calculate brightness adjustment factor
            brightness_factor = background_mean / (walnut_mean + 1e-6)
            brightness_factor = np.clip(brightness_factor, 0.7, 1.3)  # Limit adjustment
            
            # Apply only brightness adjustment, preserve original color ratios
            matched_channel = walnut_channel * brightness_factor
            
            # Clamp values to valid range
            matched_channel = np.clip(matched_channel, 0, 255)
            
            matched_patch[:, :, channel] = matched_channel
        
        return matched_patch.astype(np.uint8)
    
    @staticmethod
    def add_shadow(walnut_patch: np.ndarray, alpha_mask: np.ndarray, 
                   shadow_offset: Tuple[int, int] = (2, 3), 
                   shadow_strength: float = 0.3) -> np.ndarray:
        """
        Add soft shadow to walnut
        
        Args:
            walnut_patch: Walnut image patch
            alpha_mask: Alpha mask for walnut
            shadow_offset: Shadow offset (dx, dy)
            shadow_strength: Shadow opacity strength
            
        Returns:
            Walnut patch with shadow
        """
        
        # Create shadow mask
        shadow_mask = alpha_mask.astype(np.float32) / 255.0
        shadow_mask = cv2.GaussianBlur(shadow_mask, (5, 5), 0)
        shadow_mask *= shadow_strength
        
        # Apply shadow offset
        dx, dy = shadow_offset
        h, w = shadow_mask.shape
        shadow_offset_mask = np.zeros_like(shadow_mask)
        
        # Calculate shadow region
        y1 = max(0, dy)
        y2 = min(h, h + dy)
        x1 = max(0, dx)
        x2 = min(w, w + dx)
        
        sy1 = max(0, -dy)
        sy2 = sy1 + (y2 - y1)
        sx1 = max(0, -dx)
        sx2 = sx1 + (x2 - x1)
        
        if y2 > y1 and x2 > x1 and sy2 <= h and sx2 <= w:
            shadow_offset_mask[y1:y2, x1:x2] = shadow_mask[sy1:sy2, sx1:sx2]
        
        # Apply shadow to image
        result = walnut_patch.copy().astype(np.float32)
        for channel in range(3):
            result[:, :, channel] *= (1 - shadow_offset_mask)
        
        return np.clip(result, 0, 255).astype(np.uint8)

class SyntheticImageGenerator:
    """Generates synthetic walnut images with realistic compositing"""
    
    def __init__(self, walnut_instances: List, canopy_plates: List, output_dir: str, 
                 use_photometric_matching: bool = True):
        self.walnut_instances = walnut_instances
        self.canopy_plates = canopy_plates
        self.output_dir = output_dir
        self.photometric_matcher = PhotometricMatcher()
        self.use_photometric_matching = use_photometric_matching
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'synthetic_images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'density_maps'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
        
        # Load walnut statistics and instances
        self._load_walnut_statistics()
        self._load_walnut_instances()
    
    def _load_walnut_statistics(self):
        """Load walnut diameter and brightness statistics"""
        metadata_path = os.path.join(self.output_dir, 'metadata', 'walnut_instances.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.diameter_stats = metadata.get('diameter_stats', {})
                self.brightness_stats = metadata.get('brightness_stats', {})
        else:
            # Default statistics
            self.diameter_stats = {'mean': 20.0, 'std': 5.0, 'min': 8.0, 'max': 32.0}
            self.brightness_stats = {'mean': 100.0, 'std': 30.0, 'min': 50.0, 'max': 200.0}
    
    def _load_walnut_instances(self):
        """Load walnut instances from metadata"""
        metadata_path = os.path.join(self.output_dir, 'metadata', 'walnut_instances.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                instances_data = metadata.get('instances', [])
                
                # Convert to WalnutInstance objects
                self.walnut_instances = []
                for instance_data in instances_data:
                    # Load the actual walnut cutout image
                    walnut_cutout = cv2.imread(instance_data['image_path'], cv2.IMREAD_UNCHANGED)
                    if walnut_cutout is not None:
                        # Extract alpha channel
                        alpha = walnut_cutout[:, :, 3] if walnut_cutout.shape[2] == 4 else None
                        
                        if alpha is not None:
                            walnut_instance = WalnutInstance(
                                image_path=instance_data['image_path'],
                                center_x=instance_data['center_x'],
                                center_y=instance_data['center_y'],
                                diameter=instance_data['diameter'],
                                brightness=instance_data['brightness'],
                                alpha_mask=alpha,
                                bbox=tuple(instance_data['bbox'])
                            )
                            self.walnut_instances.append(walnut_instance)
                
                print(f"📦 Loaded {len(self.walnut_instances)} walnut instances from metadata")
        else:
            print("⚠️  No walnut instances metadata found, using empty list")
            self.walnut_instances = []
    
    def generate_synthetic_images(self, num_images: int = 1000, 
                                image_size: Tuple[int, int] = (512, 512),
                                min_walnuts: int = 10, max_walnuts: int = 50,
                                density_patch_size: int = 32) -> List[SyntheticImage]:
        """
        Generate synthetic walnut images
        
        Args:
            num_images: Number of synthetic images to generate
            image_size: Size of synthetic images (width, height)
            min_walnuts: Minimum number of walnuts per image
            max_walnuts: Maximum number of walnuts per image
            density_patch_size: Size of density map patches
            
        Returns:
            List of generated synthetic images
        """
        
        print(f"🎨 Generating {num_images} synthetic images...")
        
        synthetic_images = []
        
        for i in range(num_images):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_images} images...")
            
            # Select random canopy plate
            canopy_plate = random.choice(self.canopy_plates)
            
            # Resize canopy plate to target size
            background = cv2.resize(canopy_plate.image, image_size)
            placement_mask = cv2.resize(canopy_plate.mask, image_size)
            
            # Generate number of walnuts
            num_walnuts = random.randint(min_walnuts, max_walnuts)
            
            # Place walnuts
            result_image, walnut_centers, walnut_diameters = self._place_walnuts(
                background, placement_mask, num_walnuts
            )
            
            # Create density map
            density_map = self._create_density_map(
                walnut_centers, image_size, density_patch_size
            )
            
            # Save synthetic image
            image_path = os.path.join(self.output_dir, 'synthetic_images', f"synthetic_{i:06d}.png")
            cv2.imwrite(image_path, result_image)
            
            # Save density map
            density_path = os.path.join(self.output_dir, 'density_maps', f"density_{i:06d}.npy")
            np.save(density_path, density_map)
            
            # Save labels (center points)
            label_path = os.path.join(self.output_dir, 'labels', f"label_{i:06d}.txt")
            self._save_labels(walnut_centers, walnut_diameters, label_path)
            
            synthetic_images.append(SyntheticImage(
                image=result_image,
                density_map=density_map,
                walnut_centers=walnut_centers,
                walnut_diameters=walnut_diameters,
                image_path=image_path,
                label_path=label_path
            ))
        
        print(f"✅ Generated {len(synthetic_images)} synthetic images")
        
        # Save generation metadata
        self._save_generation_metadata(synthetic_images)
        
        return synthetic_images
    
    def _place_walnuts(self, background: np.ndarray, placement_mask: np.ndarray, 
                      num_walnuts: int) -> Tuple[np.ndarray, List[Tuple[int, int]], List[float]]:
        """
        Place walnuts on background image
        
        Args:
            background: Background image
            placement_mask: Mask indicating valid placement areas
            num_walnuts: Number of walnuts to place
            
        Returns:
            Tuple of (result_image, walnut_centers, walnut_diameters)
        """
        
        result_image = background.copy()
        walnut_centers = []
        walnut_diameters = []
        
        h, w = background.shape[:2]
        
        for _ in range(num_walnuts):
            # Select random walnut instance
            walnut_instance = random.choice(self.walnut_instances)
            
            # Load walnut cutout
            walnut_cutout = cv2.imread(walnut_instance.image_path, cv2.IMREAD_UNCHANGED)
            if walnut_cutout is None:
                continue
            
            # Extract alpha channel
            alpha = walnut_cutout[:, :, 3] if walnut_cutout.shape[2] == 4 else None
            if alpha is None:
                continue
            
            # Scale walnut based on diameter statistics (more conservative scaling)
            target_diameter = self._sample_diameter()
            scale_factor = target_diameter / walnut_instance.diameter
            scale_factor = np.clip(scale_factor, 0.3, 1.5)  # More conservative scaling for realism
            
            # Resize walnut
            new_size = (int(walnut_cutout.shape[1] * scale_factor), 
                       int(walnut_cutout.shape[0] * scale_factor))
            if new_size[0] < 1 or new_size[1] < 1:
                continue
                
            walnut_resized = cv2.resize(walnut_cutout, new_size)
            alpha_resized = cv2.resize(alpha, new_size)
            
            # Find valid placement position
            placement_pos = self._find_placement_position(
                placement_mask, new_size, walnut_centers, min_distance=30
            )
            
            if placement_pos is None:
                continue
            
            x, y = placement_pos
            
            # Apply photometric matching if enabled
            if self.use_photometric_matching:
                # Extract background region for photometric matching
                bg_region = self._extract_background_region(
                    background, x, y, new_size, padding=10
                )
                
                # Apply conservative photometric matching
                walnut_matched = self.photometric_matcher.match_walnut_to_background(
                    walnut_resized[:, :, :3], bg_region, alpha_resized
                )
            else:
                # Use original walnut colors without matching
                walnut_matched = walnut_resized[:, :, :3]
            
            # Add shadow
            walnut_with_shadow = self.photometric_matcher.add_shadow(
                walnut_matched, alpha_resized
            )
            
            # Composite walnut onto background
            result_image = self._composite_walnut(
                result_image, walnut_with_shadow, alpha_resized, x, y
            )
            
            # Record walnut information
            center_x = x + new_size[0] // 2
            center_y = y + new_size[1] // 2
            walnut_centers.append((center_x, center_y))
            walnut_diameters.append(target_diameter)
        
        return result_image, walnut_centers, walnut_diameters
    
    def _sample_diameter(self) -> float:
        """Sample walnut diameter from learned distribution (adjusted for realism)"""
        # Use more realistic walnut sizes regardless of extracted sizes
        mean = 10.0  # Target mean diameter of 10 pixels
        std = 3.0    # Smaller standard deviation
        min_d = 4.0  # Minimum realistic walnut size
        max_d = 16.0 # Maximum realistic walnut size
        
        # Sample from normal distribution and clip to valid range
        diameter = np.random.normal(mean, std)
        diameter = np.clip(diameter, min_d, max_d)
        
        return diameter
    
    def _find_placement_position(self, placement_mask: np.ndarray, walnut_size: Tuple[int, int],
                               existing_centers: List[Tuple[int, int]], 
                               min_distance: int = 30) -> Optional[Tuple[int, int]]:
        """Find valid position to place walnut"""
        
        h, w = placement_mask.shape
        walnut_w, walnut_h = walnut_size
        
        # Ensure walnut fits in image
        if walnut_w >= w or walnut_h >= h:
            return None
        
        max_attempts = 100
        for _ in range(max_attempts):
            # Random position
            x = random.randint(0, w - walnut_w)
            y = random.randint(0, h - walnut_h)
            
            # Check if position is in valid placement area
            if not self._is_valid_placement(placement_mask, x, y, walnut_w, walnut_h):
                continue
            
            # Check distance from existing walnuts
            center_x = x + walnut_w // 2
            center_y = y + walnut_h // 2
            
            too_close = False
            for existing_x, existing_y in existing_centers:
                distance = math.sqrt((center_x - existing_x)**2 + (center_y - existing_y)**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                return (x, y)
        
        return None
    
    def _is_valid_placement(self, placement_mask: np.ndarray, x: int, y: int, 
                          w: int, h: int) -> bool:
        """Check if placement position is valid (green canopy areas only)"""
        
        # Check if region is mostly in valid placement area
        region = placement_mask[y:y+h, x:x+w]
        valid_pixels = np.sum(region > 128)
        total_pixels = w * h
        
        # At least 90% of pixels should be in valid green canopy area
        # This is stricter to ensure walnuts are only placed on foliage
        return valid_pixels / total_pixels > 0.9
    
    def _extract_background_region(self, background: np.ndarray, x: int, y: int, 
                                 walnut_size: Tuple[int, int], padding: int = 10) -> np.ndarray:
        """Extract background region for photometric matching"""
        
        h, w = background.shape[:2]
        walnut_w, walnut_h = walnut_size
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + walnut_w + padding)
        y2 = min(h, y + walnut_h + padding)
        
        return background[y1:y2, x1:x2]
    
    def _composite_walnut(self, background: np.ndarray, walnut: np.ndarray, 
                         alpha: np.ndarray, x: int, y: int) -> np.ndarray:
        """Composite walnut onto background using alpha blending"""
        
        result = background.copy()
        h, w = background.shape[:2]
        walnut_h, walnut_w = walnut.shape[:2]
        
        # Calculate region bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + walnut_w)
        y2 = min(h, y + walnut_h)
        
        # Calculate walnut region bounds
        wx1 = max(0, -x)
        wy1 = max(0, -y)
        wx2 = wx1 + (x2 - x1)
        wy2 = wy1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1 and wx2 > wx1 and wy2 > wy1:
            # Extract regions
            bg_region = result[y1:y2, x1:x2]
            walnut_region = walnut[wy1:wy2, wx1:wx2]
            alpha_region = alpha[wy1:wy2, wx1:wx2].astype(np.float32) / 255.0
            
            # Alpha blend
            for c in range(3):
                bg_region[:, :, c] = (bg_region[:, :, c] * (1 - alpha_region) + 
                                    walnut_region[:, :, c] * alpha_region).astype(np.uint8)
        
        return result
    
    def _create_density_map(self, walnut_centers: List[Tuple[int, int]], 
                          image_size: Tuple[int, int], patch_size: int) -> np.ndarray:
        """Create density map for training"""
        
        w, h = image_size
        density_w = w // patch_size
        density_h = h // patch_size
        
        density_map = np.zeros((density_h, density_w), dtype=np.float32)
        
        for center_x, center_y in walnut_centers:
            # Calculate which patch this center belongs to
            patch_x = center_x // patch_size
            patch_y = center_y // patch_size
            
            # Ensure within bounds
            patch_x = min(patch_x, density_w - 1)
            patch_y = min(patch_y, density_h - 1)
            
            # Add to density map
            density_map[patch_y, patch_x] += 1.0
        
        return density_map
    
    def _save_labels(self, walnut_centers: List[Tuple[int, int]], 
                    walnut_diameters: List[float], label_path: str):
        """Save label file with walnut centers and diameters"""
        
        with open(label_path, 'w') as f:
            f.write("# Synthetic walnut labels\n")
            f.write("# Format: center_x center_y diameter\n")
            f.write(f"# Total walnuts: {len(walnut_centers)}\n")
            
            for (center_x, center_y), diameter in zip(walnut_centers, walnut_diameters):
                f.write(f"{center_x} {center_y} {diameter:.1f}\n")
    
    def _save_generation_metadata(self, synthetic_images: List[SyntheticImage]):
        """Save metadata about generated synthetic images"""
        
        metadata = {
            'total_images': len(synthetic_images),
            'total_walnuts': sum(len(img.walnut_centers) for img in synthetic_images),
            'avg_walnuts_per_image': np.mean([len(img.walnut_centers) for img in synthetic_images]),
            'diameter_stats': self.diameter_stats,
            'brightness_stats': self.brightness_stats
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata', 'synthetic_generation.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Synthetic Dataset Statistics:")
        print(f"  Total images: {metadata['total_images']}")
        print(f"  Total walnuts: {metadata['total_walnuts']}")
        print(f"  Average walnuts per image: {metadata['avg_walnuts_per_image']:.1f}")

def main():
    """Main function to generate synthetic images"""
    
    parser = argparse.ArgumentParser(description="Generate synthetic walnut images")
    parser.add_argument("--walnut_instances", required=True,
                       help="Path to walnut instances metadata")
    parser.add_argument("--canopy_plates", required=True,
                       help="Path to canopy plates directory")
    parser.add_argument("--output", default="./synthetic_data",
                       help="Output directory for synthetic data")
    parser.add_argument("--num_images", type=int, default=1000,
                       help="Number of synthetic images to generate")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512],
                       help="Size of synthetic images (width height)")
    parser.add_argument("--min_walnuts", type=int, default=10,
                       help="Minimum walnuts per image")
    parser.add_argument("--max_walnuts", type=int, default=50,
                       help="Maximum walnuts per image")
    parser.add_argument("--no_photometric_matching", action="store_true",
                       help="Disable photometric matching to preserve original walnut colors")
    
    args = parser.parse_args()
    
    print("🎨 Synthetic Image Generator")
    print("=" * 40)
    
    # Load walnut instances (will be loaded by the generator)
    walnut_instances = []  # Will be loaded from metadata in the generator
    
    # Load canopy plates
    canopy_plates = []
    for plate_file in Path(args.canopy_plates).glob("*.png"):
        image = cv2.imread(str(plate_file))
        if image is not None:
            # Create simple placement mask (all areas valid)
            placement_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            canopy_plates.append(type('CanopyPlate', (), {
                'image_path': str(plate_file),
                'image': image,
                'mask': placement_mask
            })())
    
    if not canopy_plates:
        print("Error: No canopy plates found")
        return
    
    # Generate synthetic images
    use_photometric = not args.no_photometric_matching
    generator = SyntheticImageGenerator(walnut_instances, canopy_plates, args.output, use_photometric)
    synthetic_images = generator.generate_synthetic_images(
        num_images=args.num_images,
        image_size=tuple(args.image_size),
        min_walnuts=args.min_walnuts,
        max_walnuts=args.max_walnuts
    )
    
    print(f"✅ Generated {len(synthetic_images)} synthetic images")

if __name__ == "__main__":
    main()
