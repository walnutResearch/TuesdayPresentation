#!/usr/bin/env python3
"""
Walnut Detector
===============

Sliding window detector using trained binary classifier for walnut detection.
Applies the trained model to full images to detect and count walnuts.

Author: Walnut Counting Project
Date: 2025
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class WalnutDetector:
    """Sliding window walnut detector using trained binary classifier"""
    
    def __init__(self, model_path: str, patch_size: int = 32, stride: int = 16, 
                 confidence_threshold: float = 0.5, device: str = 'auto'):
        
        self.patch_size = patch_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        
        # Device selection
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"🖥️  Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Transform for patches
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        from binary_classifier import WalnutClassifier
        
        # Create model architecture
        model = WalnutClassifier(input_size=self.patch_size, num_classes=2)
        
        # Load weights
        # PyTorch 2.6 defaults weights_only=True; allow full checkpoint loading
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"📦 Loaded model from {model_path}")
        val_acc = checkpoint.get('val_acc', None)
        try:
            if val_acc is not None:
                print(f"📊 Model accuracy: {float(val_acc):.2f}%")
            else:
                print("📊 Model accuracy: Unknown")
        except Exception:
            print("📊 Model accuracy: Unknown")
        
        return model.to(self.device)
    
    def detect_walnuts(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float], np.ndarray]:
        """
        Detect walnuts in an image using sliding window approach
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            centers: List of (x, y) coordinates of detected walnuts
            confidences: List of confidence scores for each detection
            confidence_map: 2D array of confidence scores for each pixel
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Initialize confidence map
        confidence_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.int32)
        
        # Extract patches using sliding window
        patches = []
        patch_coords = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Extract patch
                patch = image_rgb[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                patch_coords.append((x, y))
        
        # Process patches in batches
        batch_size = 32
        all_confidences = []
        
        for i in tqdm(range(0, len(patches), batch_size), desc="Processing patches"):
            batch_patches = patches[i:i+batch_size]
            batch_coords = patch_coords[i:i+batch_size]
            
            # Transform patches
            batch_tensors = []
            for patch in batch_patches:
                tensor = self.transform(patch)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidences = probabilities[:, 1].cpu().numpy()  # Probability of class 1 (walnut)
            
            all_confidences.extend(confidences)
            
            # Update confidence map
            for j, (x, y) in enumerate(batch_coords):
                confidence = confidences[j]
                confidence_map[y:y+self.patch_size, x:x+self.patch_size] += confidence
                count_map[y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Normalize confidence map
        confidence_map = np.divide(confidence_map, count_map, 
                                 out=np.zeros_like(confidence_map), 
                                 where=count_map != 0)
        
        # Find high-confidence regions
        high_conf_mask = confidence_map > self.confidence_threshold
        
        # Find local maxima
        centers, confidences = self._find_local_maxima(confidence_map, high_conf_mask)
        
        return centers, confidences, confidence_map
    
    def _find_local_maxima(self, confidence_map: np.ndarray, mask: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Find local maxima in confidence map"""
        from scipy.ndimage import maximum_filter
        
        # Apply maximum filter
        local_maxima = maximum_filter(confidence_map, size=3) == confidence_map
        
        # Combine with high confidence mask
        peaks = local_maxima & mask
        
        # Get coordinates and confidences
        y_coords, x_coords = np.where(peaks)
        confidences = confidence_map[peaks]
        
        # Sort by confidence
        sorted_indices = np.argsort(confidences)[::-1]
        
        centers = [(int(x_coords[i]), int(y_coords[i])) for i in sorted_indices]
        confidences = [float(confidences[i]) for i in sorted_indices]
        
        return centers, confidences
    
    def cluster_detections(self, centers: List[Tuple[int, int]], confidences: List[float], 
                          eps: float = 20.0, min_samples: int = 1) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Cluster nearby detections to avoid duplicates"""
        if len(centers) == 0:
            return [], []
        
        # Convert to numpy array
        centers_array = np.array(centers)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_array)
        
        # Get cluster centers
        clustered_centers = []
        clustered_confidences = []
        
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
            
            # Get points in this cluster
            cluster_mask = clustering.labels_ == cluster_id
            cluster_centers = centers_array[cluster_mask]
            cluster_confidences = np.array(confidences)[cluster_mask]
            
            # Use weighted average for cluster center
            weights = cluster_confidences
            weighted_center = np.average(cluster_centers, axis=0, weights=weights)
            max_confidence = np.max(cluster_confidences)
            
            clustered_centers.append((int(weighted_center[0]), int(weighted_center[1])))
            clustered_confidences.append(float(max_confidence))
        
        return clustered_centers, clustered_confidences
    
    def visualize_detections(self, image: np.ndarray, centers: List[Tuple[int, int]], 
                           confidences: List[float], confidence_map: np.ndarray = None,
                           save_path: str = None) -> np.ndarray:
        """Visualize detections on image"""
        vis_image = image.copy()
        
        # Draw confidence map if provided
        if confidence_map is not None:
            # Create heatmap
            heatmap = cv2.applyColorMap(
                (confidence_map * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 0.7, heatmap, 0.3, 0)
        
        # Draw detections
        for i, (x, y) in enumerate(centers):
            confidence = confidences[i]
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw circle
            cv2.circle(vis_image, (x, y), 8, color, 2)
            
            # Draw confidence text
            cv2.putText(vis_image, f"{confidence:.2f}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add count text
        cv2.putText(vis_image, f"Walnuts: {len(centers)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def process_image(self, image_path: str, output_dir: str = None, 
                     cluster: bool = True) -> Dict:
        """Process a single image and return detection results"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect walnuts
        centers, confidences, confidence_map = self.detect_walnuts(image)
        
        # Cluster detections if requested
        if cluster:
            centers, confidences = self.cluster_detections(centers, confidences)
        
        # Create results
        results = {
            'image_path': image_path,
            'num_walnuts': len(centers),
            'centers': centers,
            'confidences': confidences,
            'mean_confidence': np.mean(confidences) if confidences else 0.0
        }
        
        # Save visualization if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create visualization
            vis_image = self.visualize_detections(image, centers, confidences, confidence_map)
            
            # Save visualization
            image_name = Path(image_path).stem
            vis_path = os.path.join(output_dir, f"{image_name}_detections.jpg")
            cv2.imwrite(vis_path, vis_image)
            
            # Save confidence map
            conf_path = os.path.join(output_dir, f"{image_name}_confidence.png")
            cv2.imwrite(conf_path, (confidence_map * 255).astype(np.uint8))
            
            results['visualization_path'] = vis_path
            results['confidence_map_path'] = conf_path
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Detect walnuts in images")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--image_path", help="Path to single image")
    parser.add_argument("--image_dir", help="Path to directory of images")
    parser.add_argument("--output_dir", help="Output directory for results")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
    parser.add_argument("--stride", type=int, default=16, help="Stride for sliding window")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--cluster", action="store_true", help="Cluster nearby detections")
    
    args = parser.parse_args()
    
    print("🥜 Walnut Detector")
    print("=" * 30)
    
    # Create detector
    detector = WalnutDetector(
        model_path=args.model_path,
        patch_size=args.patch_size,
        stride=args.stride,
        confidence_threshold=args.threshold
    )
    
    # Process single image
    if args.image_path:
        print(f"Processing image: {args.image_path}")
        results = detector.process_image(args.image_path, args.output_dir, args.cluster)
        
        print(f"Detected {results['num_walnuts']} walnuts")
        print(f"Mean confidence: {results['mean_confidence']:.3f}")
        
        if args.output_dir:
            print(f"Results saved to: {args.output_dir}")
    
    # Process directory of images
    elif args.image_dir:
        image_files = list(Path(args.image_dir).glob("*.png")) + list(Path(args.image_dir).glob("*.jpg"))
        print(f"Processing {len(image_files)} images...")
        
        all_results = []
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                results = detector.process_image(str(image_file), args.output_dir, args.cluster)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        # Summary statistics
        total_walnuts = sum(r['num_walnuts'] for r in all_results)
        mean_confidence = np.mean([r['mean_confidence'] for r in all_results])
        
        print(f"\n📊 Summary:")
        print(f"Total walnuts detected: {total_walnuts}")
        print(f"Average walnuts per image: {total_walnuts / len(all_results):.1f}")
        print(f"Mean confidence: {mean_confidence:.3f}")
        
        # Save results
        if args.output_dir:
            results_path = os.path.join(args.output_dir, "detection_results.json")
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to: {results_path}")
    
    else:
        print("Please provide either --image_path or --image_dir")

if __name__ == "__main__":
    main()
