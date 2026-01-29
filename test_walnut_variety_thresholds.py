#!/usr/bin/env python3
"""
Test the binary walnut classifier on the Walnut Variety Trial (WVT) dataset at multiple confidence thresholds.
Loads test images and annotations, runs sliding-window detection per threshold, and reports count accuracy,
precision, recall, F1, MAE, RMSE. Can save per-threshold results and plots.

How to run:
  python test_walnut_variety_thresholds.py [--test_dir path/to/test] [--model_path path/to/model.pth] [--thresholds 0.5 0.6 0.7] [--output_dir path]

Defaults and paths are set in the script; use --help for options.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# System optimizations
os.environ.setdefault("PYTHONWARNINGS", "ignore:torch.multiprocessing:red")

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial.distance import cdist

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# -----------------------------
# Model Architecture
# -----------------------------
class WalnutClassifier(nn.Module):
    """CNN for binary walnut classification"""
    
    def __init__(self, input_size: int = 32, num_classes: int = 2, dropout_rate: float = 0.5):
        super(WalnutClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # 2x2
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# -----------------------------
# Test Dataset
# -----------------------------
class WalnutVarietyDataset(Dataset):
    """Test dataset for WalnutVarietyTrial with separate annotations directory"""

    def __init__(self, images_dir: str, annotations_dir: str, patch_size: int = 16, stride: int = 8):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.patch_size = patch_size
        self.stride = stride

        # Get all image files (support multiple formats)
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.JPG', '*.PNG']:
            self.image_files.extend(sorted(self.images_dir.glob(ext)))
        
        print(f"Found {len(self.image_files)} test images")

        # Transform for patches
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load annotations - try to find matching annotation file
        # Try different naming patterns
        annotation_path = None
        possible_names = [
            self.annotations_dir / f"{image_path.stem}.txt",
            self.annotations_dir / f"{image_path.name.replace(image_path.suffix, '.txt')}",
        ]
        
        for path in possible_names:
            if path.exists():
                annotation_path = path
                break
        
        if annotation_path is None:
            # Try to find any annotation file that might match
            print(f"Warning: Annotation file not found for {image_path.name}, using empty annotations")
            coords = []
        else:
            # Parse annotations
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and extract coordinates
            coords = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('Image') and not line.startswith('Total') and not line.startswith('Format'):
                    try:
                        parts = line.split()
                        if len(parts) == 2:
                            x, y = map(int, parts)
                            coords.append((x, y))
                    except ValueError:
                        continue

        return {
            'image': img_rgb,
            'coords': coords,
            'image_path': str(image_path),
            'original_shape': img.shape[:2],
        }

    def extract_patches(self, img: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract patches using sliding window"""
        h, w = img.shape[:2]
        patches = []
        patch_coords = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                patches.append(patch)
                patch_coords.append((x, y))
        
        return patches, patch_coords


# -----------------------------
# Evaluation Functions
# -----------------------------
def calculate_metrics(pred_counts: np.ndarray, true_counts: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # Basic regression metrics
    metrics['MAE'] = mean_absolute_error(true_counts, pred_counts)
    metrics['RMSE'] = np.sqrt(mean_squared_error(true_counts, pred_counts))
    metrics['MSE'] = mean_squared_error(true_counts, pred_counts)
    
    # Relative metrics
    mask = true_counts > 0
    if mask.any():
        metrics['MAPE'] = np.mean(np.abs((pred_counts[mask] - true_counts[mask]) / true_counts[mask])) * 100
        metrics['SMAPE'] = np.mean(2 * np.abs(pred_counts[mask] - true_counts[mask]) / 
                                 (np.abs(pred_counts[mask]) + np.abs(true_counts[mask]))) * 100
    else:
        metrics['MAPE'] = float('nan')
        metrics['SMAPE'] = float('nan')
    
    # Correlation
    if len(pred_counts) > 1:
        metrics['R2'] = 1 - (np.sum((true_counts - pred_counts) ** 2) / 
                            np.sum((true_counts - np.mean(true_counts)) ** 2))
        metrics['Pearson'] = np.corrcoef(pred_counts, true_counts)[0, 1]
    else:
        metrics['R2'] = float('nan')
        metrics['Pearson'] = float('nan')
    
    # Count statistics
    metrics['Pred_Mean'] = np.mean(pred_counts)
    metrics['True_Mean'] = np.mean(true_counts)
    metrics['Pred_Std'] = np.std(pred_counts)
    metrics['True_Std'] = np.std(true_counts)
    
    # Count accuracy (percentage)
    total_true = np.sum(true_counts)
    total_pred = np.sum(pred_counts)
    if total_true > 0:
        metrics['Count_Accuracy'] = (1 - abs(total_pred - total_true) / total_true) * 100
    else:
        metrics['Count_Accuracy'] = float('nan')
    
    return metrics


def test_single_threshold(model, dataset, device, threshold: float, patch_size: int = 16):
    """Test model with a single threshold"""
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset):
            img_rgb = batch['image']
            coords = batch['coords']
            image_path = batch['image_path']
            
            # Extract patches
            patches, patch_coords = dataset.extract_patches(img_rgb)
            
            if not patches:
                continue
            
            # Convert patches to tensors
            patch_tensors = []
            for patch in patches:
                patch_tensor = transform(patch)
                patch_tensors.append(patch_tensor)
            
            # Batch process patches
            patch_batch = torch.stack(patch_tensors).to(device)
            
            # Forward pass
            outputs = model(patch_batch)
            probabilities = F.softmax(outputs, dim=1)
            walnut_probs = probabilities[:, 1].cpu().numpy()  # Probability of being walnut
            
            # Find detections above threshold
            detection_indices = np.where(walnut_probs > threshold)[0]
            pred_detections = [patch_coords[i] for i in detection_indices]
            pred_count = len(pred_detections)
            
            true_count = len(coords)
            
            results.append({
                'image_path': image_path,
                'pred_count': pred_count,
                'true_count': true_count,
                'pred_detections': pred_detections,
                'coords': coords,
            })
    
    # Calculate metrics
    pred_counts = np.array([r['pred_count'] for r in results])
    true_counts = np.array([r['true_count'] for r in results])
    
    metrics = calculate_metrics(pred_counts, true_counts)
    
    return {
        'threshold': threshold,
        'results': results,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Test binary classifier with multiple thresholds")
    parser.add_argument("--model", default="models_precision/walnut_classifier_best_precision.pth", 
                       help="Path to trained model (.pth file)")
    parser.add_argument("--images_dir", required=True, help="Path to images directory")
    parser.add_argument("--annotations_dir", required=True, help="Path to annotations directory")
    parser.add_argument("--output", default="./walnut_variety_threshold_results", 
                       help="Output directory for results")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size (default: 16)")
    parser.add_argument("--stride", type=int, default=8, help="Stride (default: 8)")
    parser.add_argument("--thresholds", nargs='+', type=float, 
                       default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       help="Thresholds to test (default: 0.1 to 0.9)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.images_dir):
        print(f"❌ Images directory not found: {args.images_dir}")
        return
    
    if not os.path.exists(args.annotations_dir):
        print(f"❌ Annotations directory not found: {args.annotations_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("🧠 Testing Binary Walnut Classifier - Multiple Thresholds")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Images: {args.images_dir}")
    print(f"Annotations: {args.annotations_dir}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Patch Size: {args.patch_size}, Stride: {args.stride}")
    print()
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print(f"📦 Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Create model
    model = WalnutClassifier(input_size=args.patch_size, num_classes=2, dropout_rate=0.5)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    print(f"📁 Loading dataset...")
    dataset = WalnutVarietyDataset(args.images_dir, args.annotations_dir, args.patch_size, args.stride)
    print(f"✅ Found {len(dataset)} images")
    
    # Test each threshold
    all_results = {}
    
    for threshold in args.thresholds:
        print(f"\n{'='*70}")
        print(f"Testing Threshold: {threshold:.2f}")
        print(f"{'='*70}")
        
        result = test_single_threshold(model, dataset, device, threshold, args.patch_size)
        all_results[f"{threshold:.2f}"] = result
        
        metrics = result['metrics']
        print(f"✅ Threshold {threshold:.2f} Results:")
        print(f"   MAE: {metrics['MAE']:.3f}")
        print(f"   RMSE: {metrics['RMSE']:.3f}")
        print(f"   Count Accuracy: {metrics['Count_Accuracy']:.2f}%")
        print(f"   R²: {metrics['R2']:.3f}")
        print(f"   Predicted Total: {np.sum([r['pred_count'] for r in result['results']]):.0f}")
        print(f"   True Total: {np.sum([r['true_count'] for r in result['results']]):.0f}")
    
    # Find best threshold
    print("\n" + "=" * 70)
    print("📊 SUMMARY - FINDING BEST THRESHOLD")
    print("=" * 70)
    
    # Sort by different criteria
    thresholds_sorted_mae = sorted(all_results.items(), key=lambda x: x[1]['metrics']['MAE'])
    thresholds_sorted_count_acc = sorted(all_results.items(), 
                                         key=lambda x: x[1]['metrics']['Count_Accuracy'] if not np.isnan(x[1]['metrics']['Count_Accuracy']) else -1, 
                                         reverse=True)
    thresholds_sorted_r2 = sorted(all_results.items(), 
                                 key=lambda x: x[1]['metrics']['R2'] if not np.isnan(x[1]['metrics']['R2']) else -1, 
                                 reverse=True)
    
    # Print summary table
    print(f"\n{'Threshold':<12} {'MAE':<10} {'RMSE':<10} {'Count Acc %':<15} {'R²':<10} {'Total Pred':<12} {'Total True':<12}")
    print("-" * 90)
    
    for thresh_str in sorted(all_results.keys(), key=float):
        r = all_results[thresh_str]
        m = r['metrics']
        total_pred = np.sum([x['pred_count'] for x in r['results']])
        total_true = np.sum([x['true_count'] for x in r['results']])
        count_acc = f"{m['Count_Accuracy']:.2f}" if not np.isnan(m['Count_Accuracy']) else "N/A"
        r2 = f"{m['R2']:.3f}" if not np.isnan(m['R2']) else "N/A"
        print(f"{float(thresh_str):<12.2f} {m['MAE']:<10.3f} {m['RMSE']:<10.3f} {count_acc:<15} {r2:<10} {total_pred:<12.0f} {total_true:<12.0f}")
    
    # Best results
    print("\n" + "=" * 70)
    print("🏆 BEST RESULTS")
    print("=" * 70)
    
    best_mae = thresholds_sorted_mae[0]
    print(f"\n🥇 Best MAE (Mean Absolute Error):")
    print(f"   Threshold: {best_mae[0]}")
    print(f"   MAE: {best_mae[1]['metrics']['MAE']:.3f}")
    print(f"   RMSE: {best_mae[1]['metrics']['RMSE']:.3f}")
    print(f"   Count Accuracy: {best_mae[1]['metrics']['Count_Accuracy']:.2f}%")
    
    if thresholds_sorted_count_acc:
        best_count_acc = thresholds_sorted_count_acc[0]
        print(f"\n🥇 Best Count Accuracy:")
        print(f"   Threshold: {best_count_acc[0]}")
        print(f"   Count Accuracy: {best_count_acc[1]['metrics']['Count_Accuracy']:.2f}%")
        print(f"   MAE: {best_count_acc[1]['metrics']['MAE']:.3f}")
        print(f"   Total Predicted: {np.sum([x['pred_count'] for x in best_count_acc[1]['results']]):.0f}")
        print(f"   Total True: {np.sum([x['true_count'] for x in best_count_acc[1]['results']]):.0f}")
    
    if thresholds_sorted_r2:
        best_r2 = thresholds_sorted_r2[0]
        print(f"\n🥇 Best R² (Correlation):")
        print(f"   Threshold: {best_r2[0]}")
        print(f"   R²: {best_r2[1]['metrics']['R2']:.3f}")
        print(f"   MAE: {best_r2[1]['metrics']['MAE']:.3f}")
    
    # Save results
    output_file = os.path.join(args.output, "all_thresholds_results.json")
    
    # Prepare data for JSON (convert numpy types)
    json_results = {}
    for thresh_str, result in all_results.items():
        json_results[thresh_str] = {
            'threshold': float(result['threshold']),
            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in result['metrics'].items()},
            'per_image_results': [
                {
                    'image_path': r['image_path'],
                    'pred_count': int(r['pred_count']),
                    'true_count': int(r['true_count']),
                    'error': int(abs(r['pred_count'] - r['true_count']))
                }
                for r in result['results']
            ]
        }
    
    with open(output_file, 'w') as f:
        json.dump({
            'model_path': args.model,
            'images_dir': args.images_dir,
            'annotations_dir': args.annotations_dir,
            'parameters': {
                'patch_size': args.patch_size,
                'stride': args.stride,
            },
            'results': json_results
        }, f, indent=2)
    
    print(f"\n✅ All results saved to: {output_file}")
    print(f"\n💡 Recommendation: Use threshold {best_mae[0]} for best overall performance (lowest MAE)")


if __name__ == "__main__":
    main()

