#!/usr/bin/env python3
"""
Test the trained binary walnut classifier on a test image directory: sliding-window detection with configurable
patch size, stride, and threshold. Compares predicted counts to ground-truth annotations and reports MAE, RMSE,
R², precision, recall, F1. Can save detection overlays and JSON.

How to run:
  python test_binary_classifier.py --model path/to/model.pth --test_dir path/to/test/images [--output path/to/output] [--patch_size 32] [--stride 16] [--threshold 0.5]

Use --help for all options. Annotations are expected in a sibling annotations/ directory or as specified.
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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import cdist

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# -----------------------------
# Model Architecture (copied from training code)
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
class TestBinaryDataset(Dataset):
    """Test dataset for binary walnut classification using sliding window"""

    def __init__(self, test_dir: str, patch_size: int = 16, stride: int = 8):
        self.test_dir = Path(test_dir)
        self.images_dir = self.test_dir
        self.annotations_dir = self.test_dir / "annotations"
        self.patch_size = patch_size
        self.stride = stride

        # Get all image files
        self.image_files = sorted(self.images_dir.glob("*.png"))
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

        # Load annotations
        annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            raise ValueError(f"Annotation file not found: {annotation_path}")

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

        # Create ground truth density map
        density_map = self._create_density_map(img.shape[:2], coords)

        return {
            'image': img_rgb,
            'coords': coords,
            'image_path': str(image_path),
            'original_shape': img.shape[:2],
            'density_map': density_map
        }

    def _create_density_map(self, img_shape: Tuple[int, int], coords: List[Tuple[int, int]]) -> np.ndarray:
        """Create density map from coordinates - each point contributes exactly 1.0 to total count"""
        h, w = img_shape
        density_map = np.zeros((h, w), dtype=np.float32)
        
        for x, y in coords:
            # Clamp to valid range
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            density_map[y, x] += 1.0
        
        return density_map

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
    
    return metrics


def create_visualizations(results: List[Dict], output_dir: str):
    """Create comprehensive visualizations of results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    pred_counts = [r['pred_count'] for r in results]
    true_counts = [r['true_count'] for r in results]
    image_names = [Path(r['image_path']).stem for r in results]
    
    # 1. Scatter plot: Predicted vs True counts
    plt.figure(figsize=(10, 8))
    plt.scatter(true_counts, pred_counts, alpha=0.7, s=50)
    
    # Perfect prediction line
    min_val = min(min(true_counts), min(pred_counts))
    max_val = max(max(true_counts), max(pred_counts))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Count')
    plt.ylabel('Predicted Count')
    plt.title('Predicted vs True Walnut Counts (Binary Classifier)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = np.corrcoef(true_counts, pred_counts)[0, 1] ** 2
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'count_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Residual plot
    residuals = np.array(pred_counts) - np.array(true_counts)
    plt.figure(figsize=(10, 6))
    plt.scatter(true_counts, residuals, alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('True Count')
    plt.ylabel('Residuals (Predicted - True)')
    plt.title('Residual Plot (Binary Classifier)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Binary Classifier)')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Per-image comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    x_pos = np.arange(len(image_names))
    width = 0.35
    
    ax1.bar(x_pos - width/2, true_counts, width, label='True Count', alpha=0.8)
    ax1.bar(x_pos + width/2, pred_counts, width, label='Predicted Count', alpha=0.8)
    ax1.set_xlabel('Image')
    ax1.set_ylabel('Count')
    ax1.set_title('Per-Image Count Comparison (Binary Classifier)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.split('_')[-1] for name in image_names], rotation=45)
    
    # Error bars
    errors = np.array(pred_counts) - np.array(true_counts)
    ax2.bar(x_pos, errors, alpha=0.7, color='red')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Image')
    ax2.set_ylabel('Error (Predicted - True)')
    ax2.set_title('Per-Image Prediction Errors (Binary Classifier)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.split('_')[-1] for name in image_names], rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_image_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def create_detection_visualizations(results: List[Dict], output_dir: str, num_samples: int = None):
    """Create visualizations showing detection results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # If num_samples is None, process all images
    if num_samples is None:
        num_samples = len(results)
    
    # Select samples with largest errors for visualization (if limiting)
    if num_samples < len(results):
        errors = [abs(r['pred_count'] - r['true_count']) for r in results]
        sorted_indices = np.argsort(errors)[::-1]  # Largest errors first
    else:
        # Process all images in order
        sorted_indices = list(range(len(results)))
    
    for i in range(min(num_samples, len(results))):
        idx = sorted_indices[i]
        result = results[idx]
        
        # Load original image
        img = cv2.imread(result['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with true annotations
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title(f"Original Image\nTrue Count: {result['true_count']:.0f}")
        axes[0, 0].axis('off')
        
        # Add true annotation points
        for x, y in result['coords']:
            axes[0, 0].plot(x, y, 'ro', markersize=4, alpha=0.8)
        
        # Predicted detections
        axes[0, 1].imshow(img_rgb)
        axes[0, 1].set_title(f"Predicted Detections\nCount: {result['pred_count']:.0f}")
        axes[0, 1].axis('off')
        
        # Add predicted detection points
        for x, y in result['pred_detections']:
            axes[0, 1].plot(x, y, 'go', markersize=4, alpha=0.8)
        
        # Overlay comparison
        axes[1, 0].imshow(img_rgb)
        axes[1, 0].set_title(f"Overlay Comparison\nError: {abs(result['pred_count'] - result['true_count']):.0f}")
        axes[1, 0].axis('off')
        
        # Add both true and predicted points
        for x, y in result['coords']:
            axes[1, 0].plot(x, y, 'ro', markersize=4, alpha=0.8, label='True' if x == result['coords'][0][0] and y == result['coords'][0][1] else "")
        for x, y in result['pred_detections']:
            axes[1, 0].plot(x, y, 'go', markersize=4, alpha=0.8, label='Predicted' if x == result['pred_detections'][0][0] and y == result['pred_detections'][0][1] else "")
        
        if result['coords'] or result['pred_detections']:
            axes[1, 0].legend()
        
        # Count comparison
        axes[1, 1].bar(['True', 'Predicted'], [result['true_count'], result['pred_count']], 
                      color=['red', 'green'], alpha=0.7)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Count Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Sample {i+1}: {Path(result['image_path']).stem}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'detection_sample_{i+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Detection visualizations saved to {output_dir}")


# -----------------------------
# Main Testing Function
# -----------------------------
def test_binary_classifier(model_path: str, test_dir: str, output_dir: str = "./binary_test_results", 
                          patch_size: int = 16, stride: int = 8, threshold: float = 0.2,
                          num_samples: int = None):
    """Test the binary classifier model and generate comprehensive evaluation"""
    
    print("🧠 Testing Binary Walnut Classifier")
    print("=" * 50)
    print(f"📊 Parameters:")
    print(f"   Patch size: {patch_size}")
    print(f"   Stride: {stride}")
    print(f"   Threshold: {threshold}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device selection - prioritize CUDA
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print(f"📦 Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    model = WalnutClassifier(input_size=patch_size, num_classes=2, dropout_rate=0.5)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test dataset
    print(f"📁 Loading test dataset from {test_dir}")
    test_dataset = TestBinaryDataset(test_dir, patch_size, stride)
    
    # Transform for patches
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Run inference
    print("🔍 Running inference...")
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataset):
            img_rgb = batch['image']
            coords = batch['coords']
            image_path = batch['image_path']
            original_shape = batch['original_shape']
            
            # Extract patches
            patches, patch_coords = test_dataset.extract_patches(img_rgb)
            
            if not patches:
                print(f"  Image {batch_idx + 1}/{len(test_dataset)}: No patches extracted")
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
            
            # Store results
            result = {
                'image_path': image_path,
                'pred_count': pred_count,
                'true_count': true_count,
                'pred_detections': pred_detections,
                'coords': coords,
                'original_shape': original_shape,
                'walnut_probs': walnut_probs,
                'patch_coords': patch_coords
            }
            results.append(result)
            
            print(f"  Image {batch_idx + 1}/{len(test_dataset)}: "
                  f"True={true_count}, Pred={pred_count}, "
                  f"Error={abs(pred_count - true_count)}")
    
    # Calculate overall metrics
    print("\n📊 Calculating metrics...")
    pred_counts = np.array([r['pred_count'] for r in results])
    true_counts = np.array([r['true_count'] for r in results])
    
    metrics = calculate_metrics(pred_counts, true_counts)
    
    # Print results
    print("\n" + "="*50)
    print("📈 EVALUATION RESULTS")
    print("="*50)
    print(f"Total Images: {len(results)}")
    print(f"MAE (Mean Absolute Error): {metrics['MAE']:.3f}")
    print(f"RMSE (Root Mean Square Error): {metrics['RMSE']:.3f}")
    print(f"MSE (Mean Square Error): {metrics['MSE']:.3f}")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.1f}%")
    print(f"SMAPE (Symmetric MAPE): {metrics['SMAPE']:.1f}%")
    print(f"R² (Coefficient of Determination): {metrics['R2']:.3f}")
    print(f"Pearson Correlation: {metrics['Pearson']:.3f}")
    print(f"Predicted Count Mean ± Std: {metrics['Pred_Mean']:.2f} ± {metrics['Pred_Std']:.2f}")
    print(f"True Count Mean ± Std: {metrics['True_Mean']:.2f} ± {metrics['True_Std']:.2f}")
    
    # Save detailed results
    detailed_results = {
        'metrics': metrics,
        'parameters': {
            'patch_size': patch_size,
            'stride': stride,
            'threshold': threshold
        },
        'per_image_results': [
            {
                'image_path': r['image_path'],
                'pred_count': float(r['pred_count']),
                'true_count': float(r['true_count']),
                'error': float(abs(r['pred_count'] - r['true_count'])),
                'relative_error': float(abs(r['pred_count'] - r['true_count']) / max(r['true_count'], 1e-6) * 100)
            }
            for r in results
        ],
        'model_info': {
            'model_path': model_path,
            'input_size': patch_size,
            'num_classes': 2,
            'dropout_rate': 0.5,
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
    }
    
    results_path = os.path.join(output_dir, 'binary_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to {results_path}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    create_visualizations(results, os.path.join(output_dir, 'plots'))
    create_detection_visualizations(results, os.path.join(output_dir, 'detections'), num_samples)
    
    print(f"\n✅ Testing completed! Results saved to {output_dir}")
    return detailed_results


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Test binary walnut classifier")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--test_dir", required=True, help="Path to test directory")
    parser.add_argument("--output", default="./binary_test_results", help="Output directory for results")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for sliding window (default: 16)")
    parser.add_argument("--stride", type=int, default=8, help="Stride for sliding window (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Classification threshold (default: 0.2)")
    parser.add_argument("--num_detection_viz", type=int, default=None, help="Number of detection visualizations to generate (default: all images)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory not found: {args.test_dir}")
        return
    
    # Run testing
    test_binary_classifier(args.model, args.test_dir, args.output, 
                          args.patch_size, args.stride, args.threshold, args.num_detection_viz)


if __name__ == "__main__":
    main()
