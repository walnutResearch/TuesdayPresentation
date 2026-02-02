#!/usr/bin/env python3
"""
Test the trained density estimation model on a test image directory: loads model, runs patch-based density
prediction, sums to get per-image count, and compares to ground truth. Reports MAE, RMSE, correlation, etc.

How to run:
  python test_density_model.py --model path/to/density_model.pth --test_dir path/to/test [--output path] [--patch_size 32]

Use --help for all options. Expects annotations in a sibling annotations/ or as specified.
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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial.distance import cdist

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# -----------------------------
# Model Architecture (copied from training code)
# -----------------------------
class MultiScaleDensityNet(nn.Module):
    """Multiscale CNN for density estimation"""

    def __init__(self, input_channels: int = 6, hidden_dim: int = 64, num_scales: int = 2, 
                 dropout_rate: float = 0.2, no_bn: bool = False):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.dropout_rate = dropout_rate
        self.no_bn = no_bn

        # Scale blocks
        self.scale1 = self._make_scale_block(input_channels, 32, scale=1)
        self.scale2 = self._make_scale_block(input_channels, 32, scale=2)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(32 * 2, hidden_dim, 3, padding=1),
            self._norm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

        # Density head
        head_in = hidden_dim
        head_mid = 16 if head_in >= 16 else head_in
        self.density_head = nn.Sequential(
            nn.Conv2d(head_in, head_mid, 3, padding=1),
            self._norm(head_mid),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(head_mid, 1, 1),
        )

        # Learnable calibration params
        self.scale_param = nn.Parameter(torch.tensor(-1.8))
        self.bias_param = nn.Parameter(torch.tensor(-7.2))

    def _norm(self, c: int) -> nn.Module:
        if self.no_bn:
            groups = min(8, c)
            groups = max(1, groups)
            return nn.GroupNorm(num_groups=groups, num_channels=c)
        else:
            return nn.BatchNorm2d(c)

    def _make_scale_block(self, in_channels: int, out_channels: int, scale: int) -> nn.Module:
        if scale == 1:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                self._norm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                self._norm(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                self._norm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                self._norm(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        h, w = feat1.size(2), feat1.size(3)
        feat2 = F.interpolate(feat2, size=(h, w), mode="bilinear", align_corners=False)
        fused_features = torch.cat([feat1, feat2], dim=1)
        fused = self.fusion(fused_features)
        logits = self.density_head(fused)
        out = F.softplus(logits) * F.softplus(self.scale_param) + F.softplus(self.bias_param)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=0.0)
        return out


# -----------------------------
# Transform (copied from training code)
# -----------------------------
class MultiChannelTransform:
    """Transform images to multi-channel input for density estimation"""

    def __init__(self, patch_size: int = 32, max_side: int = 512):
        self.patch_size = patch_size
        self.max_side = max_side

    def _resize_to_multiple(self, image, target_w, target_h):
        new_w = (target_w // self.patch_size) * self.patch_size
        new_h = (target_h // self.patch_size) * self.patch_size
        new_w = max(new_w, self.patch_size)
        new_h = max(new_h, self.patch_size)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to multi-channel input"""
        h, w = image.shape[:2]

        # Cap longest side
        scale = max(h, w) / float(self.max_side)
        if scale > 1.0:
            target_w = int(w / scale)
            target_h = int(h / scale)
        else:
            target_w, target_h = w, h

        image_resized = self._resize_to_multiple(image, target_w, target_h)

        # Normalize to [0,1]
        image_float = image_resized.astype(np.float32) / 255.0

        # Build channels
        channels = []
        # Convert BGR->RGB
        rgb = image_float[:, :, ::-1]
        channels.extend([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])

        # Grayscale (L channel from Lab)
        lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
        grayscale = lab[:, :, 0].astype(np.float32) / 255.0
        channels.append(grayscale)

        # Sobel edges
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        den = float(edge_magnitude.max())
        if den > 1e-8:
            edge_magnitude /= den
        else:
            edge_magnitude[:] = 0.0
        channels.append(edge_magnitude.astype(np.float32))

        edge_direction = np.arctan2(sobel_y, sobel_x)
        edge_direction = (edge_direction + np.pi) / (2 * np.pi)
        channels.append(edge_direction.astype(np.float32))

        multi_channel = np.stack(channels, axis=2)
        return multi_channel


# -----------------------------
# Test Dataset
# -----------------------------
class TestDensityDataset(Dataset):
    """Test dataset for density estimation"""

    def __init__(self, test_dir: str, patch_size: int = 32, max_side: int = 512):
        self.test_dir = Path(test_dir)
        self.images_dir = self.test_dir
        self.annotations_dir = self.test_dir / "annotations"
        self.patch_size = patch_size
        self.transform = MultiChannelTransform(patch_size, max_side)

        # Get all image files
        self.image_files = sorted(self.images_dir.glob("*.png"))
        print(f"Found {len(self.image_files)} test images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

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
                    elif len(parts) == 1 and parts[0].isdigit():
                        # Skip single numbers (might be line numbers or other data)
                        continue
                except ValueError:
                    continue
        
        # Debug: print first few coords
        if len(coords) > 0:
            print(f"  Parsed {len(coords)} coordinates, first few: {coords[:3]}")
        else:
            print(f"  Warning: No coordinates parsed from {annotation_path}")

        # Create ground truth density map
        density_map = self._create_density_map(img.shape[:2], coords)
        
        # Verify count matches annotation
        calculated_count = density_map.sum()
        expected_count = len(coords)
        if abs(calculated_count - expected_count) > 0.1:
            print(f"  Warning: Count mismatch for {image_path.name}: expected {expected_count}, got {calculated_count:.1f}")

        # Transform image
        multi_channel = self.transform(img)

        # To tensors
        image_tensor = torch.from_numpy(multi_channel).permute(2, 0, 1).float()
        density_tensor = torch.from_numpy(density_map).float()

        return {
            'image': image_tensor,
            'density': density_tensor,
            'coords': coords,
            'image_path': str(image_path),
            'original_shape': img.shape[:2]
        }

    def _create_density_map(self, img_shape: Tuple[int, int], coords: List[Tuple[int, int]]) -> np.ndarray:
        """Create density map from coordinates - each point contributes exactly 1.0 to total count"""
        h, w = img_shape
        
        # Resize to match model output (16x16)
        target_h, target_w = 16, 16
        scale_y = target_h / h
        scale_x = target_w / w
        
        density_map = np.zeros((target_h, target_w), dtype=np.float32)
        
        for x, y in coords:
            # Scale coordinates
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            
            # Clamp to valid range
            scaled_x = max(0, min(target_w - 1, scaled_x))
            scaled_y = max(0, min(target_h - 1, scaled_y))
            
            # Add exactly 1.0 at the scaled coordinate
            density_map[scaled_y, scaled_x] += 1.0
        
        return density_map


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
    plt.title('Predicted vs True Walnut Counts')
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
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
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
    ax1.set_title('Per-Image Count Comparison')
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
    ax2.set_title('Per-Image Prediction Errors')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.split('_')[-1] for name in image_names], rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_image_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def create_density_visualizations(results: List[Dict], output_dir: str, num_samples: int = None):
    """Create visualizations showing density maps"""
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
        
        # Get density maps
        pred_density = result['pred_density'].numpy().squeeze()
        true_density = result['true_density'].numpy().squeeze()
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image with annotations
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title(f"Original Image\nTrue Count: {result['true_count']:.1f}")
        axes[0, 0].axis('off')
        
        # Add annotation points
        for x, y in result['coords']:
            axes[0, 0].plot(x, y, 'ro', markersize=3, alpha=0.8)
        
        # Predicted density map
        im1 = axes[0, 1].imshow(pred_density, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title(f"Predicted Density\nCount: {result['pred_count']:.1f}")
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # True density map
        im2 = axes[0, 2].imshow(true_density, cmap='hot', interpolation='nearest')
        axes[0, 2].set_title(f"True Density\nCount: {result['true_count']:.1f}")
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Difference map
        diff_map = pred_density - true_density
        im3 = axes[1, 0].imshow(diff_map, cmap='RdBu_r', interpolation='nearest')
        axes[1, 0].set_title(f"Difference Map\nError: {abs(result['pred_count'] - result['true_count']):.1f}")
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Histogram comparison
        axes[1, 1].hist(pred_density.flatten(), bins=20, alpha=0.7, label='Predicted', density=True)
        axes[1, 1].hist(true_density.flatten(), bins=20, alpha=0.7, label='True', density=True)
        axes[1, 1].set_xlabel('Density Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Density Value Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Count comparison
        axes[1, 2].bar(['True', 'Predicted'], [result['true_count'], result['pred_count']], 
                      color=['blue', 'red'], alpha=0.7)
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Count Comparison')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f"Sample {i+1}: {Path(result['image_path']).stem}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'density_sample_{i+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Density visualizations saved to {output_dir}")


# -----------------------------
# Main Testing Function
# -----------------------------
def test_density_model(model_path: str, test_dir: str, output_dir: str = "./density_test_results", 
                      scale_factor: float = 1.0, bias_factor: float = 0.0, 
                      min_prediction: float = 0.0, max_prediction: float = float('inf'),
                      num_samples: int = None):
    """Test the density model and generate comprehensive evaluation
    
    Args:
        model_path: Path to trained model
        test_dir: Path to test directory
        output_dir: Output directory for results
        scale_factor: Multiplier to scale predictions (default: 1.0)
        bias_factor: Additive bias to predictions (default: 0.0)
        min_prediction: Minimum allowed prediction value (default: 0.0)
        max_prediction: Maximum allowed prediction value (default: no limit)
    """
    
    print("🧠 Testing Density Model")
    print("=" * 50)
    print(f"📊 Prediction adjustments:")
    print(f"   Scale factor: {scale_factor}")
    print(f"   Bias factor: {bias_factor}")
    print(f"   Min prediction: {min_prediction}")
    print(f"   Max prediction: {max_prediction if max_prediction != float('inf') else 'No limit'}")
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
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config from checkpoint
    config_dict = checkpoint.get('config', {})
    input_channels = config_dict.get('input_channels', 6)
    hidden_dim = config_dict.get('hidden_dim', 64)
    dropout_rate = config_dict.get('dropout_rate', 0.2)
    no_bn = config_dict.get('no_bn', False)
    
    # Create model
    model = MultiScaleDensityNet(
        input_channels=input_channels,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        no_bn=no_bn
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test dataset
    print(f"📁 Loading test dataset from {test_dir}")
    test_dataset = TestDensityDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Run inference
    print("🔍 Running inference...")
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            image = batch['image'].to(device)
            true_density = batch['density']
            coords = batch['coords'][0]  # Remove batch dimension
            image_path = batch['image_path'][0]
            original_shape = batch['original_shape'][0]
            
            # Fix coords: convert from list of tensors back to list of tuples
            if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], torch.Tensor):
                # Coords are stored as alternating x, y tensors
                coords_list = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        x = coords[i].item()
                        y = coords[i + 1].item()
                        coords_list.append((x, y))
                coords = coords_list
            
            # Forward pass
            pred_density = model(image)
            
            # Calculate counts
            pred_count = pred_density.squeeze().sum().item()
            true_count = true_density.sum().item()
            
            # Apply scaling and bias adjustments
            pred_count = pred_count * scale_factor + bias_factor
            pred_count = max(min_prediction, min(max_prediction, pred_count))
            
            # Store results
            result = {
                'image_path': image_path,
                'pred_count': pred_count,
                'true_count': true_count,
                'pred_density': pred_density.cpu(),
                'true_density': true_density,
                'coords': coords,
                'original_shape': original_shape
            }
            results.append(result)
            
            print(f"  Image {batch_idx + 1}/{len(test_loader)}: "
                  f"True={true_count:.1f}, Pred={pred_count:.1f}, "
                  f"Error={abs(pred_count - true_count):.1f}")
    
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
            'input_channels': input_channels,
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate,
            'no_bn': no_bn,
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
    }
    
    results_path = os.path.join(output_dir, 'density_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to {results_path}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    create_visualizations(results, os.path.join(output_dir, 'plots'))
    create_density_visualizations(results, os.path.join(output_dir, 'density_maps'), num_samples)
    
    print(f"\n✅ Testing completed! Results saved to {output_dir}")
    return detailed_results


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Test density estimation model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--test_dir", required=True, help="Path to test directory")
    parser.add_argument("--output", default="./density_test_results", help="Output directory for results")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Multiplier to scale predictions (default: 1.0)")
    parser.add_argument("--bias_factor", type=float, default=0.0, help="Additive bias to predictions (default: 0.0)")
    parser.add_argument("--min_prediction", type=float, default=0.0, help="Minimum allowed prediction value (default: 0.0)")
    parser.add_argument("--max_prediction", type=float, default=float('inf'), help="Maximum allowed prediction value (default: no limit)")
    parser.add_argument("--num_density_viz", type=int, default=None, help="Number of density visualizations to generate (default: all images)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory not found: {args.test_dir}")
        return
    
    # Run testing
    test_density_model(args.model, args.test_dir, args.output, 
                      args.scale_factor, args.bias_factor, 
                      args.min_prediction, args.max_prediction, args.num_density_viz)


if __name__ == "__main__":
    main()
