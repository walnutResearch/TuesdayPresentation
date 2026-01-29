#!/usr/bin/env python3
"""
Improved density estimation model: multi-channel CNN for walnut density (count) prediction, balancing
accuracy and speed. Defines ModelConfig, MultiChannelTransform, and training script. Alternative to density_model.py.

How to run (training):
  python improved_density_model.py --synthetic_data path/to/data [--output path] [--epochs 25]

Use as module: from improved_density_model import ModelConfig, MultiChannelTransform, ...
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

@dataclass
class ModelConfig:
    """Configuration for the improved density model"""
    input_channels: int = 6  # RGB + Grayscale + Edge magnitude + Edge direction
    hidden_dim: int = 64
    dropout_rate: float = 0.2
    patch_size: int = 32
    learning_rate: float = 0.001
    batch_size: int = 12
    num_epochs: int = 25

class MultiChannelTransform:
    """Transform to create multi-channel input"""
    
    def __init__(self, patch_size: int = 32):
        self.patch_size = patch_size
    
    def __call__(self, image):
        """Convert BGR image to multi-channel format"""
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to patch size
        image = cv2.resize(image, (self.patch_size, self.patch_size))
        
        # Create grayscale channel
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create edge magnitude and direction
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_direction = np.arctan2(sobely, sobelx)
        
        # Normalize
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)
        edge_direction = (edge_direction + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        
        # Stack channels: [R, G, B, Grayscale, Edge_Magnitude, Edge_Direction]
        multi_channel = np.stack([
            image[:, :, 0],  # R
            image[:, :, 1],  # G
            image[:, :, 2],  # B
            gray,            # Grayscale
            edge_magnitude,  # Edge magnitude
            edge_direction   # Edge direction
        ], axis=2)
        
        return multi_channel.astype(np.float32)

class DensityDataset(Dataset):
    """Dataset for density estimation training"""
    
    def __init__(self, images_dir: str, density_maps_dir: str, 
                 patch_size: int = 32, transform=None):
        self.images_dir = Path(images_dir)
        self.density_maps_dir = Path(density_maps_dir)
        self.patch_size = patch_size
        self.transform = transform or MultiChannelTransform(patch_size)
        
        # Get all image files
        self.image_files = list(self.images_dir.glob("*.png"))
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in dataset")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Load corresponding density map
        density_path = self.density_maps_dir / f"density_{image_path.stem.split('_')[-1]}.npy"
        if not density_path.exists():
            raise ValueError(f"Density map not found: {density_path}")
        
        density_map = np.load(density_path)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        density_tensor = torch.from_numpy(density_map).float()
        
        return image_tensor, density_tensor

class ImprovedDensityNet(nn.Module):
    """
    Improved CNN for density estimation with better architecture:
    - Multi-scale feature extraction
    - Residual connections
    - Attention mechanism
    - Better feature fusion
    """
    
    def __init__(self, config: ModelConfig):
        super(ImprovedDensityNet, self).__init__()
        self.config = config
        
        # Multi-scale feature extractors
        self.scale1 = self._make_scale_block(config.input_channels, 32, 1)   # 1x scale
        self.scale2 = self._make_scale_block(config.input_channels, 32, 2)   # 2x scale
        self.scale3 = self._make_scale_block(config.input_channels, 32, 4)   # 4x scale
        
        # Feature fusion with attention
        total_channels = 32 * 3
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, config.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(config.dropout_rate)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Residual blocks for better feature learning
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(config.hidden_dim, config.hidden_dim)
            for _ in range(2)
        ])
        
        # Density prediction head with skip connections
        self.density_head = nn.Sequential(
            nn.Conv2d(config.hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(config.dropout_rate),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(config.dropout_rate),
            
            # Adaptive pooling to match target size (16x16 for density maps)
            nn.AdaptiveAvgPool2d((16, 16)),
            
            nn.Conv2d(32, 16, 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, 1),  # Output density map
            nn.ReLU(inplace=True)  # Ensure non-negative densities
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_scale_block(self, in_channels: int, out_channels: int, scale: int) -> nn.Module:
        """Create a scale-specific feature extraction block"""
        layers = []
        
        # Downsampling if needed
        if scale > 1:
            layers.append(nn.AvgPool2d(scale, stride=scale))
        
        # Feature extraction
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.config.dropout_rate),
            
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        return nn.Sequential(*layers)
    
    def _make_residual_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.config.dropout_rate),
            
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Multi-scale feature extraction
        feat1 = self.scale1(x)  # 1x scale
        feat2 = self.scale2(x)  # 2x scale
        feat3 = self.scale3(x)  # 4x scale
        
        # Upsample features to same size
        feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat3 = F.interpolate(feat3, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate multi-scale features
        fused = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Feature fusion
        fused = self.fusion(fused)
        
        # Apply attention
        attention_weights = self.attention(fused)
        fused = fused * attention_weights
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            residual = residual_block(fused)
            fused = fused + residual  # Skip connection
            fused = F.relu(fused)
        
        # Density prediction
        density = self.density_head(fused)
        
        return density

class ImprovedTrainer:
    """Improved trainer for density estimation model"""
    
    def __init__(self, model: ImprovedDensityNet, config: ModelConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 8
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for images, density_maps in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions.squeeze(1), density_maps)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, density_maps in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions.squeeze(1), density_maps)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              model_path: str) -> Tuple[List[float], List[float]]:
        """Train the model with early stopping"""
        print(f"🚀 Starting improved training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_path)
                self.patience_counter = 0
                print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        
        return self.train_losses, self.val_losses

def create_data_loaders(synthetic_data_dir: str, config: ModelConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Create dataset
    dataset = DensityDataset(
        images_dir=os.path.join(synthetic_data_dir, "synthetic_images"),
        density_maps_dir=os.path.join(synthetic_data_dir, "density_maps"),
        patch_size=config.patch_size
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"📊 Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def plot_training_history(train_losses: List[float], val_losses: List[float], 
                         output_dir: str):
    """Plot training history"""
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Improved Density Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss difference
    plt.subplot(2, 1, 2)
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    plt.plot(loss_diff, label='|Train - Val| Loss', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Indicator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_improved.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Improved Density Estimation Model Training')
    parser.add_argument('--synthetic_data', type=str, required=True,
                       help='Path to synthetic data directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=12,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create config
    config = ModelConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    print("🧠 Improved Density Estimation Model Training")
    print("=" * 50)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args.synthetic_data, config)
    
    # Create model
    model = ImprovedDensityNet(config)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer - prioritize MPS for Apple Silicon, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"🖥️  Using device: {device}")
    
    trainer = ImprovedTrainer(model, config, device)
    
    # Train model
    model_path = os.path.join(args.output, "improved_density_model.pth")
    train_losses, val_losses = trainer.train(train_loader, val_loader, model_path)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, args.output)
    
    print(f"\n✅ Training completed!")
    print(f"💾 Best model saved to: {model_path}")
    print(f"📈 Final validation loss: {trainer.best_val_loss:.4f}")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
