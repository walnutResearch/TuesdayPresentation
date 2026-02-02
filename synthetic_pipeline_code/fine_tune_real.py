#!/usr/bin/env python3
"""
Fine-tune the density estimation model on real annotated images to close the synthetic–real domain gap.
Loads a pretrained model (e.g. from density_model.py training), trains on real patches with density targets.

How to run:
  python fine_tune_real.py --pretrained_model path/to/density_model.pth --real_data path/to/annotated_dataset [--output path] [--epochs 15]

Use --help for all options. Run from synthetic_pipeline_code/ so density_model is importable.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import argparse
from dataclasses import dataclass
import random

# Import from density_model
from density_model import MultiScaleDensityNet, ModelConfig, MultiChannelTransform

class RealDataDataset(Dataset):
    """Dataset for real annotated data fine-tuning"""
    
    def __init__(self, images_dir: str, annotations_dir: str, 
                 patch_size: int = 32, transform=None, augment: bool = True):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.patch_size = patch_size
        self.transform = transform or MultiChannelTransform(patch_size)
        self.augment = augment
        
        # Get all image files
        self.image_files = list(self.images_dir.glob("*.png"))
        self.image_files.extend(list(self.images_dir.glob("*.jpg")))
        self.image_files.extend(list(self.images_dir.glob("*.JPG")))
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} real images for fine-tuning")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Load annotations
        annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            raise ValueError(f"Annotation file not found: {annotation_path}")
        
        walnut_centers = self._load_annotations(annotation_path)
        
        # Create density map from annotations
        density_map = self._create_density_map(image, walnut_centers)
        
        # Apply data augmentation if enabled
        if self.augment:
            image, density_map = self._augment_data(image, density_map)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        density_tensor = torch.from_numpy(density_map).float()
        
        return image_tensor, density_tensor
    
    def _load_annotations(self, annotation_path: Path) -> List[Tuple[int, int]]:
        """Load walnut center annotations from file"""
        walnut_centers = []
        
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = int(float(parts[0])), int(float(parts[1]))
                        walnut_centers.append((x, y))
        
        return walnut_centers
    
    def _create_density_map(self, image: np.ndarray, walnut_centers: List[Tuple[int, int]]) -> np.ndarray:
        """Create density map from walnut center annotations"""
        
        h, w = image.shape[:2]
        density_h = h // self.patch_size
        density_w = w // self.patch_size
        
        density_map = np.zeros((density_h, density_w), dtype=np.float32)
        
        for center_x, center_y in walnut_centers:
            # Calculate which patch this center belongs to
            patch_x = center_x // self.patch_size
            patch_y = center_y // self.patch_size
            
            # Ensure within bounds
            patch_x = min(patch_x, density_w - 1)
            patch_y = min(patch_y, density_h - 1)
            
            # Add to density map
            density_map[patch_y, patch_x] += 1.0
        
        return density_map
    
    def _augment_data(self, image: np.ndarray, density_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to image and density map"""
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            density_map = np.flip(density_map, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            density_map = np.flip(density_map, 0)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                density_map = np.rot90(density_map, -1)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
                density_map = np.rot90(density_map, 2)
            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                density_map = np.rot90(density_map, 1)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        return image, density_map

class FineTuner:
    """Fine-tune density estimation model on real data"""
    
    def __init__(self, model: nn.Module, config: ModelConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function - MSE for density regression
        self.criterion = nn.MSELoss()
        
        # Use lower learning rate for fine-tuning
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate * 0.1)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, density_maps) in enumerate(train_loader):
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(images)
            
            # Calculate loss
            loss = self.criterion(predictions.squeeze(1), density_maps)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, density_maps in val_loader:
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                loss = self.criterion(predictions.squeeze(1), density_maps)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def fine_tune(self, train_loader: DataLoader, val_loader: DataLoader, 
                  num_epochs: int = 20, save_path: str = "fine_tuned_model.pth"):
        """Fine-tune the model on real data"""
        
        print("🔧 Starting fine-tuning on real data...")
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nFine-tuning Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'fine_tuned': True
                }, save_path)
                print(f"💾 Saved best fine-tuned model (Val Loss: {val_loss:.6f})")
        
        print("✅ Fine-tuning completed!")
        return self.train_losses, self.val_losses

def load_pretrained_model(model_path: str, config: ModelConfig) -> nn.Module:
    """Load pre-trained model for fine-tuning"""
    
    print(f"📥 Loading pre-trained model from: {model_path}")
    
    # Create model
    model = MultiScaleDensityNet(config)
    
    # Load pre-trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✅ Pre-trained model loaded successfully")
    
    return model

def create_real_data_loaders(real_data_dir: str, config: ModelConfig, 
                           val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for real data fine-tuning"""
    
    # Use validation data from the original dataset for fine-tuning
    val_images_dir = os.path.join(real_data_dir, 'val', 'images')
    val_annotations_dir = os.path.join(real_data_dir, 'val', 'annotations')
    
    if not os.path.exists(val_images_dir) or not os.path.exists(val_annotations_dir):
        raise ValueError("Real data directories not found")
    
    # Create full dataset
    full_dataset = RealDataDataset(
        val_images_dir, val_annotations_dir, config.patch_size, augment=True
    )
    
    # Split dataset for fine-tuning
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders with smaller batch size for fine-tuning
    train_loader = DataLoader(
        train_dataset, 
        batch_size=max(1, config.batch_size // 2),  # Smaller batch size
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=max(1, config.batch_size // 2),  # Smaller batch size
        shuffle=False, 
        num_workers=2
    )
    
    print(f"📊 Real data split: {train_size} train, {val_size} val")
    
    return train_loader, val_loader

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cuda') -> Dict:
    """Evaluate model on test data"""
    
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0
    num_batches = 0
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for images, density_maps in test_loader:
            images = images.to(device)
            density_maps = density_maps.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Calculate metrics
            loss = criterion(predictions.squeeze(1), density_maps)
            total_loss += loss.item()
            
            # Calculate MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(predictions.squeeze(1) - density_maps))
            total_mae += mae.item()
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = torch.mean(torch.abs((predictions.squeeze(1) - density_maps) / (density_maps + 1e-6)))
            total_mape += mape.item()
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_mape = total_mape / num_batches
    
    return {
        'mse_loss': avg_loss,
        'mae': avg_mae,
        'mape': avg_mape
    }

def main():
    """Main function for fine-tuning on real data"""
    
    parser = argparse.ArgumentParser(description="Fine-tune density model on real data")
    parser.add_argument("--pretrained_model", required=True,
                       help="Path to pre-trained model")
    parser.add_argument("--real_data", required=True,
                       help="Path to real annotated data directory")
    parser.add_argument("--output", default="./models",
                       help="Output directory for fine-tuned models")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for fine-tuning")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--patch_size", type=int, default=32,
                       help="Patch size for density maps")
    
    args = parser.parse_args()
    
    print("🔧 Fine-tuning on Real Data")
    print("=" * 40)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create model configuration
    config = ModelConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patch_size=args.patch_size
    )
    
    # Load pre-trained model
    model = load_pretrained_model(args.pretrained_model, config)
    
    # Create real data loaders
    train_loader, val_loader = create_real_data_loaders(args.real_data, config)
    
    # Device selection - prioritize CUDA
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"🖥️  Using device: {device}")
    
    fine_tuner = FineTuner(model, config, device)
    
    # Fine-tune model
    model_path = os.path.join(args.output, "fine_tuned_density_model.pth")
    train_losses, val_losses = fine_tuner.fine_tune(
        train_loader, val_loader, args.epochs, model_path
    )
    
    # Evaluate on test data
    print("\n📊 Evaluating on test data...")
    test_images_dir = os.path.join(args.real_data, 'test', 'images')
    test_annotations_dir = os.path.join(args.real_data, 'test', 'annotations')
    
    if os.path.exists(test_images_dir) and os.path.exists(test_annotations_dir):
        test_dataset = RealDataDataset(
            test_images_dir, test_annotations_dir, config.patch_size, augment=False
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        metrics = evaluate_model(model, test_loader, device)
        print(f"Test MSE Loss: {metrics['mse_loss']:.6f}")
        print(f"Test MAE: {metrics['mae']:.6f}")
        print(f"Test MAPE: {metrics['mape']:.6f}")
    
    # Save fine-tuning history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config.__dict__,
        'fine_tuning_epochs': args.epochs
    }
    
    history_path = os.path.join(args.output, "fine_tuning_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Fine-tuning completed! Model saved to: {model_path}")

if __name__ == "__main__":
    main()
