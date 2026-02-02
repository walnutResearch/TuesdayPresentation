#!/usr/bin/env python3
"""
Train the binary walnut classifier (CNN) on positive/negative 32x32 patches. Reads from positive/ and negative/
directories, uses data augmentation, and saves the best model (e.g. walnut_classifier.pth) and training metrics.

Features:
- Class weight balancing: Automatically calculates and applies class weights to handle imbalanced datasets
- Hard negative mining: Optionally finds and saves difficult negative samples after training
- Data augmentation: Random flips, rotations, and color jitter

IMPORTANT NOTE ON DATA SPLITTING:
The current train/val split is at PATCH level, not IMAGE level. If multiple patches come from
the same source image, they may be split between train and validation sets, causing data leakage.
For proper image-level splitting, patches should preserve source image information (e.g., in filename).

How to run:
  python binary_classifier.py --dataset_dir path/to/dataset [--output_dir models] [--epochs 50]
  python binary_classifier.py --dataset_dir path/to/dataset --mine_hard_negatives  # Mine hard negatives after training

Use --help for all options. Optimized for CUDA/GPU.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class BinaryWalnutDataset(Dataset):
    """Dataset for binary walnut classification"""
    
    def __init__(self, positive_dir: str, negative_dir: str, transform=None, augment=True):
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        self.transform = transform
        self.augment = augment
        
        # Get all image files
        self.positive_files = list(self.positive_dir.glob("*.png"))
        self.negative_files = list(self.negative_dir.glob("*.png"))
        
        print(f"Found {len(self.positive_files)} positive samples")
        print(f"Found {len(self.negative_files)} negative samples")
        
        # Create labels
        self.images = []
        self.labels = []
        
        # Add positive samples (label = 1)
        for img_path in self.positive_files:
            self.images.append(str(img_path))
            self.labels.append(1)
        
        # Add negative samples (label = 0)
        for img_path in self.negative_files:
            self.images.append(str(img_path))
            self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get label
        label = self.labels[idx]
        
        return img, label

class AugmentationTransform:
    """Custom augmentation for walnut classification"""
    
    def __init__(self, patch_size: int = 32, training: bool = True):
        self.patch_size = patch_size
        self.training = training
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms
        if training:
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((patch_size, patch_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.augment_transform = self.base_transform
    
    def __call__(self, img):
        if self.training and random.random() < 0.7:  # 70% chance of augmentation
            return self.augment_transform(img)
        else:
            return self.base_transform(img)

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

def calculate_class_weights(num_positive: int, num_negative: int, device: str = 'cpu'):
    """Calculate weights to handle class imbalance
    
    Args:
        num_positive: Number of positive (walnut) samples
        num_negative: Number of negative (background) samples
        device: Device to place weights on
        
    Returns:
        torch.Tensor: Class weights [weight_negative, weight_positive]
    """
    total = num_positive + num_negative
    if num_positive == 0 or num_negative == 0:
        # If one class is missing, use equal weights
        return torch.tensor([1.0, 1.0], device=device)
    
    weight_positive = total / (2 * num_positive)
    weight_negative = total / (2 * num_negative)
    return torch.tensor([weight_negative, weight_positive], device=device)

class BinaryTrainer:
    """Trainer for binary walnut classification"""
    
    def __init__(self, model: WalnutClassifier, device: str, learning_rate: float = 0.001,
                 class_weights: torch.Tensor = None):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Loss and optimizer
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print(f"📊 Using class weights: negative={class_weights[0]:.3f}, positive={class_weights[1]:.3f}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_precision = 0.0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1 (walnut)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Calculate additional metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='binary'),
            'recall': recall_score(all_labels, all_predictions, average='binary'),
            'f1': f1_score(all_labels, all_predictions, average='binary'),
            'auc': roc_auc_score(all_labels, all_probabilities)
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, save_path: str = "walnut_classifier.pth"):
        """Train the model"""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
            print(f"F1: {metrics['f1']:.3f}, AUC: {metrics['auc']:.3f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Track best validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            
            # Save best model based on precision
            if metrics['precision'] > self.best_precision:
                self.best_precision = metrics['precision']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'metrics': metrics
                }, save_path)
                print(f"💾 Saved best model (Precision: {metrics['precision']:.3f}, Val Acc: {val_acc:.2f}%)")
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"   Best precision: {self.best_precision:.3f}")
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

def mine_hard_negatives(model: WalnutClassifier, negative_dir: str, transform, 
                        device: str, threshold: float = 0.5, batch_size: int = 32):
    """Find negative patches the model incorrectly classifies as positive (hard negatives)
    
    Args:
        model: Trained model to use for mining
        negative_dir: Directory containing negative patches
        transform: Transform to apply to images
        device: Device to run inference on
        threshold: Confidence threshold above which to consider a false positive
        batch_size: Batch size for inference
        
    Returns:
        List of tuples: [(path, confidence_score), ...] sorted by confidence (highest first)
    """
    model.eval()
    hard_negatives = []
    
    negative_path = Path(negative_dir)
    negative_files = list(negative_path.glob("*.png"))
    
    print(f"🔍 Mining hard negatives from {len(negative_files)} negative patches...")
    print(f"   Threshold: {threshold:.2f}")
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(negative_files), batch_size), desc="Mining"):
        batch_files = negative_files[i:i+batch_size]
        batch_tensors = []
        batch_paths = []
        
        for path in batch_files:
            try:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img)
                batch_tensors.append(img_tensor)
                batch_paths.append(path)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                continue
        
        if len(batch_tensors) == 0:
            continue
        
        # Stack and move to device
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            walnut_probs = probabilities[:, 1].cpu().numpy()  # P(walnut)
        
        # Find hard negatives (high confidence false positives)
        for j, prob in enumerate(walnut_probs):
            if prob > threshold:
                hard_negatives.append((batch_paths[j], float(prob)))
    
    # Sort by confidence (most confident mistakes first)
    hard_negatives.sort(key=lambda x: x[1], reverse=True)
    
    print(f"✅ Found {len(hard_negatives)} hard negatives (out of {len(negative_files)} total)")
    if len(hard_negatives) > 0:
        print(f"   Confidence range: {hard_negatives[-1][1]:.3f} - {hard_negatives[0][1]:.3f}")
    
    return hard_negatives

def create_data_loaders(dataset_dir: str, batch_size: int = 32, 
                       patch_size: int = 32, val_split: float = 0.2,
                       device: str = 'cpu') -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Create training and validation data loaders
    
    NOTE: Current implementation splits at PATCH level, not IMAGE level.
    If multiple patches come from the same source image, they may be split
    between train/val, causing data leakage. For proper image-level splitting,
    patches need to preserve source image information (e.g., in filename).
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader  
        class_weights: Tensor of class weights for loss function
    """
    
    # Create datasets
    positive_dir = os.path.join(dataset_dir, "positive")
    negative_dir = os.path.join(dataset_dir, "negative")
    
    # Check if real data exists
    real_pos_dir = os.path.join(dataset_dir, "real_positive")
    real_neg_dir = os.path.join(dataset_dir, "real_negative")
    
    if os.path.exists(real_pos_dir) and os.path.exists(real_neg_dir) and \
       len(list(Path(real_pos_dir).glob("*.png"))) > 0 and len(list(Path(real_neg_dir).glob("*.png"))) > 0:
        print("Using real validation data")
        # Use real data for validation
        val_positive_dir = real_pos_dir
        val_negative_dir = real_neg_dir
    else:
        print("⚠️  Using synthetic data split for validation")
        print("⚠️  WARNING: Split is at PATCH level, not IMAGE level!")
        print("⚠️  Patches from the same source image may be in both train and val.")
        # Split synthetic data
        val_positive_dir = positive_dir
        val_negative_dir = negative_dir
    
    # Create full dataset first
    full_dataset = BinaryWalnutDataset(
        positive_dir, negative_dir, 
        transform=AugmentationTransform(patch_size, training=True),
        augment=True
    )
    
    # Calculate class weights from full dataset
    num_positive = len(full_dataset.positive_files)
    num_negative = len(full_dataset.negative_files)
    class_weights = calculate_class_weights(num_positive, num_negative, device)
    
    print(f"📊 Dataset balance: {num_positive} positives, {num_negative} negatives")
    print(f"📊 Class ratio: {num_negative/num_positive:.2f}:1 (neg:pos)")
    
    # Split dataset if no real validation data
    if val_positive_dir == positive_dir:  # Using synthetic data split
        from torch.utils.data import random_split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset.transform = AugmentationTransform(patch_size, training=False)
        val_dataset.dataset.augment = False
    else:
        # Use separate validation dataset
        train_dataset = full_dataset
        val_dataset = BinaryWalnutDataset(
            val_positive_dir, val_negative_dir,
            transform=AugmentationTransform(patch_size, training=False),
            augment=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, class_weights

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_hard_negatives(hard_negatives: List[Tuple[Path, float]], output_dir: str, 
                        num_to_save: int = None, duplicate_factor: int = 1):
    """Save or duplicate hard negative patches
    
    Args:
        hard_negatives: List of (path, confidence) tuples
        output_dir: Directory to save hard negatives
        num_to_save: Number of top hard negatives to save (None = all)
        duplicate_factor: Number of times to duplicate each hard negative (for oversampling)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Limit number if specified
    if num_to_save is not None:
        hard_negatives = hard_negatives[:num_to_save]
    
    print(f"💾 Saving {len(hard_negatives)} hard negatives (duplicating {duplicate_factor}x)...")
    
    saved_count = 0
    for path, confidence in tqdm(hard_negatives, desc="Saving"):
        # Read original image
        img = cv2.imread(str(path))
        if img is None:
            continue
        
        # Save multiple copies if duplicating
        for dup_idx in range(duplicate_factor):
            # Create filename with confidence score
            base_name = path.stem
            if duplicate_factor > 1:
                output_name = f"{base_name}_hardneg_{confidence:.3f}_dup{dup_idx}.png"
            else:
                output_name = f"{base_name}_hardneg_{confidence:.3f}.png"
            
            output_file = output_path / output_name
            cv2.imwrite(str(output_file), img)
            saved_count += 1
    
    print(f"✅ Saved {saved_count} hard negative patches to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train binary walnut classifier")
    parser.add_argument("--dataset_dir", required=True, help="Path to binary dataset directory")
    parser.add_argument("--output_dir", default="./models", help="Output directory for models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--mine_hard_negatives", action="store_true", 
                        help="After training, mine hard negatives and save them")
    parser.add_argument("--hard_negative_threshold", type=float, default=0.5,
                        help="Confidence threshold for hard negative mining")
    parser.add_argument("--hard_negative_output", type=str, default=None,
                        help="Directory to save hard negatives (default: dataset_dir/hard_negatives)")
    parser.add_argument("--hard_negative_duplicate", type=int, default=1,
                        help="Number of times to duplicate each hard negative (for oversampling)")
    
    args = parser.parse_args()
    
    print("🥜 Binary Walnut Classifier Training")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device selection - prioritize CUDA
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, class_weights = create_data_loaders(
        args.dataset_dir, args.batch_size, args.patch_size, device=device
    )
    
    # Create model
    model = WalnutClassifier(
        input_size=args.patch_size, 
        num_classes=2, 
        dropout_rate=args.dropout
    )
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer with class weights
    trainer = BinaryTrainer(model, device, args.learning_rate, class_weights=class_weights)
    
    # Train model (saves model with best precision)
    model_path = os.path.join(args.output_dir, "walnut_classifier_best_precision.pth")
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        train_loader, val_loader, args.epochs, model_path
    )
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, "training_history.png")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs,
        'best_val_acc': trainer.best_val_acc,
        'best_precision': trainer.best_precision
    }
    
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Training completed! Model saved to: {model_path}")
    
    # Hard negative mining (optional)
    if args.mine_hard_negatives:
        print("\n" + "=" * 50)
        print("🔍 HARD NEGATIVE MINING")
        print("=" * 50)
        
        # Load the best model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create transform for mining
        mining_transform = AugmentationTransform(args.patch_size, training=False)
        
        # Mine hard negatives
        negative_dir = os.path.join(args.dataset_dir, "negative")
        hard_negatives = mine_hard_negatives(
            model, negative_dir, mining_transform, device, 
            threshold=args.hard_negative_threshold, batch_size=args.batch_size
        )
        
        if len(hard_negatives) > 0:
            # Determine output directory
            if args.hard_negative_output:
                output_dir = args.hard_negative_output
            else:
                output_dir = os.path.join(args.dataset_dir, "hard_negatives")
            
            # Save hard negatives
            save_hard_negatives(
                hard_negatives, output_dir, 
                duplicate_factor=args.hard_negative_duplicate
            )
            
            print(f"\n💡 Next steps:")
            print(f"   1. Review hard negatives in: {output_dir}")
            print(f"   2. Optionally copy them to {negative_dir} to retrain")
            print(f"   3. Or use --hard_negative_duplicate > 1 to oversample them")
        else:
            print("ℹ️  No hard negatives found. Model is performing well on negative samples!")

if __name__ == "__main__":
    main()
