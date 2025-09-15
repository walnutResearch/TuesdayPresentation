#!/usr/bin/env python3
"""
Density Estimation Model (MPS/Mac-friendly, Fast Mode)
=====================================================

Multiscale CNN model for walnut density estimation based on patch-wise counting.
- Apple Silicon (MPS) optimized: mixed precision, channels_last, fewer workers
- Fast mode: slimmer net, BN-free (GroupNorm), smaller defaults
- Input pipeline caching to avoid per-epoch OpenCV cost
- Safer Sobel normalization, controllable resize cap
- Count-aware loss (absolute + relative) to prevent under-count shrinkage

Author: Walnut Counting Project (revamped)
Date: 2025
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

# --- System/threading tweaks (must be before heavy imports) ---
os.environ.setdefault("PYTHONWARNINGS", "ignore:torch.multiprocessing:red")

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# -----------------------------
# Config
# -----------------------------
@dataclass
class ModelConfig:
    """Configuration for density estimation model"""
    input_channels: int = 6          # RGB + Grayscale + EdgeMag + EdgeDir
    patch_size: int = 32
    hidden_dim: int = 64
    num_scales: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 16
    num_epochs: int = 30
    no_bn: bool = False              # Fast mode sets this True (uses GroupNorm instead)
    max_side: int = 512              # Longest side cap for resizing (set smaller if needed)
    print_every: int = 100           # Step logging frequency
    seed: int = 42                   # Reproducibility


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# -----------------------------
# Transforms / Dataset
# -----------------------------
class MultiChannelTransform:
    """Transform images to multi-channel input for density estimation"""

    def __init__(self, patch_size: int = 32, max_side: int = 512):
        self.patch_size = patch_size
        self.max_side = max_side

    def _resize_to_multiple(self, image, target_w, target_h):
        # Ensure dims are multiples of patch_size
        new_w = (target_w // self.patch_size) * self.patch_size
        new_h = (target_h // self.patch_size) * self.patch_size
        new_w = max(new_w, self.patch_size)
        new_h = max(new_h, self.patch_size)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to multi-channel input
        Returns (H, W, 6): RGB, Grayscale(L), EdgeMag, EdgeDir
        """
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
        # Convert BGR->RGB quickly
        rgb = image_float[:, :, ::-1]
        channels.extend([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])

        # Grayscale (L channel from Lab on BGR input)
        lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
        grayscale = lab[:, :, 0].astype(np.float32) / 255.0
        channels.append(grayscale)

        # Sobel edges from GRAY
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

        edge_direction = np.arctan2(sobel_y, sobel_x)  # [-pi, pi]
        edge_direction = (edge_direction + np.pi) / (2 * np.pi)  # [0,1]
        channels.append(edge_direction.astype(np.float32))

        multi_channel = np.stack(channels, axis=2)  # (H, W, 6)
        return multi_channel


class DensityDataset(Dataset):
    """Dataset for density estimation training with simple caching"""

    def __init__(
        self,
        images_dir: str,
        density_maps_dir: str,
        patch_size: int = 32,
        max_side: int = 512,
        cache: bool = True,
        transform: Optional[MultiChannelTransform] = None,
    ):
        self.images_dir = Path(images_dir)
        self.density_maps_dir = Path(density_maps_dir)
        self.patch_size = patch_size
        self.transform = transform or MultiChannelTransform(patch_size, max_side)
        self.cache = cache

        self.image_files = sorted(self.images_dir.glob("*.png"))
        print(f"Found {len(self.image_files)} images in dataset")

        self.cache_dir = self.images_dir.parent / "cache_mc"
        if self.cache:
            self.cache_dir.mkdir(exist_ok=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Load density map
        density_path = self.density_maps_dir / f"density_{image_path.stem.split('_')[-1]}.npy"
        if not density_path.exists():
            raise ValueError(f"Density map not found: {density_path}")
        density_map = np.load(density_path).astype(np.float32)

        # Cache key
        if self.cache:
            cache_path = self.cache_dir / f"{image_path.stem}_mc.npy"
            if cache_path.exists():
                mc = np.load(cache_path).astype(np.float32)
            else:
                mc = self.transform(img)
                np.save(cache_path, mc)
        else:
            mc = self.transform(img)

        # To tensors (C,H,W) and (H,W)
        image_tensor = torch.from_numpy(mc).permute(2, 0, 1).float()
        density_tensor = torch.from_numpy(density_map).float()
        return image_tensor, density_tensor


# -----------------------------
# Model
# -----------------------------
class MultiScaleDensityNet(nn.Module):
    """Multiscale CNN for density estimation"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Scale blocks (reduced channels for speed)
        self.scale1 = self._make_scale_block(config.input_channels, 32, scale=1)
        self.scale2 = self._make_scale_block(config.input_channels, 32, scale=2)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(32 * 2, config.hidden_dim, 3, padding=1),
            self._norm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(config.dropout_rate),
        )

        # Density head (lightweight)
        head_in = config.hidden_dim
        head_mid = 16 if head_in >= 16 else head_in
        self.density_head = nn.Sequential(
            nn.Conv2d(head_in, head_mid, 3, padding=1),
            self._norm(head_mid),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(head_mid, 1, 1),
            # no ReLU here; Softplus applied in forward()
        )

        self._initialize_weights()

        # ---- Learnable calibration params (balanced start) ----
        self.scale_param = nn.Parameter(torch.tensor(-1.8))  # softplus ≈ 0.14–0.15
        self.bias_param  = nn.Parameter(torch.tensor(-7.2))  # softplus ≈ 0.00075

    def _norm(self, c: int) -> nn.Module:
        # Fast mode: GroupNorm (stable for tiny batches) instead of BatchNorm
        if self.config.no_bn:
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
            # Downsampled scale (avoid MaxPool2d on MPS/inductor; use stride-2 conv)
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                self._norm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                self._norm(out_channels),
                nn.ReLU(inplace=True),
            )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Final conv bias set to 0 (we use learnable softplus bias instead)
        for m in self.density_head.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        h, w = feat1.size(2), feat1.size(3)
        feat2 = F.interpolate(feat2, size=(h, w), mode="bilinear", align_corners=False)
        fused_features = torch.cat([feat1, feat2], dim=1)
        fused = self.fusion(fused_features)
        logits = self.density_head(fused)
        # Non-negative with learnable calibration
        out = F.softplus(logits) * F.softplus(self.scale_param) + F.softplus(self.bias_param)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=0.0)
        return out


# -----------------------------
# Training
# -----------------------------
class DensityTrainer:
    """Trainer for density estimation model"""

    def __init__(self, model: nn.Module, config: ModelConfig, device: str = "cuda"):
        self.model = model.to(device)
        # channels_last on model params/buffers
        self.model = self.model.to(memory_format=torch.channels_last)

        self.config = config
        self.device = device

        # Robust pixel loss
        self.pix_loss = nn.SmoothL1Loss(beta=0.5)  # Huber

        # ---- Optimizer: different LR + no weight decay for calibration ----
        calib = [p for n, p in self.model.named_parameters() if ("scale_param" in n) or ("bias_param" in n)]
        backbone = [p for n, p in self.model.named_parameters() if ("scale_param" not in n) and ("bias_param" not in n)]
        self.optimizer = optim.Adam(
            [
                {"params": backbone, "lr": config.learning_rate, "weight_decay": 1e-4},
                {"params": calib,    "lr": config.learning_rate * 4.0, "weight_decay": 0.0},
            ]
        )

        # Quicker LR drops
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # Optional compile (CUDA only)
        if device == "cuda":
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
            except Exception:
                pass

        self.train_losses = []
        self.val_losses = []

        # Autocast dtype
        self.autocast_dtype = torch.float16 if device != "cpu" else torch.float32

    def _move_batch(self, images, density_maps):
        images = images.to(self.device, non_blocking=False).to(memory_format=torch.channels_last)
        density_maps = density_maps.to(self.device, non_blocking=False)
        return images, density_maps

    # ---- Count metrics for better monitoring ----
    def _count_metrics(self, preds: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            p = preds.squeeze(1).to(torch.float32)  # (N,16,16)
            pred_counts = p.sum(dim=(1, 2))
            true_counts = targets.to(torch.float32).sum(dim=(1, 2))
            abs_err = (pred_counts - true_counts).abs()
            sq_err = (pred_counts - true_counts) ** 2
            mae = abs_err.mean().item()
            rmse = torch.sqrt(sq_err.mean()).item()
            mask = true_counts > 0
            mape = (abs_err[mask] / (true_counts[mask] + 1e-6)).mean().item() * 100 if mask.any() else float("nan")
            return mae, rmse, mape, pred_counts.mean().item(), true_counts.mean().item()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _count_weights(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Returns (cw_abs, cw_rel): absolute count L1 weight, relative count weight.
        Stronger early to push magnitude up, then taper.
        """
        if epoch_idx < 2:      # epochs 0–1
            return 0.40, 0.40
        elif epoch_idx < 8:    # epochs 2–7
            return 0.25, 0.20
        else:
            return 0.10, 0.10

    def _compute_loss_fp32(self, preds: torch.Tensor, targets: torch.Tensor, cw_abs: float, cw_rel: float):
        # Compute losses in full fp32 for stability
        preds32 = preds.float().squeeze(1)
        targets32 = targets.float()

        # Pixelwise robust loss
        l_pix = self.pix_loss(preds32, targets32)

        # Counts
        pred_counts = preds32.sum(dim=(1, 2))
        true_counts = targets32.sum(dim=(1, 2))
        l_cnt_abs = F.l1_loss(pred_counts, true_counts)  # absolute
        l_cnt_rel = (pred_counts - true_counts).abs() / (true_counts.abs() + 1.0)  # relative, stabilized
        l_cnt_rel = l_cnt_rel.mean()

        return l_pix + cw_abs * l_cnt_abs + cw_rel * l_cnt_rel

    def train_epoch(self, train_loader: DataLoader, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        cw_abs, cw_rel = self._count_weights(epoch_idx)

        for batch_idx, (images, density_maps) in enumerate(train_loader):
            images, density_maps = self._move_batch(images, density_maps)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward in mixed precision
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                preds = self.model(images)

            # Loss in fp32
            loss = self._compute_loss_fp32(preds, density_maps, cw_abs, cw_rel)

            loss.backward()
            # Clip to prevent any early spikes from blowing up grads
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

            if self.config.print_every > 0 and (batch_idx % self.config.print_every == 0):
                print(f"  Batch {batch_idx}/{len(train_loader)}  Loss: {loss.item():.6f}")

        return total_loss / max(1, num_batches)

    def validate_epoch(self, val_loader: DataLoader, epoch_idx: int) -> Tuple[float, Tuple[float, float, float], Tuple[float, float]]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        mae_sum = rmse_sum = mape_sum = 0.0
        mape_n = 0
        pred_mean_sum = 0.0
        true_mean_sum = 0.0
        cw_abs, cw_rel = self._count_weights(epoch_idx)

        with torch.no_grad():
            for images, density_maps in val_loader:
                images, density_maps = self._move_batch(images, density_maps)

                # Forward in mixed precision
                with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                    preds = self.model(images)

                # Loss in fp32
                loss = self._compute_loss_fp32(preds, density_maps, cw_abs, cw_rel)

                total_loss += float(loss.item())
                num_batches += 1

                mae, rmse, mape, pred_mean, true_mean = self._count_metrics(preds, density_maps)
                mae_sum += mae
                rmse_sum += rmse
                pred_mean_sum += pred_mean
                true_mean_sum += true_mean
                if not np.isnan(mape):
                    mape_sum += mape
                    mape_n += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_mae = mae_sum / max(1, num_batches)
        avg_rmse = rmse_sum / max(1, num_batches)
        avg_mape = (mape_sum / mape_n) if mape_n > 0 else float("nan")
        pred_mean_epoch = pred_mean_sum / max(1, num_batches)
        true_mean_epoch = true_mean_sum / max(1, num_batches)
        return avg_loss, (avg_mae, avg_rmse, avg_mape), (pred_mean_epoch, true_mean_epoch)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_path: str = "density_model.pth"):
        print("🚀 Starting training...")
        best_val = float("inf")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            tr = self.train_epoch(train_loader, epoch_idx=epoch)
            va, (mae, rmse, mape), (pred_mean, true_mean) = self.validate_epoch(val_loader, epoch_idx=epoch)
            self.scheduler.step(va)

            self.train_losses.append(tr)
            self.val_losses.append(va)

            print(f"Train Loss: {tr:.6f} | Val Loss: {va:.6f} | "
                  f"Count MAE: {mae:.3f} | RMSE: {rmse:.3f} | MAPE: {mape:.1f}% | "
                  f"LR: {self.get_lr():.6g}")
            print(f"Val counts → pred mean: {pred_mean:.2f}, true mean: {true_mean:.2f}")

            if va < best_val:
                best_val = va
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": epoch,
                        "val_loss": va,
                        "config": self.config.__dict__,
                    },
                    save_path,
                )
                print(f"💾 Saved best model to {save_path} (Val {va:.6f})")

        print("✅ Training completed!")
        return self.train_losses, self.val_losses


# -----------------------------
# Dataloaders
# -----------------------------
def create_data_loaders(synthetic_data_dir: str, config: ModelConfig, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    images_dir = os.path.join(synthetic_data_dir, "synthetic_images")
    density_maps_dir = os.path.join(synthetic_data_dir, "density_maps")

    # On macOS (MPS), multi-process workers hurt. Use 0.
    is_macos = sys.platform == "darwin"
    use_workers = 0 if is_macos else 2

    # Enable caching to avoid per-epoch OpenCV work
    full_dataset = DensityDataset(images_dir, density_maps_dir, config.patch_size, config.max_side, cache=True)

    dataset_size = len(full_dataset)
    val_size = max(1, int(val_split * dataset_size))
    train_size = max(1, dataset_size - val_size)

    # Edge case: very tiny dataset
    if train_size <= 0:
        train_size, val_size = dataset_size, 0

    # Make split reproducible
    set_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

    # Build kwargs without persistent_workers if workers=0 (older torch versions error)
    def loader_kwargs():
        kw = dict(batch_size=config.batch_size, pin_memory=False)
        kw["num_workers"] = use_workers
        if use_workers > 0:
            kw["persistent_workers"] = False
        return kw

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs())
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs())

    print(f"📊 Dataset split: {train_size} train, {val_size} val")
    return train_loader, val_loader


# -----------------------------
# Main / CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train density estimation model (Mac/MPS friendly)")
    parser.add_argument("--synthetic_data", required=True, help="Path to synthetic data directory")
    parser.add_argument("--output", default="./models", help="Output directory for trained models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size for density maps")
    parser.add_argument("--max_side", type=int, default=512, help="Max long-side pixels for input resize")
    parser.add_argument("--print_every", type=int, default=100, help="Batches between progress prints (0=off)")
    parser.add_argument("--fast", action="store_true", help="Speed-optimized settings for low-RAM Macs")
    args = parser.parse_args()

    print("🧠 Density Estimation Model Training")
    print("=" * 50)

    os.makedirs(args.output, exist_ok=True)

    # Build config
    config = ModelConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patch_size=args.patch_size,
        max_side=args.max_side,
        print_every=args.print_every,
    )

    if args.fast:
        # Fast mode tweaks
        config.hidden_dim = 32
        config.batch_size = min(config.batch_size, 4)
        config.num_epochs = min(config.num_epochs, 20)
        config.dropout_rate = 0.0    # no dropout in fast mode
        config.no_bn = True
        config.max_side = min(config.max_side, 512)

    # Reproducibility
    set_seed(config.seed)

    # Device selection: prefer MPS on Apple Silicon, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"🖥️  Using device: {device}")

    # Dataloaders
    train_loader, val_loader = create_data_loaders(args.synthetic_data, config)

    # Model
    model = MultiScaleDensityNet(config)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {model_params:,}")

    # Trainer
    trainer = DensityTrainer(model, config, device)

    # Dry run on a tiny batch to catch shape/memory issues early
    try:
        small_batch = next(iter(train_loader))
        if isinstance(small_batch, (list, tuple)) and len(small_batch) == 2:
            images, targets = small_batch
            images = images[:2]
            targets = targets[:2]
            images = images.to(device).to(memory_format=torch.channels_last)
            targets = targets.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16 if device != "cpu" else torch.float32):
                out = trainer.model(images)
            print("✅ Dry run output:", tuple(out.shape), " target:", tuple(targets.shape))
    except Exception as e:
        print("⚠️ Dry run failed (continuing):", repr(e))

    # Train
    model_path = os.path.join(args.output, "density_model.pth")
    train_losses, val_losses = trainer.train(train_loader, val_loader, model_path)

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": config.__dict__,
    }
    history_path = os.path.join(args.output, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Training completed! Model saved to: {model_path}")


if __name__ == "__main__":
    main()