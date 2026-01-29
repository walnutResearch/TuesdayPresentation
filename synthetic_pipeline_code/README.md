# Synthetic Walnut Generation Pipeline

This folder contains a complete pipeline for generating synthetic walnut images and training density estimation models.

## 🌰 Overview

The pipeline creates realistic synthetic walnut images by:
1. **Extracting walnut instances** from annotated training data
2. **Generating canopy background plates** by removing walnuts from real images
3. **Creating synthetic images** with realistic walnut placement and photometric matching
4. **Training a density estimation model** on synthetic data
5. **Fine-tuning on real data** to close the domain gap

## 📁 File Structure

```
synthetic_walnut_generation/
├── synthetic_pipeline.py      # Main pipeline for data extraction
├── synthetic_generator.py     # Synthetic image generation
├── density_model.py          # Density estimation model
├── fine_tune_real.py         # Fine-tuning on real data
├── run_pipeline.py           # Complete pipeline runner
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Run Complete Pipeline
```bash
python run_pipeline.py --train_data ../walnut_annotated_dataset --output ./synthetic_data
```

### 2. Run Individual Steps

#### Step 1: Extract Walnuts and Generate Canopy Plates
```bash
python synthetic_pipeline.py --train_data ../walnut_annotated_dataset --output ./synthetic_data --num_canopy_plates 30
```

#### Step 2: Generate Synthetic Images
```bash
python synthetic_generator.py --walnut_instances ./synthetic_data/metadata --canopy_plates ./synthetic_data/canopy_plates --output ./synthetic_data --num_images 2000
```

#### Step 3: Train Density Model
```bash
python density_model.py --synthetic_data ./synthetic_data --output ./models --epochs 50
```

#### Step 4: Fine-tune on Real Data
```bash
python fine_tune_real.py --pretrained_model ./models/density_model.pth --real_data ../walnut_annotated_dataset --output ./models --epochs 15
```

## 🔧 Configuration

### Pipeline Parameters
- `--train_data`: Path to annotated training dataset
- `--output`: Output directory for synthetic data
- `--num_synthetic`: Number of synthetic images to generate (default: 2000)
- `--num_canopy_plates`: Number of canopy plates to generate (default: 30)
- `--training_epochs`: Training epochs for density model (default: 50)
- `--fine_tune_epochs`: Fine-tuning epochs on real data (default: 15)

### Model Parameters
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patch_size`: Patch size for density maps (default: 32)

## 📊 Output Structure

```
synthetic_data/
├── synthetic_images/          # Generated synthetic images
├── density_maps/             # Density maps for training
├── canopy_plates/            # Background plates (walnut-free)
├── walnut_cutouts/           # Individual walnut instances
├── models/                   # Trained models
│   ├── density_model.pth     # Pre-trained model
│   └── fine_tuned_density_model.pth  # Fine-tuned model
└── metadata/                 # Statistics and configurations
    ├── walnut_instances.json
    └── synthetic_generation.json
```

## 🧠 Model Architecture

### Multi-Scale Density Network
- **Input**: 6-channel images (RGB + Grayscale + Edge maps)
- **Architecture**: Multi-scale CNN with feature fusion
- **Output**: Density map for patch-wise counting
- **Loss**: MSE loss for density regression

### Key Features
- **Multi-scale processing**: Handles walnuts of different sizes
- **Edge-aware input**: Includes edge maps for better feature extraction
- **Photometric matching**: Realistic walnut appearance on backgrounds
- **Data augmentation**: Robust training with various transformations

## 📈 Training Process

### 1. Synthetic Data Generation
- Extract walnut instances from annotated training data
- Generate canopy background plates by inpainting
- Create synthetic images with realistic walnut placement
- Apply photometric matching and shadows

### 2. Model Training
- Train on large synthetic dataset (2000+ images)
- Use multi-scale features for size variation
- Apply data augmentation for robustness

### 3. Fine-tuning
- Fine-tune on real annotated data (validation set)
- Lower learning rate to preserve synthetic knowledge
- Close domain gap between synthetic and real images

## 🎯 Usage for Inference

After training, use the fine-tuned model for walnut counting:

```python
import torch
from density_model import MultiScaleDensityNet, ModelConfig, MultiChannelTransform

# Load model
config = ModelConfig()
model = MultiScaleDensityNet(config)
checkpoint = torch.load('fine_tuned_density_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process image
transform = MultiChannelTransform(patch_size=32)
image = cv2.imread('your_image.jpg')
multi_channel = transform(image)
input_tensor = torch.from_numpy(multi_channel).permute(2, 0, 1).float().unsqueeze(0)

# Get density prediction
with torch.no_grad():
    density_map = model(input_tensor)
    total_count = density_map.sum().item()

print(f"Estimated walnut count: {total_count:.1f}")
```

## 🔍 Key Features

### Photometric Matching
- Matches walnut appearance to local background statistics
- Applies per-channel affine transformations
- Adds realistic shadows for depth

### Realistic Placement
- Places walnuts in valid foliage areas
- Maintains minimum distance between walnuts
- Allows partial occlusion for realism

### Multi-Channel Input
- RGB channels for color information
- Grayscale (Lab L channel) for luminance
- Edge magnitude and direction for texture

### Data Augmentation
- Random flips and rotations
- Brightness and contrast adjustments
- Maintains density map consistency

## 📊 Performance Metrics

The model is evaluated using:
- **MSE Loss**: Mean squared error for density regression
- **MAE**: Mean absolute error for count estimation
- **MAPE**: Mean absolute percentage error

## 🛠️ Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
PIL>=8.0.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
```

## 📝 Notes

- The pipeline is designed to work with the annotated dataset structure
- Synthetic data generation can be computationally intensive
- Fine-tuning requires a small amount of real annotated data
- The model outputs density maps that can be summed for total counts

## 🤝 Contributing

When modifying the pipeline:
1. Maintain the modular structure
2. Add comprehensive comments
3. Update this README with changes
4. Test with small datasets first

## 📄 License

Part of the Walnut Counting Project - 2024
