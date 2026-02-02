# Walnut Detection and Counting System

A comprehensive deep learning system for automated walnut detection and counting in aerial/drone imagery. Achieves **99.18% count accuracy** using CNN-based binary classification with sliding window detection.

## Table of Contents

- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Script Documentation](#script-documentation)
- [Common Workflows](#common-workflows)
- [Model Performance](#model-performance)

## Features


- ✅ **Binary Classifier Training** - CNN-based patch classification with class balancing
- ✅ **Hard Negative Mining** - Automatic identification of difficult samples
- ✅ **Sliding Window Detection** - Full-image processing with confidence maps
- ✅ **Synthetic Data Generation** - Advanced data augmentation pipeline
- ✅ **Density Estimation Models** - Alternative counting approaches
- ✅ **Complete Evaluation Suite** - Comprehensive metrics and visualizations
- ✅ **GIS Integration** - GeoJSON export with GPS coordinates
- ✅ **CUDA/GPU Support** - Optimized for GPU acceleration

## Setup Instructions

### 1. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Deactivate Virtual Environment (when done)

```bash
deactivate
```

## Project Structure

```
TuesdayPresentation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
│
├── Training Scripts
│   ├── binary_classifier.py          # Main binary classifier training
│   ├── extract_training_patches.py   # Extract patches from annotated images
│   ├── trainingModels/                # Alternative training implementations
│   │   ├── binary_classifier.py
│   │   ├── density_model.py
│   │   └── fast_density_model.py
│   └── synthetic_pipeline_code/      # Synthetic data generation
│       ├── synthetic_generator.py
│       ├── density_model.py
│       └── ...
│
├── Detection & Inference
│   ├── walnut_detector.py            # Main detector (sliding window)
│   ├── run_full_pipeline.py         # Complete pipeline (images → GeoJSON)
│   ├── run_wvt_detector_geojson.py  # Walnut variety detector with GeoJSON
│   └── run_walnut_detector_multiple_thresholds.py
│
├── Evaluation Scripts
│   ├── evaluate_detector.py          # Evaluate detector on test set
│   ├── evaluate_detections.py        # Evaluate detection results
│   ├── test_binary_classifier.py    # Test binary classifier
│   ├── test_density_model.py        # Test density model
│   ├── test_all_thresholds.py       # Test multiple thresholds
│   └── test_walnut_variety_thresholds.py
│
├── Analysis & Visualization
│   ├── create_confusion_matrix.py
│   ├── generate_graphs.py
│   ├── generate_threshold_graphs.py
│   ├── calculate_count_accuracy.py
│   └── ...
│
├── Data Processing
│   ├── process_all_original_images.py
│   ├── process_quadrants_all_images.py
│   ├── combine_quadrant_annotations.py
│   └── ...
│
└── models/                           # Trained model checkpoints
    ├── models_new/
    └── models_precision/
```

## Script Documentation

### Training Scripts

#### `binary_classifier.py`
**Purpose:** Train a binary CNN classifier to distinguish walnut patches from background patches.

**Features:**
- Automatic class weight balancing for imbalanced datasets
- Hard negative mining (optional)
- Data augmentation (flips, rotations, color jitter)
- Saves best model based on precision

**Usage:**
```bash
# Basic training
python binary_classifier.py --dataset_dir ./data --output_dir ./models --epochs 50

# With hard negative mining
python binary_classifier.py --dataset_dir ./data --mine_hard_negatives --hard_negative_threshold 0.6

# Full options
python binary_classifier.py \
    --dataset_dir ./data \
    --output_dir ./models \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --patch_size 32 \
    --dropout 0.5 \
    --mine_hard_negatives \
    --hard_negative_threshold 0.5 \
    --hard_negative_duplicate 2
```

**Arguments:**
- `--dataset_dir`: Path to dataset directory (must contain `positive/` and `negative/` subdirectories)
- `--output_dir`: Output directory for models (default: `./models`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patch_size`: Patch size in pixels (default: 32)
- `--dropout`: Dropout rate (default: 0.5)
- `--mine_hard_negatives`: Enable hard negative mining after training
- `--hard_negative_threshold`: Confidence threshold for hard negatives (default: 0.5)
- `--hard_negative_output`: Directory to save hard negatives
- `--hard_negative_duplicate`: Number of times to duplicate each hard negative

**Output:**
- `walnut_classifier_best_precision.pth`: Best model checkpoint
- `training_history.png`: Training curves
- `training_history.json`: Training metrics

---

#### `extract_training_patches.py`
**Purpose:** Extract 32x32 patches from annotated images for training the binary classifier.

**Usage:**
```bash
python extract_training_patches.py \
    --images_dir path/to/images \
    --annotations_dir path/to/annotations \
    --output_dir path/to/output \
    --patch_size 32 \
    --num_negatives_per_image 50
```

**Arguments:**
- `--images_dir`: Directory containing source images
- `--annotations_dir`: Directory containing annotation files (.txt with x y coordinates)
- `--output_dir`: Output directory (creates `positive/` and `negative/` subdirectories)
- `--patch_size`: Patch size in pixels (default: 32)
- `--num_negatives_per_image`: Number of negative patches per image (default: matches positives)

**Output:**
- `positive/`: Directory with positive patches (centered on annotations)
- `negative/`: Directory with negative patches (background, avoiding walnut areas)

---

### Detection & Inference Scripts

#### `walnut_detector.py`
**Purpose:** Main detector class for detecting walnuts in full images using sliding window approach.

**Usage (as library):**
```python
from walnut_detector import WalnutDetector

detector = WalnutDetector(
    model_path="models_new/walnut_classifier.pth",
    patch_size=32,
    stride=16,
    confidence_threshold=0.6,
    device='auto'  # or 'cuda', 'mps', 'cpu'
)

centers, confidences, confidence_map = detector.detect_walnuts(image)
```

**Usage (CLI):**
```bash
python walnut_detector.py \
    --model_path models_new/walnut_classifier.pth \
    --image_dir path/to/images \
    --output_dir path/to/output \
    --threshold 0.6 \
    --patch_size 32 \
    --stride 16 \
    --cluster
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--image_dir`: Directory containing images to process
- `--output_dir`: Output directory for results
- `--threshold`: Confidence threshold (default: 0.6, recommended: 0.5-0.6)
- `--patch_size`: Patch size (default: 32)
- `--stride`: Sliding window stride (default: 16)
- `--cluster`: Enable DBSCAN clustering to merge nearby detections

**Output:**
- Detection overlays with bounding boxes
- Confidence maps
- JSON files with detection coordinates

---

#### `run_full_pipeline.py`
**Purpose:** Complete end-to-end pipeline: count walnuts per image and generate GeoJSON with GPS coordinates.

**Usage:**
```bash
python run_full_pipeline.py --image_dir path/to/images
```

**Arguments:**
- `--image_dir`: Directory containing images with GPS metadata (EXIF)

**Output:**
- `*_image_counts.json`: Per-image walnut counts
- `*_walnut_counts.geojson`: GeoJSON with one point per walnut

**Features:**
- Divides images into quadrants for processing
- Extracts GPS coordinates from EXIF data
- Generates GeoJSON compatible with GIS software

---

### Evaluation Scripts

#### `evaluate_detector.py`
**Purpose:** Evaluate detector performance on annotated test set with comprehensive metrics.

**Usage:**
```bash
python evaluate_detector.py \
    --model_path models_new/walnut_classifier.pth \
    --images_dir path/to/test/images \
    --labels_dir path/to/test/annotations \
    --output_dir path/to/output \
    --threshold 0.6 \
    --patch_size 32 \
    --stride 16 \
    --match_distance 20
```

**Arguments:**
- `--model_path`: Path to trained model
- `--images_dir`: Test images directory
- `--labels_dir`: Ground truth annotations directory (.txt files)
- `--output_dir`: Output directory for results
- `--threshold`: Detection confidence threshold
- `--patch_size`: Patch size used during training
- `--stride`: Stride used during detection
- `--match_distance`: Distance threshold for matching predictions to ground truth (pixels)

**Output:**
- `per_image_metrics.json`: Per-image precision, recall, F1, TP/FP/FN
- `summary.json`: Overall metrics (precision, recall, F1, count accuracy)
- `confusion_summary.png`: Visualization of confusion matrix

**Metrics Calculated:**
- Precision, Recall, F1 Score
- True Positives, False Positives, False Negatives
- Count Accuracy (predicted vs ground truth counts)
- Per-image breakdown

---

#### `test_binary_classifier.py`
**Purpose:** Test binary classifier on a test directory with comprehensive evaluation.

**Usage:**
```bash
python test_binary_classifier.py \
    --model_path models_new/walnut_classifier.pth \
    --test_dir path/to/test \
    --output_dir ./binary_test_results \
    --patch_size 32 \
    --stride 16 \
    --threshold 0.5
```

**Arguments:**
- `--model_path`: Path to trained model
- `--test_dir`: Test directory (should contain `images/` and `annotations/` subdirectories)
- `--output_dir`: Output directory for results
- `--patch_size`: Patch size (default: 32)
- `--stride`: Stride for sliding window (default: 16)
- `--threshold`: Confidence threshold (default: 0.5)

**Output:**
- Detection visualizations
- Performance metrics
- Confusion matrices
- Per-image analysis

---

#### `test_all_thresholds.py`
**Purpose:** Test detector at multiple confidence thresholds (0.1-0.9) and compile results.

**Usage:**
```bash
python test_all_thresholds.py
```

**Note:** Edit configuration variables at the top of the script:
- `MODEL_PATH`: Path to model
- `TEST_IMAGES_DIR`: Test images directory
- `TEST_LABELS_DIR`: Test annotations directory
- `THRESHOLDS`: List of thresholds to test

**Output:**
- `threshold_evaluations/th{threshold}/`: Results for each threshold
- `all_threshold_results.json`: Compiled results for all thresholds

---

### Analysis & Visualization Scripts

#### `create_confusion_matrix.py`
**Purpose:** Create confusion matrix visualization from evaluation results.

**Usage:**
```bash
python create_confusion_matrix.py --results_dir path/to/results
```

---

#### `generate_graphs.py`
**Purpose:** Generate various performance graphs and visualizations.

**Usage:**
```bash
python generate_graphs.py --input_dir path/to/results --output_dir path/to/output
```

---

#### `calculate_count_accuracy.py`
**Purpose:** Calculate count accuracy metrics from detection results.

**Usage:**
```bash
python calculate_count_accuracy.py --results_file path/to/results.json
```

---

### Synthetic Data Generation

#### `synthetic_pipeline_code/synthetic_generator.py`
**Purpose:** Generate synthetic training data with realistic walnut placements.

**Usage:**
```bash
python synthetic_pipeline_code/synthetic_generator.py \
    --output_dir ./synthetic_data \
    --num_images 1000 \
    --num_walnuts_per_image 50
```

---

#### `synthetic_pipeline_code/density_model.py`
**Purpose:** Train density estimation model for counting walnuts.

**Usage:**
```bash
python synthetic_pipeline_code/density_model.py \
    --synthetic_data ./synthetic_data \
    --output ./models \
    --epochs 100 \
    --batch_size 16
```

---

## Common Workflows

### Workflow 1: Train Binary Classifier from Scratch

```bash
# 1. Extract training patches from annotated images
python extract_training_patches.py \
    --images_dir ./training_data/images \
    --annotations_dir ./training_data/annotations \
    --output_dir ./patches

# 2. Train binary classifier
python binary_classifier.py \
    --dataset_dir ./patches \
    --output_dir ./models \
    --epochs 50 \
    --mine_hard_negatives

# 3. Test the trained model
python test_binary_classifier.py \
    --model_path ./models/walnut_classifier_best_precision.pth \
    --test_dir ./test_data \
    --output_dir ./test_results
```

### Workflow 2: Detect Walnuts in New Images

```bash
# Single image directory
python walnut_detector.py \
    --model_path models_new/walnut_classifier.pth \
    --image_dir ./new_images \
    --output_dir ./detections \
    --threshold 0.6 \
    --cluster

# Full pipeline with GeoJSON output
python run_full_pipeline.py --image_dir ./new_images
```

### Workflow 3: Evaluate Model Performance

```bash
# Evaluate on test set
python evaluate_detector.py \
    --model_path models_new/walnut_classifier.pth \
    --images_dir ./test/images \
    --labels_dir ./test/annotations \
    --output_dir ./evaluation_results \
    --threshold 0.6

# Test multiple thresholds
python test_all_thresholds.py

# Generate visualizations
python create_confusion_matrix.py --results_dir ./evaluation_results
python generate_graphs.py --input_dir ./evaluation_results
```

### Workflow 4: Hard Negative Mining & Retraining

```bash
# 1. Train initial model
python binary_classifier.py --dataset_dir ./patches --output_dir ./models --epochs 30

# 2. Mine hard negatives (automatic if --mine_hard_negatives is used)
# Or manually:
python -c "
from binary_classifier import mine_hard_negatives, WalnutClassifier, AugmentationTransform
import torch
model = WalnutClassifier()
model.load_state_dict(torch.load('./models/walnut_classifier_best_precision.pth'))
hard_negatives = mine_hard_negatives(model, './patches/negative', ...)
"

# 3. Copy hard negatives to training set
cp ./patches/hard_negatives/* ./patches/negative/

# 4. Retrain with hard negatives
python binary_classifier.py --dataset_dir ./patches --output_dir ./models --epochs 50
```

## Model Performance

### Best Performing Configuration

- **Model:** `models_new/walnut_classifier.pth`
- **Threshold:** 0.5
- **Count Accuracy:** 99.18%
- **Precision:** High (varies by threshold)
- **Recall:** High (varies by threshold)

### Recommended Thresholds

- **Best Count Accuracy:** 0.5 (99.18% accuracy)
- **Balanced Precision/Recall:** 0.6 (97.13% accuracy, fewer false positives)
- **High Precision:** 0.7+ (fewer false positives, may miss some walnuts)

## System Requirements

- **Python:** 3.7+
- **CUDA:** Optional but recommended for GPU acceleration
- **RAM:** 8GB+ recommended
- **Storage:** 10GB+ for models and datasets

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in training scripts
- Reduce `patch_size` or increase `stride` in detection
- Process images in smaller batches

### Low Detection Accuracy
- Try different confidence thresholds (0.5, 0.6, 0.7)
- Enable clustering with `--cluster` flag
- Check if model was trained on similar data
- Consider hard negative mining and retraining

### Class Imbalance Issues
- Class weights are automatically calculated and applied
- Check training logs for class balance statistics
- Use `--mine_hard_negatives` to improve negative sample quality

### Data Split Warnings
- Current split is at patch level, not image level
- For proper image-level splitting, modify `extract_training_patches.py` to preserve source image info in filenames

## Additional Resources

- `PROJECT_ACHIEVEMENTS.md`: Detailed performance metrics and achievements
- `threshold_explanation.md`: Explanation of threshold selection
- `understanding_metrics_explanation.md`: Detailed metric explanations
- `synthetic_pipeline_code/README.md`: Synthetic data generation documentation

## License

[Add your license information here]

## Contact

[Add contact information here]
