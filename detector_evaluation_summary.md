# Walnut Detector Evaluation Results

## Model Information
- **Model Path**: `./models_new/walnut_classifier.pth`
- **Model Validation Accuracy**: 97.19%
- **Test Dataset**: `/Users/kalpit/TuesdayPresentation/Walnut_Research/test`
- **Number of Test Images**: 10
- **Total Ground Truth Walnuts**: 487

## Detection Parameters
- **Patch Size**: 32x32 pixels
- **Stride**: 16 pixels
- **Match Distance**: 20 pixels (for TP/FP/FN matching)
- **Clustering**: Enabled (DBSCAN)

## Results by Confidence Threshold

### Threshold 0.5
- **True Positives (TP)**: 286
- **False Positives (FP)**: 404
- **False Negatives (FN)**: 201
- **Precision**: 41.4%
- **Recall**: 58.7%
- **F1 Score**: 48.6%

### Threshold 0.6 ⭐ (Best Balance)
- **True Positives (TP)**: 263
- **False Positives (FP)**: 238
- **False Negatives (FN)**: 224
- **Precision**: 52.5%
- **Recall**: 54.0%
- **F1 Score**: 53.2%

### Threshold 0.7 ⭐⭐ (Best F1)
- **True Positives (TP)**: 237
- **False Positives (FP)**: 148
- **False Negatives (FN)**: 250
- **Precision**: 61.6%
- **Recall**: 48.7%
- **F1 Score**: 54.4%

### Threshold 0.8
- **True Positives (TP)**: 164
- **False Positives (FP)**: 58
- **False Negatives (FN)**: 323
- **Precision**: 73.9%
- **Recall**: 33.7%
- **F1 Score**: 46.3%

## Summary

**Best Overall Performance**: Threshold 0.7
- Highest F1 Score: **54.4%**
- Good precision-recall balance
- Precision: 61.6% (fewer false positives)
- Recall: 48.7% (moderate detection rate)

## Per-Image Metrics (Threshold 0.7)

| Image | Predicted | Ground Truth | TP | FP | FN |
|-------|-----------|--------------|----|----|----|
| DJI_20250926104000_0001_D_q11.JPG | 47 | 52 | 27 | 20 | 25 |
| DJI_20250926104012_0007_D_q01.JPG | 50 | 58 | 33 | 17 | 25 |
| DJI_20250926104012_0007_D_q11.JPG | 60 | 46 | 24 | 36 | 22 |
| DJI_20250926104028_0015_D_q11.JPG | 35 | 50 | 25 | 10 | 25 |
| DJI_20250926104036_0019_D_q00.JPG | 30 | 41 | 24 | 6 | 17 |
| DJI_20250926104042_0022_D_q01.JPG | 40 | 33 | 21 | 19 | 12 |
| DJI_20250926104044_0023_D_q11.JPG | 58 | 53 | 28 | 30 | 25 |
| DJI_20250926104048_0025_D_q01.JPG | 75 | 66 | 40 | 35 | 26 |
| DJI_20250926104048_0025_D_q11.JPG | 28 | 45 | 15 | 13 | 30 |
| DJI_20250926104052_0027_D_q11.JPG | 58 | 43 | 26 | 32 | 17 |

## Recommendations

1. **Optimal Threshold**: Use **0.7** for best F1 score (54.4%)
2. **For Higher Precision**: Use threshold **0.8** (73.9% precision, but lower recall)
3. **For Higher Recall**: Use threshold **0.5** (58.7% recall, but many false positives)

## Areas for Improvement

1. **False Positives**: Still significant at all thresholds - consider:
   - Post-processing with non-maximum suppression
   - Larger match distance for clustering
   - Additional filtering based on confidence distribution

2. **False Negatives**: Missing ~50% of walnuts - consider:
   - Lower threshold for difficult cases
   - Multi-scale detection
   - Data augmentation during training

3. **Model Training**: 
   - The model achieved 97.19% validation accuracy on patches, but detection performance is lower
   - This suggests the sliding window approach may need refinement
   - Consider training with hard negative mining

## Detailed Results

Detailed per-image metrics are saved in:
- `./detector_evaluation_results_new_th05/per_image_metrics.json` (threshold 0.5)
- `./detector_evaluation_results_new_th06/per_image_metrics.json` (threshold 0.6)
- `./detector_evaluation_results_new_th07/per_image_metrics.json` (threshold 0.7) ⭐
- `./detector_evaluation_results_new_th08/per_image_metrics.json` (threshold 0.8)

