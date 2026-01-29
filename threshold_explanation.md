# What is the Threshold Parameter For?

## Overview

The **threshold** parameter is a **confidence threshold** that controls how confident the model needs to be before it considers a detection as a "walnut".

## How It Works

### 1. **Model Output**
When the model processes a 32x32 pixel patch, it outputs a **confidence score** between 0.0 and 1.0:
- **0.0** = Definitely NOT a walnut (100% background)
- **1.0** = Definitely a walnut (100% confidence)
- **0.5** = Uncertain (50% walnut, 50% background)

### 2. **Threshold Filtering**
The threshold acts as a **filter**:
- Only patches with confidence **≥ threshold** are considered as detections
- Patches with confidence **< threshold** are ignored

### 3. **Example**

```
Patch Confidence: 0.85 (85% confident it's a walnut)

Threshold 0.5: ✅ DETECTED (0.85 > 0.5)
Threshold 0.6: ✅ DETECTED (0.85 > 0.6)
Threshold 0.7: ✅ DETECTED (0.85 > 0.7)
Threshold 0.8: ✅ DETECTED (0.85 > 0.8)
Threshold 0.9: ❌ REJECTED (0.85 < 0.9)
```

## Code Implementation

In `walnut_detector.py`, line 172:
```python
# Find high-confidence regions
high_conf_mask = confidence_map > self.confidence_threshold
```

This creates a mask where only pixels with confidence **above the threshold** are considered.

## Effect of Different Thresholds

### **Low Threshold (0.5)**
- ✅ **More detections** - accepts lower confidence scores
- ✅ **Higher recall** - catches more actual walnuts
- ⚠️ **More false positives** - also accepts uncertain patches
- **Result**: More detections, but some are wrong

### **High Threshold (0.8)**
- ✅ **Fewer false positives** - only very confident detections
- ✅ **Higher precision** - most detections are correct
- ⚠️ **Fewer detections** - misses uncertain walnuts
- ⚠️ **Lower recall** - misses more actual walnuts
- **Result**: Fewer detections, but most are correct

## Real Example from Your Tests

### Threshold 0.5
- **Predicted**: 483 walnuts
- **Precision**: 56.94% (275 correct, 208 wrong)
- **Recall**: 56.47% (caught 275 out of 487)
- **Result**: More detections, but ~43% are false positives

### Threshold 0.8
- **Predicted**: 144 walnuts
- **Precision**: 82.64% (119 correct, 25 wrong)
- **Recall**: 24.44% (caught only 119 out of 487)
- **Result**: Fewer detections, but ~83% are correct (missed many walnuts)

## The Trade-off

```
Lower Threshold (0.5)          Higher Threshold (0.8)
├─ More detections              ├─ Fewer detections
├─ Higher recall                ├─ Lower recall
├─ More false positives         ├─ Fewer false positives
└─ Lower precision              └─ Higher precision
```

## Choosing the Right Threshold

### Use **Low Threshold (0.5)** when:
- ✅ You want to catch as many walnuts as possible
- ✅ Missing walnuts is worse than false positives
- ✅ You can manually verify detections
- ✅ **Counting accuracy is critical**

### Use **High Threshold (0.8)** when:
- ✅ False positives are costly
- ✅ You need reliable detections
- ✅ Quality over quantity
- ✅ **Precision is critical**

### Use **Medium Threshold (0.6-0.7)** when:
- ✅ You want a balance
- ✅ Both precision and recall matter
- ✅ **F1 score is important**

## Summary

**Threshold = Minimum confidence required to consider a detection as a walnut**

- **Low threshold** = More lenient, more detections, more mistakes
- **High threshold** = More strict, fewer detections, fewer mistakes
- **Optimal threshold** = Balance between catching walnuts and avoiding false positives

For your precision model, **threshold 0.5** gives the best count accuracy (99.18%)!

