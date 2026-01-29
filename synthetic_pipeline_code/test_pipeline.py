#!/usr/bin/env python3
"""
Smoke tests for the synthetic pipeline: walnut extraction, alpha mask, placement mask, etc.
Runs a few checks to ensure pipeline components run without full data.

How to run:
  python test_pipeline.py

Run from synthetic_pipeline_code/. Creates temporary test_output/ if needed.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

def test_walnut_extraction():
    """Test walnut extraction functionality"""
    print("🧪 Testing walnut extraction...")
    
    # Create a simple test image with a "walnut"
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Add a circular "walnut" in the center
    cv2.circle(test_image, (50, 50), 15, (100, 150, 200), -1)
    cv2.circle(test_image, (50, 50), 15, (80, 120, 180), 2)
    
    # Test alpha mask creation
    from synthetic_pipeline import WalnutExtractor
    
    # Create temporary directories
    os.makedirs("test_output/walnut_cutouts", exist_ok=True)
    os.makedirs("test_output/metadata", exist_ok=True)
    
    # Test alpha mask creation
    extractor = WalnutExtractor("dummy", "test_output")
    alpha_mask = extractor._create_alpha_mask(test_image, 50, 50)
    
    # Check if alpha mask was created
    assert alpha_mask is not None, "Alpha mask creation failed"
    assert alpha_mask.shape == test_image.shape[:2], "Alpha mask shape mismatch"
    
    print("✅ Walnut extraction test passed")
    return True

def test_photometric_matching():
    """Test photometric matching functionality"""
    print("🧪 Testing photometric matching...")
    
    from synthetic_generator import PhotometricMatcher
    
    # Create test patches
    walnut_patch = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    background_patch = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
    alpha_mask = np.ones((32, 32), dtype=np.uint8) * 255
    
    # Test photometric matching
    matcher = PhotometricMatcher()
    matched = matcher.match_walnut_to_background(walnut_patch, background_patch, alpha_mask)
    
    # Check if matching worked
    assert matched is not None, "Photometric matching failed"
    assert matched.shape == walnut_patch.shape, "Matched patch shape mismatch"
    
    print("✅ Photometric matching test passed")
    return True

def test_density_map_creation():
    """Test density map creation"""
    print("🧪 Testing density map creation...")
    
    from synthetic_generator import SyntheticImageGenerator
    
    # Create test data
    walnut_centers = [(100, 100), (200, 150), (300, 200)]
    image_size = (512, 512)
    patch_size = 32
    
    # Create generator instance (without loading real data)
    generator = SyntheticImageGenerator([], [], "test_output")
    
    # Test density map creation
    density_map = generator._create_density_map(walnut_centers, image_size, patch_size)
    
    # Check density map
    expected_h = image_size[1] // patch_size
    expected_w = image_size[0] // patch_size
    
    assert density_map.shape == (expected_h, expected_w), "Density map shape mismatch"
    assert np.sum(density_map) == len(walnut_centers), "Density map count mismatch"
    
    print("✅ Density map creation test passed")
    return True

def test_multi_channel_transform():
    """Test multi-channel image transformation"""
    print("🧪 Testing multi-channel transform...")
    
    from density_model import MultiChannelTransform
    
    # Create test image
    test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Test transform
    transform = MultiChannelTransform(patch_size=32)
    multi_channel = transform(test_image)
    
    # Check output
    assert multi_channel is not None, "Multi-channel transform failed"
    assert multi_channel.shape[2] == 6, "Expected 6 channels"
    assert multi_channel.dtype == np.float32, "Expected float32 output"
    
    print("✅ Multi-channel transform test passed")
    return True

def test_model_creation():
    """Test model creation"""
    print("🧪 Testing model creation...")
    
    from density_model import MultiScaleDensityNet, ModelConfig
    
    # Create model config
    config = ModelConfig(
        input_channels=6,
        patch_size=32,
        hidden_dim=64,  # Smaller for testing
        num_scales=2    # Smaller for testing
    )
    
    # Create model
    model = MultiScaleDensityNet(config)
    
    # Test forward pass
    batch_size = 2
    input_tensor = np.random.randn(batch_size, 6, 32, 32).astype(np.float32)
    
    import torch
    input_torch = torch.from_numpy(input_tensor)
    
    with torch.no_grad():
        output = model(input_torch)
    
    # Check output
    assert output is not None, "Model forward pass failed"
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == 1, "Expected single channel output"
    
    print("✅ Model creation test passed")
    return True

def cleanup_test_files():
    """Clean up test files"""
    import shutil
    
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
        print("🧹 Cleaned up test files")

def main():
    """Run all tests"""
    print("🧪 Running Synthetic Pipeline Tests")
    print("=" * 40)
    
    tests = [
        test_walnut_extraction,
        test_photometric_matching,
        test_density_map_creation,
        test_multi_channel_transform,
        test_model_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Pipeline is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    # Cleanup
    cleanup_test_files()
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
