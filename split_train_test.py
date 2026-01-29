#!/usr/bin/env python3
"""
Split Walnut Research images and annotations into train/ and test/ subdirectories (images + annotations).
Randomly selects a fixed number of images for test; rest go to train. Preserves directory structure.

How to run:
  python split_train_test.py [--source_dir path/to/walnut_research] [--num_test 10] [--seed 42]

Expects source_dir/cropped_images and source_dir/annotated_images; creates source_dir/train/images, train/annotations, test/images, test/annotations.
"""

import os
import shutil
from pathlib import Path
import random

def get_base_image_name(filename):
    """Extract base image name (e.g., DJI_20250926104000_0001_D from DJI_20250926104000_0001_D_q01.JPG)"""
    # Remove extension
    name = Path(filename).stem
    # Remove the _q01, _q11, etc. suffix
    if '_q' in name:
        base = '_'.join(name.split('_')[:-1])
    else:
        base = name
    return base

def split_data(source_dir, num_test=10, seed=42):
    """Split images and annotations into train/test directories"""
    
    random.seed(seed)
    
    source_path = Path(source_dir)
    images_dir = source_path / "cropped_images"
    annotations_dir = source_path / "annotated_images"
    
    # Create output directories
    train_images_dir = source_path / "train" / "images"
    train_annotations_dir = source_path / "train" / "annotations"
    test_images_dir = source_path / "test" / "images"
    test_annotations_dir = source_path / "test" / "annotations"
    
    for dir_path in [train_images_dir, train_annotations_dir, test_images_dir, test_annotations_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.jpg"))
    
    print(f"Total image files: {len(image_files)}")
    
    # Randomly select exactly num_test individual images for testing
    test_images = random.sample(image_files, min(num_test, len(image_files)))
    train_images = [img for img in image_files if img not in test_images]
    
    print(f"\nSelected {len(test_images)} images for testing:")
    for img in sorted(test_images, key=lambda x: x.name):
        print(f"  - {img.name}")
    
    print(f"\nRemaining {len(train_images)} images for training")
    
    # Move test images and annotations
    test_image_count = 0
    test_annotation_count = 0
    for img_file in test_images:
        dest = test_images_dir / img_file.name
        shutil.copy2(img_file, dest)
        test_image_count += 1
        
        # Find and copy corresponding annotation
        annotation_file = annotations_dir / f"{img_file.stem}.txt"
        if annotation_file.exists():
            dest_ann = test_annotations_dir / annotation_file.name
            shutil.copy2(annotation_file, dest_ann)
            test_annotation_count += 1
    
    # Move train images and annotations
    train_image_count = 0
    train_annotation_count = 0
    for img_file in train_images:
        dest = train_images_dir / img_file.name
        shutil.copy2(img_file, dest)
        train_image_count += 1
        
        # Find and copy corresponding annotation
        annotation_file = annotations_dir / f"{img_file.stem}.txt"
        if annotation_file.exists():
            dest_ann = train_annotations_dir / annotation_file.name
            shutil.copy2(annotation_file, dest_ann)
            train_annotation_count += 1
    
    print(f"\n✅ Split complete!")
    print(f"Test: {test_image_count} images, {test_annotation_count} annotations")
    print(f"Train: {train_image_count} images, {train_annotation_count} annotations")
    print(f"\nDirectories created:")
    print(f"  - {train_images_dir}")
    print(f"  - {train_annotations_dir}")
    print(f"  - {test_images_dir}")
    print(f"  - {test_annotations_dir}")

if __name__ == "__main__":
    import sys
    
    source_dir = "/Users/kalpit/TuesdayPresentation/Walnut_Research"
    num_test = 10
    
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    if len(sys.argv) > 2:
        num_test = int(sys.argv[2])
    
    split_data(source_dir, num_test)

