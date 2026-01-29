#!/usr/bin/env python3
"""
Run the full synthetic walnut pipeline: extract walnuts + canopy plates, generate synthetic images,
train the density model on synthetic data, and optionally fine-tune on real data. Orchestrates
synthetic_pipeline.py, synthetic_generator.py, density_model.py, fine_tune_real.py.

How to run:
  python run_pipeline.py --train_data path/to/annotated_dataset --output path/to/synthetic_data [--num_synthetic 2000] [--num_canopy_plates 30] [--training_epochs 50] [--fine_tune_epochs 15]

Use --help for all options.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔄 {description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Use actual import names rather than pip package names where they differ
    required_packages = [
        'torch',
        'torchvision',
        'cv2',          # opencv-python installs as cv2
        'numpy',
        'PIL',
        'matplotlib',
        'sklearn'       # scikit-learn installs as sklearn
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install opencv-python scikit-learn (and other listed packages if missing)")
        return False
    
    print("✅ All dependencies available")
    return True

def main():
    """Main pipeline runner"""
    
    parser = argparse.ArgumentParser(description="Synthetic Walnut Generation Pipeline")
    parser.add_argument("--train_data", default="../walnut_annotated_dataset",
                       help="Path to training dataset")
    parser.add_argument("--output", default="./synthetic_data",
                       help="Output directory for synthetic data")
    parser.add_argument("--num_synthetic", type=int, default=2000,
                       help="Number of synthetic images to generate")
    parser.add_argument("--num_canopy_plates", type=int, default=30,
                       help="Number of canopy plates to generate")
    parser.add_argument("--training_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=15,
                       help="Number of fine-tuning epochs")
    parser.add_argument("--skip_synthetic", action="store_true",
                       help="Skip synthetic data generation (use existing)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip model training (use existing)")
    parser.add_argument("--skip_fine_tuning", action="store_true",
                       help="Skip fine-tuning (use existing)")
    
    args = parser.parse_args()
    
    print("🌰 Synthetic Walnut Generation Pipeline")
    print("=" * 50)
    print(f"Training data: {args.train_data}")
    print(f"Output directory: {args.output}")
    print(f"Synthetic images: {args.num_synthetic}")
    print(f"Canopy plates: {args.num_canopy_plates}")
    print(f"Training epochs: {args.training_epochs}")
    print(f"Fine-tuning epochs: {args.fine_tune_epochs}")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Step 1: Generate synthetic data
    if not args.skip_synthetic:
        print("\n📦 Step 1: Generating synthetic data...")
        
        # Extract walnut instances and generate canopy plates
        synthetic_cmd = f"python synthetic_pipeline.py --train_data {args.train_data} --output {args.output} --num_canopy_plates {args.num_canopy_plates}"
        
        if not run_command(synthetic_cmd, "Synthetic data generation"):
            print("❌ Synthetic data generation failed. Stopping.")
            return False
        
        # Generate synthetic images
        generator_cmd = f"python synthetic_generator.py --walnut_instances {args.output}/metadata --canopy_plates {args.output}/canopy_plates --output {args.output} --num_images {args.num_synthetic}"
        
        if not run_command(generator_cmd, "Synthetic image generation"):
            print("❌ Synthetic image generation failed. Stopping.")
            return False
    
    # Step 2: Train density estimation model
    if not args.skip_training:
        print("\n🧠 Step 2: Training density estimation model...")
        
        training_cmd = f"python density_model.py --synthetic_data {args.output} --output {args.output}/models --epochs {args.training_epochs}"
        
        if not run_command(training_cmd, "Model training"):
            print("❌ Model training failed. Stopping.")
            return False
    
    # Step 3: Fine-tune on real data
    if not args.skip_fine_tuning:
        print("\n🔧 Step 3: Fine-tuning on real data...")
        
        pretrained_model = f"{args.output}/models/density_model.pth"
        if not os.path.exists(pretrained_model):
            print(f"❌ Pre-trained model not found: {pretrained_model}")
            return False
        
        fine_tune_cmd = f"python fine_tune_real.py --pretrained_model {pretrained_model} --real_data {args.train_data} --output {args.output}/models --epochs {args.fine_tune_epochs}"
        
        if not run_command(fine_tune_cmd, "Fine-tuning on real data"):
            print("❌ Fine-tuning failed. Stopping.")
            return False
    
    # Step 4: Generate summary report
    print("\n📊 Step 4: Generating summary report...")
    
    # Collect statistics
    stats = {
        'pipeline_completed': True,
        'synthetic_images_generated': args.num_synthetic,
        'canopy_plates_generated': args.num_canopy_plates,
        'training_epochs': args.training_epochs,
        'fine_tuning_epochs': args.fine_tune_epochs,
        'output_directory': args.output
    }
    
    # Check if files exist
    synthetic_images_dir = f"{args.output}/synthetic_images"
    density_maps_dir = f"{args.output}/density_maps"
    models_dir = f"{args.output}/models"
    
    if os.path.exists(synthetic_images_dir):
        stats['synthetic_images_count'] = len(list(Path(synthetic_images_dir).glob("*.png")))
    
    if os.path.exists(density_maps_dir):
        stats['density_maps_count'] = len(list(Path(density_maps_dir).glob("*.npy")))
    
    if os.path.exists(f"{models_dir}/density_model.pth"):
        stats['pretrained_model_exists'] = True
    
    if os.path.exists(f"{models_dir}/fine_tuned_density_model.pth"):
        stats['fine_tuned_model_exists'] = True
    
    # Save statistics
    stats_path = f"{args.output}/pipeline_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 50)
    print("📁 Output structure:")
    print(f"  {args.output}/")
    print(f"  ├── synthetic_images/     ({stats.get('synthetic_images_count', 0)} images)")
    print(f"  ├── density_maps/         ({stats.get('density_maps_count', 0)} maps)")
    print(f"  ├── canopy_plates/        ({args.num_canopy_plates} plates)")
    print(f"  ├── walnut_cutouts/       (walnut instances)")
    print(f"  ├── models/               (trained models)")
    print(f"  └── metadata/             (statistics and configs)")
    print("\n🚀 Ready for inference!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
