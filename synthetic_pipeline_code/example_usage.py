#!/usr/bin/env python3
"""
Example usage and commands for the synthetic walnut pipeline: shows how to run the full pipeline,
individual steps (extract, generate, train, fine-tune), and typical argument values.

How to run:
  python example_usage.py

Prints example commands and descriptions; does not execute the full pipeline by default.
"""

import os
import sys
import argparse
from pathlib import Path

def example_1_basic_pipeline():
    """Example 1: Run the complete pipeline"""
    print("🌰 Example 1: Complete Pipeline")
    print("-" * 40)
    
    command = """
    # Run the complete pipeline
    python run_pipeline.py \\
        --train_data ../walnut_annotated_dataset \\
        --output ./synthetic_data \\
        --num_synthetic 1000 \\
        --num_canopy_plates 20 \\
        --training_epochs 30 \\
        --fine_tune_epochs 10
    """
    
    print("Command:")
    print(command)
    print("\nThis will:")
    print("1. Extract walnut instances from training data")
    print("2. Generate canopy background plates")
    print("3. Create 1000 synthetic images")
    print("4. Train density estimation model")
    print("5. Fine-tune on real validation data")

def example_2_step_by_step():
    """Example 2: Run pipeline step by step"""
    print("\n🔧 Example 2: Step-by-Step Pipeline")
    print("-" * 40)
    
    commands = [
        "# Step 1: Extract walnuts and generate canopy plates",
        "python synthetic_pipeline.py --train_data ../walnut_annotated_dataset --output ./synthetic_data --num_canopy_plates 20",
        "",
        "# Step 2: Generate synthetic images",
        "python synthetic_generator.py --walnut_instances ./synthetic_data/metadata --canopy_plates ./synthetic_data/canopy_plates --output ./synthetic_data --num_images 1000",
        "",
        "# Step 3: Train density model",
        "python density_model.py --synthetic_data ./synthetic_data --output ./models --epochs 30",
        "",
        "# Step 4: Fine-tune on real data",
        "python fine_tune_real.py --pretrained_model ./models/density_model.pth --real_data ../walnut_annotated_dataset --output ./models --epochs 10"
    ]
    
    for cmd in commands:
        print(cmd)

def example_3_custom_parameters():
    """Example 3: Custom parameters for different use cases"""
    print("\n⚙️  Example 3: Custom Parameters")
    print("-" * 40)
    
    examples = [
        {
            "name": "High-quality synthetic data",
            "command": "python run_pipeline.py --num_synthetic 5000 --num_canopy_plates 50 --training_epochs 100",
            "description": "Generate more synthetic data and train longer for better quality"
        },
        {
            "name": "Quick testing",
            "command": "python run_pipeline.py --num_synthetic 100 --num_canopy_plates 5 --training_epochs 10",
            "description": "Quick test run with minimal data and training"
        },
        {
            "name": "Large images",
            "command": "python run_pipeline.py --num_synthetic 2000 --image_size 1024 1024",
            "description": "Generate larger synthetic images (1024x1024)"
        },
        {
            "name": "Skip synthetic generation",
            "command": "python run_pipeline.py --skip_synthetic --skip_training",
            "description": "Only run fine-tuning on existing models"
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  {example['command']}")
        print(f"  {example['description']}")

def example_4_inference():
    """Example 4: Using trained models for inference"""
    print("\n🔮 Example 4: Model Inference")
    print("-" * 40)
    
    inference_code = '''
import torch
import cv2
import numpy as np
from density_model import MultiScaleDensityNet, ModelConfig, MultiChannelTransform

def count_walnuts_in_image(image_path, model_path):
    """Count walnuts in an image using trained model"""
    
    # Load model
    config = ModelConfig()
    model = MultiScaleDensityNet(config)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    transform = MultiChannelTransform(patch_size=32)
    multi_channel = transform(image)
    
    # Convert to tensor
    input_tensor = torch.from_numpy(multi_channel).permute(2, 0, 1).float().unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        density_map = model(input_tensor)
        total_count = density_map.sum().item()
    
    return total_count, density_map

# Usage
image_path = "your_walnut_image.jpg"
model_path = "models/fine_tuned_density_model.pth"
count, density_map = count_walnuts_in_image(image_path, model_path)
print(f"Estimated walnut count: {count:.1f}")
    '''
    
    print("Python code for inference:")
    print(inference_code)

def example_5_troubleshooting():
    """Example 5: Common troubleshooting tips"""
    print("\n🔧 Example 5: Troubleshooting")
    print("-" * 40)
    
    issues = [
        {
            "problem": "Out of memory during training",
            "solution": "Reduce batch_size: --batch_size 8 or --batch_size 4"
        },
        {
            "problem": "Synthetic images look unrealistic",
            "solution": "Increase num_canopy_plates: --num_canopy_plates 50"
        },
        {
            "problem": "Model not converging",
            "solution": "Increase training_epochs: --training_epochs 100"
        },
        {
            "problem": "Fine-tuning overfitting",
            "solution": "Reduce fine_tune_epochs: --fine_tune_epochs 5"
        },
        {
            "problem": "Dependencies missing",
            "solution": "pip install torch torchvision opencv-python numpy PIL matplotlib scikit-learn"
        }
    ]
    
    for issue in issues:
        print(f"\n❌ {issue['problem']}")
        print(f"✅ {issue['solution']}")

def main():
    """Main function to show examples"""
    
    parser = argparse.ArgumentParser(description="Synthetic Pipeline Examples")
    parser.add_argument("--example", type=int, choices=[1, 2, 3, 4, 5], 
                       help="Show specific example (1-5)")
    
    args = parser.parse_args()
    
    print("🌰 Synthetic Walnut Generation Pipeline - Examples")
    print("=" * 60)
    
    if args.example:
        examples = {
            1: example_1_basic_pipeline,
            2: example_2_step_by_step,
            3: example_3_custom_parameters,
            4: example_4_inference,
            5: example_5_troubleshooting
        }
        examples[args.example]()
    else:
        # Show all examples
        example_1_basic_pipeline()
        example_2_step_by_step()
        example_3_custom_parameters()
        example_4_inference()
        example_5_troubleshooting()
    
    print("\n" + "=" * 60)
    print("📚 For more details, see README.md")
    print("🧪 Run tests with: python test_pipeline.py")

if __name__ == "__main__":
    main()
