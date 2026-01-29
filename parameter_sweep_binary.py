#!/usr/bin/env python3
"""
Run a parameter sweep for the binary classifier: run test_binary_classifier.py for multiple patch_size,
stride, and threshold combinations, collect MAE/RMSE/R² and optional outputs, and save results to JSON.

How to run:
  python parameter_sweep_binary.py

Model path, test dir, parameter grid, and output path are set inside the script.
"""

import subprocess
import json
import os
from pathlib import Path

def run_test(patch_size, stride, threshold, output_suffix):
    """Run a single test with given parameters"""
    cmd = [
        "python3", "test_binary_classifier.py",
        "--model", "models/walnut_classifier_1.pth",
        "--test_dir", "test",
        "--output", f"binary_test_{output_suffix}",
        "--patch_size", str(patch_size),
        "--stride", str(stride),
        "--threshold", str(threshold)
    ]
    
    print(f"Testing: patch_size={patch_size}, stride={stride}, threshold={threshold}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Extract metrics from output
            lines = result.stdout.split('\n')
            mae = rmse = r2 = pred_mean = true_mean = None
            
            for line in lines:
                if "MAE (Mean Absolute Error):" in line:
                    mae = float(line.split(":")[1].strip())
                elif "RMSE (Root Mean Square Error):" in line:
                    rmse = float(line.split(":")[1].strip())
                elif "R² (Coefficient of Determination):" in line:
                    r2 = float(line.split(":")[1].strip())
                elif "Predicted Count Mean ± Std:" in line:
                    pred_mean = float(line.split(":")[1].strip().split("±")[0].strip())
                elif "True Count Mean ± Std:" in line:
                    true_mean = float(line.split(":")[1].strip().split("±")[0].strip())
            
            return {
                'patch_size': patch_size,
                'stride': stride,
                'threshold': threshold,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'pred_mean': pred_mean,
                'true_mean': true_mean,
                'success': True
            }
        else:
            print(f"Error: {result.stderr}")
            return {
                'patch_size': patch_size,
                'stride': stride,
                'threshold': threshold,
                'success': False,
                'error': result.stderr
            }
    except subprocess.TimeoutExpired:
        print(f"Timeout for patch_size={patch_size}, stride={stride}, threshold={threshold}")
        return {
            'patch_size': patch_size,
            'stride': stride,
            'threshold': threshold,
            'success': False,
            'error': 'Timeout'
        }

def main():
    print("🧪 Binary Classifier Parameter Sweep")
    print("=" * 50)
    
    # Parameter combinations to test
    patch_sizes = [8, 16, 24, 32]
    strides = [2, 4, 8, 16]
    thresholds = [0.1, 0.2, 0.3, 0.5]
    
    results = []
    total_tests = len(patch_sizes) * len(strides) * len(thresholds)
    current_test = 0
    
    print(f"Total tests to run: {total_tests}")
    print()
    
    for patch_size in patch_sizes:
        for stride in strides:
            # Skip if stride >= patch_size (would cause issues)
            if stride >= patch_size:
                continue
                
            for threshold in thresholds:
                current_test += 1
                output_suffix = f"p{patch_size}_s{stride}_t{threshold}"
                
                print(f"[{current_test}/{total_tests}] ", end="")
                result = run_test(patch_size, stride, threshold, output_suffix)
                results.append(result)
                
                if result['success']:
                    print(f"✅ MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}, R²: {result['r2']:.3f}")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                print()
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful tests!")
        return
    
    # Sort by MAE (lower is better)
    successful_results.sort(key=lambda x: x['mae'])
    
    print("=" * 50)
    print("📊 TOP 10 RESULTS (sorted by MAE)")
    print("=" * 50)
    
    for i, result in enumerate(successful_results[:10]):
        print(f"{i+1:2d}. Patch: {result['patch_size']:2d}, Stride: {result['stride']:2d}, "
              f"Threshold: {result['threshold']:.1f}")
        print(f"    MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}, "
              f"R²: {result['r2']:.3f}, Pred Mean: {result['pred_mean']:.1f}")
        print()
    
    # Find best overall
    best = successful_results[0]
    print("🏆 BEST CONFIGURATION:")
    print(f"   Patch Size: {best['patch_size']}")
    print(f"   Stride: {best['stride']}")
    print(f"   Threshold: {best['threshold']}")
    print(f"   MAE: {best['mae']:.2f}")
    print(f"   RMSE: {best['rmse']:.2f}")
    print(f"   R²: {best['r2']:.3f}")
    
    # Save results
    results_file = "binary_parameter_sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': results,
            'successful_results': successful_results,
            'best_configuration': best
        }, f, indent=2)
    
    print(f"\n💾 All results saved to {results_file}")

if __name__ == "__main__":
    main()

