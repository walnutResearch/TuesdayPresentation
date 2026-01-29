#!/usr/bin/env python3
"""
Run a quick parameter sweep: call test_binary_classifier.py with several patch_size, stride, and threshold
combinations, parse MAE/RMSE/R² from stdout, and print or save a summary. Used to find good detector settings.

How to run:
  python quick_parameter_sweep.py

Model path, test dir, and parameter lists are set inside the script (subprocess calls test_binary_classifier.py).
"""

import subprocess
import json

def run_test(patch_size, stride, threshold):
    """Run a single test with given parameters"""
    cmd = [
        "python3", "test_binary_classifier.py",
        "--model", "models/walnut_classifier_1.pth",
        "--test_dir", "test",
        "--output", f"binary_test_p{patch_size}_s{stride}_t{threshold}",
        "--patch_size", str(patch_size),
        "--stride", str(stride),
        "--threshold", str(threshold)
    ]
    
    print(f"Testing: patch_size={patch_size}, stride={stride}, threshold={threshold}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            # Extract metrics from output
            lines = result.stdout.split('\n')
            mae = rmse = r2 = pred_mean = None
            
            for line in lines:
                if "MAE (Mean Absolute Error):" in line:
                    mae = float(line.split(":")[1].strip())
                elif "RMSE (Root Mean Square Error):" in line:
                    rmse = float(line.split(":")[1].strip())
                elif "R² (Coefficient of Determination):" in line:
                    r2 = float(line.split(":")[1].strip())
                elif "Predicted Count Mean ± Std:" in line:
                    pred_mean = float(line.split(":")[1].strip().split("±")[0].strip())
            
            return {
                'patch_size': patch_size,
                'stride': stride,
                'threshold': threshold,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'pred_mean': pred_mean,
                'success': True
            }
        else:
            return {'success': False, 'error': result.stderr}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    print("🧪 Quick Binary Classifier Parameter Sweep")
    print("=" * 50)
    
    # Key parameter combinations to test
    test_configs = [
        # (patch_size, stride, threshold)
        (8, 4, 0.1),
        (8, 4, 0.2),
        (16, 4, 0.1),
        (16, 4, 0.2),
        (16, 8, 0.1),
        (16, 8, 0.2),
        (24, 8, 0.1),
        (24, 8, 0.2),
        (24, 12, 0.1),
        (24, 12, 0.2),
        (32, 8, 0.1),
        (32, 8, 0.2),
        (32, 16, 0.1),
        (32, 16, 0.2),
    ]
    
    results = []
    
    for i, (patch_size, stride, threshold) in enumerate(test_configs):
        print(f"[{i+1}/{len(test_configs)}] ", end="")
        result = run_test(patch_size, stride, threshold)
        results.append(result)
        
        if result['success']:
            print(f"✅ MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}, R²: {result['r2']:.3f}")
        else:
            print(f"❌ Failed")
        print()
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful tests!")
        return
    
    # Sort by MAE (lower is better)
    successful_results.sort(key=lambda x: x['mae'])
    
    print("=" * 50)
    print("📊 RESULTS (sorted by MAE)")
    print("=" * 50)
    
    for i, result in enumerate(successful_results):
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
    with open("quick_parameter_sweep_results.json", 'w') as f:
        json.dump({
            'all_results': results,
            'successful_results': successful_results,
            'best_configuration': best
        }, f, indent=2)
    
    print(f"\n💾 Results saved to quick_parameter_sweep_results.json")

if __name__ == "__main__":
    main()

