#!/usr/bin/env python3
"""
Summarize threshold evaluation results: reads all_threshold_results.json and prints precision, count accuracy,
recall, F1, TP/FP/FN by threshold for Walnut Research and Glenn datasets. Also prints best-by-count-accuracy
and best-by-precision.

How to run:
  python summarize_threshold_results.py

Expects all_threshold_results.json in project root (path set inside script).
"""

import json
from pathlib import Path

# Load the results
results_file = Path("/Users/kalpit/TuesdayPresentation/all_threshold_results.json")
with open(results_file, 'r') as f:
    data = json.load(f)

# Organize by dataset
walnut_research_results = []
glenn_results = []

for result in data['all_results']:
    if result['dataset'] == 'walnut_research':
        walnut_research_results.append(result)
    elif result['dataset'] == 'glenn':
        glenn_results.append(result)

# Sort by threshold
walnut_research_results.sort(key=lambda x: x['threshold'] if x['threshold'] else 0)
glenn_results.sort(key=lambda x: x['threshold'] if x['threshold'] else 0)

print("=" * 100)
print("PRECISION & COUNT ACCURACY BY THRESHOLD")
print("=" * 100)

# Walnut Research Dataset
if walnut_research_results:
    print("\n" + "=" * 100)
    print("WALNUT RESEARCH DATASET (Ground Truth: 487 walnuts)")
    print("=" * 100)
    print(f"{'Threshold':<12} {'Predicted':<12} {'Count Acc %':<15} {'Precision %':<15} {'Recall %':<15} {'F1 %':<15} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 100)
    
    for r in walnut_research_results:
        threshold = f"{r['threshold']:.2f}" if r['threshold'] else "N/A"
        count_acc = f"{r['count_accuracy']:.2f}" if r['count_accuracy'] else "N/A"
        print(f"{threshold:<12} {r['total_predicted']:<12} {count_acc:<15} {r['precision']:<15.2f} {r['recall']:<15.2f} {r['f1']:<15.2f} {r['TP']:<8} {r['FP']:<8} {r['FN']:<8}")
    
    # Best results
    print("\n" + "-" * 100)
    print("BEST RESULTS FOR WALNUT RESEARCH:")
    print("-" * 100)
    
    best_count_acc = max([r for r in walnut_research_results if r['count_accuracy']], 
                         key=lambda x: x['count_accuracy'], default=None)
    if best_count_acc:
        print(f"\n🏆 Best Count Accuracy:")
        print(f"   Threshold: {best_count_acc['threshold']:.2f}")
        print(f"   Count Accuracy: {best_count_acc['count_accuracy']:.2f}%")
        print(f"   Precision: {best_count_acc['precision']:.2f}%")
        print(f"   Recall: {best_count_acc['recall']:.2f}%")
        print(f"   F1: {best_count_acc['f1']:.2f}%")
        print(f"   Predicted: {best_count_acc['total_predicted']} vs Ground Truth: {best_count_acc['ground_truth']}")
    
    best_precision = max(walnut_research_results, key=lambda x: x['precision'])
    print(f"\n🎯 Best Precision:")
    print(f"   Threshold: {best_precision['threshold']:.2f}")
    print(f"   Precision: {best_precision['precision']:.2f}%")
    print(f"   Count Accuracy: {best_precision['count_accuracy']:.2f}%" if best_precision['count_accuracy'] else "   Count Accuracy: N/A")
    print(f"   Recall: {best_precision['recall']:.2f}%")
    print(f"   F1: {best_precision['f1']:.2f}%")
    
    best_f1 = max(walnut_research_results, key=lambda x: x['f1'])
    print(f"\n⚖️  Best F1 Score:")
    print(f"   Threshold: {best_f1['threshold']:.2f}")
    print(f"   F1: {best_f1['f1']:.2f}%")
    print(f"   Precision: {best_f1['precision']:.2f}% | Recall: {best_f1['recall']:.2f}%")
    print(f"   Count Accuracy: {best_f1['count_accuracy']:.2f}%" if best_f1['count_accuracy'] else "   Count Accuracy: N/A")

# Glenn Dataset
if glenn_results:
    print("\n\n" + "=" * 100)
    print("GLENN DATASET (Ground Truth: 1782 walnuts)")
    print("=" * 100)
    print(f"{'Threshold':<12} {'Predicted':<12} {'Count Acc %':<15} {'Precision %':<15} {'Recall %':<15} {'F1 %':<15} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 100)
    
    for r in glenn_results:
        threshold = f"{r['threshold']:.2f}" if r['threshold'] else "N/A"
        count_acc = f"{r['count_accuracy']:.2f}" if r['count_accuracy'] else "N/A"
        print(f"{threshold:<12} {r['total_predicted']:<12} {count_acc:<15} {r['precision']:<15.2f} {r['recall']:<15.2f} {r['f1']:<15.2f} {r['TP']:<8} {r['FP']:<8} {r['FN']:<8}")
    
    # Best results
    print("\n" + "-" * 100)
    print("BEST RESULTS FOR GLENN:")
    print("-" * 100)
    
    best_count_acc = max([r for r in glenn_results if r['count_accuracy']], 
                         key=lambda x: x['count_accuracy'], default=None)
    if best_count_acc:
        print(f"\n🏆 Best Count Accuracy:")
        print(f"   Threshold: {best_count_acc['threshold']:.2f}")
        print(f"   Count Accuracy: {best_count_acc['count_accuracy']:.2f}%")
        print(f"   Precision: {best_count_acc['precision']:.2f}%")
        print(f"   Recall: {best_count_acc['recall']:.2f}%")
        print(f"   F1: {best_count_acc['f1']:.2f}%")
        print(f"   Predicted: {best_count_acc['total_predicted']} vs Ground Truth: {best_count_acc['ground_truth']}")
    
    best_precision = max(glenn_results, key=lambda x: x['precision'])
    print(f"\n🎯 Best Precision:")
    print(f"   Threshold: {best_precision['threshold']:.2f}")
    print(f"   Precision: {best_precision['precision']:.2f}%")
    print(f"   Count Accuracy: {best_precision['count_accuracy']:.2f}%" if best_precision['count_accuracy'] else "   Count Accuracy: N/A")
    print(f"   Recall: {best_precision['recall']:.2f}%")
    print(f"   F1: {best_precision['f1']:.2f}%")

print("\n" + "=" * 100)

