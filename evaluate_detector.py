#!/usr/bin/env python3
"""
Evaluate the walnut detector on an annotated test set: run sliding-window detection on each test image,
load GT center-point annotations, match predictions to GT with a distance threshold, and compute TP/FP/FN,
precision, recall, F1. Saves per_image_metrics.json and summary.json (and optional confusion plot).

How to run:
  python evaluate_detector.py --model_path models_new/walnut_classifier.pth --images_dir path/to/images --labels_dir path/to/annotations --output_dir path/to/output [--threshold 0.6] [--patch_size 32] [--stride 16] [--match_distance 20]

Use --help for all options.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# Local import
from walnut_detector import WalnutDetector


def _parse_numeric_pair(tokens: List[str]) -> Optional[Tuple[float, float]]:
    try:
        x = float(tokens[0])
        y = float(tokens[1])
        return x, y
    except Exception:
        return None


def load_annotations(
    label_path: Path,
    image_wh: Tuple[int, int],
    ann_format: str = "auto",
) -> List[Tuple[int, int]]:
    """
    Load center points from a label file.
    Accepted formats (auto-detected unless specified):
      - pixels: "x y" (ints/floats in pixel coords)
      - class-prefixed: "cls x y" (we take last 2 numbers)
      - YOLO-normalized: x,y in [0,1] → multiplied by (W,H)
      - CSV: "x,y" (comma separated)
    """
    points: List[Tuple[int, int]] = []
    if not label_path.exists():
        return points

    W, H = image_wh
    try:
        with open(label_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                # CSV case
                if "," in line and (ann_format == "auto" or ann_format == "csv"):
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 2:
                        parsed = _parse_numeric_pair(parts[:2])
                        if parsed is None:
                            continue
                        x, y = parsed
                    else:
                        continue
                else:
                    parts = line.split()
                    # class-prefixed → take last two numeric tokens
                    if len(parts) >= 3 and (ann_format == "auto" or ann_format == "points"):
                        parsed = _parse_numeric_pair(parts[-2:])
                        if parsed is not None:
                            x, y = parsed
                        else:
                            continue
                    elif len(parts) >= 2:
                        parsed = _parse_numeric_pair(parts[:2])
                        if parsed is None:
                            continue
                        x, y = parsed
                    else:
                        continue

                # Normalize handling
                if ann_format == "yolo" or (ann_format == "auto" and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    x_px = int(round(x * W))
                    y_px = int(round(y * H))
                else:
                    x_px = int(round(x))
                    y_px = int(round(y))

                # Bounds clamp
                x_px = max(0, min(W - 1, x_px))
                y_px = max(0, min(H - 1, y_px))
                points.append((x_px, y_px))
    except Exception:
        pass
    return points


def greedy_match(preds: List[Tuple[int, int]], gts: List[Tuple[int, int]], max_dist: float) -> Tuple[int, int, int]:
    """
    Greedy matching between predicted centers and ground-truth centers using a distance threshold.
    Returns (TP, FP, FN).
    """
    if len(preds) == 0 and len(gts) == 0:
        return 0, 0, 0

    preds_array = np.array(preds, dtype=np.float32) if preds else np.zeros((0, 2), dtype=np.float32)
    gts_array = np.array(gts, dtype=np.float32) if gts else np.zeros((0, 2), dtype=np.float32)

    if len(preds_array) == 0:
        return 0, 0, int(len(gts_array))
    if len(gts_array) == 0:
        return 0, int(len(preds_array)), 0

    # Compute distance matrix
    dists = np.linalg.norm(preds_array[:, None, :] - gts_array[None, :, :], axis=2)  # (P, G)

    # Greedy: repeatedly take the minimal distance pair under threshold
    TP = 0
    used_preds = set()
    used_gts = set()

    while True:
        # Set large value for already used rows/cols
        masked = dists.copy()
        if used_preds:
            masked[list(used_preds), :] = np.inf
        if used_gts:
            masked[:, list(used_gts)] = np.inf

        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        min_val = masked[i, j]
        if not np.isfinite(min_val) or min_val > max_dist:
            break

        TP += 1
        used_preds.add(i)
        used_gts.add(j)

    FP = len(preds_array) - len(used_preds)
    FN = len(gts_array) - len(used_gts)
    return TP, FP, FN


def evaluate(
    model_path: str,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    patch_size: int = 32,
    stride: int = 16,
    threshold: float = 0.6,
    match_dist: float = 12.0,
    cluster: bool = True,
    ann_format: str = "auto",
) -> Dict:
    """Run evaluation over a folder of images with annotation TXT files."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize detector
    detector = WalnutDetector(
        model_path=model_path,
        patch_size=patch_size,
        stride=stride,
        confidence_threshold=threshold,
    )

    image_paths = sorted(list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg")) + 
                        list(Path(images_dir).glob("*.JPG")) + list(Path(images_dir).glob("*.PNG")))
    print(f"Found {len(image_paths)} test images")

    per_image = []
    total_TP = total_FP = total_FN = 0

    for img_path in tqdm(image_paths, desc="Evaluating"):
        img_path = Path(img_path)
        gt_path = Path(labels_dir) / f"{img_path.stem}.txt"

        # Run detection
        try:
            results = detector.process_image(str(img_path), None, cluster)
            preds: List[Tuple[int, int]] = results["centers"]
        except Exception as e:
            print(f"Error on {img_path.name}: {e}")
            preds = []

        # Load GT
        # Need image size for potential normalized coords
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: could not read image for size: {img_path.name}")
            H = W = 0
        else:
            H, W = img.shape[:2]
        gts = load_annotations(gt_path, (W, H), ann_format=ann_format)

        if len(gts) == 0:
            print(f"Warning: no GT points for {img_path.stem} (looked for {gt_path.name})")

        # Match
        TP, FP, FN = greedy_match(preds, gts, match_dist)
        total_TP += TP
        total_FP += FP
        total_FN += FN

        per_image.append({
            "image": img_path.name,
            "num_preds": len(preds),
            "num_gts": len(gts),
            "TP": TP,
            "FP": FP,
            "FN": FN,
        })

    # Aggregate metrics
    precision = total_TP / (total_TP + total_FP + 1e-6)
    recall = total_TP / (total_TP + total_FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    summary = {
        "images": len(image_paths),
        "TP": int(total_TP),
        "FP": int(total_FP),
        "FN": int(total_FN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "match_distance_px": match_dist,
        "patch_size": patch_size,
        "stride": stride,
        "threshold": threshold,
    }

    # Save reports
    with open(Path(output_dir) / "per_image_metrics.json", "w") as f:
        json.dump(per_image, f, indent=2)
    with open(Path(output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Confusion matrix-like visualization (TP/FP/FN)
    fig, ax = plt.subplots(figsize=(4, 3))
    data = np.array([[summary["TP"], summary["FP"]], [summary["FN"], 0]])
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["TP", "FP"])  # columns
    ax.set_yticklabels(["FN", "-"])   # rows (TN undefined for point-detection)
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
    ax.set_title("Detection Confusion Summary")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "confusion_summary.png", dpi=200)
    plt.close()

    print("\nEvaluation Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved detailed results to {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate walnut detector on annotated test set")
    parser.add_argument("--model_path", required=True, help="Path to trained model .pth")
    parser.add_argument("--test_dir", required=True, help="Path to test dataset root containing images/ and annotations/")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--match_dist", type=float, default=12.0, help="Matching distance (pixels)")
    parser.add_argument("--no_cluster", action="store_true", help="Disable clustering of nearby detections")
    parser.add_argument("--ann_format", type=str, default="auto", choices=["auto", "points", "yolo", "csv"], help="Annotation file format")

    args = parser.parse_args()

    # Check if images are in a subdirectory (must match extensions used in evaluate())
    images_subdir = os.path.join(args.test_dir, "images")
    subdir_images = (list(Path(images_subdir).glob("*.png")) + list(Path(images_subdir).glob("*.jpg")) +
                     list(Path(images_subdir).glob("*.JPG")) + list(Path(images_subdir).glob("*.PNG")))
    if os.path.exists(images_subdir) and len(subdir_images) > 0:
        images_dir = images_subdir
    else:
        images_dir = args.test_dir  # Images are directly in test directory
    labels_dir = os.path.join(args.test_dir, "annotations")

    evaluate(
        model_path=args.model_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        threshold=args.threshold,
        match_dist=args.match_dist,
        cluster=not args.no_cluster,
        ann_format=args.ann_format,
    )


if __name__ == "__main__":
    main()
