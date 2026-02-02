#!/usr/bin/env python3
"""
Evaluate the density estimation model on a dataset: load model, run inference on images, compare
predicted counts to ground truth. Optionally applies style pretuning (grayworld, CLAHE, etc.).
Outputs MSE, MAE, correlation, and optional CSV/plots.

How to run:
  python eval_density.py --model path/to/model.pth --images_dir path/to/images [--annotations_dir path/to/annotations] [--output_dir path] [--style_pretune]

Use --help for all options. Run from synthetic_pipeline_code/ or set PYTHONPATH so density_model is importable.
"""
import os
from pathlib import Path
import argparse
import numpy as np
import cv2
import csv
import torch
from scipy.stats import pearsonr

from density_model import MultiScaleDensityNet, ModelConfig, MultiChannelTransform


# -----------------------------
# Style pretune (to match synthetic look)
# -----------------------------
def _grayworld(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    m = img.reshape(-1, 3).mean(axis=0) + 1e-6
    g = m.mean()
    scale = g / m
    out = (img * scale).clip(0, 255).astype(np.uint8)
    return out

def _clahe_L(img_bgr: np.ndarray, clip=2.2, tile=8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    Lc = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([Lc, a, b]), cv2.COLOR_LAB2BGR)

def _unsharp(img_bgr: np.ndarray, sigma=1.2, amount=0.9, threshold=0) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    k = int(max(3, round(sigma*3)*2+1))
    blur = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    high = img - blur
    if threshold > 0:
        mask = (np.max(np.abs(high), axis=2) > threshold).astype(np.float32)[..., None]
        high = high * mask
    sharp = img + amount * high
    return np.clip(sharp, 0, 255).astype(np.uint8)

def _gamma(img_bgr: np.ndarray, gamma=0.94) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    lut = ((np.arange(256) / 255.0) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)

def style_pretune(
    img_bgr: np.ndarray,
    grayworld=True,
    clahe=True, clahe_clip=2.2, clahe_tile=8,
    unsharp=True, us_sigma=1.2, us_amount=0.9, us_thresh=0,
    gamma=True, gamma_val=0.94
) -> np.ndarray:
    out = img_bgr
    if grayworld: out = _grayworld(out)
    if clahe:     out = _clahe_L(out, clip=clahe_clip, tile=clahe_tile)
    if unsharp:   out = _unsharp(out, sigma=us_sigma, amount=us_amount, threshold=us_thresh)
    if gamma:     out = _gamma(out, gamma_val)
    return out


# -----------------------------
# Model inference
# -----------------------------
@torch.no_grad()
def predict_density_map(model, transform, device, img_bgr: np.ndarray) -> np.ndarray:
    mc = transform(img_bgr)  # (H,W,6)
    x = torch.from_numpy(mc).permute(2, 0, 1).unsqueeze(0)  # (1,6,H,W)
    x = x.to(device).to(memory_format=torch.channels_last)

    devtype = "cuda" if device == "cuda" else "mps" if device == "mps" else "cpu"
    dtype = torch.float16 if devtype in ("cuda", "mps") else torch.float32

    with torch.autocast(device_type=devtype, dtype=dtype):
        y = model(x)  # (1,1,16,16)

    dens = y.squeeze(0).squeeze(0).to(torch.float32).cpu().numpy()  # (16,16)
    return dens


# -----------------------------
# Peak finding with NMS
# -----------------------------
def find_peaks_nms(density_up: np.ndarray,
                   thresh_rel: float = 0.25,
                   nms_radius: int = 10,
                   max_peaks: int = 1000) -> np.ndarray:
    """Local maxima + NMS on upsampled density map. Returns [[y,x], ...]"""
    p99 = np.percentile(density_up, 99) if density_up.size else 1.0
    norm = (density_up / (p99 + 1e-8)).clip(0, 1)
    norm_blur = cv2.GaussianBlur(norm.astype(np.float32), (0, 0), 1.0)
    den8 = (norm_blur * 255).astype(np.uint8)
    dil = cv2.dilate(den8, np.ones((3, 3), np.uint8))
    maxima = (den8 == dil) & (den8 >= int(thresh_rel * 255))

    ys, xs = np.where(maxima)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.int32)

    scores = density_up[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    keep_y, keep_x = [], []
    for y, x, s in zip(ys, xs, scores):
        if len(keep_y) >= max_peaks:
            break
        if len(keep_y) == 0:
            keep_y.append(y); keep_x.append(x); continue
        d2 = (np.array(keep_y) - y) ** 2 + (np.array(keep_x) - x) ** 2
        if np.all(d2 >= (nms_radius ** 2)):
            keep_y.append(y); keep_x.append(x)

    return np.stack([keep_y, keep_x], axis=1).astype(np.int32)


# -----------------------------
# GT parsing (your commented TXT format)
# -----------------------------
def parse_txt_total(txt_path: Path) -> int:
    """
    Reads '# Total walnuts: N' from the file.
    Returns -1 if not found.
    """
    if not txt_path.exists():
        return -1
    total = None
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("# total walnuts"):
                parts = s.split(":")
                if len(parts) >= 2:
                    try:
                        total = int(parts[1].strip())
                    except Exception:
                        pass
                break
    return -1 if total is None else total


# -----------------------------
# Heatmap overlay / circle drawing
# -----------------------------
def overlay_heatmap(img_bgr: np.ndarray, dens_up: np.ndarray, alpha=0.5) -> np.ndarray:
    p99 = np.percentile(dens_up, 99) if dens_up.size else 1.0
    norm = (dens_up / (p99 + 1e-8)).clip(0, 1)
    heat = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)

def draw_circles(img_bgr: np.ndarray, peaks: np.ndarray, color=(0,255,255), radius=0):
    H, W = img_bgr.shape[:2]
    rad = radius if radius > 0 else max(6, int(min(H, W) / 40))
    out = img_bgr.copy()
    for (y, x) in peaks:
        cv2.circle(out, (int(x), int(y)), rad, color, 2)
    return out


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(preds, gts):
    preds = np.array(preds, dtype=np.float64)
    gts = np.array(gts, dtype=np.float64)
    abs_err = np.abs(preds - gts)
    mae = float(abs_err.mean()) if len(abs_err) else float("nan")
    rmse = float(np.sqrt(((preds - gts) ** 2).mean())) if len(abs_err) else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.where(gts > 0, abs_err / gts, np.nan)) * 100.0)
    try:
        r = float(pearsonr(preds, gts)[0]) if len(preds) >= 2 else float("nan")
    except Exception:
        r = float("nan")
    return mae, rmse, mape, r


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate density model with optional pretune + peak NMS")
    ap.add_argument("--test_root", required=True, help="Folder with test/{images,annotations}")
    ap.add_argument("--ckpt", required=True, help="Path to trained model (.pth)")
    ap.add_argument("--out_csv", required=True, help="Where to write per-image CSV")

    # Counting mode
    ap.add_argument("--count_mode", choices=["sum", "peaks"], default="sum",
                    help="Which prediction to score against GT: density sum or peak count")
    # Peaks/NMS
    ap.add_argument("--peak_thresh", type=float, default=0.25, help="Relative threshold vs p99")
    ap.add_argument("--nms_radius", type=int, default=10, help="NMS radius in pixels on full-res density")
    ap.add_argument("--max_peaks", type=int, default=1000, help="Max kept peaks per image")

    # Pretune
    ap.add_argument("--pretune", action="store_true", help="Apply style pretune before inference")

    # Visual outputs
    ap.add_argument("--save_dir", default="", help="If set, saves heatmaps and circles here")
    ap.add_argument("--save_heatmap", action="store_true", help="Save heatmap overlay")
    ap.add_argument("--save_circles", action="store_true", help="Save circle overlay")
    ap.add_argument("--circle_color", type=str, default="0,255,255", help="BGR '0,255,255'")
    ap.add_argument("--circle_radius", type=int, default=0, help="Circle radius (0=auto)")
    args = ap.parse_args()

    test_root = Path(args.test_root)
    img_dir = test_root / "images"
    ann_dir = test_root / "annotations"

    # Device selection - prioritize CUDA
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🖥️ Using device: {device}")

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = MultiScaleDensityNet(cfg).to(device)
    model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = MultiChannelTransform(cfg.patch_size, cfg.max_side)

    # Images
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        img_paths += sorted(img_dir.glob(ext))
    if not img_paths:
        raise SystemExit(f"No images found in {img_dir}")

    # Visual dirs
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Parse circle color
    try:
        bgr = tuple(int(v) for v in args.circle_color.split(","))
        circle_color = (bgr[0], bgr[1], bgr[2])
    except Exception:
        circle_color = (0, 255, 255)

    rows = []
    pred_for_metrics = []
    gts_for_metrics = []

    for i, img_path in enumerate(img_paths, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[{i}/{len(img_paths)}] ❌ read fail {img_path.name}")
            continue

        # optional pretune
        img_in = style_pretune(img_bgr) if args.pretune else img_bgr

        # predict density (16x16), upsample
        dens = predict_density_map(model, transform, device, img_in)
        pred_sum = float(dens.sum())

        H, W = img_bgr.shape[:2]
        dens_up = cv2.resize(dens, (W, H), interpolation=cv2.INTER_CUBIC)

        # peaks
        peaks = find_peaks_nms(
            dens_up,
            thresh_rel=args.peak_thresh,
            nms_radius=args.nms_radius,
            max_peaks=args.max_peaks,
        )
        pred_peaks = int(peaks.shape[0])

        # GT
        txt = ann_dir / f"{img_path.stem}.txt"
        gt = parse_txt_total(txt)
        gt_src = f"txt_total:{txt.name}" if gt >= 0 else "none"
        if gt < 0:
            print(f"[{i}/{len(img_paths)}] {img_path.name}: ⚠️ missing/invalid GT, skipping metrics row")
            # still store row with gt=-1
        else:
            # Decide which prediction is scored
            chosen = pred_sum if args.count_mode == "sum" else pred_peaks
            pred_for_metrics.append(chosen)
            gts_for_metrics.append(gt)

        # Save visuals if requested
        if save_dir:
            if args.save_heatmap:
                hm = overlay_heatmap(img_in, dens_up, alpha=0.5)
                cv2.imwrite(str(save_dir / f"{img_path.stem}_heat.png"), hm)
            if args.save_circles:
                circles = draw_circles(img_in, peaks, circle_color, args.circle_radius)
                # annotate counts
                cv2.putText(circles, f"sum={pred_sum:.1f} peaks={pred_peaks:d}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(circles, f"sum={pred_sum:.1f} peaks={pred_peaks:d}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                if gt >= 0:
                    cv2.putText(circles, f"GT={gt}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(circles, f"GT={gt}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90,255,90), 2, cv2.LINE_AA)
                cv2.imwrite(str(save_dir / f"{img_path.stem}_circles.png"), circles)

        # CSV row (record both kinds of preds)
        row = {
            "image": img_path.name,
            "pred_sum": f"{pred_sum:.4f}",
            "pred_peaks": int(pred_peaks),
            "gt_count": f"{gt:.4f}" if gt >= 0 else "",
            "abs_err_sum": f"{abs(pred_sum - gt):.4f}" if gt >= 0 else "",
            "abs_err_peaks": f"{abs(pred_peaks - gt):.4f}" if gt >= 0 else "",
            "gt_source": gt_src,
        }
        rows.append(row)

        print(f"[{i}/{len(img_paths)}] {img_path.name}: sum={pred_sum:.2f} peaks={pred_peaks}"
              + (f" gt={gt}" if gt >= 0 else ""))

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Metrics summary (only images with valid GT)
    if len(gts_for_metrics) > 0:
        mae, rmse, mape, r = compute_metrics(pred_for_metrics, gts_for_metrics)
        print("\n🔎 Summary on test set")
        print(f"  Mode: {args.count_mode}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE_%: {mape:.4f}")
        print(f"  Pred_mean: {np.mean(pred_for_metrics):.4f}")
        print(f"  True_mean: {np.mean(gts_for_metrics):.4f}")
        print(f"  PearsonR: {r:.4f}")
        print(f"  N: {len(gts_for_metrics)}")
    else:
        print("\nNo valid images with GT were found for metrics.")

    print(f"\n🧾 Wrote per-image results to: {out_csv}")
    if save_dir:
        print(f"🖼️  Visuals saved in: {save_dir}")


if __name__ == "__main__":
    main()