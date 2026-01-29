#!/usr/bin/env python3
"""
Visualize density model predictions: load a trained density model, run on images, and save density
maps or overlay visualizations. Useful for debugging and presentation.

How to run:
  python visualize_density.py --model path/to/model.pth --image path/to/image.jpg [--output_dir path]

Use --help for options. Run from synthetic_pipeline_code/ so density_model is importable.
"""
import os
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch

from density_model import MultiScaleDensityNet, ModelConfig, MultiChannelTransform


# -----------------------------
# Optional style pre-tuning (to match synthetic look)
# -----------------------------
def _grayworld(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    m = img.reshape(-1, 3).mean(axis=0) + 1e-6
    g = m.mean()
    scale = g / m
    out = (img * scale).clip(0, 255).astype(np.uint8)
    return out

def _clahe_L(img_bgr: np.ndarray, clip=2.0, tile=8) -> np.ndarray:
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

def _gamma(img_bgr: np.ndarray, gamma=0.95) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    lut = ((np.arange(256) / 255.0) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)

def style_pretune(img_bgr: np.ndarray,
                  grayworld=True, clahe=True, clahe_clip=2.2, clahe_tile=8,
                  unsharp=True, us_sigma=1.2, us_amount=0.9, us_thresh=0,
                  gamma=True, gamma_val=0.94) -> np.ndarray:
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

    dens = y.squeeze(0).squeeze(0).to(torch.float32).cpu().numpy()
    return dens  # (16,16)


# -----------------------------
# Peak finding with NMS
# -----------------------------
def find_peaks_nms(density_up: np.ndarray,
                   thresh_rel: float = 0.25,
                   nms_radius: int = 9,
                   max_peaks: int = 500) -> np.ndarray:
    """
    Local maxima with simple NMS on the *upsampled* density map.
    - thresh_rel applied to 99th-percentile normalized map.
    - NMS keeps strong peaks separated by >= nms_radius pixels.
    Returns [[y, x], ...]
    """
    # Normalize by p99 for stability across images
    p99 = np.percentile(density_up, 99) if density_up.size else 1.0
    norm = (density_up / (p99 + 1e-8)).clip(0, 1)

    # Smooth a little to avoid tiny noise peaks
    norm_blur = cv2.GaussianBlur(norm.astype(np.float32), (0, 0), 1.0)

    # Initial local maxima
    den8 = (norm_blur * 255).astype(np.uint8)
    dil = cv2.dilate(den8, np.ones((3, 3), np.uint8))
    maxima = (den8 == dil) & (den8 >= int(thresh_rel * 255))

    ys, xs = np.where(maxima)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.int32)

    scores = density_up[ys, xs]  # use original (not normalized) as score
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    # NMS in image plane
    keep_y, keep_x = [], []
    for y, x, s in zip(ys, xs, scores):
        if len(keep_y) >= max_peaks:
            break
        if len(keep_y) == 0:
            keep_y.append(y); keep_x.append(x); continue
        d2 = (np.array(keep_y) - y) ** 2 + (np.array(keep_x) - x) ** 2
        if np.all(d2 >= (nms_radius ** 2)):
            keep_y.append(y); keep_x.append(x)

    coords = np.stack([keep_y, keep_x], axis=1).astype(np.int32)
    return coords


# -----------------------------
# Ground-truth loader (your commented TXT format)
# -----------------------------
def parse_txt_total(txt_path: Path) -> int:
    total = None
    if not txt_path.exists():
        return -1
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("# total walnuts"):
                # "# Total walnuts: 62"
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        total = int(parts[1].strip())
                    except Exception:
                        pass
                break
    return -1 if total is None else total


# -----------------------------
# Heatmap overlay
# -----------------------------
def overlay_heatmap(img_bgr: np.ndarray, dens_up: np.ndarray, alpha=0.5) -> np.ndarray:
    p99 = np.percentile(dens_up, 99) if dens_up.size else 1.0
    norm = (dens_up / (p99 + 1e-8)).clip(0, 1)
    heat = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize walnuts with circles (pred + optional GT)")
    ap.add_argument("--images_dir", required=True, help="Folder with test images")
    ap.add_argument("--ckpt", required=True, help="Path to trained model (.pth)")
    ap.add_argument("--out_dir", default="./viz_walnuts", help="Where to save outputs")

    # Peaks & drawing
    ap.add_argument("--peak_thresh", type=float, default=0.25, help="Relative threshold (0-1) vs p99")
    ap.add_argument("--nms_radius", type=int, default=10, help="NMS radius (pixels) on full-res map")
    ap.add_argument("--max_peaks", type=int, default=500, help="Max circles per image")
    ap.add_argument("--circle_radius", type=int, default=0, help="Circle radius px (0 = auto)")
    ap.add_argument("--circle_color", type=str, default="0,255,255", help="BGR like '0,255,255'")

    # Extras
    ap.add_argument("--save_heatmap", action="store_true", help="Also save heatmap overlay")
    ap.add_argument("--save_density", action="store_true", help="Also save raw density .npy")
    ap.add_argument("--show_gt", action="store_true", help="Try overlaying GT count from matching TXT")

    # Pretune
    ap.add_argument("--pretune", action="store_true", help="Apply style pre-tuning before inference")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
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

    # Gather images
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_paths += sorted(Path(args.images_dir).glob(ext))
    if not img_paths:
        raise SystemExit(f"No images found in {args.images_dir}")

    # Parse circle color
    try:
        bgr = tuple(int(v) for v in args.circle_color.split(","))
        circle_color = (bgr[0], bgr[1], bgr[2])
    except Exception:
        circle_color = (0, 255, 255)

    for i, img_path in enumerate(img_paths, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[{i}/{len(img_paths)}] ❌ Failed to read {img_path.name}")
            continue

        # Optional pretune
        img_in = style_pretune(img_bgr) if args.pretune else img_bgr

        # Predict density (16x16)
        dens = predict_density_map(model, transform, device, img_in)
        pred_sum = float(dens.sum())

        # Upsample to image size
        H, W = img_bgr.shape[:2]
        dens_up = cv2.resize(dens, (W, H), interpolation=cv2.INTER_CUBIC)

        # Peaks with NMS
        peaks = find_peaks_nms(
            dens_up,
            thresh_rel=args.peak_thresh,
            nms_radius=args.nms_radius,
            max_peaks=args.max_peaks,
        )

        # Draw circles
        vis = img_in.copy()  # what the model "saw"
        # auto radius: ~1/40 of min dimension, capped
        rad = args.circle_radius if args.circle_radius > 0 else max(6, int(min(H, W) / 40))
        for (y, x) in peaks:
            cv2.circle(vis, (int(x), int(y)), rad, circle_color, 2)

        # Overlay info text
        cv2.putText(vis, f"pred_sum={pred_sum:.1f} | peaks={len(peaks)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"pred_sum={pred_sum:.1f} | peaks={len(peaks)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        # If show_gt: try to read *_samebasename*.txt in sibling annotations folder or same folder
        if args.show_gt:
            # Try common locations: images_dir/../annotations or images_dir
            cand_txt = []
            cand_txt.append(img_path.with_suffix(".txt"))
            cand_txt.append(img_path.parent.parent / "annotations" / f"{img_path.stem}.txt")
            gt_total = -1
            src = None
            for t in cand_txt:
                n = parse_txt_total(t)
                if n >= 0:
                    gt_total = n
                    src = t.name
                    break
            if gt_total >= 0:
                msg = f"GT={gt_total} ({src})"
                cv2.putText(vis, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90,255,90), 2, cv2.LINE_AA)

        # Save main visualization
        out_path = out_dir / f"{img_path.stem}_circled.png"
        cv2.imwrite(str(out_path), vis)

        # Optional heatmap overlay
        if args.save_heatmap:
            hm = overlay_heatmap(img_in, dens_up, alpha=0.5)
            hm_path = out_dir / f"{img_path.stem}_heat.png"
            cv2.imwrite(str(hm_path), hm)

        # Optional density dump
        if args.save_density:
            npy_path = out_dir / f"{img_path.stem}_density.npy"
            np.save(npy_path, dens)

        print(f"[{i}/{len(img_paths)}] {img_path.name}: pred_sum={pred_sum:.1f}, peaks={len(peaks)} -> {out_path}")

    print(f"\n✅ Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()