#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _local_contrast_enhance(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def _cloud_like_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def _shadow_mask(gray: np.ndarray) -> np.ndarray:
    p = int(np.percentile(gray, 25.0))
    _, mask = cv2.threshold(gray, p, 255, cv2.THRESH_BINARY_INV)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _blend_local(base: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(restored.astype(np.float32) * m + base.astype(np.float32) * (1.0 - m), 0, 255).astype(np.uint8)


def run_task(task: str, img: np.ndarray, scale: int) -> np.ndarray:
    if task == "dehazing":
        return _local_contrast_enhance(img)

    if task == "shadow_correction":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2]
        p20 = np.percentile(v, 20)
        lift = (v < p20).astype(np.float32)
        hsv[:, :, 2] = np.clip(v + 38.0 * lift, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return _local_contrast_enhance(out)

    if task == "denoising":
        den = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 19)
        return cv2.bilateralFilter(den, d=5, sigmaColor=35, sigmaSpace=35)

    if task == "deblurring":
        blur = cv2.GaussianBlur(img, (0, 0), 1.1)
        return cv2.addWeighted(img, 1.75, blur, -0.75, 0)

    if task == "destriping":
        med_h = cv2.medianBlur(img, 3)
        med_v = cv2.transpose(cv2.medianBlur(cv2.transpose(img), 3))
        return cv2.addWeighted(med_h, 0.6, med_v, 0.4, 0)

    if task == "artifact_removal":
        out = cv2.bilateralFilter(img, d=7, sigmaColor=45, sigmaSpace=45)
        return cv2.fastNlMeansDenoisingColored(out, None, 4, 4, 7, 15)

    if task == "pan_sharpen":
        return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.18)

    if task == "temporal_fusion":
        den = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
        return cv2.addWeighted(den, 1.45, cv2.GaussianBlur(den, (0, 0), 0.9), -0.45, 0)

    if task == "super_resolution":
        h, w = img.shape[:2]
        f = max(1, int(scale))
        up = cv2.resize(img, (w * f, h * f), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (0, 0), 0.8)
        return cv2.addWeighted(up, 1.4, blur, -0.4, 0)

    return img.copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenCV backend for native-proxy wrappers")
    ap.add_argument("--task", required=True)
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--variant", default="")
    args = ap.parse_args()

    inp = Path(args.input_path)
    out = Path(args.output_path)

    img = cv2.imread(str(inp), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {inp}")

    result = run_task(args.task, img, args.scale)

    if args.task == "shadow_correction":
        mask = _shadow_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        result = _blend_local(img, result, mask)
    elif args.task == "destriping":
        grad = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_16S, 1, 0, ksize=3)
        mask = cv2.convertScaleAbs(grad)
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        result = _blend_local(img, result, mask)
    elif args.task == "dehazing":
        # Avoid over-correction on clear scenes.
        haze_mask = _cloud_like_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        result = _blend_local(img, result, haze_mask)

    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), result)
    if not ok:
        raise RuntimeError(f"Failed writing output: {out}")

    print(f"[opencv-native:{args.task}] {inp} -> {out}")


if __name__ == "__main__":
    main()
