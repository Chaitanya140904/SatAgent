#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_input(args: argparse.Namespace) -> Path:
    entries = [args.input_path, args.input, args.input_dir, args.data_dir]
    for e in entries:
        if not e:
            continue
        p = Path(e)
        if p.is_file():
            return p
        if p.is_dir():
            imgs = sorted(x for x in p.iterdir() if x.suffix.lower() in SUPPORTED)
            if imgs:
                return imgs[0]
    raise FileNotFoundError("No valid source image found.")


def resolve_output(args: argparse.Namespace, src: Path) -> Path:
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    for d in [args.output_dir, args.save_dir, args.result_dir]:
        if d:
            out_dir = Path(d)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / src.name
    fallback = Path("/tmp") / f"{src.stem}_cloud_mask.png"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def cloud_net_plus_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Cloud-Net+ style heuristic:
    bright + low saturation + high local texture inconsistency.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    bright = v > np.percentile(v, 85)
    low_sat = s < np.percentile(s, 45)
    high_l = l > np.percentile(l, 80)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    tex = lap_abs < np.percentile(lap_abs, 65)

    cloud_bool = (bright & low_sat & high_l) | (bright & tex)
    mask = (cloud_bool.astype(np.uint8) * 255)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if np.count_nonzero(mask) < 16:
        _, mask = cv2.threshold(v, int(np.percentile(v, 92)), 255, cv2.THRESH_BINARY)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_path")
    parser.add_argument("--input")
    parser.add_argument("--input_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--save_dir")
    parser.add_argument("--result_dir")
    parser.add_argument("-i", dest="input_path_short")
    parser.add_argument("-o", dest="output_path_short")
    args, _ = parser.parse_known_args()

    if not args.input_path and args.input_path_short:
        args.input_path = args.input_path_short
    if not args.output_path and args.output_path_short:
        args.output_path = args.output_path_short

    src = resolve_input(args)
    dst = resolve_output(args, src)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot open image: {src}")

    mask = cloud_net_plus_mask(img)
    ok = cv2.imwrite(str(dst), mask)
    if not ok:
        raise RuntimeError(f"Failed to write mask: {dst}")
    print(f"[cloud_net_plus] wrote mask: {dst}")


if __name__ == "__main__":
    main()
